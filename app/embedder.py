"""InsightFace SCRFD (detect) + ArcFace (recognise) on CPU, lazy-loaded, plus an
eDifFIQA face-image-quality scorer.

Uses the maintained `insightface` package with the buffalo_l model pack:
  * SCRFD-10G detector  -> verifies a real face is present (kills non-faces such
    as the infamous car wheel) and provides the 5 keypoints used for alignment
  * a ResNet50 ArcFace recogniser -> a 512-d L2-normalised embedding for the
    largest detected face, used for person matching and reject clustering.

Face quality is scored with eDifFIQA (https://github.com/yakhyo/face-image-quality-assessment),
a SOTA no-reference FIQA model that ranks #1 on the NIST FATE Quality leaderboard
(with the `l` variant). It predicts *recognition utility* from the aligned face,
so unlike a pixel-sharpness heuristic it correctly scores tiny, dark, off-angle or
low-detail crops as poor. It is a small extra ONNX run on CPU via onnxruntime
(already a dependency), downloaded + cached once next to the InsightFace models.

Heavy (~0.5 GB resident, ~100-300 ms/crop on CPU) but strictly event-driven, so
idle cost is ~0. Models auto-download once into MODEL_HOME and are cached.

The models are loaded lazily on first use so the web server starts instantly and
unit tests can run without the models present.
"""
from __future__ import annotations

import ctypes
import gc
import hashlib
import io
import logging
import os
import threading
import time
import urllib.request
from typing import Optional, Tuple

import numpy as np

log = logging.getLogger("bfr.embedder")

# eDifFIQA ONNX weights, published as GitHub release assets (MIT-licensed repo;
# weights from the upstream eDifFIQA authors). sha256 pins the variants we ship +
# validate; an unknown variant downloads without a checksum (logged).
_FIQA_BASE = "https://github.com/yakhyo/face-image-quality-assessment/releases/download/weights/"
_FIQA_SHA = {
    "t": "7a83be63e6583ec5800fa0762e219b65b4b9f1721c4d210bfe2f16c0478832bb",
    "l": "72238239298cf645d3f5954d657b4aca7b64fd25bc808c0260778386da9b00a1",
}


class Embedder:
    def __init__(self, model_name: str = "buffalo_l", det_size: int = 640,
                 det_thresh: float = 0.5, model_root: Optional[str] = None,
                 fiqa_variant: str = "l"):
        self.model_name = model_name
        self.det_size = (det_size, det_size)
        self.det_thresh = det_thresh  # lower = catch more borderline/blurry faces (+ more junk)
        self.model_root = model_root
        self.fiqa_variant = (fiqa_variant or "l").strip().lower()
        self._app = None
        self._init_lock = threading.Lock()
        self._infer_lock = threading.Lock()  # insightface pre/post-proc is not reentrant
        self._fiqa = None  # (session, input_name, output_name) once resolved
        self._fiqa_lock = threading.Lock()
        self._last_use = 0.0  # time.monotonic() of the last embed, for idle release

    def _ensure(self):
        if self._app is None:
            with self._init_lock:
                if self._app is None:
                    from insightface.app import FaceAnalysis

                    kwargs = {}
                    if self.model_root:
                        kwargs["root"] = self.model_root
                    app = FaceAnalysis(
                        name=self.model_name,
                        providers=["CPUExecutionProvider"],
                        allowed_modules=["detection", "recognition"],
                        **kwargs,
                    )
                    app.prepare(ctx_id=-1, det_size=self.det_size, det_thresh=self.det_thresh)
                    self._app = app
                    log.info("insightface %s ready (CPU)", self.model_name)
        return self._app

    def warmup(self):
        """Force model load (e.g. at startup, off the request path)."""
        self._ensure()
        self._fiqa_session()

    def embed(self, image_bytes: bytes) -> Tuple[bool, float, Optional[float], Optional[np.ndarray]]:
        """Return (has_face, det_score, quality, normed_embedding) for the largest
        face. (False, 0.0, None, None) means SCRFD found no face.

        quality is a 0-100 eDifFIQA face-image-quality score (higher = more usable
        for recognition), comparable across crops, or None if the quality model is
        unavailable.
        """
        from PIL import Image, ImageOps

        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            log.debug("undecodable image: %s", e)
            return False, 0.0, None, None

        # Frigate crops are TIGHT face boxes; SCRFD needs margin/context to detect
        # a face that fills the frame (a tight crop alone yields zero detections),
        # so pad with a black border before detection. Verified: 69x86 / 78x99 face
        # crops go from no-detection to det_score ~0.8 once padded, while a genuine
        # non-face still yields no detection. The padding only affects detection +
        # the landmark-based alignment; the embedding remains correct.
        m = max(img.size) // 2
        if m > 0:
            img = ImageOps.expand(img, border=m, fill=(0, 0, 0))

        # insightface expects BGR (cv2 convention)
        arr = np.asarray(img)[:, :, ::-1].copy()
        app = self._ensure()
        self._last_use = time.monotonic()
        with self._infer_lock:
            faces = app.get(arr)
        if not faces:
            return False, 0.0, None, None

        def _area(f):
            x1, y1, x2, y2 = f.bbox
            return float((x2 - x1) * (y2 - y1))

        f = max(faces, key=_area)
        det_score = float(getattr(f, "det_score", 0.0))
        quality = self._quality(arr, f)
        emb = getattr(f, "normed_embedding", None)
        if emb is None:
            emb = getattr(f, "embedding", None)
        if emb is None:
            return True, det_score, quality, None
        emb = np.asarray(emb, dtype=np.float32).reshape(-1)
        n = float(np.linalg.norm(emb))
        if n > 0:
            emb = emb / n
        return True, det_score, quality, emb

    # ----------------------------------------------------------- idle lifecycle
    @property
    def loaded(self) -> bool:
        return self._app is not None

    def idle_seconds(self) -> float:
        return time.monotonic() - self._last_use

    def release(self):
        """Drop the loaded models so an idle process keeps only the web server's
        footprint; they reload lazily on the next embed(). malloc_trim returns the
        freed arenas to the OS (glibc otherwise retains them, so RSS would not fall)."""
        with self._init_lock, self._fiqa_lock:
            self._app = None
            self._fiqa = None
        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

    # ---------------------------------------------------------------- quality
    def _quality(self, arr, face) -> Optional[float]:
        """eDifFIQA learned face image quality (0-100, higher = more usable for
        recognition) on the aligned 112x112 crop. Because it predicts recognition
        utility rather than pixel sharpness, it correctly scores tiny, dark or
        low-detail faces as poor where a Laplacian/contrast heuristic does not.
        None if the quality model is unavailable."""
        sess, iname, oname = self._fiqa_session()
        if sess is None:
            return None
        try:
            import cv2
            from insightface.utils import face_align

            aligned = face_align.norm_crop(arr, landmark=face.kps, image_size=112)  # BGR uint8
            # eDifFIQA preprocessing: RGB, (x-127.5)/127.5 -> [-1,1], CHW, NCHW.
            blob = cv2.dnn.blobFromImage(aligned, 1.0 / 127.5, (112, 112),
                                         (127.5, 127.5, 127.5), swapRB=True)
            raw = float(np.squeeze(sess.run([oname], {iname: blob})[0]))
            return round(max(0.0, min(100.0, raw * 100.0)), 1)
        except Exception as e:
            log.debug("FIQA scoring failed: %s", e)
            return None

    def _fiqa_session(self):
        """Lazily download (once) and load the eDifFIQA ONNX quality model. Returns
        (session, input_name, output_name); session is None if the model could not
        be obtained, in which case quality scoring degrades to unavailable."""
        if self._fiqa is not None:
            return self._fiqa
        with self._fiqa_lock:
            if self._fiqa is not None:
                return self._fiqa
            try:
                import onnxruntime as ort

                root = self.model_root or os.path.join(
                    os.path.expanduser("~"), ".insightface", "models")
                os.makedirs(root, exist_ok=True)
                path = os.path.join(root, f"ediffiqa_{self.fiqa_variant}.onnx")
                if not os.path.exists(path):
                    self._download_fiqa(self.fiqa_variant, path)
                sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
                self._fiqa = (sess, sess.get_inputs()[0].name, sess.get_outputs()[0].name)
                log.info("eDifFIQA(%s) quality model ready (CPU)", self.fiqa_variant)
            except Exception as e:
                log.warning("eDifFIQA model unavailable (%s); quality scoring disabled", e)
                self._fiqa = (None, None, None)
        return self._fiqa

    @staticmethod
    def _download_fiqa(variant: str, path: str):
        url = f"{_FIQA_BASE}ediffiqa_{variant}.onnx"
        tmp = f"{path}.part"
        log.info("downloading eDifFIQA(%s) from %s ...", variant, url)
        urllib.request.urlretrieve(url, tmp)
        want = _FIQA_SHA.get(variant)
        if want:
            h = hashlib.sha256()
            with open(tmp, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            if h.hexdigest() != want:
                os.remove(tmp)
                raise ValueError(f"eDifFIQA({variant}) checksum mismatch")
        else:
            log.warning("eDifFIQA(%s) has no pinned checksum; downloaded unverified", variant)
        os.replace(tmp, path)
