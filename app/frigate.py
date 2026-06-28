"""Thin async client for Frigate's face API (verified against Frigate v0.17.1).

Talks to Frigate's internal HTTP API (default http://frigate:5000), which is
unauthenticated, so there is no login flow. Only the face endpoints we use are
wrapped:

  GET  /api/faces                         -> {name: [filenames]} including "train"
  GET  /clips/faces/<dir>/<file>          -> crop image bytes
  POST /api/faces/train/<name>/classify   -> move a train crop into <name>/ and
                                             retrain Frigate's recogniser
                                             body: {"training_file": "<filename>"}
  POST /api/faces/<name>/delete           -> delete crop(s) from <name>/ and clear
                                             them from Frigate's vector DB
                                             body: {"ids": ["<filename>", ...]}

Train-crop filenames look like:
  <event_ts>-<rand>-<crop_ts>-<label>-<score>.webp
where a Frigate **event id is itself "<event_ts>-<rand>"** (a timestamp and a
short random token joined by '-', e.g. "1782260557.077255-8eqa9h"). So the event
id spans the first TWO dash-delimited fields; `TrainCrop.event_id` rejoins them,
which is the form the snapshot endpoint (`/api/events/<id>/snapshot.jpg`) needs.
Frigate replaces '-' inside a label with '_', so the remaining fields are
unambiguous. Parsing is best-effort: an unrecognised filename is still ingested
(only the metadata is missing), never skipped.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional
from urllib.parse import quote

import httpx

_IMG_EXTS = (".webp", ".png", ".jpg", ".jpeg")

_FN = re.compile(
    r"^(?P<ets>\d+(?:\.\d+)?)-(?P<eid>[^-]+)-(?P<cts>\d+(?:\.\d+)?)-"
    r"(?P<label>[^-]+)-(?P<score>[\d.]+)\.(?:webp|png|jpg|jpeg)$",
    re.IGNORECASE,
)


@dataclass
class TrainCrop:
    filename: str
    event_id: str = ""  # full Frigate event id "<event_ts>-<rand>" (snapshot lookups)
    event_ts: float = 0.0
    crop_ts: float = 0.0
    label: str = ""
    score: float = 0.0


def label_key(name: str) -> str:
    """Normalize a person name for matching that survives Frigate's train-crop
    mangling. Frigate rewrites '-' inside a label to '_' (the label field in a
    crop filename is '-'-delimited), so a person enrolled as "Jan-Peter" appears
    as "Jan_Peter" in a crop's parsed label. Folding '-' and '_' together (and
    lower-casing) lets both map to the same key, so reconciling a parsed label
    against the real Frigate person reuses the existing folder instead of
    creating a spurious duplicate. Two people who differ only by '-' vs '_' are
    indistinguishable in crop filenames anyway, so collapsing them is the best
    (and only) recoverable behaviour."""
    return name.strip().lower().replace("-", "_")


def parse_filename(fn: str) -> TrainCrop:
    """Best-effort parse; always returns a TrainCrop (filename is the unique id)."""
    m = _FN.match(fn)
    if not m:
        return TrainCrop(filename=fn)
    try:
        return TrainCrop(
            filename=fn,
            # A Frigate event id is "<event_ts>-<rand>"; rejoin the two raw fields
            # exactly as written (don't reformat the float -> precision drift).
            event_id=f"{m.group('ets')}-{m.group('eid')}",
            event_ts=float(m.group("ets")),
            crop_ts=float(m.group("cts")),
            label=m.group("label"),
            score=float(m.group("score")),
        )
    except (ValueError, TypeError):
        return TrainCrop(filename=fn)


class FrigateClient:
    def __init__(self, base_url: str, *, verify_tls: bool = False, timeout: float = 20.0):
        self.base = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            verify=verify_tls, timeout=timeout, follow_redirects=True
        )

    async def aclose(self):
        await self._client.aclose()

    async def version(self) -> str:
        try:
            r = await self._client.get(f"{self.base}/api/version", timeout=5.0)
            return r.text.strip()
        except Exception:
            return ""

    async def list_faces(self) -> dict:
        r = await self._client.get(f"{self.base}/api/faces")
        r.raise_for_status()
        data = r.json()
        return data if isinstance(data, dict) else {}

    async def list_train(self) -> list[TrainCrop]:
        faces = await self.list_faces()
        return [parse_filename(fn) for fn in faces.get("train", [])]

    async def fetch_crop(self, filename: str, folder: str = "train") -> bytes:
        r = await self._client.get(f"{self.base}/clips/faces/{quote(folder)}/{quote(filename)}")
        r.raise_for_status()
        return r.content

    async def event_snapshot(self, event_id: str, *, bbox: bool = True) -> bytes:
        """Full-frame snapshot for the tracked-object event a face crop came
        from -- the whole scene, not just the face crop (Frigate's own faces
        page shows this when you click a recognition's magnifier). The face
        filename's middle field is that event id.

        `bbox` overlays the object box while the event is still in progress;
        once it has ended Frigate serves the snapshot per its config and the
        params are ignored. Returns b"" when Frigate has no snapshot for the
        event (snapshots disabled for the camera, or the event already aged
        out), so callers can surface "unavailable" rather than a 500."""
        params = {"bbox": 1} if bbox else None
        r = await self._client.get(
            f"{self.base}/api/events/{quote(event_id)}/snapshot.jpg", params=params
        )
        if r.status_code == 404:
            return b""
        r.raise_for_status()
        return r.content

    async def assign(self, filename: str, name: str) -> dict:
        """Move train/<filename> into <name>/ and retrain Frigate."""
        r = await self._client.post(
            f"{self.base}/api/faces/train/{quote(name)}/classify",
            json={"training_file": filename},
        )
        r.raise_for_status()
        return r.json()

    async def delete_crop(self, filename: str, folder: str = "train") -> dict:
        """Delete a crop from <folder>/ (also clears Frigate's vector DB)."""
        r = await self._client.post(
            f"{self.base}/api/faces/{quote(folder)}/delete",
            json={"ids": [filename]},
        )
        r.raise_for_status()
        return r.json()

    async def list_person_names(self) -> list:
        """Known people enrolled in Frigate (the face folders, minus 'train')."""
        faces = await self.list_faces()
        return sorted((k for k in faces.keys() if k != "train"), key=str.lower)

    async def rename_person(self, old: str, new: str) -> dict:
        r = await self._client.put(
            f"{self.base}/api/faces/{quote(old)}/rename", json={"new_name": new}
        )
        r.raise_for_status()
        return r.json()

    async def delete_person(self, name: str) -> dict:
        """Delete a whole Frigate person folder by deleting all of its images
        (Frigate removes the folder when its last image is deleted)."""
        faces = await self.list_faces()
        folder = next((k for k in faces if k != "train" and k.lower() == name.lower()), name)
        ids = faces.get(folder, [])
        if not ids:
            return {"success": True, "message": "no images"}
        r = await self._client.post(
            f"{self.base}/api/faces/{quote(folder)}/delete", json={"ids": ids}
        )
        r.raise_for_status()
        return r.json()

    async def login_ok(self, user: str, password: str) -> bool:
        """Validate credentials against Frigate's own login (so the companion can
        piggyback on Frigate's user database). 200 = valid, 401 = wrong."""
        try:
            r = await self._client.post(
                f"{self.base}/api/login",
                json={"user": user, "password": password},
                timeout=10.0,
            )
            return r.status_code == 200
        except Exception:
            return False
