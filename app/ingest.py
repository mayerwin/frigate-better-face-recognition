"""Background poller.

Every poll_interval it lists Frigate's train crops, and for each crop it has not
seen before: fetches the image, embeds it (SCRFD + ArcFace), classifies it
against the human-built galleries, stores a small thumbnail + the decision.

Per the human-in-the-loop policy, ingestion NEVER writes to Frigate on its own
unless auto_label is explicitly enabled (off by default), in which case a
high-confidence match is assigned (which also trains Frigate). Auto-rejected
crops are merely recorded + hidden from the review queue; they are not deleted
from Frigate here (a human "not a face" tap is what deletes, via the API).
"""
from __future__ import annotations

import asyncio
import io
import logging
import time

log = logging.getLogger("bfr.ingest")


def _make_thumb(image_bytes: bytes, size: int) -> bytes:
    from PIL import Image

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img.thumbnail((size, size))
        out = io.BytesIO()
        img.save(out, format="WEBP", quality=82, method=4)
        return out.getvalue()
    except Exception:
        # Undecodable crop: store an empty thumb so the row is still recorded and
        # seen() dedupes it, instead of re-fetching + re-embedding it every poll.
        return b""


class Ingestor:
    def __init__(self, cfg, store, embedder, frigate, *, is_active=None, model_ttl=180.0):
        self.cfg = cfg
        self.store = store
        self.embedder = embedder
        self.frigate = frigate
        # is_active() -> True while the Review tab is open; when given, ingest only
        # runs then (or when auto_label is on) so the models can unload otherwise.
        self.is_active = is_active
        self.model_ttl = model_ttl
        self._task = None
        self._backfill_task = None
        self._stop = asyncio.Event()
        self.last_run = 0.0
        self.last_error = ""
        self.ingested = 0

    def start(self):
        self._stop.clear()
        self._task = asyncio.create_task(self._loop())
        self._backfill_task = asyncio.create_task(self.backfill_blur())

    async def stop(self):
        self._stop.set()
        tasks = [t for t in (self._task, self._backfill_task) if t]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def backfill_blur(self):
        """One-shot: fill blur_score for crops ingested before the blur feature,
        by re-fetching + re-measuring the ones still present in Frigate."""
        try:
            missing = await asyncio.to_thread(self.store.crops_missing_blur)
        except Exception:
            return
        for cid, fn in missing:
            if self._stop.is_set():
                break
            try:
                img = await self.frigate.fetch_crop(fn)
                _, _, blur, _ = await asyncio.to_thread(self.embedder.embed, img)
                await asyncio.to_thread(self.store.update_blur, cid, blur)
            except Exception:
                # crop gone (FIFO-evicted) or undecodable: write 0 so we stop retrying
                try:
                    await asyncio.to_thread(self.store.update_blur, cid, 0.0)
                except Exception:
                    pass

    async def _loop(self):
        while not self._stop.is_set():
            try:
                if await self._should_ingest():
                    await self.run_once()
                    self.last_error = ""
                await self._maybe_release_idle()
            except Exception as e:  # never let the loop die
                self.last_error = str(e)
                log.warning("ingest cycle failed: %s", e)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.cfg.poll_interval)
            except asyncio.TimeoutError:
                pass

    async def _should_ingest(self) -> bool:
        """Ingest new crops only while someone is reviewing (the UI's Review tab is
        open) or when auto_label is on. Otherwise stay idle so the face models can
        unload and the background footprint is just the web server. Frigate keeps a
        rolling buffer of recent face attempts (save_attempts, default 200, oldest
        deleted first), so on-demand scanning misses nothing as long as you review
        within that window."""
        if self.is_active is None:
            return True
        if self.is_active():
            return True
        settings = await asyncio.to_thread(self.store.get_settings)
        return bool(settings.get("auto_label", False))

    async def _maybe_release_idle(self):
        e = self.embedder
        if getattr(e, "loaded", False) and e.idle_seconds() > self.model_ttl:
            idle = e.idle_seconds()
            await asyncio.to_thread(e.release)
            log.info("released face models after %.0fs idle", idle)

    async def run_once(self) -> int:
        crops = await self.frigate.list_train()
        # Read settings + rebuild the galleries ONCE per cycle, off the event loop,
        # so a large library never stalls the API. Galleries only contain human
        # decisions, so they don't change within a single ingest cycle.
        settings = await asyncio.to_thread(self.store.get_settings)
        pos = await asyncio.to_thread(self.store.positive_gallery)
        neg = await asyncio.to_thread(self.store.negative_gallery)
        n = 0
        for c in crops:
            if c is None:
                continue
            if await asyncio.to_thread(self.store.seen, c.filename):
                continue
            try:
                await self._ingest_one(c, settings, pos, neg)
                n += 1
            except Exception as e:
                log.warning("ingest %s failed: %s", c.filename, e)
        self.last_run = time.time()
        self.ingested += n
        return n

    async def _ingest_one(self, c, settings, pos, neg):
        from .classifier import classify

        img = await self.frigate.fetch_crop(c.filename)
        has_face, det_score, blur_score, emb = await asyncio.to_thread(self.embedder.embed, img)
        d = classify(
            emb,
            has_face,
            pos,
            neg,
            match_threshold=float(settings.get("match_threshold", 0.5)),
            reject_threshold=float(settings.get("reject_threshold", 0.5)),
            auto_reject=bool(settings.get("auto_reject", True)),
            auto_label=bool(settings.get("auto_label", False)),
            blur_score=blur_score,
            blur_threshold=float(settings.get("blur_threshold", 0.0)),
        )
        thumb = await asyncio.to_thread(_make_thumb, img, self.cfg.thumb_size)
        await asyncio.to_thread(
            self.store.add_crop,
            frigate_id=c.filename,
            camera="",
            event_ts=c.event_ts,
            det_score=det_score,
            blur_score=blur_score,
            has_face=has_face,
            embedding=emb,
            thumb=thumb,
            status=d.status,
            reason=d.reason,
            person_id=d.person_id,
            suggested_person_id=d.suggested_person_id,
            match_score=d.score,
            source_path=f"train/{c.filename}",
        )
        # Opt-in only: auto-label assigns + trains Frigate without a human click.
        if d.status == "auto_labeled" and d.person_id and self.cfg.frigate_writeback:
            person = await asyncio.to_thread(self.store.get_person, d.person_id)
            if person:
                try:
                    await self.frigate.assign(c.filename, person["name"])
                except Exception as e:
                    log.warning("auto-label assign of %s failed: %s", c.filename, e)
