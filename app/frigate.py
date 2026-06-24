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
  <event_ts>-<event_id>-<crop_ts>-<label>-<score>.webp
Frigate replaces '-' inside a label with '_', so the five fields are
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
    event_id: str = ""
    event_ts: float = 0.0
    crop_ts: float = 0.0
    label: str = ""
    score: float = 0.0


def parse_filename(fn: str) -> TrainCrop:
    """Best-effort parse; always returns a TrainCrop (filename is the unique id)."""
    m = _FN.match(fn)
    if not m:
        return TrainCrop(filename=fn)
    try:
        return TrainCrop(
            filename=fn,
            event_id=m.group("eid"),
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
