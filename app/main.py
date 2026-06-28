"""FastAPI app: the review UI + JSON API on top of Frigate's face library.

`create_app()` is a factory that takes its dependencies (store, embedder,
frigate client) injected, so tests can wire in fakes and skip the background
poller. `build_default_app()` does the production wiring from env config.

Human-in-the-loop policy, enforced here:
  * assigning a name to a crop is the ONLY way a person label reaches Frigate,
    and it always requires this explicit call (a human click in the UI);
  * rejecting a crop ("not a face") records the embedding in the negative
    gallery and deletes the crop from Frigate's train folder;
  * the classifier's auto decisions only triage what the UI shows.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from typing import Optional
from urllib.parse import quote

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import auth as authmod
from . import nginx_inject
from .classifier import classify
from .config import TUNABLE_KEYS, Config
from .frigate import label_key, parse_filename
from .ingest import Ingestor
from .inject_assets import INJECT_JS

log = logging.getLogger("bfr.api")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


class AssignBody(BaseModel):
    person_id: Optional[int] = None
    name: Optional[str] = None


class PersonBody(BaseModel):
    name: str


class SettingBody(BaseModel):
    key: str
    value: object


class LoginBody(BaseModel):
    user: str = ""
    password: str = ""


class CropFileBody(BaseModel):
    filename: str


def _coerce_setting(key: str, value):
    if key in ("auto_reject", "auto_label"):
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("1", "true", "yes", "on")
    if key in ("match_threshold", "reject_threshold"):
        v = float(value)
        return max(0.0, min(1.0, v))
    if key == "blur_threshold":
        return max(0.0, float(value))
    return value


def create_app(cfg: Config, store, embedder, frigate, *, run_ingest: bool = True) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        store.seed_settings(cfg.tunable_defaults())

        def is_active():  # the Review tab is open if it polled /api/review recently
            return (time.monotonic() - app.state.last_review_view) < cfg.ui_active_window

        ingestor = Ingestor(cfg, store, embedder, frigate,
                            is_active=is_active, model_ttl=cfg.model_idle_ttl)
        app.state.ingestor = ingestor
        if run_ingest:
            ingestor.start()
        # frigate-ext: make Frigate's own nginx inject our /faces button + route
        # /__betterfaces/* back here. Background task -> never blocks startup; it is
        # best-effort and self-heals across Frigate restarts. Skipped under tests
        # (run_ingest=False), where Docker is absent.
        nginx_task = asyncio.create_task(nginx_inject.run()) if run_ingest else None
        try:
            yield
        finally:
            if nginx_task is not None:
                nginx_task.cancel()
                try:
                    await nginx_task
                except (asyncio.CancelledError, Exception):
                    pass
            if run_ingest:
                await ingestor.stop()
            try:
                await frigate.aclose()
            except Exception:
                pass
            store.checkpoint()

    app = FastAPI(title="frigate-better-face-recognition", version="0.1.0", lifespan=lifespan)
    app.state.cfg = cfg
    app.state.store = store
    app.state.embedder = embedder
    app.state.frigate = frigate
    app.state.frigate_version = ""  # cached after first success; version is stable within a run
    app.state.last_review_view = 0.0  # time.monotonic() of the last Review-tab poll (gates ingest)

    async def _frigate_version():
        if not app.state.frigate_version:
            v = await frigate.version()
            if v:
                app.state.frigate_version = v
        return app.state.frigate_version

    # ---- auth (optional; piggybacks on Frigate's user database) ----
    secret = authmod.load_secret(cfg.data_dir) if cfg.auth == "frigate" else b""

    if cfg.auth == "frigate":
        @app.middleware("http")
        async def require_auth(request, call_next):
            p = request.url.path
            # /__betterfaces/* is reached THROUGH Frigate's nginx (frigate-ext) and
            # only serves the public inject script, so it must bypass our own auth
            # (the browser carries Frigate's cookies, not ours).
            if (p in ("/login", "/api/healthz", "/favicon.ico")
                    or p.startswith("/static/") or p.startswith("/__betterfaces/")):
                return await call_next(request)
            if authmod.valid_token(secret, request.cookies.get(authmod.COOKIE, "")):
                return await call_next(request)
            if p.startswith("/api/"):
                return JSONResponse({"detail": "authentication required"}, status_code=401)
            return RedirectResponse("/login", status_code=302)

    @app.get("/login")
    async def login_page():
        return FileResponse(os.path.join(STATIC_DIR, "login.html"))

    @app.post("/login")
    async def login(body: LoginBody):
        if cfg.auth != "frigate":
            return {"ok": True}
        if not await frigate.login_ok(body.user, body.password):
            raise HTTPException(401, "invalid credentials")
        resp = JSONResponse({"ok": True})
        resp.set_cookie(authmod.COOKIE, authmod.make_token(secret), httponly=True,
                        samesite="lax", max_age=authmod.TTL, path="/")
        return resp

    @app.post("/logout")
    async def logout():
        resp = JSONResponse({"ok": True})
        resp.delete_cookie(authmod.COOKIE, path="/")
        return resp

    def _persons_by_id():
        return {p["id"]: p["name"] for p in store.list_persons()}

    def _view(row, names):
        meta = parse_filename(row["frigate_id"])
        return {
            "id": row["id"],
            "image": f"/api/crops/{row['id']}/image",
            # Full-scene snapshot of the event this face came from (the whole
            # picture, like Frigate's faces-page magnifier); null when the
            # filename has no parseable event id to look it up by.
            "snapshot": f"/api/crops/{row['id']}/snapshot" if meta.event_id else None,
            "status": row["status"],
            "reason": row.get("reason") or "",
            "frigate_label": meta.label,
            "score": round(row["match_score"], 3) if row.get("match_score") is not None else None,
            "det_score": round(row["det_score"], 3) if row.get("det_score") is not None else None,
            "blur": round(row["blur_score"], 1) if row.get("blur_score") is not None else None,
            "event_ts": row.get("event_ts"),
            "suggested": names.get(row.get("suggested_person_id")),
            "suggested_id": row.get("suggested_person_id"),
            "person": names.get(row.get("person_id")),
            "person_id": row.get("person_id"),
        }

    # ---------------------------------------------------------------- state
    @app.get("/api/state")
    async def state():
        ver = await _frigate_version()
        try:
            fp = await frigate.list_person_names()
        except Exception:
            fp = []
        return {
            "counts": store.counts(),
            "persons": store.list_persons(),
            "frigate_persons": fp,
            "settings": store.get_settings(),
            "frigate": {"url": cfg.frigate_url, "version": ver, "reachable": bool(ver),
                        "ui_url": cfg.frigate_ui_url},
            "ingest": {
                "last_run": getattr(app.state.ingestor, "last_run", 0.0),
                "last_error": getattr(app.state.ingestor, "last_error", ""),
                "ingested": getattr(app.state.ingestor, "ingested", 0),
                "models_loaded": getattr(app.state.embedder, "loaded", None),
            },
            "writeback": cfg.frigate_writeback,
            "auth": cfg.auth,
        }

    @app.get("/api/review")
    async def review(limit: int = 300, offset: int = 0):
        app.state.last_review_view = time.monotonic()  # keep ingesting while you review
        names = _persons_by_id()
        rows = store.list_by_status("review", limit=limit, offset=offset, order="created_at DESC")
        return [_view(r, names) for r in rows]

    @app.get("/api/filtered")
    async def filtered(limit: int = 300, offset: int = 0):
        names = _persons_by_id()
        rows = store.list_by_statuses(["auto_rejected", "rejected"], limit=limit, offset=offset)
        return [_view(r, names) for r in rows]

    @app.get("/api/labeled")
    async def labeled(limit: int = 1000, offset: int = 0):
        names = _persons_by_id()
        rows = store.list_by_statuses(["labeled", "auto_labeled"], limit=limit, offset=offset)
        return [_view(r, names) for r in rows]

    @app.get("/api/crops/{cid}/image")
    async def crop_image(cid: int):
        thumb = store.get_thumb(cid)
        if not thumb:
            raise HTTPException(404, "no image")
        return Response(content=thumb, media_type="image/webp",
                        headers={"Cache-Control": "public, max-age=86400"})

    async def _snapshot_response(event_id: str):
        """Proxy Frigate's full-scene event snapshot. 404s (rather than 500s)
        when Frigate has no snapshot for the event so the UI shows a tidy
        'unavailable' message instead of an error."""
        if not event_id:
            raise HTTPException(404, "no event for this crop")
        try:
            data = await frigate.event_snapshot(event_id)
        except Exception as e:
            log.warning("event snapshot %s failed: %s", event_id, e)
            raise HTTPException(404, "snapshot unavailable")
        if not data:
            raise HTTPException(404, "snapshot unavailable")
        return Response(content=data, media_type="image/jpeg",
                        headers={"Cache-Control": "public, max-age=3600"})

    @app.get("/api/crops/{cid}/snapshot")
    async def crop_snapshot(cid: int):
        # The whole picture (full camera frame) for a review/filtered crop, so
        # you can identify the person from the scene, not just the face crop.
        row = store.get_crop(cid)
        if not row:
            raise HTTPException(404, "no such crop")
        return await _snapshot_response(parse_filename(row["frigate_id"]).event_id)

    @app.get("/api/event-snapshot/{filename}")
    async def event_snapshot(filename: str):
        # Same full-scene snapshot, addressed by a Frigate crop filename (used by
        # the People detail view, whose crops live only in Frigate, not our DB).
        return await _snapshot_response(parse_filename(filename).event_id)

    # -------------------------------------------------------------- persons
    @app.get("/api/persons")
    async def persons():
        return store.list_persons()

    @app.post("/api/persons")
    async def create_person(body: PersonBody):
        name = (body.name or "").strip()
        if not name:
            raise HTTPException(400, "name required")
        try:
            pid = store.get_or_create_person(name)
        except Exception as e:
            raise HTTPException(400, str(e))
        return {"id": pid, "name": name}

    @app.get("/api/people")
    async def people():
        # The people ARE Frigate's face library; the count is how many face crops
        # Frigate has for each. There is no separate list -- labelling here writes
        # straight into Frigate.
        try:
            faces = await frigate.list_faces()
        except Exception:
            faces = {}
        out = {}
        for k, v in faces.items():
            if k == "train":
                continue
            out[k.lower()] = {"name": k, "count": len(v)}
        # surface anyone labelled here whose Frigate write has not landed yet (rare)
        for p in store.list_persons():
            lo = p["name"].lower()
            if lo not in out:
                out[lo] = {"name": p["name"], "count": p.get("labeled_count") or 0}
        return sorted(out.values(), key=lambda x: x["name"].lower())

    @app.post("/api/people/delete")
    async def delete_person(body: PersonBody):
        name = (body.name or "").strip()
        if not name:
            raise HTTPException(400, "name required")
        frigate_ok, frigate_err = True, ""
        try:
            await frigate.delete_person(name)
        except Exception as e:
            frigate_ok, frigate_err = False, str(e)
            log.warning("frigate delete_person failed for %s: %s", name, e)
        store.delete_person_by_name(name)
        return {"ok": True, "frigate_deleted": frigate_ok, "frigate_error": frigate_err}

    @app.get("/api/person/{name}/crops")
    async def person_crops(name: str):
        # All of a person's face crops = their Frigate folder (pre-existing
        # enrollments + crops trained through this tool), newest first.
        faces = await frigate.list_faces()
        folder = next((k for k in faces if k != "train" and k.lower() == name.lower()), name)
        files = faces.get(folder, [])

        def _ts(fn):
            m = re.search(r"(\d+(?:\.\d+)?)\.(?:webp|png|jpe?g)$", fn, re.IGNORECASE)
            return float(m.group(1)) if m else 0.0

        ordered = sorted(files, key=_ts, reverse=True)

        def _crop(f):
            eid = parse_filename(f).event_id
            return {
                "filename": f,
                "image": f"/api/person/{quote(folder)}/image/{quote(f)}",
                "snapshot": f"/api/event-snapshot/{quote(f)}" if eid else None,
            }

        return {"name": folder, "crops": [_crop(f) for f in ordered]}

    @app.get("/api/person/{name}/image/{filename}")
    async def person_image(name: str, filename: str):
        try:
            data = await frigate.fetch_crop(filename, folder=name)
        except Exception:
            raise HTTPException(404, "no image")
        return Response(content=data, media_type="image/webp",
                        headers={"Cache-Control": "public, max-age=3600"})

    @app.post("/api/person/{name}/crops/delete")
    async def person_crop_delete(name: str, body: CropFileBody):
        frigate_ok, frigate_err = True, ""
        try:
            await frigate.delete_crop(body.filename, folder=name)
        except Exception as e:
            frigate_ok, frigate_err = False, str(e)
            log.warning("delete %s/%s failed: %s", name, body.filename, e)
        return {"ok": True, "frigate_deleted": frigate_ok, "frigate_error": frigate_err}

    # -------------------------------------------------------------- actions
    @app.post("/api/crops/{cid}/assign")
    async def assign(cid: int, body: AssignBody):
        row = store.get_crop(cid)
        if not row:
            raise HTTPException(404, "no such crop")
        if body.person_id is not None:
            person = store.get_person(body.person_id)
            if not person:
                raise HTTPException(404, "no such person")
            name = person["name"]
        elif body.name and body.name.strip():
            name = body.name.strip()
        else:
            raise HTTPException(400, "person_id or name required")

        # Reuse an existing Frigate person to avoid duplicate folders. Match by
        # exact casing first ("Erwin" onto an existing "erwin"), then fall back
        # to a '-'/'_'-folded key: Frigate mangles '-' to '_' in train-crop
        # labels, so a crop for "Jan-Peter" prefills as "Jan_Peter" and must map
        # back onto the real person instead of creating a new "Jan_Peter".
        try:
            existing = await frigate.list_person_names()
            match = next((e for e in existing if e.lower() == name.lower()), None)
            if match is None:
                match = next((e for e in existing if label_key(e) == label_key(name)), None)
            if match is not None:
                name = match
        except Exception:
            pass
        pid = store.get_or_create_person(name)

        frigate_ok, frigate_err = True, ""
        if cfg.frigate_writeback:
            try:
                await frigate.assign(row["frigate_id"], name)
            except Exception as e:  # still record locally; surface the warning
                frigate_ok, frigate_err = False, str(e)
                log.warning("frigate assign failed for %s: %s", row["frigate_id"], e)
        store.set_decision(cid, "labeled", person_id=pid, reason="manual")
        return {"ok": True, "person_id": pid, "name": name,
                "frigate_trained": frigate_ok, "frigate_error": frigate_err}

    @app.post("/api/crops/{cid}/reject")
    async def reject(cid: int):
        row = store.get_crop(cid)
        if not row:
            raise HTTPException(404, "no such crop")
        frigate_ok, frigate_err = True, ""
        if cfg.frigate_writeback:
            try:
                await frigate.delete_crop(row["frigate_id"])
            except Exception as e:
                frigate_ok, frigate_err = False, str(e)
                log.warning("frigate delete failed for %s: %s", row["frigate_id"], e)
        store.set_decision(cid, "rejected", reason="not_a_face")
        return {"ok": True, "frigate_deleted": frigate_ok, "frigate_error": frigate_err}

    @app.post("/api/crops/{cid}/delete")
    async def delete_crop(cid: int):
        # Delete-forever: remove from Frigate + purge locally (not kept in Filtered).
        row = store.get_crop(cid)
        if not row:
            raise HTTPException(404, "no such crop")
        frigate_ok, frigate_err = True, ""
        if cfg.frigate_writeback:
            try:
                await frigate.delete_crop(row["frigate_id"])
            except Exception as e:
                frigate_ok, frigate_err = False, str(e)
                log.warning("frigate purge failed for %s: %s", row["frigate_id"], e)
        store.purge_crop(cid)
        return {"ok": True, "frigate_deleted": frigate_ok, "frigate_error": frigate_err}

    @app.post("/api/crops/{cid}/undo")
    async def undo(cid: int):
        row = store.get_crop(cid)
        if not row:
            raise HTTPException(404, "no such crop")
        # Revert our record to the review queue. A prior assign/reject that already
        # wrote to Frigate is not reversed there (the crop was moved/deleted), so we
        # flag the desync; classification state (gallery membership) is restored.
        desynced = cfg.frigate_writeback and row["status"] in ("labeled", "rejected")
        store.set_decision(cid, "review", person_id=None, reason="undo")
        return {"ok": True, "frigate_desynced": desynced}

    # ------------------------------------------------------------- settings
    @app.get("/api/settings")
    async def get_settings():
        return store.get_settings()

    @app.post("/api/settings")
    async def set_setting(body: SettingBody):
        if body.key not in TUNABLE_KEYS:
            raise HTTPException(400, f"unknown setting {body.key}")
        store.set_setting(body.key, _coerce_setting(body.key, body.value))
        return store.get_settings()

    # ------------------------------------------------------------ utilities
    @app.get("/api/healthz")
    async def healthz():
        return {"ok": True}

    # ---- frigate-ext: the script Frigate's nginx injects into its /faces page ----
    @app.get("/__betterfaces/inject.js")
    async def betterfaces_inject_js():
        cfgline = f"window.__BFR_PORT={int(cfg.public_port)};"
        if cfg.public_url:
            cfgline += f"window.__BFR_URL={json.dumps(cfg.public_url)};"
        return Response(content=cfgline + "\n" + INJECT_JS,
                        media_type="application/javascript",
                        headers={"Cache-Control": "no-cache"})

    @app.get("/__betterfaces/api/health")
    async def betterfaces_health():
        return {"ok": True, "service": "frigate-better-face-recognition",
                "inject": nginx_inject.status()}

    @app.get("/")
    async def index():
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))

    if os.path.isdir(STATIC_DIR):
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    return app


def build_default_app() -> FastAPI:
    from .db import Store
    from .embedder import Embedder
    from .frigate import FrigateClient

    cfg = Config()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    store = Store(cfg.db_path)
    embedder = Embedder(model_name=cfg.model_name, det_thresh=cfg.det_threshold,
                        model_root=os.path.join(cfg.data_dir, "models"),
                        fiqa_variant=cfg.fiqa_variant)
    frigate = FrigateClient(cfg.frigate_url, verify_tls=cfg.frigate_verify_tls)
    return create_app(cfg, store, embedder, frigate, run_ingest=True)
