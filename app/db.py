"""SQLite persistence: persons, crops (embeddings + thumbnails as blobs), and
runtime settings. One file, WAL mode, safe for concurrent FastAPI access via a
process-wide re-entrant lock (writes are tiny and infrequent).

Galleries are *derived*, never stored separately:
  * positive gallery = embeddings of crops a human labelled (status='labeled')
  * negative gallery = embeddings of crops a human marked "not a face"
    (status='rejected')
Auto decisions ('auto_labeled' / 'auto_rejected') deliberately do NOT seed the
galleries, so the classifier never trains on its own guesses (no drift).
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from typing import Optional

import numpy as np

SCHEMA_VERSION = 1
EMB_DIM = 512
BLUR_METRIC = "ediffiqa-l1"  # bump to invalidate + recompute cached quality scores

VALID_STATUS = {"review", "labeled", "rejected", "auto_rejected", "auto_labeled", "deleted"}

_DDL = """
CREATE TABLE IF NOT EXISTS persons (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT NOT NULL UNIQUE COLLATE NOCASE,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS crops (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    frigate_id          TEXT NOT NULL UNIQUE,
    camera              TEXT,
    event_ts            REAL,
    det_score           REAL,
    blur_score          REAL,
    has_face            INTEGER NOT NULL DEFAULT 0,
    embedding           BLOB,
    thumb               BLOB,
    status              TEXT NOT NULL,
    reason              TEXT,
    person_id           INTEGER REFERENCES persons(id) ON DELETE SET NULL,
    suggested_person_id INTEGER REFERENCES persons(id) ON DELETE SET NULL,
    match_score         REAL,
    source_path         TEXT,
    created_at          REAL NOT NULL,
    decided_at          REAL
);
CREATE INDEX IF NOT EXISTS idx_crops_status  ON crops(status);
CREATE INDEX IF NOT EXISTS idx_crops_person  ON crops(person_id);
CREATE INDEX IF NOT EXISTS idx_crops_created ON crops(created_at);
CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def emb_to_blob(e) -> Optional[bytes]:
    if e is None:
        return None
    return np.asarray(e, dtype=np.float32).reshape(-1).tobytes()


def blob_to_emb(b) -> Optional[np.ndarray]:
    if b is None:
        return None
    return np.frombuffer(b, dtype=np.float32)


def _now() -> float:
    return time.time()


class Store:
    def __init__(self, path: str):
        self.path = path
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        with self._lock:
            self._conn.executescript(_DDL)
            # migration: add columns missing on an older db (keeps existing data)
            try:
                self._conn.execute("ALTER TABLE crops ADD COLUMN blur_score REAL")
            except sqlite3.OperationalError:
                pass
            # if the sharpness metric changed, clear cached scores so the backfill
            # recomputes them with the new metric
            row = self._conn.execute("SELECT value FROM meta WHERE key='blur_metric'").fetchone()
            if not row or row["value"] != BLUR_METRIC:
                self._conn.execute("UPDATE crops SET blur_score=NULL")
                self._conn.execute(
                    "INSERT INTO meta(key,value) VALUES('blur_metric',?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (BLUR_METRIC,),
                )
            self._conn.execute(
                "INSERT INTO meta(key,value) VALUES('schema_version',?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(SCHEMA_VERSION),),
            )
            self._conn.commit()

    def close(self):
        with self._lock:
            self._conn.close()

    def checkpoint(self):
        """Fold the WAL back into the main db file so a file-level backup
        (butler-backup) captures a self-consistent snapshot."""
        with self._lock:
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    # ------------------------------------------------------------------ settings
    def seed_settings(self, defaults: dict):
        with self._lock:
            for k, v in defaults.items():
                self._conn.execute(
                    "INSERT OR IGNORE INTO settings(key,value) VALUES(?,?)",
                    (k, json.dumps(v)),
                )
            self._conn.commit()

    def get_settings(self) -> dict:
        with self._lock:
            rows = self._conn.execute("SELECT key,value FROM settings").fetchall()
        return {r["key"]: json.loads(r["value"]) for r in rows}

    def set_setting(self, key: str, value):
        with self._lock:
            self._conn.execute(
                "INSERT INTO settings(key,value) VALUES(?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, json.dumps(value)),
            )
            self._conn.commit()

    # ------------------------------------------------------------------- persons
    def create_person(self, name: str) -> int:
        name = (name or "").strip()
        if not name:
            raise ValueError("person name required")
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO persons(name,created_at) VALUES(?,?)", (name, _now())
            )
            self._conn.commit()
            return cur.lastrowid

    def get_or_create_person(self, name: str) -> int:
        name = (name or "").strip()
        if not name:
            raise ValueError("person name required")
        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM persons WHERE name = ? COLLATE NOCASE", (name,)
            ).fetchone()
            if row:
                return row["id"]
            try:
                cur = self._conn.execute(
                    "INSERT INTO persons(name,created_at) VALUES(?,?)", (name, _now())
                )
                self._conn.commit()
                return cur.lastrowid
            except sqlite3.IntegrityError:
                row = self._conn.execute(
                    "SELECT id FROM persons WHERE name = ? COLLATE NOCASE", (name,)
                ).fetchone()
                if row:
                    return row["id"]
                raise

    def list_persons(self) -> list:
        with self._lock:
            rows = self._conn.execute(
                "SELECT p.id, p.name, p.created_at, "
                "SUM(CASE WHEN c.status='labeled' THEN 1 ELSE 0 END) AS labeled_count "
                "FROM persons p LEFT JOIN crops c ON c.person_id = p.id "
                "GROUP BY p.id ORDER BY p.name COLLATE NOCASE"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_person(self, pid: int):
        with self._lock:
            r = self._conn.execute("SELECT * FROM persons WHERE id=?", (pid,)).fetchone()
        return dict(r) if r else None

    def rename_person(self, pid: int, name: str):
        with self._lock:
            self._conn.execute("UPDATE persons SET name=? WHERE id=?", ((name or "").strip(), pid))
            self._conn.commit()

    def delete_person(self, pid: int):
        with self._lock:
            self._conn.execute("DELETE FROM persons WHERE id=?", (pid,))
            self._conn.commit()

    def delete_person_by_name(self, name: str):
        with self._lock:
            self._conn.execute("DELETE FROM persons WHERE name = ? COLLATE NOCASE", (name,))
            self._conn.commit()

    # --------------------------------------------------------------------- crops
    def seen(self, frigate_id: str) -> bool:
        with self._lock:
            r = self._conn.execute(
                "SELECT 1 FROM crops WHERE frigate_id=?", (frigate_id,)
            ).fetchone()
        return r is not None

    def add_crop(
        self,
        *,
        frigate_id,
        camera,
        event_ts,
        det_score,
        has_face,
        embedding,
        thumb,
        status,
        blur_score=None,
        reason="",
        person_id=None,
        suggested_person_id=None,
        match_score=None,
        source_path=None,
    ) -> int:
        if status not in VALID_STATUS:
            raise ValueError(f"invalid status {status!r}")
        decided = None if status == "review" else _now()
        with self._lock:
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO crops("
                "frigate_id,camera,event_ts,det_score,blur_score,has_face,embedding,thumb,status,reason,"
                "person_id,suggested_person_id,match_score,source_path,created_at,decided_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    frigate_id, camera, event_ts, det_score, blur_score, 1 if has_face else 0,
                    emb_to_blob(embedding), thumb, status, reason, person_id,
                    suggested_person_id, match_score, source_path, _now(), decided,
                ),
            )
            self._conn.commit()
            if cur.rowcount == 0:
                # duplicate frigate_id was ignored; return the existing row id
                row = self._conn.execute(
                    "SELECT id FROM crops WHERE frigate_id=?", (frigate_id,)
                ).fetchone()
                return row["id"] if row else 0
            return cur.lastrowid

    def get_crop(self, cid: int):
        with self._lock:
            r = self._conn.execute("SELECT * FROM crops WHERE id=?", (cid,)).fetchone()
        return dict(r) if r else None

    def get_thumb(self, cid: int):
        with self._lock:
            r = self._conn.execute("SELECT thumb FROM crops WHERE id=?", (cid,)).fetchone()
        return r["thumb"] if r else None

    def purge_crop(self, cid: int):
        """Delete-forever: mark deleted and drop the thumbnail + embedding to free
        space, but keep the row so seen() still dedupes the filename (no re-ingest)."""
        with self._lock:
            self._conn.execute(
                "UPDATE crops SET status='deleted', thumb=NULL, embedding=NULL, decided_at=? WHERE id=?",
                (_now(), cid),
            )
            self._conn.commit()

    def update_blur(self, cid: int, blur_score: float):
        with self._lock:
            self._conn.execute("UPDATE crops SET blur_score=? WHERE id=?", (blur_score, cid))
            self._conn.commit()

    def crops_missing_blur(self, limit: int = 500):
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, frigate_id FROM crops WHERE blur_score IS NULL AND has_face=1 "
                "AND status IN ('review','auto_rejected') LIMIT ?",
                (limit,),
            ).fetchall()
        return [(r["id"], r["frigate_id"]) for r in rows]

    def set_decision(self, cid: int, status: str, *, person_id=None, reason="", match_score=None):
        if status not in VALID_STATUS:
            raise ValueError(f"invalid status {status!r}")
        with self._lock:
            self._conn.execute(
                "UPDATE crops SET status=?, person_id=?, reason=?, "
                "match_score=COALESCE(?,match_score), decided_at=? WHERE id=?",
                (status, person_id, reason, match_score, _now(), cid),
            )
            self._conn.commit()

    _ORDER = {
        "created_at DESC": "created_at DESC",
        "created_at ASC": "created_at ASC",
        "match_score DESC": "match_score DESC NULLS LAST",
    }

    def list_by_status(self, status, limit=200, offset=0, order="created_at DESC"):
        order_sql = self._ORDER.get(order, "created_at DESC")
        with self._lock:
            rows = self._conn.execute(
                "SELECT id,frigate_id,camera,event_ts,det_score,blur_score,has_face,status,reason,"
                "person_id,suggested_person_id,match_score,created_at,decided_at "
                f"FROM crops WHERE status=? ORDER BY {order_sql} LIMIT ? OFFSET ?",
                (status, limit, offset),
            ).fetchall()
        return [dict(r) for r in rows]

    def list_by_statuses(self, statuses, limit=300, offset=0):
        statuses = list(statuses)
        if not statuses:
            return []
        ph = ",".join("?" * len(statuses))
        with self._lock:
            rows = self._conn.execute(
                "SELECT id,frigate_id,camera,event_ts,det_score,blur_score,has_face,status,reason,"
                "person_id,suggested_person_id,match_score,created_at,decided_at "
                f"FROM crops WHERE status IN ({ph}) ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (*statuses, limit, offset),
            ).fetchall()
        return [dict(r) for r in rows]

    # Only prune enough backlog in one pass to justify a full-file VACUUM (a big
    # first cleanup); steady-state passes delete a handful of rows and skip it.
    _VACUUM_MIN_DELETED = 200

    def prune_old_crops(self, auto_rejected_days: float = 0.0, review_days: float = 0.0) -> int:
        """Retention: delete crops the classifier never uses once they age out, so
        bfr.db stays bounded. Touches ONLY non-gallery statuses -- 'auto_rejected'
        and 'deleted' tombstones (on auto_rejected_days) and 'review' (on
        review_days). NEVER 'labeled' or 'rejected' (the human galleries) or
        'auto_labeled'. Age is created_at (time since ingest); a value <= 0
        disables that bucket. Returns the number of rows deleted. Rows this old
        are long gone from Frigate's save_attempts buffer, so deleting them can't
        cause a re-ingest."""
        now = _now()
        total = 0
        with self._lock:
            if auto_rejected_days and auto_rejected_days > 0:
                cutoff = now - auto_rejected_days * 86400.0
                total += self._conn.execute(
                    "DELETE FROM crops WHERE status IN ('auto_rejected','deleted') "
                    "AND created_at < ?",
                    (cutoff,),
                ).rowcount
            if review_days and review_days > 0:
                cutoff = now - review_days * 86400.0
                total += self._conn.execute(
                    "DELETE FROM crops WHERE status='review' AND created_at < ?",
                    (cutoff,),
                ).rowcount
            self._conn.commit()
            if total >= self._VACUUM_MIN_DELETED:
                try:  # reclaim the freed pages to the OS; never let it break retention
                    self._conn.execute("VACUUM")
                except Exception:
                    pass
        return total

    def counts(self) -> dict:
        with self._lock:
            rows = self._conn.execute(
                "SELECT status, COUNT(*) AS n FROM crops GROUP BY status"
            ).fetchall()
        return {r["status"]: r["n"] for r in rows}

    # ----------------------------------------------------------------- galleries
    def positive_gallery(self) -> dict:
        with self._lock:
            rows = self._conn.execute(
                "SELECT person_id, embedding FROM crops "
                "WHERE status='labeled' AND embedding IS NOT NULL AND person_id IS NOT NULL"
            ).fetchall()
        g: dict = {}
        for r in rows:
            e = np.frombuffer(r["embedding"], dtype=np.float32)
            if e.size != EMB_DIM:  # skip any malformed blob rather than crash vstack
                continue
            g.setdefault(r["person_id"], []).append(e)
        return {pid: np.vstack(v) for pid, v in g.items() if v}

    def negative_gallery(self) -> np.ndarray:
        with self._lock:
            rows = self._conn.execute(
                "SELECT embedding FROM crops WHERE status='rejected' AND embedding IS NOT NULL"
            ).fetchall()
        embs = [np.frombuffer(r["embedding"], dtype=np.float32) for r in rows]
        embs = [e for e in embs if e.size == EMB_DIM]
        if not embs:
            return np.zeros((0, EMB_DIM), dtype=np.float32)
        return np.vstack(embs)
