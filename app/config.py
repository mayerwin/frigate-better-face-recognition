"""Runtime configuration (12-factor: everything via environment variables).

Only FRIGATE_URL is really required; every other value has a sensible default,
so `docker run -e FRIGATE_URL=...` just works.

Tunables you may want to change from the UI at runtime (thresholds, the
auto-reject / auto-label toggles) are *seeded* from these env defaults into the
SQLite `settings` table on first run. After that the DB is the source of truth
and the env defaults are ignored, so a restart never clobbers your tuning.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return int(default)


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


# Keys that live in the settings table and may be overridden at runtime.
TUNABLE_KEYS = (
    "match_threshold", "reject_threshold", "auto_reject", "auto_label", "blur_threshold",
    "retention_auto_rejected_days", "retention_review_days",
)


@dataclass(frozen=True)
class Config:
    # --- Frigate connection ---
    frigate_url: str = field(
        default_factory=lambda: os.environ.get("FRIGATE_URL", "http://frigate:5000").rstrip("/")
    )
    frigate_user: str = field(default_factory=lambda: os.environ.get("FRIGATE_USER", ""))
    frigate_password: str = field(default_factory=lambda: os.environ.get("FRIGATE_PASSWORD", ""))
    frigate_verify_tls: bool = field(default_factory=lambda: _env_bool("FRIGATE_VERIFY_TLS", False))

    # Auth: "frigate" = require a login validated against Frigate's own user
    # database (same credentials as Frigate); "none" = open (front it yourself).
    auth: str = field(default_factory=lambda: os.environ.get("AUTH", "frigate").strip().lower())

    # --- storage / models ---
    data_dir: str = field(default_factory=lambda: os.environ.get("DATA_DIR", "/app/data"))
    model_name: str = field(default_factory=lambda: os.environ.get("MODEL_NAME", "buffalo_l"))
    # eDifFIQA face-image-quality model variant: t/s/m/l (l = best, ranks #1 on the
    # NIST FATE Quality leaderboard; t is a 7 MB tiny option). Scored once per new
    # crop on CPU and cached under data_dir/models.
    fiqa_variant: str = field(default_factory=lambda: os.environ.get("FIQA_VARIANT", "l").strip().lower())

    # --- web server ---
    host: str = field(default_factory=lambda: os.environ.get("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _env_int("PORT", 8975))

    # --- Faces-page button (frigate-ext injection) ---
    # The button injected into Frigate's /faces page opens this tool in a new tab.
    # By default it targets http://<the-host-you-reach-frigate-on>:<public_port>/.
    # public_url overrides with an absolute URL (e.g. a reverse-proxied https one).
    public_port: int = field(default_factory=lambda: _env_int("BFR_PUBLIC_PORT", _env_int("PORT", 8975)))
    public_url: str = field(default_factory=lambda: os.environ.get("BFR_PUBLIC_URL", "").strip())
    # The "back to Frigate" home button on THIS tool's own dashboard. Empty => the
    # browser derives https://<this-host>:8971/ (Frigate's standard front door).
    frigate_ui_url: str = field(default_factory=lambda: os.environ.get("BFR_FRIGATE_UI_URL", "").strip())

    # --- pipeline behaviour ---
    poll_interval: float = field(default_factory=lambda: _env_float("POLL_INTERVAL", 10.0))
    thumb_size: int = field(default_factory=lambda: _env_int("THUMB_SIZE", 320))
    det_threshold: float = field(default_factory=lambda: _env_float("DET_THRESHOLD", 0.5))
    frigate_writeback: bool = field(default_factory=lambda: _env_bool("FRIGATE_WRITEBACK", True))

    # --- on-demand model lifecycle ---
    # The face models load only while the Review tab is open (or auto_label is on)
    # and are released after model_idle_ttl seconds with no face scored, so the
    # background footprint is just the web server (~150 MB) instead of ~1 GB. The UI
    # polls /api/review every 8 s while that tab is open; ui_active_window is how
    # long after the last such poll we keep ingesting new crops.
    ui_active_window: float = field(default_factory=lambda: _env_float("UI_ACTIVE_WINDOW", 30.0))
    model_idle_ttl: float = field(default_factory=lambda: _env_float("MODEL_IDLE_TTL", 180.0))

    # --- tunable defaults (seed the settings table on first run) ---
    match_threshold: float = field(default_factory=lambda: _env_float("MATCH_THRESHOLD", 0.5))
    reject_threshold: float = field(default_factory=lambda: _env_float("REJECT_THRESHOLD", 0.5))
    auto_reject: bool = field(default_factory=lambda: _env_bool("AUTO_REJECT", True))
    auto_label: bool = field(default_factory=lambda: _env_bool("AUTO_LABEL", False))
    blur_threshold: float = field(default_factory=lambda: _env_float("BLUR_THRESHOLD", 0.0))
    # Retention: auto-prune crops the classifier never uses so bfr.db doesn't grow
    # forever. Age in days; 0 = keep forever. Only ever touches non-gallery rows
    # ('auto_rejected' + 'deleted' tombstones, and 'review'); never 'labeled' or
    # 'rejected' (the human-built galleries).
    retention_auto_rejected_days: float = field(default_factory=lambda: _env_float("RETENTION_AUTO_REJECTED_DAYS", 90.0))
    retention_review_days: float = field(default_factory=lambda: _env_float("RETENTION_REVIEW_DAYS", 365.0))

    @property
    def db_path(self) -> str:
        return os.path.join(self.data_dir, "bfr.db")

    def tunable_defaults(self) -> dict:
        return {
            "match_threshold": self.match_threshold,
            "reject_threshold": self.reject_threshold,
            "auto_reject": self.auto_reject,
            "auto_label": self.auto_label,
            "blur_threshold": self.blur_threshold,
            "retention_auto_rejected_days": self.retention_auto_rejected_days,
            "retention_review_days": self.retention_review_days,
        }
