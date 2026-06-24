"""Self-healing nginx integrator implementing the shared **frigate-ext** protocol
(v2) so MULTIPLE independent extensions inject into Frigate's one nginx.conf
without conflicts.

Why v2: nginx's `sub_filter` consumes each source token once, so two extensions
both doing `sub_filter '</body>' ...` collide (only the first fires). v2 fixes
that: each extension just drops its `<script>` TAG in `inject/<ext-id>.html`, and
the integrator regenerates exactly ONE combined `sub_filter '</body>' '<all
tags></body>';` into `generated/subfilter.conf`. One directive, no collision,
scales to N extensions. Per-extension `location` blocks have distinct prefixes,
so those never collide.

Spec + reference (Node): PROTOCOL.md / src/nginx.js in
https://github.com/mayerwin/frigate-layout-sync . The SHARED constants + the
combined-sub_filter regeneration MUST stay byte-identical to that spec so
concurrent extensions converge. All Docker access is `docker exec
<frigate-container> ...` over the mounted /var/run/docker.sock.
"""
from __future__ import annotations

import asyncio
import logging
import os

log = logging.getLogger("bfr.nginx")

# ============================ SHARED PROTOCOL ============================
# Identical across every frigate-ext extension. Do NOT diverge from the spec.
LOCK = "/tmp/frigate-inject.lock"
EXT_ROOT = "/usr/local/nginx/conf/frigate-ext"
INJECT_DIR = EXT_ROOT + "/inject"        # <ext-id>.html -> one <script> tag (input)
LOCATION_DIR = EXT_ROOT + "/locations"   # <ext-id>.conf -> a location block (input)
GEN_DIR = EXT_ROOT + "/generated"        # the single combined sub_filter (output)
MANAGED = "# frigate-ext (managed)"
CONF = os.environ.get("FRIGATE_NGINX_CONF", "/usr/local/nginx/conf/nginx.conf")

# ============================= THIS EXTENSION ============================
EXT_ID = "better-frigate-face-recognition"  # unique slug -> our filenames
ROUTE = "/__betterfaces/"                    # unique nginx location prefix
# Our one <script> tag. NO single quotes (it goes inside a single-quoted nginx
# string) and NO newline (the regenerator strips CR/LF anyway).
INJECT_TAG = '<script src="' + ROUTE + 'inject.js" defer></script>'

CONTAINER = os.environ.get("FRIGATE_CONTAINER", "frigate")
# Our container's name:port, resolvable from inside Frigate (we share its docker
# network). Must match the container_name the deployment uses.
UPSTREAM = os.environ.get("BFR_UPSTREAM", "frigate-better-face-recognition:8975")
RESOLVER = os.environ.get("NGINX_RESOLVER", "127.0.0.11")
AUTO = os.environ.get("NGINX_AUTOCONFIG", "true").strip().lower() not in ("0", "false", "no", "off")

_state = {"mode": "starting" if AUTO else "disabled", "applied": False,
          "last_error": None, "container": CONTAINER, "route": ROUTE, "upstream": UPSTREAM}
_busy = False


def status() -> dict:
    return dict(_state)


# ------------------------------------------------------- file contents
def location_conf() -> str:
    # Variable proxy_pass + resolver => nginx resolves the upstream lazily at
    # request time, so this companion being down can never break Frigate's own
    # web UI (the path just 502s). Included inside `server { }`.
    return "\n".join([
        "location " + ROUTE + " {",
        "    resolver " + RESOLVER + " ipv6=off valid=10s;",
        '    set $bfr_upstream "http://' + UPSTREAM + '";',
        "    proxy_pass $bfr_upstream$request_uri;",
        "    proxy_http_version 1.1;",
        "    proxy_set_header Host $host;",
        "    proxy_set_header X-Forwarded-For $remote_addr;",
        "    proxy_read_timeout 30s;",
        "}",
        "",
    ])


# Byte-identical to every other extension's regen (so concurrent runs converge):
# concatenate all inject/*.html (sorted by the shell glob), strip CR/LF, and wrap
# them in ONE `sub_filter '</body>' '<all tags></body>';`. If no tags remain,
# remove the file (empty *.conf glob is valid nginx).
_REGEN = (
    "T=$(cat '" + INJECT_DIR + "/'*.html 2>/dev/null | tr -d '\\r\\n'); "
    "if [ -n \"$T\" ]; then "
    "printf \"sub_filter '</body>' '%s</body>';\\n\" \"$T\" "
    "> '" + GEN_DIR + "/subfilter.conf.t' && mv '" + GEN_DIR + "/subfilter.conf.t' '" + GEN_DIR + "/subfilter.conf'; "
    "else rm -f '" + GEN_DIR + "/subfilter.conf'; fi"
)


# ------------------------------------------------------- the locked critical section
def apply_script() -> str:
    """One flock'd shell pass: migrate our legacy v1 file, ensure the two include
    lines (idempotent grep -F), regenerate the single combined sub_filter, then
    validate + reload (rolling back ONLY our own two inputs on failure). awk uses
    index() with literal substrings so there is no regex escaping to get wrong."""
    loc_line = "        include " + LOCATION_DIR + "/*.conf; " + MANAGED
    gen_line = "            include " + GEN_DIR + "/*.conf; " + MANAGED
    awk_loc = (
        "awk '{ "
        'if (!L && index($0, "location / {")) { print "' + loc_line + '"; L=1 } print '
        "}' '" + CONF + "' > '" + CONF + ".bfr.t' && mv '" + CONF + ".bfr.t' '" + CONF + "'"
    )
    awk_gen = (
        "awk '{ print; "
        'if (!S && index($0, "root /opt/frigate/web")) { print "' + gen_line + '"; S=1 } '
        "}' '" + CONF + "' > '" + CONF + ".bfr.t' && mv '" + CONF + ".bfr.t' '" + CONF + "'"
    )
    return "\n".join([
        "mkdir -p '" + INJECT_DIR + "' '" + LOCATION_DIR + "' '" + GEN_DIR + "'",
        # migrate: our v1 wrote a per-ext sub_filter here; the combined one supersedes it
        "rm -f '" + EXT_ROOT + "/subfilters/" + EXT_ID + ".conf'",
        # idempotent include lines (checked individually so an upgrade adds a missing one)
        "if ! grep -qF 'frigate-ext/locations/*.conf' '" + CONF + "'; then",
        "  " + awk_loc,
        "fi",
        "if ! grep -qF 'frigate-ext/generated/*.conf' '" + CONF + "'; then",
        "  " + awk_gen,
        "fi",
        _REGEN,
        "if nginx -t 2>/tmp/bfr-nginxt; then",
        "  nginx -s reload 2>/dev/null; echo BFR_OK",
        "else",
        "  rm -f '" + INJECT_DIR + "/" + EXT_ID + ".html' '" + LOCATION_DIR + "/" + EXT_ID + ".conf'",
        "  " + _REGEN,
        "  nginx -t >/dev/null 2>&1 && nginx -s reload 2>/dev/null",
        "  echo BFR_FAIL; cat /tmp/bfr-nginxt >&2; exit 1",
        "fi",
    ])


# ------------------------------------------------------- docker plumbing
async def _docker(args, input_bytes: bytes | None = None) -> bytes:
    proc = await asyncio.create_subprocess_exec(
        "docker", *args,
        stdin=asyncio.subprocess.PIPE if input_bytes is not None else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate(input=input_bytes)
    if proc.returncode != 0:
        msg = (err.decode("utf-8", "replace").strip() or ("docker " + " ".join(args[:1]) + " failed"))
        raise RuntimeError(msg)
    return out


# flock the shared lock INSIDE Frigate for the whole command (serializes every
# extension). flock creates the lock file if missing.
async def _locked_sh(script: str) -> bytes:
    return await _docker(["exec", CONTAINER, "flock", "-w", "30", LOCK, "sh", "-c", script])


# Atomic write of an input file (mkdir -p parent, tmp + mv) so nginx never sees a
# half file. No lock needed: distinct filename per extension.
async def _write_file(path: str, content: str) -> bytes:
    d = path.rsplit("/", 1)[0]
    script = "mkdir -p '" + d + "' && cat > '" + path + ".tmp' && mv '" + path + ".tmp' '" + path + "'"
    return await _docker(["exec", "-i", CONTAINER, "sh", "-c", script], content.encode("utf-8"))


# ------------------------------------------------------- apply (idempotent + safe)
async def ensure() -> None:
    global _busy
    if not AUTO or _busy:
        return
    _busy = True
    try:
        await _write_file(INJECT_DIR + "/" + EXT_ID + ".html", INJECT_TAG)  # no trailing newline
        await _write_file(LOCATION_DIR + "/" + EXT_ID + ".conf", location_conf())
        await _locked_sh(apply_script())
        if not _state["applied"]:
            log.info("injected into %s nginx (route %s -> %s)", CONTAINER, ROUTE, UPSTREAM)
        _state.update(mode="managed", applied=True, last_error=None)
    except Exception as e:  # noqa: BLE001 - best-effort integrator, never fatal
        _state.update(applied=False, last_error=str(e))
        log.warning("nginx ensure failed: %s", e)
    finally:
        _busy = False


async def _ensure_with_retry(tries: int = 24, delay: float = 5.0) -> None:
    for _ in range(tries):
        if _state["applied"]:
            return
        await ensure()
        if not _state["applied"]:
            await asyncio.sleep(delay)


def _manual_hint() -> None:
    log.warning("auto nginx config is OFF or Docker is unreachable; wire Frigate's nginx yourself:")
    log.warning("  drop this tag in %s/%s.html:  %s", INJECT_DIR, EXT_ID, INJECT_TAG)
    log.warning("  drop a location in %s/%s.conf:  location %s { proxy_pass http://%s; }",
                LOCATION_DIR, EXT_ID, ROUTE, UPSTREAM)


# ------------------------------------------------------- lifecycle
async def run() -> None:
    """Full lifecycle; launch as a background task so it never blocks startup.
    Applies once (with retry), then re-applies on every Frigate (re)start and on a
    ~30s backstop timer. Cancellation-safe."""
    if not AUTO:
        _state["mode"] = "disabled"
        _manual_hint()
        return
    try:
        await _docker(["version", "--format", "{{.Server.Version}}"])
    except Exception as e:  # noqa: BLE001
        _state.update(mode="manual", last_error="docker unreachable: " + str(e))
        _manual_hint()
        return

    await _ensure_with_retry()

    async def watch_restarts() -> None:
        # Re-apply whenever the Frigate container (re)starts: an upgrade recreates
        # it from a fresh image and wipes the includes + frigate-ext dir. Match by
        # the event ACTOR name -- NOT `--filter container=<name>`, which binds to
        # the current id and misses the recreated container.
        while True:
            try:
                proc = await asyncio.create_subprocess_exec(
                    "docker", "events", "--filter", "type=container",
                    "--filter", "event=start", "--format", "{{.Actor.Attributes.name}}",
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
                )
                assert proc.stdout is not None
                async for line in proc.stdout:
                    if line.decode("utf-8", "replace").strip() == CONTAINER:
                        _state["applied"] = False
                        await asyncio.sleep(1.5)  # let Frigate's nginx come up first
                        await _ensure_with_retry()
                await proc.wait()
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                log.debug("docker events watch error: %s", e)
            await asyncio.sleep(5)

    async def backstop() -> None:
        # Reliable backstop if an event is ever missed: ensure() re-reads live
        # state, so a missing injection self-heals within ~30s regardless.
        while True:
            await asyncio.sleep(30)
            await ensure()

    try:
        await asyncio.gather(watch_restarts(), backstop())
    except asyncio.CancelledError:
        pass


async def remove() -> None:
    """Remove ONLY this extension's inputs, regenerate the combined sub_filter
    (now without our tag), and reload. Leaves the shared includes + other
    extensions untouched."""
    script = "\n".join([
        "rm -f '" + INJECT_DIR + "/" + EXT_ID + ".html' '" + LOCATION_DIR + "/" + EXT_ID + ".conf'",
        _REGEN,
        "nginx -t >/dev/null 2>&1 && nginx -s reload 2>/dev/null",
        "echo removed",
    ])
    await _locked_sh(script)
    log.info("removed %s injection from %s nginx", EXT_ID, CONTAINER)


# `python -m app.nginx_inject apply|remove`
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cmd = sys.argv[1] if len(sys.argv) > 1 else "apply"
    if cmd == "remove":
        asyncio.run(remove())
    else:
        asyncio.run(ensure())
        sys.exit(0 if _state["applied"] else 1)
