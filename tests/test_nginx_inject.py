"""Protocol-conformance tests for the frigate-ext nginx integrator (v2).

These assert the coexistence-critical invariants (shared constants, the single
combined sub_filter regeneration, byte-exact include anchors, our-files-only
rollback) WITHOUT touching Docker, so they run anywhere. The SHARED values +
regen logic must match PROTOCOL.md and the frigate-layout-sync reference so the
two extensions share Frigate's one nginx.conf and converge byte-for-byte.
"""
from app import nginx_inject as nx


def test_shared_constants_match_spec():
    assert nx.LOCK == "/tmp/frigate-inject.lock"
    assert nx.EXT_ROOT == "/usr/local/nginx/conf/frigate-ext"
    assert nx.INJECT_DIR == "/usr/local/nginx/conf/frigate-ext/inject"
    assert nx.LOCATION_DIR == "/usr/local/nginx/conf/frigate-ext/locations"
    assert nx.GEN_DIR == "/usr/local/nginx/conf/frigate-ext/generated"
    assert nx.MANAGED == "# frigate-ext (managed)"


def test_ext_identity_is_unique():
    # Must differ from frigate-layout-sync's ext-id and route or we'd collide.
    assert nx.EXT_ID == "better-frigate-face-recognition"
    assert nx.ROUTE == "/__betterfaces/"


def test_inject_tag_is_js_string_safe_and_has_no_quotes_or_newline():
    # The tag sits in a single-quoted nginx string AND, because Frigate's
    # sub_filter_types includes application/javascript, it can be spliced into a
    # JS string literal in a bundle that contains "</body>" (issue #5: this
    # blanked the Config page). So it must carry NO character that can terminate a
    # JS string or the nginx string: no quote of any kind, no backslash, no
    # newline. The src is therefore unquoted (valid HTML5).
    assert nx.INJECT_TAG == '<script src=/__betterfaces/inject.js defer></script>'
    for bad in ("'", '"', "`", "\\", "\n", "\r"):
        assert bad not in nx.INJECT_TAG


def test_location_conf_is_lazy_and_scoped():
    conf = nx.location_conf()
    assert conf.startswith("location /__betterfaces/ {")
    assert "resolver 127.0.0.11 ipv6=off valid=10s;" in conf
    assert 'set $bfr_upstream "http://' + nx.UPSTREAM + '";' in conf
    assert "proxy_pass $bfr_upstream$request_uri;" in conf


def test_regen_builds_one_combined_subfilter_byte_identically():
    # The combined sub_filter is generated from ALL inject/*.html, so it MUST be
    # byte-identical logic across extensions (this is what makes concurrent runs
    # converge). One '</body>' directive carrying every extension's tag.
    r = nx._REGEN
    assert "cat '/usr/local/nginx/conf/frigate-ext/inject/'*.html" in r
    assert "tr -d '\\r\\n'" in r
    assert "printf \"sub_filter '</body>' '%s</body>';\\n\" \"$T\"" in r
    assert "/usr/local/nginx/conf/frigate-ext/generated/subfilter.conf" in r


def test_apply_uses_v2_includes_and_anchors():
    s = nx.apply_script()
    # idempotent include checks on the path substrings
    assert "grep -qF 'frigate-ext/locations/*.conf'" in s
    assert "grep -qF 'frigate-ext/generated/*.conf'" in s
    # byte-exact include lines (indentation matters; must equal the reference)
    assert ('print "        include /usr/local/nginx/conf/frigate-ext/locations/*.conf; '
            '# frigate-ext (managed)"') in s
    assert ('print "            include /usr/local/nginx/conf/frigate-ext/generated/*.conf; '
            '# frigate-ext (managed)"') in s
    # anchored at Frigate's own landmarks via literal index() matches
    assert 'index($0, "location / {")' in s
    assert 'index($0, "root /opt/frigate/web")' in s
    # migrates our legacy v1 per-ext sub_filter away
    assert "rm -f '/usr/local/nginx/conf/frigate-ext/subfilters/better-frigate-face-recognition.conf'" in s


def test_apply_rolls_back_only_our_own_inputs():
    s = nx.apply_script()
    assert "nginx -t" in s and "nginx -s reload" in s
    assert "/inject/better-frigate-face-recognition.html" in s
    assert "/locations/better-frigate-face-recognition.conf" in s
    # never removes another extension's files
    assert "layoutsync" not in s.lower()
    assert "layout-sync" not in s.lower()


def test_reload_is_gated_on_an_actual_change():
    # Every `nginx -s reload` grows Frigate's nginx master RSS, so the 30s
    # backstop must NOT reload when nothing changed (it used to, and ran Frigate's
    # LXC out of memory within days). The decision is made in-script from C.
    s = nx.apply_script()
    assert s.startswith("C=0\n")
    assert 'if [ "$C" = 0 ]; then echo BFR_UNCHANGED; exit 0; fi' in s
    assert s.index("BFR_UNCHANGED") < s.index("nginx -s reload")
    assert 'cmp -s "$F.t" "$F"' in nx._REGEN


def _sandbox(tmp_path):
    """Render apply_script() against a temp dir with a fake `nginx` that logs."""
    import shutil
    import subprocess

    import pytest

    sh = shutil.which("sh")
    if not sh:
        pytest.skip("needs a POSIX sh")
    root = tmp_path.as_posix()
    (tmp_path / "bin").mkdir()
    fake = tmp_path / "bin" / "nginx"
    fake.write_text('#!/bin/sh\necho "nginx $*" >> "' + root + '/calls"\n', newline="\n")
    fake.chmod(0o755)
    (tmp_path / "nginx.conf").write_text(
        "server {\n    location / {\n            root /opt/frigate/web;\n    }\n}\n", newline="\n")
    script = (nx.apply_script()
              .replace(nx.EXT_ROOT, root + "/frigate-ext")
              .replace(nx.CONF, root + "/nginx.conf")
              .replace("/tmp/", root + "/"))
    env = {"PATH": root + "/bin:/usr/bin:/bin"}

    def run() -> str:
        r = subprocess.run([sh, "-c", script], capture_output=True, text=True, env=env)
        assert r.returncode == 0, r.stderr
        return r.stdout.strip().splitlines()[-1]

    def reloads() -> int:
        calls = tmp_path / "calls"
        return calls.read_text().count("-s reload") if calls.exists() else 0

    return run, reloads, tmp_path


def test_apply_reloads_once_then_never_while_unchanged(tmp_path):
    run, reloads, root = _sandbox(tmp_path)
    assert run() == "BFR_RELOADED" and reloads() == 1
    for _ in range(5):
        assert run() == "BFR_UNCHANGED"
    assert reloads() == 1
    sub = (root / "frigate-ext" / "generated" / "subfilter.conf").read_bytes()
    assert sub == b"sub_filter '</body>' '" + nx.INJECT_TAG.encode() + b"</body>';\n"
    assert not list((root / "frigate-ext").rglob("*.t"))


def test_apply_reloads_again_on_drift(tmp_path):
    run, reloads, root = _sandbox(tmp_path)
    run()
    # another extension's tag appears -> combined sub_filter changes -> reload
    (root / "frigate-ext" / "inject" / "zz-other.html").write_text("<script src=/x.js></script>")
    assert run() == "BFR_RELOADED" and reloads() == 2
    # Frigate recreated nginx.conf without our includes -> reload
    conf = root / "nginx.conf"
    conf.write_text("".join(l for l in conf.read_text().splitlines(True) if "frigate-ext" not in l))
    assert run() == "BFR_RELOADED" and reloads() == 3
    assert run() == "BFR_UNCHANGED" and reloads() == 3
