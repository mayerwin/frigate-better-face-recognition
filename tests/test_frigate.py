import asyncio

import httpx

from app.frigate import FrigateClient, label_key, parse_filename


def test_label_key_folds_frigate_mangled_hyphen():
    # Frigate rewrites '-' to '_' in train-crop labels, so the hyphenated person
    # and their mangled crop label must collapse to one key (issue #1).
    assert label_key("Jan-Peter") == label_key("Jan_Peter") == "jan_peter"
    assert label_key("  Erwin ") == "erwin"
    assert label_key("Mary-Jane-Watson") == "mary_jane_watson"


def test_parse_filename_standard():
    c = parse_filename("1782080480.272977-bfxa4p-1782080490.833032-Jenny-0.82.webp")
    # event_id is the full Frigate id (ts + rand), the form snapshot lookups need
    assert c.event_id == "1782080480.272977-bfxa4p"
    assert c.label == "Jenny"
    assert abs(c.score - 0.82) < 1e-6
    assert abs(c.event_ts - 1782080480.272977) < 1e-3


def test_parse_filename_unknown_label_and_low_score():
    c = parse_filename("1782101505.680761-ux06fu-1782101519.643345-unknown-0.07.webp")
    assert c.label == "unknown"
    assert abs(c.score - 0.07) < 1e-6


def test_parse_filename_unrecognised_is_not_skipped():
    c = parse_filename("weird-name.png")
    assert c.filename == "weird-name.png"
    assert c.label == ""  # best-effort: still ingestable


def _client_with(handler):
    c = FrigateClient("http://frigate:5000")
    c._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return c


def test_list_train_filters_to_train_key():
    def handler(req):
        assert req.url.path == "/api/faces"
        return httpx.Response(200, json={
            "train": ["1-a-2-Jenny-0.8.webp", "3-b-4-unknown-0.1.webp"],
            "Erwin": ["x.webp"],
        })
    c = _client_with(handler)
    crops = asyncio.run(c.list_train())
    asyncio.run(c.aclose())
    assert [x.label for x in crops] == ["Jenny", "unknown"]


def test_assign_posts_training_file():
    seen = {}
    def handler(req):
        seen["path"] = req.url.path
        seen["body"] = req.read().decode()
        return httpx.Response(200, json={"success": True})
    c = _client_with(handler)
    asyncio.run(c.assign("1-a-2-Jenny-0.8.webp", "Erwin"))
    asyncio.run(c.aclose())
    assert seen["path"] == "/api/faces/train/Erwin/classify"
    assert "training_file" in seen["body"] and "Jenny" in seen["body"]


def test_delete_posts_ids():
    seen = {}
    def handler(req):
        seen["path"] = req.url.path
        seen["body"] = req.read().decode()
        return httpx.Response(200, json={"success": True})
    c = _client_with(handler)
    asyncio.run(c.delete_crop("1-a-2-unknown-0.1.webp"))
    asyncio.run(c.aclose())
    assert seen["path"] == "/api/faces/train/delete"
    assert "1-a-2-unknown-0.1.webp" in seen["body"]


def test_event_snapshot_fetches_full_frame_and_handles_missing():
    def handler(req):
        if req.url.path == "/api/events/gone/snapshot.jpg":
            return httpx.Response(404, text="not found")
        assert req.url.path == "/api/events/bfxa4p/snapshot.jpg"
        assert req.url.params.get("bbox") == "1"
        return httpx.Response(200, content=b"JPEGDATA")
    c = _client_with(handler)
    data = asyncio.run(c.event_snapshot("bfxa4p"))
    missing = asyncio.run(c.event_snapshot("gone"))
    asyncio.run(c.aclose())
    assert data == b"JPEGDATA"
    assert missing == b""  # 404 -> b"" so the UI can show "unavailable", not error
