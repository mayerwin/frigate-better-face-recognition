import asyncio

import httpx

from app.frigate import FrigateClient, parse_filename


def test_parse_filename_standard():
    c = parse_filename("1782080480.272977-bfxa4p-1782080490.833032-Jenny-0.82.webp")
    assert c.event_id == "bfxa4p"
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
