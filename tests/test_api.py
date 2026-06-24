import numpy as np
from fastapi.testclient import TestClient

from app.config import Config
from app.db import EMB_DIM, Store
from app.main import create_app


class FakeFrigate:
    def __init__(self):
        self.assigned = []
        self.deleted = []

    async def version(self):
        return "0.17.1-test"

    async def assign(self, fn, name):
        self.assigned.append((fn, name))
        return {"success": True}

    async def delete_crop(self, fn, folder="train"):
        self.deleted.append(fn)
        return {"success": True}

    async def list_train(self):
        return []

    async def list_person_names(self):
        return ["Jenny", "erwin"]

    async def list_faces(self):
        return {"train": [], "Jenny": ["Jenny-100.5.webp", "Jenny-200.0.webp", "old.webp"], "erwin": []}

    async def login_ok(self, user, password):
        return user == "admin" and password == "secret"

    async def fetch_crop(self, fn, folder="train"):
        return b""

    async def aclose(self):
        pass


class FakeEmbedder:
    def embed(self, b):
        return True, 0.9, 120.0, np.ones(EMB_DIM, dtype=np.float32)


def unit(seed):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(EMB_DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def make(tmp_path, auth="none"):
    cfg = Config(data_dir=str(tmp_path), frigate_writeback=True, auth=auth)
    store = Store(cfg.db_path)
    frig = FakeFrigate()
    app = create_app(cfg, store, FakeEmbedder(), frig, run_ingest=False)
    return cfg, store, frig, app


def test_state_and_review_and_image(tmp_path):
    cfg, store, frig, app = make(tmp_path)
    cid = store.add_crop(frigate_id="1-a-2-unknown-0.6.webp", camera="", event_ts=1.0,
                         det_score=0.9, has_face=True, embedding=unit(1), thumb=b"webpbytes",
                         status="review", source_path="train/1-a-2-unknown-0.6.webp")
    with TestClient(app) as client:
        st = client.get("/api/state").json()
        assert st["counts"]["review"] == 1
        assert st["frigate"]["version"] == "0.17.1-test"
        rev = client.get("/api/review").json()
        assert len(rev) == 1 and rev[0]["id"] == cid
        img = client.get(f"/api/crops/{cid}/image")
        assert img.status_code == 200 and img.content == b"webpbytes"


def test_assign_trains_frigate_and_builds_gallery(tmp_path):
    cfg, store, frig, app = make(tmp_path)
    cid = store.add_crop(frigate_id="1-a-2-unknown-0.6.webp", camera="", event_ts=1.0,
                         det_score=0.9, has_face=True, embedding=unit(2), thumb=b"x",
                         status="review")
    with TestClient(app) as client:
        r = client.post(f"/api/crops/{cid}/assign", json={"name": "Bob"})
        assert r.status_code == 200 and r.json()["frigate_trained"] is True
    assert frig.assigned and frig.assigned[0][1] == "Bob"
    pg = store.positive_gallery()
    assert len(pg) == 1 and list(pg.values())[0].shape == (1, EMB_DIM)
    assert store.get_crop(cid)["status"] == "labeled"


def test_reject_deletes_in_frigate_and_builds_negative(tmp_path):
    cfg, store, frig, app = make(tmp_path)
    cid = store.add_crop(frigate_id="1-a-2-unknown-0.07.webp", camera="", event_ts=1.0,
                         det_score=0.9, has_face=True, embedding=unit(3), thumb=b"x",
                         status="review")
    with TestClient(app) as client:
        r = client.post(f"/api/crops/{cid}/reject")
        assert r.status_code == 200 and r.json()["frigate_deleted"] is True
    assert frig.deleted == ["1-a-2-unknown-0.07.webp"]
    assert store.negative_gallery().shape[0] == 1
    assert store.get_crop(cid)["status"] == "rejected"


def test_settings_update_validates_key(tmp_path):
    cfg, store, frig, app = make(tmp_path)
    with TestClient(app) as client:
        ok = client.post("/api/settings", json={"key": "match_threshold", "value": 0.62})
        assert ok.status_code == 200 and abs(ok.json()["match_threshold"] - 0.62) < 1e-6
        bad = client.post("/api/settings", json={"key": "evil", "value": 1})
        assert bad.status_code == 400


def test_assign_reuses_existing_frigate_casing(tmp_path):
    # FakeFrigate already has "erwin"; labelling "ERWIN" must reuse that exact
    # casing so a duplicate folder isn't created in Frigate.
    cfg, store, frig, app = make(tmp_path)
    cid = store.add_crop(frigate_id="x.webp", camera="", event_ts=1, det_score=0.9,
                         has_face=True, embedding=unit(4), thumb=b"x", status="review")
    with TestClient(app) as client:
        assert client.post(f"/api/crops/{cid}/assign", json={"name": "ERWIN"}).status_code == 200
    assert frig.assigned[0][1] == "erwin"


def test_auth_required_and_login(tmp_path):
    cfg, store, frig, app = make(tmp_path, auth="frigate")
    with TestClient(app) as client:
        assert client.get("/api/state").status_code == 401          # no session
        assert client.post("/login", json={"user": "x", "password": "y"}).status_code == 401
        assert client.post("/login", json={"user": "admin", "password": "secret"}).status_code == 200
        assert client.get("/api/state").status_code == 200          # session set
        client.post("/logout")
        assert client.get("/api/state").status_code == 401          # cleared


def test_people_counts_come_from_frigate(tmp_path):
    cfg, store, frig, app = make(tmp_path)
    with TestClient(app) as client:
        ppl = {p["name"]: p["count"] for p in client.get("/api/people").json()}
    assert ppl.get("Jenny") == 3   # Frigate folder count, not a separate label count
    assert ppl.get("erwin") == 0


def test_person_crops_ordered_and_delete(tmp_path):
    cfg, store, frig, app = make(tmp_path)
    with TestClient(app) as client:
        r = client.get("/api/person/Jenny/crops").json()
        assert r["name"] == "Jenny"
        # newest first by trailing timestamp (200.0 > 100.5 > none)
        assert [c["filename"] for c in r["crops"]] == ["Jenny-200.0.webp", "Jenny-100.5.webp", "old.webp"]
        assert client.post("/api/person/Jenny/crops/delete", json={"filename": "Jenny-200.0.webp"}).status_code == 200
    assert "Jenny-200.0.webp" in frig.deleted


def test_build_default_app_constructs(tmp_path, monkeypatch):
    # Smoke test: constructs the real production wiring (Config + Store + Embedder
    # + FrigateClient + create_app) without starting the server or loading models.
    # Catches wiring errors like referencing a Config field that does not exist.
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("FRIGATE_URL", "http://frigate:5000")
    monkeypatch.setenv("AUTH", "none")
    from app import main as mainmod
    app = mainmod.build_default_app()
    assert app is not None


def test_undo_returns_to_review(tmp_path):
    cfg, store, frig, app = make(tmp_path)
    cid = store.add_crop(frigate_id="z.webp", camera="", event_ts=1.0, det_score=0.0,
                         has_face=False, embedding=None, thumb=b"x", status="auto_rejected",
                         reason="no_face")
    with TestClient(app) as client:
        r = client.post(f"/api/crops/{cid}/undo")
        assert r.status_code == 200
    assert store.get_crop(cid)["status"] == "review"
