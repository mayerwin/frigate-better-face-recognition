import numpy as np

from app.db import EMB_DIM, Store


def store(tmp_path):
    return Store(str(tmp_path / "t.db"))


def unit(seed, dim=EMB_DIM):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def test_seen_and_add(tmp_path):
    s = store(tmp_path)
    assert not s.seen("evt1")
    s.add_crop(frigate_id="evt1", camera="cam", event_ts=1.0, det_score=0.9,
               has_face=True, embedding=unit(1), thumb=b"x", status="review")
    assert s.seen("evt1")
    assert s.counts().get("review") == 1


def test_duplicate_frigate_id_is_ignored(tmp_path):
    s = store(tmp_path)
    s.add_crop(frigate_id="dup", camera="c", event_ts=1, det_score=0.9,
               has_face=True, embedding=unit(1), thumb=b"x", status="review")
    s.add_crop(frigate_id="dup", camera="c", event_ts=2, det_score=0.9,
               has_face=True, embedding=unit(2), thumb=b"y", status="review")
    assert s.counts().get("review") == 1


def test_label_builds_positive_gallery(tmp_path):
    s = store(tmp_path)
    pid = s.create_person("Erwin")
    cid = s.add_crop(frigate_id="e2", camera="c", event_ts=1, det_score=0.9,
                     has_face=True, embedding=unit(2), thumb=b"x", status="review")
    s.set_decision(cid, "labeled", person_id=pid, reason="manual")
    g = s.positive_gallery()
    assert pid in g and g[pid].shape == (1, EMB_DIM)


def test_reject_builds_negative_gallery(tmp_path):
    s = store(tmp_path)
    cid = s.add_crop(frigate_id="e3", camera="c", event_ts=1, det_score=0.9,
                     has_face=True, embedding=unit(3), thumb=b"x", status="review")
    s.set_decision(cid, "rejected", reason="not a face")
    assert s.negative_gallery().shape == (1, EMB_DIM)


def test_auto_decisions_do_not_seed_galleries(tmp_path):
    s = store(tmp_path)
    pid = s.create_person("X")
    s.add_crop(frigate_id="a1", camera="c", event_ts=1, det_score=0.9, has_face=True,
               embedding=unit(4), thumb=b"x", status="auto_labeled", person_id=pid)
    s.add_crop(frigate_id="a2", camera="c", event_ts=1, det_score=0.9, has_face=True,
               embedding=unit(5), thumb=b"x", status="auto_rejected", reason="no_face")
    assert s.positive_gallery() == {}
    assert s.negative_gallery().shape == (0, EMB_DIM)


def test_person_labeled_count(tmp_path):
    s = store(tmp_path)
    pid = s.create_person("Y")
    cid = s.add_crop(frigate_id="p1", camera="c", event_ts=1, det_score=0.9,
                     has_face=True, embedding=unit(6), thumb=b"x", status="review")
    s.set_decision(cid, "labeled", person_id=pid)
    people = {p["name"]: p for p in s.list_persons()}
    assert people["Y"]["labeled_count"] == 1


def test_settings_seed_then_override(tmp_path):
    s = store(tmp_path)
    s.seed_settings({"match_threshold": 0.5, "auto_label": False})
    assert s.get_settings()["match_threshold"] == 0.5
    s.set_setting("match_threshold", 0.42)
    assert s.get_settings()["match_threshold"] == 0.42
    # re-seeding must not clobber a value the user already tuned
    s.seed_settings({"match_threshold": 0.9})
    assert s.get_settings()["match_threshold"] == 0.42


def test_invalid_status_rejected(tmp_path):
    s = store(tmp_path)
    try:
        s.add_crop(frigate_id="bad", camera="c", event_ts=1, det_score=0.9,
                   has_face=True, embedding=unit(7), thumb=b"x", status="bogus")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_duplicate_add_returns_existing_id(tmp_path):
    s = store(tmp_path)
    a = s.add_crop(frigate_id="dup2", camera="c", event_ts=1, det_score=0.9, has_face=True,
                   embedding=unit(9), thumb=b"x", status="review")
    b = s.add_crop(frigate_id="dup2", camera="c", event_ts=2, det_score=0.9, has_face=True,
                   embedding=unit(10), thumb=b"y", status="review")
    assert a > 0 and a == b


def test_get_or_create_person_is_case_insensitive(tmp_path):
    s = store(tmp_path)
    assert s.get_or_create_person("Erwin") == s.get_or_create_person("erwin")
    assert len(s.list_persons()) == 1


def _add_aged(s, fid, status, age_days, **kw):
    """Insert a crop, then backdate created_at by age_days for retention tests."""
    import time as _t
    cid = s.add_crop(frigate_id=fid, camera="c", event_ts=1, det_score=0.9,
                     has_face=True, embedding=unit(hash(fid) % 1000), thumb=b"x",
                     status=status, **kw)
    with s._lock:
        s._conn.execute("UPDATE crops SET created_at=? WHERE id=?",
                        (_t.time() - age_days * 86400.0, cid))
        s._conn.commit()
    return cid


def test_retention_prunes_only_aged_non_gallery_rows(tmp_path):
    s = store(tmp_path)
    pid = s.create_person("Keep")
    # OLD rows across every status
    _add_aged(s, "ar_old", "auto_rejected", 100, reason="no_face")
    _add_aged(s, "del_old", "deleted", 100)
    _add_aged(s, "rev_old", "review", 400)
    _add_aged(s, "lab_old", "labeled", 500, person_id=pid)      # gallery -> must survive
    _add_aged(s, "rej_old", "rejected", 500)                     # gallery -> must survive
    _add_aged(s, "auto_lab_old", "auto_labeled", 500, person_id=pid)  # not pruned
    # RECENT rows that must survive
    _add_aged(s, "ar_new", "auto_rejected", 10, reason="no_face")
    _add_aged(s, "rev_new", "review", 30)

    n = s.prune_old_crops(auto_rejected_days=90, review_days=365)
    assert n == 3  # ar_old + del_old + rev_old

    left = {r for r in ("ar_old", "del_old", "rev_old", "lab_old", "rej_old",
                        "auto_lab_old", "ar_new", "rev_new") if s.seen(r)}
    assert left == {"lab_old", "rej_old", "auto_lab_old", "ar_new", "rev_new"}
    # galleries intact
    assert s.negative_gallery().shape == (1, EMB_DIM)
    assert pid in s.positive_gallery()


def test_retention_zero_days_disables(tmp_path):
    s = store(tmp_path)
    _add_aged(s, "ar", "auto_rejected", 9999, reason="no_face")
    _add_aged(s, "rv", "review", 9999)
    assert s.prune_old_crops(auto_rejected_days=0, review_days=0) == 0
    assert s.seen("ar") and s.seen("rv")


def test_list_by_statuses_single_query(tmp_path):
    s = store(tmp_path)
    s.add_crop(frigate_id="r1", camera="c", event_ts=1, det_score=0.0, has_face=False,
               embedding=None, thumb=b"x", status="auto_rejected", reason="no_face")
    cid = s.add_crop(frigate_id="r2", camera="c", event_ts=2, det_score=0.9, has_face=True,
                     embedding=unit(11), thumb=b"x", status="review")
    s.set_decision(cid, "rejected", reason="not_a_face")
    rows = s.list_by_statuses(["auto_rejected", "rejected"])
    assert len(rows) == 2
