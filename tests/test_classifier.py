import numpy as np

from app.classifier import classify


def unit(*vals):
    v = np.array(vals, dtype=np.float32)
    return v / np.linalg.norm(v)


def gal(*vecs):
    return np.vstack(vecs).astype(np.float32)


EMPTY = np.zeros((0, 3), dtype=np.float32)
TH = dict(match_threshold=0.5, reject_threshold=0.5)


def test_no_face_is_auto_rejected():
    d = classify(unit(1, 0, 0), False, {}, EMPTY, auto_reject=True, auto_label=False, **TH)
    assert d.status == "auto_rejected" and d.reason == "no_face"


def test_none_embedding_is_auto_rejected():
    d = classify(None, True, {}, EMPTY, auto_reject=True, auto_label=False, **TH)
    assert d.status == "auto_rejected" and d.reason == "no_face"


def test_matches_reject_gallery():
    wheel = unit(0, 1, 0)
    neg = gal(unit(0, 0.98, 0.02))  # a near-duplicate wheel already rejected
    d = classify(wheel, True, {}, neg, auto_reject=True, auto_label=False, **TH)
    assert d.status == "auto_rejected" and d.reason == "matches_reject"
    assert d.score >= 0.5


def test_auto_reject_disabled_sends_repeat_junk_to_review():
    wheel = unit(0, 1, 0)
    neg = gal(unit(0, 1, 0))
    d = classify(wheel, True, {}, neg, auto_reject=False, auto_label=False, **TH)
    assert d.status == "review"


def test_suggests_person_but_never_autolabels_by_default():
    face = unit(1, 0, 0)
    pos = {7: gal(unit(0.97, 0.03, 0))}
    d = classify(face, True, pos, EMPTY, auto_reject=True, auto_label=False, **TH)
    assert d.status == "review"
    assert d.suggested_person_id == 7
    assert d.person_id is None
    assert d.score > 0.5


def test_autolabel_only_when_explicitly_enabled():
    face = unit(1, 0, 0)
    pos = {7: gal(unit(0.97, 0.03, 0))}
    d = classify(face, True, pos, EMPTY, auto_reject=True, auto_label=True, **TH)
    assert d.status == "auto_labeled" and d.person_id == 7


def test_unknown_face_goes_to_review_without_suggestion():
    face = unit(1, 0, 0)
    pos = {7: gal(unit(0, 1, 0))}  # orthogonal -> low similarity
    d = classify(face, True, pos, EMPTY, auto_reject=True, auto_label=False, **TH)
    assert d.status == "review" and d.suggested_person_id is None


def test_reject_takes_precedence_over_person_match():
    # A crop that is close to BOTH a person and a reject should be rejected:
    # protecting Frigate from training on something you flagged as junk.
    v = unit(1, 1, 0)
    pos = {3: gal(unit(1, 1, 0))}
    neg = gal(unit(1, 1, 0))
    d = classify(v, True, pos, neg, auto_reject=True, auto_label=True, **TH)
    assert d.status == "auto_rejected" and d.reason == "matches_reject"


def test_too_blurry_is_auto_rejected_when_threshold_set():
    face = unit(1, 0, 0)
    # sharp enough -> not blurry-rejected
    d = classify(face, True, {}, EMPTY, auto_reject=True, auto_label=False,
                 blur_score=200.0, blur_threshold=100.0, **TH)
    assert d.status == "review"
    # below the sharpness threshold -> auto-rejected as too_blurry
    d2 = classify(face, True, {}, EMPTY, auto_reject=True, auto_label=False,
                  blur_score=40.0, blur_threshold=100.0, **TH)
    assert d2.status == "auto_rejected" and d2.reason == "too_blurry"


def test_no_autolabel_of_known_junk_when_auto_reject_off():
    # auto_reject off + auto_label on: a crop matching BOTH a reject and a person
    # must NOT be auto-labelled into Frigate; it goes to review instead.
    v = unit(1, 1, 0)
    pos = {3: gal(unit(1, 1, 0))}
    neg = gal(unit(1, 1, 0))
    d = classify(v, True, pos, neg, auto_reject=False, auto_label=True, **TH)
    assert d.status == "review"
