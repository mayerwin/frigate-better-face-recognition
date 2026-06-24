"""The 3-way classifier: the whole "brain" of the cascade, kept as a pure
function over numpy arrays so it is trivially testable and has no I/O.

Given one face embedding it returns exactly one Decision:

  * no face present (SCRFD found nothing)        -> auto_rejected (no_face)
  * matches a human "not a face" reject          -> auto_rejected (matches_reject)
  * matches a known person above the threshold   -> review + suggestion
                                                    (or auto_labeled, opt-in)
  * none of the above (a genuine, unknown face)  -> review (unknown)

Embeddings are assumed L2-normalised (InsightFace `normed_embedding`), so a
dot product IS cosine similarity.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Decision:
    status: str  # review | auto_rejected | auto_labeled
    reason: str = ""  # no_face | matches_reject | suggested | match | unknown
    person_id: Optional[int] = None
    suggested_person_id: Optional[int] = None
    score: float = 0.0


def _max_sim(emb: np.ndarray, gallery: Optional[np.ndarray]):
    if gallery is None or len(gallery) == 0:
        return 0.0, None
    sims = gallery @ emb
    i = int(np.argmax(sims))
    return float(sims[i]), i


def classify(
    emb: Optional[np.ndarray],
    has_face: bool,
    pos_gallery: dict,
    neg_gallery: np.ndarray,
    *,
    match_threshold: float,
    reject_threshold: float,
    auto_reject: bool,
    auto_label: bool,
    blur_score: Optional[float] = None,
    blur_threshold: float = 0.0,
) -> Decision:
    # 1. SCRFD verification: no face in the crop -> it is junk (a wheel, a wall).
    #    This is always auto-rejected; it is the "is it really a face" gate.
    #    Such crops stay auditable + undoable, never silently destroyed.
    if not has_face or emb is None:
        return Decision(status="auto_rejected", reason="no_face")

    # 1b. Optional quality gate: auto-discard faces below the chosen quality. A
    #     missing score (quality model unavailable) is never auto-rejected here.
    if blur_threshold and blur_threshold > 0 and blur_score is not None and blur_score < blur_threshold:
        return Decision(status="auto_rejected", reason="too_blurry", score=float(blur_score))

    # 2. Negative learning: does it match something you already called junk?
    neg_sim, _ = _max_sim(emb, neg_gallery)
    looks_like_reject = neg_sim >= reject_threshold
    if auto_reject and looks_like_reject:
        return Decision(status="auto_rejected", reason="matches_reject", score=neg_sim)

    # 3. Person matching: nearest enrolled person.
    best_pid, best = None, 0.0
    for pid, gal in (pos_gallery or {}).items():
        s, _ = _max_sim(emb, gal)
        if s > best:
            best, best_pid = s, pid

    if best_pid is not None and best >= match_threshold:
        # Never auto-label something that also looks like known junk, even if
        # auto_reject is off -- that would write a wrong identity into Frigate.
        if auto_label and not looks_like_reject:
            return Decision(status="auto_labeled", reason="match", person_id=best_pid, score=best)
        # Default: only SUGGEST. A human must confirm before we tell Frigate.
        return Decision(status="review", reason="suggested", suggested_person_id=best_pid, score=best)

    # 4. A genuine face we do not recognise -> queue it for labelling.
    return Decision(status="review", reason="unknown", score=best)
