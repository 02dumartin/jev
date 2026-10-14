"""① Existing detector: the DINO label when its score >= t, otherwise background."""

from jevgate.core.config import BACKGROUND
from jevgate.judges.base import Judge, decision


def detector_probs(dino_label: str, score: float, classes: list[str]) -> dict[str, float]:
    """Pseudo-probabilities used for ROC: DINO label = score, background = 1 - score, others 0."""
    probs = dict.fromkeys(classes, 0.0)
    if dino_label in probs and dino_label != BACKGROUND:
        probs[dino_label] = score
    probs[BACKGROUND] = 1.0 - score
    return probs


class ThresholdJudge(Judge):
    name = "threshold"

    def __init__(self, t: float, classes: list[str], detect_ms: dict[tuple, float]):
        self.t = t
        self.classes = classes
        self.detect_ms = detect_ms

    def judge(self, frames, candidates, evidence):
        rows = []
        for c in candidates:
            row = decision(c["cand_id"], detector_probs(c["dino_label"], c["dino_score"], self.classes), 0.0,
                           t=self.t)
            row["pred"] = c["dino_label"] if c["dino_score"] >= self.t and c["dino_label"] in self.classes \
                else BACKGROUND
            row["confidence"] = row["probs"][row["pred"]]
            rows.append(row)
        return rows
