"""① Existing detector: keep a box when its Grounding DINO score >= t."""

from .base import Judge, decision


class ThresholdJudge(Judge):
    name = "threshold"

    def __init__(self, t: float):
        self.t = t

    def judge(self, frames, candidates, evidence):
        return [decision(c["cand_id"], c["dino_score"] >= self.t, None, 0.0, score=c["dino_score"], t=self.t)
                for c in candidates]
