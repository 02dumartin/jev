"""S1-S8 orchestration. Each stage writes into the run folder; the next stage reads only those files."""

import shutil
import time
from collections import Counter
from functools import partial
from pathlib import Path

import yaml
from dotenv import load_dotenv

from . import viz
from .data import build_frames, load_gt
from .evaluate import evaluate_run
from .evidence import build_evidence
from .gate import combine, fit_detector_threshold, fit_tau
from .judges.base import Budget
from .judges.threshold import ThresholdJudge
from .label import detector_recall_upper_bound, label_candidates
from .runio import RunDir

# Config sections a reused output depends on. Reuse is refused when any of them differ.
REUSE_DEPENDS = {
    "candidates": ["data", "clips", "run.split", "run.max_frames_per_clip", "detector"],
    "vlm": ["data", "clips", "run.split", "run.max_frames_per_clip", "detector", "vlm", "prompts"],
    "jev": ["data", "clips", "run.split", "run.max_frames_per_clip", "detector", "evidence", "jev", "prompts"],
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _section(cfg: dict, dotted: str):
    node = cfg
    for part in dotted.split("."):
        node = node.get(part) if isinstance(node, dict) else None
    return node


def reuse_output(run: RunDir, source: RunDir, what: str, cfg: dict) -> None:
    """Copy one stage output from an earlier run after checking that its inputs were identical."""
    src_cfg = source.config()
    diffs = [k for k in REUSE_DEPENDS[what] if _section(cfg, k) != _section(src_cfg, k)]
    if diffs:
        raise ValueError(f"cannot reuse {what} from {source.run_id}: config differs in {diffs}")
    src = source.file("candidates") if what == "candidates" else source.decisions_file(what)
    dst = run.file("candidates") if what == "candidates" else run.decisions_file(what)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)
    if what != "candidates" and (source.path / "timing.json").exists():
        old = source.read_json("timing.json")
        if what in old:
            timing = run.read_json("timing.json") if (run.path / "timing.json").exists() else {}
            run.write_json("timing.json", timing | {what: old[what]})
    log(f"reused {what} from {source.run_id}")


def resolve_thresholds(cfg: dict, run: RunDir, labels: list[dict], cands: list[dict], jev_rows) -> dict:
    """Dev: fit t and tau on this run. Test: load the dev-fitted file and refuse anything else."""
    if cfg["run"]["split"] == "test":
        th = yaml.safe_load(Path(cfg["thresholds_file"]).read_text())
        if th.get("fit_split") != "dev" or th.get("fit_clips") != cfg["clips"]["dev"]:
            raise ValueError("thresholds_file was not fitted on the configured dev clips; refusing test run")
        return th
    scored = {r["cand_id"]: r["y"] for r in labels if r["scored"]}
    score = {c["cand_id"]: c["dino_score"] for c in cands}
    ids = list(scored)
    th = {
        "t": fit_detector_threshold([scored[i] for i in ids], [score[i] for i in ids]),
        "fit_split": "dev",
        "fit_clips": cfg["clips"]["dev"],
        "fit_run": run.run_id,
        "fit_commit": run.git().get("commit"),
        "max_frames_per_clip": cfg["run"].get("max_frames_per_clip"),
    }
    if jev_rows is not None:
        p = {r["cand_id"]: r["p_real"] for r in jev_rows}
        th["tau"] = fit_tau([scored[i] for i in ids], [p[i] for i in ids],
                            cfg["gate"]["max_auto_error"], cfg["gate"]["tau_grid_step"])
        th["max_auto_error"] = cfg["gate"]["max_auto_error"]
    return th


class Pipeline:
    """Runs the stages for one config. `reuse` maps 'candidates'/'vlm'/'jev' to an earlier RunDir."""

    def __init__(self, cfg: dict, repo: Path, reuse: dict[str, RunDir] | None = None):
        self.cfg = cfg
        self.repo = repo
        self.reuse = reuse or {}
        self.systems = cfg["run"]["systems"]
        self.budget = Budget(cfg)
        load_dotenv(repo / ".env")

    def _timed(self, run: RunDir, name: str, fn):
        started = time.perf_counter()
        out = fn()
        timing = run.read_json("timing.json") if (run.path / "timing.json").exists() else {}
        run.write_json("timing.json", timing | {name: round(time.perf_counter() - started, 3)})
        return out

    def run(self) -> RunDir:
        cfg = self.cfg
        run = RunDir.create(cfg, self.repo)
        log(f"run folder: {run.path}")
        gt = load_gt(cfg)

        frames = build_frames(cfg, gt)
        run.write("frames", frames)
        log(f"S1 frames: {len(frames)} ({dict(Counter(f['gt_status'] for f in frames))})")

        if "candidates" in self.reuse:
            reuse_output(run, self.reuse["candidates"], "candidates", cfg)
        else:
            from .detect import GroundingDinoDetector

            detector = GroundingDinoDetector(cfg)
            run.write("candidates", self._timed(run, "detect", partial(detector.detect, cfg, frames)))
            del detector
            _free_gpu()
        cands = run.read("candidates")
        log(f"S2 candidates: {len(cands)}")

        labels = label_candidates(cands, frames, gt, cfg)
        run.write("labels", labels)
        recall_ub = detector_recall_upper_bound(cands, frames, gt, cfg["label"]["real_iou"])
        run.write_json("detector_recall.json", recall_ub)
        log(f"S3 labels: real {sum(r['y'] == 1 for r in labels)}, fake {sum(r['y'] == 0 for r in labels)}, "
            f"ambiguous {sum(r['y'] is None for r in labels)}; DINO recall upper bound "
            f"{recall_ub['recall_upper_bound']}")

        evidence = build_evidence(cands, cfg)
        run.write("evidence", evidence)
        log(f"S4 evidence: {len(evidence)}")

        jev_rows = self._judge_jev(run, frames, cands, evidence) if "jev" in self.systems else None
        if "vlm" in self.systems:
            self._judge_vlm(run, frames, cands, evidence)

        th = resolve_thresholds(cfg, run, labels, cands, jev_rows)
        run.write_json("thresholds.json", th)
        log(f"thresholds: t={th['t']:.4f}" + (f", tau={th['tau']:.2f}" if "tau" in th else ""))
        if "threshold" in self.systems:
            run.write_decisions("threshold", ThresholdJudge(th["t"]).judge(frames, cands, evidence))
            run.write_json("timing.json", run.read_json("timing.json") | {"threshold": 0.0})
        if "gate" in self.systems:
            run.write_decisions("gate", combine(jev_rows, run.read_decisions("vlm"), th["tau"]))
        log(f"S5-S6 decisions: {run.systems_done()}")

        gt_count = {(f["clip"], f["frame_idx"]): len(gt.frame_boxes(f["file"])) for f in frames}
        metrics = evaluate_run(run, gt_count, recall_ub)
        metrics["budget_used"] = self.budget.summary()
        run.write_json("metrics.json", metrics)
        log("S7 metrics written")

        viz.plot_run(run)
        log("S8 plots written")
        return run

    def _judge_jev(self, run, frames, cands, evidence):
        if "jev" in self.reuse:
            reuse_output(run, self.reuse["jev"], "jev", self.cfg)
        else:
            from .judges.jev import JevJudge

            judge = JevJudge(self.cfg, self.budget)
            run.write_decisions("jev", self._timed(run, "jev", partial(judge.judge, frames, cands, evidence)))
            log(f"S5 jev: {self.budget.jev_requests} requests, {self.budget.jev_tokens} tokens")
        return run.read_decisions("jev")

    def _judge_vlm(self, run, frames, cands, evidence):
        if "vlm" in self.reuse:
            reuse_output(run, self.reuse["vlm"], "vlm", self.cfg)
            return
        from .judges.vlm import VLMJudge

        judge = VLMJudge(self.cfg, self.budget)
        run.write_decisions("vlm", self._timed(run, "vlm", partial(judge.judge, frames, cands, evidence)))
        log(f"S5 vlm: {self.budget.vlm_calls} calls")
        del judge
        _free_gpu()


def _free_gpu() -> None:
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
