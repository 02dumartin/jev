"""S0-S8 orchestration. Each stage writes into the run folder; the next stage reads only those files."""

import platform
import shutil
import time
from collections import Counter
from functools import partial
from pathlib import Path

import yaml
from dotenv import load_dotenv

from . import plots, viz
from .config import class_names
from .data import available_frames, build_frames, context_frames, load_gt
from .detect import DetectionCache
from .evaluate import evaluate_run, metrics_markdown
from .evidence import build_evidence
from .gate import GATES, combine, fit_detector_threshold, fit_gate_thresholds
from .judges.base import Budget
from .judges.threshold import ThresholdJudge
from .label import detector_coverage, label_candidates, label_summary, summary_markdown
from .runio import RunDir, write_jsonl

# Config sections a reused judge output depends on. Reuse is refused when any of them differ.
REUSE_DEPENDS = {
    "vlm": ["data", "splits", "run.split", "run.max_frames", "classes", "detector", "vlm", "prompts"],
    "jev": ["data", "splits", "run.split", "run.max_frames", "classes", "detector", "evidence", "jev", "prompts"],
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _section(cfg: dict, dotted: str):
    node = cfg
    for part in dotted.split("."):
        node = node.get(part) if isinstance(node, dict) else None
    return node


def env_info(cfg: dict) -> dict:
    """Package versions and GPU, for env.json."""
    import torch
    import transformers

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
            "gpu": gpu, "detector_model": cfg["detector"]["model"], "vlm_model": cfg["vlm"]["model"],
            "jev_model_requested": cfg["jev"]["model"]}


def resolve_thresholds(cfg: dict, run: RunDir, labels: list[dict], cands: list[dict], jev_rows) -> dict:
    """Dev: fit t, tau_need_vlm and tau_conf on this run. Test: load the dev-fitted file, refuse anything else."""
    if cfg["run"]["split"] == "test":
        th = yaml.safe_load(Path(cfg["thresholds_file"]).read_text())
        if th.get("fit_split") != "dev" or th.get("fit_splits") != cfg["splits"]["dev"] or th.get("max_frames"):
            raise ValueError("thresholds_file was not fitted on the full dev split; refusing test run")
        return th
    scored = [(c, r["y"]) for c, r in zip(cands, labels, strict=True) if r["scored"]]
    th = {
        "t": fit_detector_threshold([y for _, y in scored], [c["dino_label"] for c, _ in scored],
                                    [c["dino_score"] for c, _ in scored]),
        "fit_split": "dev", "fit_splits": cfg["splits"]["dev"], "fit_run": run.run_id,
        "fit_commit": run.git().get("commit"), "max_frames": cfg["run"].get("max_frames"),
    }
    if jev_rows is not None:
        jev = {r["cand_id"]: r for r in jev_rows}
        th |= fit_gate_thresholds([y for _, y in scored], [jev[c["cand_id"]] for c, _ in scored], cfg)
    return th


class Pipeline:
    """Runs the stages for one config. `reuse` maps 'vlm'/'jev' to an earlier RunDir with identical inputs."""

    def __init__(self, cfg: dict, repo: Path, reuse: dict[str, RunDir] | None = None):
        self.cfg = cfg
        self.repo = repo
        self.reuse = reuse or {}
        self.systems = cfg["run"]["systems"]
        self.budget = Budget(cfg)
        load_dotenv(repo / ".env")

    def _record_time(self, run: RunDir, name: str, seconds: float) -> None:
        timing = run.read_json("timing.json") if (run.path / "timing.json").exists() else {}
        run.write_json("timing.json", timing | {name: round(seconds, 3)})

    def _timed(self, run: RunDir, name: str, fn):
        started = time.perf_counter()
        out = fn()
        self._record_time(run, name, time.perf_counter() - started)
        return out

    def run(self) -> RunDir:
        cfg = self.cfg
        classes = class_names(cfg)
        run = RunDir.create(cfg, self.repo)
        run.write_json("env.json", env_info(cfg))
        log(f"run folder: {run.path}")
        gt = load_gt(cfg)

        # S1 frames
        frames = build_frames(cfg, gt)
        run.write("frames", frames)
        log(f"S1 frames: {len(frames)} ({dict(Counter(f['gt_status'] for f in frames))})")

        # S2 detect (target frames + previous K frames as causal context), cached per frame
        ctx = context_frames(frames, cfg["evidence"]["history_frames"], available_frames(cfg))
        cache = DetectionCache(cfg, self.repo)
        todo = cache.missing(ctx)
        log(f"S2 DINO: {len(ctx)} frames needed ({len(frames)} target + context), {len(todo)} not cached")
        if todo:
            started = time.perf_counter()
            from .detect import GroundingDinoDetector

            detector = GroundingDinoDetector(cfg)
            self._record_time(run, "detect_load", time.perf_counter() - started)
            cache.add(self._timed(run, "detect", lambda: detector.detect_frames(cfg, todo, log)))
            del detector
            _free_gpu()
        targets = [(f["clip"], f["frame_idx"]) for f in frames]
        cands = cache.candidates(targets)
        run.write("candidates", cands)
        run.write_json("detect_ms.json", [{"clip": c, "frame_idx": i, "detect_ms": ms}
                                          for (c, i), ms in cache.detect_ms(targets).items()])
        per_frame = Counter((c["clip"], c["frame_idx"]) for c in cands)
        run.write_json("detect_summary.json", {
            "target_frames": len(frames), "context_frames": len(ctx), "candidates": len(cands),
            "candidates_per_frame_hist": dict(sorted(Counter(per_frame.get(k, 0) for k in targets).items())),
            "dino_label_counts": dict(Counter(c["dino_label"] for c in cands)),
            "score_hist": dict(Counter(round(c["dino_score"], 1) for c in cands)),
            "cache_file": str(cache.path.relative_to(self.repo)),
        })
        log(f"S2 candidates: {len(cands)}")

        # S3 labels
        labels = label_candidates(cands, frames, gt, cfg)
        run.write("labels", labels)
        coverage = detector_coverage(cands, frames, gt, cfg["label"]["match_iou"], classes)
        summary = label_summary(cands, labels, coverage, classes)
        run.write_json("label_summary.json", summary)
        (run.path / "label_summary.md").write_text(summary_markdown(summary, classes))
        log(f"S3 labels: {summary['y_per_class']}, ambiguous {summary['ambiguous']}; "
            f"DINO recall upper bound {coverage['recall_upper_bound']:.3f}")

        # S4 evidence
        evidence = build_evidence(cands, cache.candidates(ctx), ctx, cfg)
        run.write("evidence", evidence)
        log(f"S4 evidence: {len(evidence)}")

        # S5 judges
        jev_rows = self._judge_jev(run, frames, cands, evidence) if "jev" in self.systems else None
        if "vlm" in self.systems:
            self._judge_vlm(run, frames, cands, evidence)
        run.write_json("usage.json", self.budget.summary())

        th = resolve_thresholds(cfg, run, labels, cands, jev_rows)
        run.write_json("thresholds.json", th)
        if cfg["run"]["split"] == "dev":
            (run.path / "thresholds.yaml").write_text(yaml.safe_dump(th, sort_keys=False))
        log("thresholds: " + ", ".join(f"{k}={v}" for k, v in th.items() if k == "t" or k.startswith("tau")))
        if "threshold" in self.systems:
            threshold = ThresholdJudge(th["t"], classes, {})
            run.write_decisions("threshold", self._timed(run, "threshold",
                                                         partial(threshold.judge, frames, cands, evidence)))
        for g in GATES:
            if g in self.systems:
                run.write_decisions(g, combine(g, jev_rows, run.read_decisions("vlm"), th))
        log(f"S5-S6 decisions: {run.systems_done()}")

        # S7 evaluation
        gt_count = {(f["clip"], f["frame_idx"]): len(gt.frame_boxes(f["file"])) for f in frames}
        metrics = self._timed(run, "evaluate", lambda: evaluate_run(run, gt_count, summary))
        metrics["budget_used"] = self.budget.summary()
        run.write_json("metrics.json", metrics)
        (run.path / "metrics.md").write_text(metrics_markdown(metrics))
        log("S7 metrics written")

        # S8 figures and overlays
        plots.plot_run(run)
        viz.render_samples(run, gt, cfg["viz"]["sample_frames"], cfg["viz"]["seed"])
        log("S8 plots and overlays written")
        return run

    def _reuse(self, run: RunDir, what: str) -> None:
        source = self.reuse[what]
        diffs = [k for k in REUSE_DEPENDS[what] if _section(self.cfg, k) != _section(source.config(), k)]
        if diffs:
            raise ValueError(f"cannot reuse {what} from {source.run_id}: config differs in {diffs}")
        run.decisions_file(what).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source.decisions_file(what), run.decisions_file(what))
        old = source.read_json("timing.json") if (source.path / "timing.json").exists() else {}
        if what in old:
            self._record_time(run, what, old[what])
        log(f"reused {what} from {source.run_id}")

    def _judge_jev(self, run, frames, cands, evidence):
        if "jev" in self.reuse:
            self._reuse(run, "jev")
        else:
            from .judges.jev import JevJudge

            judge = JevJudge(self.cfg, self.budget, log)
            run.write_decisions("jev", self._timed(run, "jev", partial(judge.judge, frames, cands, evidence)))
            write_jsonl(run.path / "raw" / "jev_responses.jsonl", judge.raw)
            log(f"S5 jev: {self.budget.jev_requests} requests, {self.budget.jev_tokens} tokens")
        return run.read_decisions("jev")

    def _judge_vlm(self, run, frames, cands, evidence):
        if "vlm" in self.reuse:
            self._reuse(run, "vlm")
            return
        from .judges.vlm import VLMJudge

        started = time.perf_counter()
        judge = VLMJudge(self.cfg, self.budget, log)
        self._record_time(run, "vlm_load", time.perf_counter() - started)
        (run.path / "raw").mkdir(exist_ok=True)
        (run.path / "raw" / "vlm_prompt.txt").write_text(judge.prompt)
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
