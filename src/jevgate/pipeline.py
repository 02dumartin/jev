"""S0-S8 orchestration for one config. The run folder layout is fixed in runio.py."""

import platform
import shutil
import time
from collections import Counter
from functools import partial
from pathlib import Path

import yaml
from dotenv import load_dotenv

from jevgate.core.config import class_names
from jevgate.core.runio import RunDir, read_jsonl, write_jsonl
from jevgate.data.brackish import BrackishGT, available_frames, build_frames, context_frames, frame_file, load_gt
from jevgate.data.labels import detector_coverage, label_candidates, label_summary
from jevgate.evaluation.evaluate import evaluate_run
from jevgate.judges.base import Budget
from jevgate.judges.evidence import build_evidence
from jevgate.judges.gate import GATES, combine, fit_detector_threshold, fit_gate_thresholds
from jevgate.judges.threshold import ThresholdJudge
from jevgate.outputs import plots, report, viz

# Config sections a reused judge output depends on. Reuse is refused when any of them differ.
REUSE_DEPENDS = {
    "vlm": ["data", "splits", "run.split", "run.max_frames", "candidates", "classes", "detector", "vlm", "prompts"],
    "jev": ["data", "splits", "run.split", "run.max_frames", "candidates", "classes", "detector", "evidence", "jev",
            "prompts"],
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _section(cfg: dict, dotted: str):
    node = cfg
    for part in dotted.split("."):
        node = node.get(part) if isinstance(node, dict) else None
    return node


def env_info(cfg: dict) -> dict:
    """Package versions, GPU and model names for run.yaml."""
    import torch
    import transformers

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "transformers": str(transformers.__version__), "gpu": gpu, "vlm_model": cfg["vlm"]["model"],
            "jev_model_requested": cfg["jev"]["model"],
            "detector_model": cfg["detector"]["model"] if cfg["candidates"]["source"] == "dino" else None}


def gt_candidates(gt: BrackishGT, keys) -> list[dict]:
    """GT boxes as candidates (no detector fields). Ordered by (clip, frame, AAU row order)."""
    out = []
    for clip, idx in keys:
        for k, g in enumerate(gt.frame_boxes(frame_file(clip, idx))):
            out.append({"cand_id": f"{clip[-25:]}-{idx:04d}-g{k:02d}", "clip": clip, "frame_idx": idx,
                        "box_xyxy": [round(v, 1) for v in g.box_xyxy], "gt_label": g.label,
                        "gt_object_id": g.object_id})
    return out


def gt_labels(cands: list[dict]) -> list[dict]:
    """S3 for GT candidates: the truth is the box's own AAU class."""
    return [{"cand_id": c["cand_id"], "max_iou": 1.0, "y": c["gt_label"], "matched_object_id": c["gt_object_id"],
             "matched_gt_label": c["gt_label"], "scored": True} for c in cands]


def resolve_thresholds(cfg: dict, run: RunDir, boxes: list[dict], jev_rows) -> dict:
    """Dev: fit thresholds on this run. Test: load the dev-fitted file and refuse anything inconsistent."""
    source = cfg["candidates"]["source"]
    if cfg["run"]["split"] == "test":
        th = yaml.safe_load(Path(cfg["thresholds_file"]).read_text())
        if (th.get("fit_split") != "dev" or th.get("fit_splits") != cfg["splits"]["dev"] or th.get("max_frames")
                or th.get("candidate_source") != source):
            raise ValueError("thresholds_file was not fitted on the full dev split with the same candidate source")
        return th
    scored = [b for b in boxes if b["scored"]]
    th = {"fit_split": "dev", "fit_splits": cfg["splits"]["dev"], "candidate_source": source, "fit_run": run.run_id,
          "fit_commit": run.git().get("commit"), "max_frames": cfg["run"].get("max_frames")}
    if source == "dino":
        th["t"] = fit_detector_threshold([b["y"] for b in scored], [b["dino_label"] for b in scored],
                                         [b["dino_score"] for b in scored])
    if jev_rows is not None:
        jev = {r["cand_id"]: r for r in jev_rows}
        th |= fit_gate_thresholds([b["y"] for b in scored], [jev[b["cand_id"]] for b in scored], cfg)
    return th


class Pipeline:
    """Runs the stages for one config. `reuse` maps 'vlm'/'jev' to an earlier RunDir with identical inputs."""

    def __init__(self, cfg: dict, repo: Path, reuse: dict[str, RunDir] | None = None,
                 resume_jev: list[Path] | None = None):
        self.cfg = cfg
        self.repo = repo
        self.reuse = reuse or {}
        self.resume_jev = resume_jev or []
        self.systems = cfg["run"]["systems"]
        self.budget = Budget(cfg)
        self.timing: dict[str, float] = {}
        load_dotenv(repo / ".env")

    def _timed(self, name: str, fn):
        started = time.perf_counter()
        out = fn()
        self.timing[name] = round(time.perf_counter() - started, 3)
        return out

    # S2 ------------------------------------------------------------------------------------
    def _candidates(self, frames, gt, ctx):
        """(target candidates, context candidates, detect ms per target frame)."""
        targets = [(f["clip"], f["frame_idx"]) for f in frames]
        if self.cfg["candidates"]["source"] == "gt":
            return gt_candidates(gt, targets), gt_candidates(gt, ctx), {}
        from jevgate.detectors.dino import DetectionCache

        cache = DetectionCache(self.cfg, self.repo)
        todo = cache.missing(ctx)
        log(f"S2 DINO: {len(ctx)} frames needed ({len(frames)} target + context), {len(todo)} not cached")
        if todo:
            from jevgate.detectors.dino import GroundingDinoDetector

            detector = self._timed("detect_load", partial(GroundingDinoDetector, self.cfg))
            cache.add(self._timed("detect", partial(detector.detect_frames, self.cfg, todo, log)))
            del detector
            _free_gpu()
        return cache.candidates(targets), cache.candidates(ctx), cache.detect_ms(targets)

    def run(self) -> RunDir:
        cfg = self.cfg
        classes = class_names(cfg)
        source = cfg["candidates"]["source"]
        run = RunDir.create(cfg, self.repo)
        run.update_meta({"env": env_info(cfg)})
        log(f"run folder: {run.path}")
        gt = load_gt(cfg)

        # S1-S2 frames and candidate boxes (target frames + previous K frames as causal context)
        frames = build_frames(cfg, gt)
        ctx = context_frames(frames, cfg["evidence"]["history_frames"], available_frames(cfg))
        cands, ctx_cands, detect_ms = self._candidates(frames, gt, ctx)
        for f in frames:
            f["detect_ms"] = detect_ms.get((f["clip"], f["frame_idx"]))
        run.write("frames", frames)
        log(f"S1-S2 frames {len(frames)} ({dict(Counter(f['gt_status'] for f in frames))}), "
            f"candidates {len(cands)} from {source}")

        # S3-S4 truth and evidence -> boxes.jsonl
        labels = gt_labels(cands) if source == "gt" else label_candidates(cands, frames, gt, cfg)
        evidence = build_evidence(cands, ctx_cands, ctx, cfg)
        file_of = {(f["clip"], f["frame_idx"]): f["file"] for f in frames}
        boxes = [{**{k: v for k, v in c.items() if k not in ("gt_label", "gt_object_id")},
                  "file": file_of[(c["clip"], c["frame_idx"])],
                  **{k: v for k, v in lab.items() if k != "cand_id"},
                  "evidence": {k: v for k, v in ev.items() if k != "cand_id"}}
                 for c, lab, ev in zip(cands, labels, evidence, strict=True)]
        run.write("boxes", boxes)
        coverage = detector_coverage(cands, frames, gt, cfg["label"]["match_iou"], classes)
        n_flow = label_summary(cands, labels, coverage, classes)
        log(f"S3 y: {n_flow['y_per_class']}, ambiguous {n_flow['ambiguous']}, "
            f"recall upper bound {coverage['recall_upper_bound']:.3f}")

        # S5 judges (expensive outputs go to cache/)
        jev_rows = self._judge_jev(run, frames, cands, evidence) if "jev" in self.systems else None
        vlm_rows = self._judge_vlm(run, frames, cands, evidence) if "vlm" in self.systems else None

        # S6 thresholds and cheap systems
        th = resolve_thresholds(cfg, run, boxes, jev_rows)
        run.update_meta({"thresholds": th})
        log("thresholds: " + ", ".join(f"{k}={v}" for k, v in th.items() if k == "t" or k.startswith("tau")))
        rows = {}
        if "threshold" in self.systems:
            rows["threshold"] = self._timed("threshold", partial(ThresholdJudge(th["t"], classes, {}).judge,
                                                                 frames, cands, evidence))
        if vlm_rows is not None:
            rows["vlm"] = vlm_rows
        if jev_rows is not None:
            rows["jev"] = jev_rows
        for g in GATES:
            if g in self.systems:
                rows[g] = combine(g, jev_rows, vlm_rows, th)
        by_sys = {s: {r["cand_id"]: {k: v for k, v in r.items() if k != "cand_id"} for r in rs}
                  for s, rs in rows.items()}
        run.write("judgments", ({"cand_id": c["cand_id"], "systems": {s: by_sys[s][c["cand_id"]] for s in rows}}
                                for c in cands))
        run.update_meta({"timing_s": self.timing, "usage": self.budget.summary()})

        # S7 evaluation
        metrics = self._timed("evaluate", partial(evaluate_run, run, n_flow))
        run.update_meta({"timing_s": self.timing})
        run.write_json("metrics", metrics)
        log("S7 metrics written")

        # S8 figures, samples, README
        figures = plots.plot_run(run)
        viz.render_samples(run, gt, cfg["viz"]["sample_frames"], cfg["viz"]["seed"])
        run.file("report").write_text(report.render(run, metrics, figures))
        log("S8 figures, viz and README written")
        return run

    # S5 helpers ------------------------------------------------------------------------------
    def _reuse(self, run: RunDir, what: str) -> list[dict]:
        source = self.reuse[what]
        diffs = [k for k in REUSE_DEPENDS[what] if _section(self.cfg, k) != _section(source.config(), k)]
        if diffs:
            raise ValueError(f"cannot reuse {what} from {source.run_id}: config differs in {diffs}")
        run.cache_file(f"{what}.jsonl").parent.mkdir(exist_ok=True)
        shutil.copy(source.cache_file(f"{what}.jsonl"), run.cache_file(f"{what}.jsonl"))
        old = source.meta().get("timing_s", {})
        if what in old:
            self.timing[what] = old[what]
        log(f"reused {what} from {source.run_id}")
        return read_jsonl(run.cache_file(f"{what}.jsonl"))

    def _judge_jev(self, run, frames, cands, evidence) -> list[dict]:
        if "jev" in self.reuse:
            return self._reuse(run, "jev")
        from jevgate.judges.jev import JevJudge

        checkpoint = run.cache_file("jev_responses.jsonl")
        checkpoint.parent.mkdir(exist_ok=True)
        judge = JevJudge(self.cfg, self.budget, log, checkpoint=checkpoint, resume=self.resume_jev)
        if judge.stored:
            log(f"S5 jev: {len(judge.stored)} stored responses available")
        rows = self._timed("jev", partial(judge.judge, frames, cands, evidence))
        write_jsonl(run.cache_file("jev.jsonl"), rows)
        log(f"S5 jev: {self.budget.jev_requests} new requests, {self.budget.jev_tokens} tokens")
        return rows

    def _judge_vlm(self, run, frames, cands, evidence) -> list[dict]:
        if "vlm" in self.reuse:
            return self._reuse(run, "vlm")
        from jevgate.judges.vlm import VLMJudge

        judge = self._timed("vlm_load", partial(VLMJudge, self.cfg, self.budget, log))
        run.cache_file("vlm_prompt.txt").parent.mkdir(exist_ok=True)
        run.cache_file("vlm_prompt.txt").write_text(judge.prompt)
        rows = self._timed("vlm", partial(judge.judge, frames, cands, evidence))
        write_jsonl(run.cache_file("vlm.jsonl"), rows)
        log(f"S5 vlm: {self.budget.vlm_calls} calls")
        del judge
        _free_gpu()
        return rows


def _free_gpu() -> None:
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
