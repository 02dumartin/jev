"""Stage order of one run (PLAN.md v6). Each stage writes into the run folder; later stages read those files.

S1 yolo     YOLO + ByteTrack on every frame of the split's videos -> detections, track summaries
S2 labels   y per box (AAU IoU) and per track; GT objects the detector missed
S3 describer  small VLM: description + p_yes per track (representative frame)
S4 jev      Jev track_ok per track (track summary + description)
S4 vlm      large VLM p_yes per track (same images)
S5 eval     thresholds (dev: fit, test: frozen file), metrics, tables, figures, README
S6 viz      sample frames and one video per split video
"""

import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv

from jevgate.core.runio import RunDir, link, read_jsonl, write_jsonl
from jevgate.data.brackish import build_frames, load_gt
from jevgate.data.labels import label_boxes, label_tracks, missed_objects
from jevgate.data.splits import run_videos
from jevgate.detectors.track import summarize_tracks
from jevgate.evaluation.evaluate import RunData, evaluate
from jevgate.judges.base import Budget
from jevgate.judges.gate import fit_accept_threshold, fit_band, fit_conf_threshold, fit_tau
from jevgate.outputs import plots, report
from jevgate.outputs.viz import Renderer


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def yolo_cache_key(cfg: dict, videos: list[str]) -> str:
    """Detector outputs depend only on these settings; identical keys reuse runs/_cache/yolo/<key>/."""
    blob = json.dumps({"track": cfg["track"], "videos": videos,
                       "max_frames": cfg["run"].get("max_frames_per_video")}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def free_gpu() -> None:
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class Pipeline:
    def __init__(self, cfg: dict, repo: Path):
        self.cfg, self.repo = cfg, repo
        self.systems = set(cfg["run"]["systems"])
        self.budget = Budget(cfg)
        self.timing: dict[str, float] = {}
        load_dotenv(repo / ".env")

    def _time(self, key: str, started: float) -> None:
        self.timing[key] = round(time.perf_counter() - started, 3)

    # S1 -----------------------------------------------------------------------------------
    def yolo(self, run: RunDir, frames: list[dict], videos: list[str]) -> tuple[list[dict], list[dict]]:
        import ultralytics

        cfg = self.cfg
        run.file("videos").write_text("\n".join(videos) + "\n")
        link(Path(ultralytics.__file__).parent / "cfg" / "trackers" / cfg["track"]["tracker"],
             run.path / "yolo" / "input" / "tracker.yaml")
        cache = self.repo / cfg["run"]["out_root"] / "_cache" / "yolo" / yolo_cache_key(cfg, videos)
        if (cache / "detections.jsonl").exists():
            detections = read_jsonl(cache / "detections.jsonl")
            meta = json.loads((cache / "timing.json").read_text())
            log(f"S1 yolo: reused {cache.relative_to(self.repo)}")
        else:
            from jevgate.detectors.track import VideoTracker

            tracker, detections, frame_ms = VideoTracker(cfg, self.repo), [], []
            for k, video in enumerate(videos, 1):
                files = [f["file"] for f in frames if f["video"] == video]
                rows, ms = tracker.run(video, files)
                detections += rows
                frame_ms += ms
                log(f"  yolo+track {k}/{len(videos)} {video}: {len(files)} frames, {len(rows)} boxes")
            meta = {"yolo_track_s": round(sum(frame_ms) / 1000, 3),
                    "yolo_track_ms_per_frame_median": round(float(np.median(frame_ms)), 2)}
            cache.mkdir(parents=True, exist_ok=True)
            write_jsonl(cache / "detections.jsonl", detections)
            (cache / "timing.json").write_text(json.dumps(meta))
            del tracker
            free_gpu()
        self.timing |= meta
        run.write("detections", detections)
        tracks = summarize_tracks(detections, frames, cfg["data"]["frame_wh"])
        run.write("tracks", tracks)
        log(f"S1 yolo: {len(detections)} boxes, {len(tracks)} tracks, "
            f"{sum(d['track_uid'] is None for d in detections)} untracked")
        return detections, tracks

    # S3, S4 -------------------------------------------------------------------------------
    def describer(self, run: RunDir, tracks: list[dict], detections: list[dict]) -> list[dict]:
        rows, prompts = self._vlm_stage(run, tracks, detections, "describer", describe=True)
        run.write("describer", rows)
        run.write("describer_prompts", prompts)
        log(f"S3 describer: {len(rows)} tracks described")
        return rows

    def _vlm_stage(self, run, tracks, detections, section: str, describe: bool):
        from jevgate.judges.vlm import TrackVLM

        started = time.perf_counter()
        judge = TrackVLM(self.cfg, section, self.budget if section == "vlm" else None, run.dir("track_images"),
                         describe, log)
        self._time(f"{section}_load_s", started)
        started = time.perf_counter()
        out = judge.judge(tracks, {d["box_id"]: d for d in detections})
        self._time(f"{section}_s", started)
        del judge
        free_gpu()
        return out

    def jev(self, run: RunDir, tracks: list[dict], small_rows: list[dict] | None) -> list[dict]:
        from jevgate.judges.jev import JevJudge

        own = run.path / "jev" / "output" / "responses.jsonl"
        resume = [p for p in (self.repo / self.cfg["run"]["out_root"]).glob("*/jev/output/responses.jsonl")
                  if p.resolve() != own.resolve()]
        descriptions = {r["track_uid"]: r["description"] for r in small_rows} if small_rows else None
        judge = JevJudge(self.cfg, self.repo, self.budget, run.file("jev_requests"), run.file("jev_responses"),
                         resume, descriptions, log)
        started = time.perf_counter()
        rows = judge.judge(tracks)
        self._time("jev_s", started)
        run.write("jev", rows)
        log(f"S3 jev: {len(tracks)} tracks, {self.budget.jev_requests} new requests, "
            f"{sum(r['reused'] for r in rows)} tracks from stored responses, {self.budget.jev_tokens} new tokens")
        return rows

    def vlm(self, run: RunDir, tracks: list[dict], detections: list[dict]) -> list[dict]:
        rows, prompts = self._vlm_stage(run, tracks, detections, "vlm", describe=False)
        run.write("vlm", rows)
        run.write("vlm_prompts", prompts)
        log(f"S4 vlm: {len(rows)} tracks")
        return rows

    # S5 -----------------------------------------------------------------------------------
    def thresholds(self, run, videos, detections, box_labels, track_labels, jev_rows, vlm_rows, small_rows) -> dict:
        """Dev: fit t, v, v_small, tau, small_lo/hi. Test: read the frozen dev file and refuse a mismatch."""
        cfg = self.cfg
        if cfg["run"]["split"] == "test":
            th = yaml.safe_load((self.repo / cfg["thresholds_file"]).read_text())
            if th.get("fit_split") != "dev" or th.get("max_frames_per_video") or th.get("max_videos"):
                raise ValueError("thresholds_file is not a full dev fit; refusing the test run")
            dev_videos = yaml.safe_load((self.repo / cfg["video_split"]["file"]).read_text())["videos"][
                cfg["video_split"]["run_split_to_video_split"]["dev"]]
            if th.get("fit_videos") != dev_videos:
                raise ValueError("thresholds_file was fitted on other videos; refusing the test run")
            return th
        scored = [d for d in detections if box_labels[d["box_id"]]["scored"]]
        tracks_scored = [u for u, r in track_labels.items() if r["y_track"] is not None]
        th = {"fit_split": "dev", "fit_videos": videos, "fit_run": run.run_id, "fit_commit": run.git().get("commit"),
              "max_videos": cfg["run"].get("max_videos"),
              "max_frames_per_video": cfg["run"].get("max_frames_per_video"),
              "t": fit_conf_threshold([box_labels[d["box_id"]]["y"] for d in scored], [d["cls"] for d in scored],
                                      [d["conf"] for d in scored])}
        y_track = [track_labels[u]["y_track"] for u in tracks_scored]
        if vlm_rows:
            p_yes = {r["track_uid"]: r["p_yes"] for r in vlm_rows}
            th["v"] = fit_accept_threshold(y_track, [p_yes[u] for u in tracks_scored])
        if small_rows:
            p_small = [{r["track_uid"]: r["p_yes"] for r in small_rows}[u] for u in tracks_scored]
            th["v_small"] = fit_accept_threshold(y_track, p_small)
            th["small_lo"], th["small_hi"] = fit_band(y_track, p_small, cfg["gate"]["max_auto_error"])
        if jev_rows:
            p_ok = {r["track_uid"]: r["p_ok"] for r in jev_rows}
            th["tau"] = fit_tau(y_track, [p_ok[u] for u in tracks_scored], cfg["gate"]["max_auto_error"],
                                cfg["gate"]["tau_grid_step"])
            th["max_auto_error"] = cfg["gate"]["max_auto_error"]
        return th

    def run(self) -> RunDir:
        cfg = self.cfg
        run = RunDir.create(cfg, self.repo)
        log(f"run folder: {run.path}")
        gt = load_gt(cfg)
        videos = run_videos(cfg, self.repo)
        frames = build_frames(cfg, gt, videos)
        log(f"split {cfg['run']['split']}: {len(videos)} videos, {len(frames)} frames")

        detections, tracks = self.yolo(run, frames, videos)

        box_labels = label_boxes(detections, gt, cfg)
        track_labels = label_tracks(tracks, {r["box_id"]: r for r in box_labels}, cfg["label"]["track_ok_share"])
        run.write("box_labels", box_labels)
        run.write("track_labels", track_labels)
        missed = missed_objects(detections, frames, gt, cfg["label"]["match_iou"])
        kinds = ("correct", "misclassified", "false", "unscored")
        types = {k: sum(r["type"] == k for r in track_labels) for k in kinds}
        log(f"S2 labels: {sum(r['scored'] for r in box_labels)} scored boxes, track types {types}, "
            f"detector recall upper bound {missed['recall_upper_bound']:.3f}")

        small_rows = self.describer(run, tracks, detections) if {"small_vlm", "jev", "gate", "gate_small"} & \
            self.systems else None
        jev_rows = self.jev(run, tracks, small_rows) if {"jev", "gate"} & self.systems else None
        vlm_rows = self.vlm(run, tracks, detections) if {"vlm_all", "gate", "gate_small"} & self.systems else None

        bl = {r["box_id"]: r for r in box_labels}
        tl = {r["track_uid"]: r for r in track_labels}
        th = self.thresholds(run, videos, detections, bl, tl, jev_rows, vlm_rows, small_rows)
        if cfg["run"]["split"] == "dev":
            run.file("thresholds").write_text(yaml.safe_dump(th, sort_keys=False))
        log("S5 thresholds: " + ", ".join(f"{k}={th[k]}" for k in ("t", "v", "v_small", "tau", "small_lo", "small_hi")
                                          if k in th))

        started = time.perf_counter()
        data = RunData(cfg, frames, detections, tracks, box_labels, track_labels, jev_rows, vlm_rows, th, self.timing,
                       small_rows)
        metrics, judgments, curves = evaluate(data, missed, self.budget.summary(), cfg)
        run.write("judgments", judgments)
        run.write_json("metrics", metrics)
        if cfg["run"]["split"] == "test":
            run.write_json("endpoints", {"primary": metrics["primary"], "endpoints": metrics["endpoints"]})
        if data.gate_rows:
            run.write("gate", data.gate_rows)
            report.write_errors(run.file("errors"), tracks, tl, data)
        plots.all_figures(run, metrics, curves, data)
        self._time("evaluate_s", started)
        metrics["timing"] = self.timing
        run.write_json("metrics", metrics)
        run.file("metrics_md").write_text(report.metrics_md(metrics))
        run.file("readme").write_text(report.readme(run.run_id, metrics))
        run.update_meta({"thresholds": th, "timing": self.timing, "budget_used": self.budget.summary()})
        log("S5 evaluation written")

        if cfg["viz"]["enabled"]:
            route = {r["track_uid"]: r["routed_to"] for r in data.gate_rows} if jev_rows and vlm_rows else None
            renderer = Renderer(cfg, frames, detections, judgments, gt, route)
            renderer.samples(run.dir("viz"), cfg["viz"]["sample_frames"], cfg["viz"]["seed"])
            if cfg["viz"]["videos"] and "gate" in self.systems:
                for video in videos:
                    renderer.video(run.dir("videos"), video, *cfg["viz"]["video_panels"], cfg["track"]["fps"])
            log("S6 viz written")
        return run


def copy_thresholds(run: RunDir, dest: Path) -> None:
    """Freeze a dev run's thresholds for the test run."""
    shutil.copy(run.file("thresholds"), dest)
