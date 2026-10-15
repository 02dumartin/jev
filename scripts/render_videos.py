"""Re-draw evaluation/videos/ (and viz/) of an existing run from its files; no model is run again."""

import argparse
from pathlib import Path

from jevgate.core.runio import RunDir
from jevgate.data.brackish import build_frames, load_gt
from jevgate.evaluation.evaluate import RunData
from jevgate.outputs.viz import Renderer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--samples", action="store_true", help="also redraw evaluation/viz/ sample frames")
    args = ap.parse_args()
    run = RunDir(args.run_dir)
    cfg = run.config()
    gt = load_gt(cfg)
    videos = run.file("videos").read_text().split()
    frames = build_frames(cfg, gt, videos)
    detections = run.read("detections")
    if run.exists("gate"):
        route = {r["track_uid"]: r["routed_to"] for r in run.read("gate")}
    else:  # runs made before gate_decisions.jsonl existed: rebuild the routing from the stored judges
        data = RunData(cfg, frames, detections, run.read("tracks"), run.read("box_labels"), run.read("track_labels"),
                       run.read("jev"), run.read("vlm"), run.meta()["thresholds"], {})
        data.accepts()
        run.write("gate", data.gate_rows)
        route = {r["track_uid"]: r["routed_to"] for r in data.gate_rows}
    renderer = Renderer(cfg, frames, detections, run.read("judgments"), gt, route)
    panels = cfg["viz"].get("video_panels", ["gt", "gate"])
    for video in videos:
        print(renderer.video(run.dir("videos"), video, *panels, cfg["track"]["fps"]))
    if args.samples:
        renderer.samples(run.dir("viz"), cfg["viz"]["sample_frames"], cfg["viz"]["seed"])


if __name__ == "__main__":
    main()
