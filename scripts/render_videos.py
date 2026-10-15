"""Re-draw evaluation/videos/, viz/ and confusion/ of an existing run from its files; no model is run again.

Drawing-only settings (viz.*) come from the current configs/base.yaml, so a new colour scheme applies to old runs.
"""

import argparse
from pathlib import Path

from jevgate.core.config import load_config
from jevgate.core.runio import RunDir
from jevgate.data.brackish import build_frames, load_gt
from jevgate.evaluation.evaluate import RunData
from jevgate.outputs.colors import class_colors
from jevgate.outputs.plots import confusion
from jevgate.outputs.viz import Renderer

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    args = ap.parse_args()
    run = RunDir(args.run_dir)
    cfg = run.config()
    cfg["viz"] = load_config(REPO / "configs" / "base.yaml")["viz"]
    gt = load_gt(cfg)
    videos = run.file("videos").read_text().split()
    frames = build_frames(cfg, gt, videos)
    detections = run.read("detections")
    if run.exists("gate"):
        route = {r["track_uid"]: r["routed_to"] for r in run.read("gate")}
    else:  # runs made before gate_decisions.jsonl existed: rebuild the routing from the stored judges
        data = RunData(cfg, frames, detections, run.read("tracks"), run.read("box_labels"), run.read("track_labels"),
                       run.read("jev"), run.read("vlm"), run.meta()["thresholds"], {},
                       run.read("describer") if run.exists("describer") else None)
        data.accepts()
        run.write("gate", data.gate_rows)
        route = {r["track_uid"]: r["routed_to"] for r in data.gate_rows}
    renderer = Renderer(cfg, frames, detections, run.read("judgments"), gt, route)
    panels = cfg["viz"].get("video_panels", ["gt", "gate"])
    for video in videos:
        print(renderer.video(run.dir("videos"), video, *panels, cfg["track"]["fps"]))
    renderer.samples(run.dir("viz"), cfg["viz"]["sample_frames"], cfg["viz"]["seed"])
    for system, row in run.read_json("metrics")["systems"].items():
        confusion(run.dir("confusion") / f"{system}.png", system, row["confusion"], class_colors(cfg))


if __name__ == "__main__":
    main()
