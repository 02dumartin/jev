"""Run folders: naming (runs/<YYMMDD>_<name>) and the fixed layout of one run (PLAN.md §7).

runs/<run_id>/
├── README.md, run.yaml
├── data/                 symlinks: images, annotations, video_split.yaml
├── yolo/{input,output}/  weights.pt + tracker.yaml (links), videos.txt | detections.jsonl, tracks.jsonl
├── jev/{input,output}/   requests.jsonl | responses.jsonl, decisions.jsonl
├── vlm/{input,output}/   prompts.jsonl, images/ | decisions.jsonl
└── evaluation/           labels/, judgments.jsonl, tables/, confusion/, plots/, viz/, videos/
"""

import datetime as dt
import json
import os
import subprocess
from pathlib import Path

import yaml

FILES = {
    "readme": "README.md",
    "meta": "run.yaml",
    "videos": "yolo/input/videos.txt",
    "detections": "yolo/output/detections.jsonl",
    "tracks": "yolo/output/tracks.jsonl",
    "jev_requests": "jev/input/requests.jsonl",
    "jev_responses": "jev/output/responses.jsonl",
    "jev": "jev/output/decisions.jsonl",
    "vlm_prompts": "vlm/input/prompts.jsonl",
    "vlm": "vlm/output/decisions.jsonl",
    "box_labels": "evaluation/labels/boxes.jsonl",
    "track_labels": "evaluation/labels/tracks.jsonl",
    "judgments": "evaluation/judgments.jsonl",
    "metrics": "evaluation/tables/metrics.json",
    "metrics_md": "evaluation/tables/metrics.md",
    "errors": "evaluation/tables/errors.csv",
    "thresholds": "evaluation/tables/thresholds.yaml",
    "endpoints": "evaluation/tables/endpoints.json",
}
DIRS = {
    "vlm_images": "vlm/input/images",
    "confusion": "evaluation/confusion",
    "plots": "evaluation/plots",
    "viz": "evaluation/viz",
    "videos": "evaluation/videos",
}


def write_jsonl(path: Path, rows) -> int:
    """Write rows as one JSON object per line; return the row count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file into a list of dicts."""
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def git_info(repo: Path) -> dict:
    """Commit, branch and dirty flag of the working tree (None values outside a git repo)."""

    def run(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=repo, text=True, stderr=subprocess.DEVNULL).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    status = run("status", "--porcelain", "--untracked-files=no")
    return {"commit": run("rev-parse", "--short", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(status) if status is not None else None}


def next_run_dir(out_root: Path, name: str, today: dt.date | None = None) -> Path:
    """runs/<YYMMDD>_<name>, or <...>_2, _3 if that folder already exists."""
    stem = f"{(today or dt.date.today()).strftime('%y%m%d')}_{name}"
    candidate, k = out_root / stem, 2
    while candidate.exists():
        candidate, k = out_root / f"{stem}_{k}", k + 1
    return candidate


def link(target: Path, at: Path) -> None:
    """Symlink `at` -> `target` (absolute), replacing an existing link."""
    at.parent.mkdir(parents=True, exist_ok=True)
    if at.is_symlink() or at.exists():
        at.unlink()
    os.symlink(Path(target).resolve(), at)


class RunDir:
    """One experiment run on disk. Stages read and write only through this object."""

    def __init__(self, path: Path):
        self.path = Path(path)

    @classmethod
    def create(cls, cfg: dict, repo: Path) -> "RunDir":
        """Fresh run folder: run.yaml (config + git), data/ links and the yolo/input links."""
        path = next_run_dir(repo / cfg["run"]["out_root"], cfg["run"]["name"])
        path.mkdir(parents=True)
        run = cls(path)
        git = git_info(repo) | {"time": dt.datetime.now().isoformat(timespec="seconds")}
        run.update_meta({"run_id": path.name, "git": git, "config": cfg})
        data_root = repo / cfg["data"]["root"]
        link(data_root / cfg["data"]["images_dir"], path / "data" / "images")
        link(data_root / cfg["data"]["annotations_dir"], path / "data" / "annotations")
        link(repo / cfg["video_split"]["file"], path / "data" / "video_split.yaml")
        link(repo / cfg["track"]["weights"], path / "yolo" / "input" / "weights.pt")
        return run

    @property
    def run_id(self) -> str:
        return self.path.name

    def meta(self) -> dict:
        p = self.file("meta")
        return yaml.safe_load(p.read_text()) if p.exists() else {}

    def update_meta(self, updates: dict) -> None:
        """Merge top-level sections into run.yaml (dict sections merge one level deep, config replaces)."""
        meta = self.meta()
        for key, value in updates.items():
            if isinstance(value, dict) and isinstance(meta.get(key), dict) and key != "config":
                meta[key] = meta[key] | value
            else:
                meta[key] = value
        self.file("meta").write_text(yaml.safe_dump(meta, allow_unicode=True, sort_keys=False))

    def config(self) -> dict:
        return self.meta()["config"]

    def git(self) -> dict:
        return self.meta().get("git", {})

    def file(self, name: str) -> Path:
        p = self.path / FILES[name]
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def dir(self, name: str) -> Path:
        p = self.path / DIRS[name]
        p.mkdir(parents=True, exist_ok=True)
        return p

    def exists(self, name: str) -> bool:
        return (self.path / FILES[name]).exists()

    def write(self, name: str, rows) -> int:
        return write_jsonl(self.file(name), rows)

    def read(self, name: str) -> list[dict]:
        return read_jsonl(self.file(name))

    def write_json(self, name: str, obj) -> None:
        self.file(name).write_text(json.dumps(obj, indent=2, ensure_ascii=False))

    def read_json(self, name: str):
        return json.loads(self.file(name).read_text())
