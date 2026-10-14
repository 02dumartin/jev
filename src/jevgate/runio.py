"""Run folders: naming (runs/<YYMMDD>_<name>) and the fixed file layout of one run.

runs/<run_id>/
├── README.md         report (results, n flow, time and cost, what each figure shows)
├── run.yaml          config + git + env + thresholds + stage times + usage
├── frames.jsonl      one row per frame
├── boxes.jsonl       one row per candidate box: position, (detector), truth y, Jev evidence
├── judgments.jsonl   one row per box: every system's call
├── metrics.json      every number, plus curve data for the figures
├── errors.csv        boxes a main system got wrong
├── cache/            expensive judge outputs (vlm.jsonl, jev.jsonl, jev_responses.jsonl, vlm_prompt.txt)
├── figures/          fig1..fig4 (+ appendix/ when enabled)
└── samples/          one comparison image per sampled frame
"""

import datetime as dt
import json
import subprocess
from pathlib import Path

import yaml

FILES = {
    "frames": "frames.jsonl",
    "boxes": "boxes.jsonl",
    "judgments": "judgments.jsonl",
    "metrics": "metrics.json",
    "errors": "errors.csv",
    "report": "README.md",
    "meta": "run.yaml",
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
    return {
        "commit": run("rev-parse", "--short", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def next_run_dir(out_root: Path, name: str, today: dt.date | None = None) -> Path:
    """runs/<YYMMDD>_<name>, or <...>_2, _3 if that folder already exists."""
    stem = f"{(today or dt.date.today()).strftime('%y%m%d')}_{name}"
    candidate, k = out_root / stem, 2
    while candidate.exists():
        candidate, k = out_root / f"{stem}_{k}", k + 1
    return candidate


class RunDir:
    """One experiment run on disk. Stages read and write only through this object."""

    def __init__(self, path: Path):
        self.path = Path(path)

    @classmethod
    def create(cls, cfg: dict, repo: Path) -> "RunDir":
        """Make a fresh run folder and start run.yaml with the resolved config and git provenance."""
        path = next_run_dir(repo / cfg["run"]["out_root"], cfg["run"]["name"])
        path.mkdir(parents=True)
        run = cls(path)
        git = git_info(repo) | {"time": dt.datetime.now().isoformat(timespec="seconds")}
        run.update_meta({"run_id": path.name, "git": git, "config": cfg})
        return run

    @property
    def run_id(self) -> str:
        return self.path.name

    # run.yaml -----------------------------------------------------------------------------
    def meta(self) -> dict:
        p = self.file("meta")
        return yaml.safe_load(p.read_text()) if p.exists() else {}

    def update_meta(self, updates: dict) -> None:
        """Merge top-level sections into run.yaml (dict sections merge one level deep)."""
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

    # files --------------------------------------------------------------------------------
    def file(self, name: str) -> Path:
        return self.path / FILES[name]

    def cache_file(self, name: str) -> Path:
        return self.path / "cache" / name

    def write(self, name: str, rows) -> int:
        return write_jsonl(self.file(name), rows)

    def read(self, name: str) -> list[dict]:
        return read_jsonl(self.file(name))

    def write_json(self, name: str, obj) -> None:
        self.file(name).write_text(json.dumps(obj, indent=2, ensure_ascii=False))

    def read_json(self, name: str):
        return json.loads(self.file(name).read_text())

    def system_rows(self) -> dict[str, list[dict]]:
        """judgments.jsonl as {system: [row per box in box order]} (cand_id included in each row)."""
        out: dict[str, list[dict]] = {}
        for j in self.read("judgments"):
            for system, row in j["systems"].items():
                out.setdefault(system, []).append({"cand_id": j["cand_id"], **row})
        return out
