"""Run folders: naming (runs/<YYMMDD>_<name>), JSONL I/O, resolved config and git provenance."""

import datetime as dt
import json
import subprocess
from pathlib import Path

import yaml

STAGE_FILES = {
    "frames": "frames.jsonl",
    "candidates": "candidates.jsonl",
    "labels": "labels.jsonl",
    "evidence": "evidence.jsonl",
    "metrics": "metrics.json",
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
        """Make a fresh run folder and record the resolved config and git provenance."""
        path = next_run_dir(repo / cfg["run"]["out_root"], cfg["run"]["name"])
        path.mkdir(parents=True)
        run = cls(path)
        (path / "config.resolved.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
        info = git_info(repo) | {"time": dt.datetime.now().isoformat(timespec="seconds"), "run_id": path.name}
        (path / "git.json").write_text(json.dumps(info, indent=2))
        return run

    @property
    def run_id(self) -> str:
        return self.path.name

    def config(self) -> dict:
        return yaml.safe_load((self.path / "config.resolved.yaml").read_text())

    def git(self) -> dict:
        return json.loads((self.path / "git.json").read_text())

    def file(self, stage: str) -> Path:
        return self.path / STAGE_FILES[stage]

    def decisions_file(self, system: str) -> Path:
        return self.path / "decisions" / f"{system}.jsonl"

    def write(self, stage: str, rows) -> int:
        return write_jsonl(self.file(stage), rows)

    def read(self, stage: str) -> list[dict]:
        return read_jsonl(self.file(stage))

    def write_decisions(self, system: str, rows) -> int:
        return write_jsonl(self.decisions_file(system), rows)

    def read_decisions(self, system: str) -> list[dict]:
        return read_jsonl(self.decisions_file(system))

    def systems_done(self) -> list[str]:
        folder = self.path / "decisions"
        return sorted(p.stem for p in folder.glob("*.jsonl")) if folder.exists() else []

    def write_json(self, name: str, obj) -> None:
        (self.path / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False))

    def read_json(self, name: str):
        return json.loads((self.path / name).read_text())
