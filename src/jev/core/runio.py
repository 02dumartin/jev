"""Run naming, provenance, JSONL storage, and data links."""

import datetime as dt
import json
import os
import subprocess
from pathlib import Path


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


def link(target: Path, at: Path) -> None:
    """Symlink `at` -> `target` (absolute), replacing an existing link."""
    at.parent.mkdir(parents=True, exist_ok=True)
    if at.is_symlink() or at.exists():
        at.unlink()
    os.symlink(Path(target).resolve(), at)
