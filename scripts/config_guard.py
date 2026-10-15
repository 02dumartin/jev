"""⓪ Intent check: list watched config keys that differ from a base git ref (PLAN.md §7).

Exit code 2 when a question must be asked before running; 0 otherwise. Compares each configs/*.yaml
file as written (without following `inherit`) against the same file at --base.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
WATCH = [
    (("thresholds_file", "gate.max_auto_error", "gate.tau_grid_step"),
     "Was this threshold change made after looking at test videos? If so, revert and refit on dev."),
    (("video_split", "run.split"), "Was the video split fixed before seeing any result? If not, revert it."),
    (("vlm",), "The VLM input changed: ③ must be re-run as the new baseline. Proceed?"),
    (("jev.model", "jev.include_detector_reliability", "jev.max_tracks_per_request"),
     "Jev input changed: re-run ④ and ⑤?"),
    (("prompts", "classes", "not_an_animal"), "The prompt changed: do VLM and Jev still share the same text?"),
    (("track",), "The detector/tracker changed: tracks differ, so every system will be re-run."),
    (("label",), "The labelling rule changed: every metric moves; explain why in the PR."),
]


def flatten(d, prefix=""):
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out |= flatten(v, key + ".")
        else:
            out[key] = v
    return out


def at_ref(ref: str, rel: str):
    try:
        text = subprocess.check_output(["git", "show", f"{ref}:{rel}"], cwd=REPO, text=True, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return None
    return yaml.safe_load(text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="main")
    args = ap.parse_args()
    questions = {}
    for path in sorted((REPO / "configs").rglob("*.yaml")):
        rel = path.relative_to(REPO).as_posix()
        old, new = flatten(at_ref(args.base, rel)), flatten(yaml.safe_load(path.read_text()))
        if at_ref(args.base, rel) is None:
            print(f"{rel}: new file (not in {args.base})")
        changed = sorted(k for k in old.keys() | new.keys() if old.get(k) != new.get(k))
        for key in changed:
            print(f"{rel}: {key}: {old.get(key)!r} -> {new.get(key)!r}")
            for prefixes, question in WATCH:
                if any(key == p or key.startswith(p + ".") for p in prefixes):
                    questions.setdefault(question, []).append(f"{rel}:{key}")
    if not questions:
        print("no watched keys changed")
        sys.exit(0)
    print("\nQUESTIONS FOR THE USER:")
    for q, keys in questions.items():
        print(f"- {q}  [{', '.join(keys)}]")
    sys.exit(2)


if __name__ == "__main__":
    main()
