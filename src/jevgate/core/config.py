"""YAML config loading: `inherit` chains, `--set a.b=v` overrides, validation and hashing."""

import copy
import hashlib
import json
from pathlib import Path

import yaml

REQUIRED_KEYS = [
    "run.name", "run.out_root", "run.split", "run.systems",
    "data.root", "data.images_dir", "data.annotations_dir", "data.aau_csvs", "data.split_lists", "data.frame_wh",
    "video_split.file", "video_split.run_split_to_video_split", "classes", "not_an_animal",
    "track.weights", "track.tracker", "track.conf", "track.iou", "track.imgsz", "track.device",
    "label.match_iou", "label.background_iou", "label.track_ok_share",
    "prompts.context", "prompts.question", "prompts.true_means", "prompts.false_means", "prompts.describe",
    "jev.model", "jev.endpoint", "jev.max_tracks_per_request",
    "describer.model", "describer.max_new_tokens", "vlm.model", "gate.max_auto_error", "gate.tau_grid_step",
    "metrics.ece_bins", "metrics.bootstrap", "endpoints.primary_pair", "endpoints.sequential",
    "budget.max_jev_requests", "budget.max_jev_tokens", "budget.max_vlm_calls",
]
KNOWN_SYSTEMS = ("yolo", "track_vote", "vlm_all", "small_vlm", "jev", "gate", "gate_small")
BACKGROUND = "background"


class ConfigError(ValueError):
    """Raised when a config is missing keys or holds inconsistent values."""


def deep_merge(base: dict, override: dict) -> dict:
    """Return a copy of `base` with `override` merged in; nested dicts merge, everything else replaces."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def get(cfg: dict, dotted: str, default=KeyError):
    """Read `a.b.c` from a nested dict. Raise KeyError unless a default is given."""
    node = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            if default is KeyError:
                raise KeyError(dotted)
            return default
        node = node[part]
    return node


def set_dotted(cfg: dict, dotted: str, value) -> None:
    """Write `value` at `a.b.c`, creating intermediate dicts."""
    parts = dotted.split(".")
    node = cfg
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def parse_overrides(items: list[str]) -> dict:
    """Turn ['a.b=1', 'c=[x, y]'] into a nested dict; values are parsed as YAML scalars/lists."""
    out: dict = {}
    for item in items or []:
        if "=" not in item:
            raise ConfigError(f"override must look like key=value: {item!r}")
        key, raw = item.split("=", 1)
        set_dotted(out, key.strip(), yaml.safe_load(raw))
    return out


def _load_with_inherit(path: Path, seen: tuple = ()) -> dict:
    path = path.resolve()
    if path in seen:
        raise ConfigError(f"inherit cycle: {' -> '.join(map(str, seen + (path,)))}")
    raw = yaml.safe_load(path.read_text()) or {}
    parent = raw.pop("inherit", None)
    if parent is None:
        return raw
    return deep_merge(_load_with_inherit(path.parent / parent, seen + (path,)), raw)


def validate(cfg: dict) -> None:
    """Fail fast on missing keys and on values that would silently corrupt an experiment."""
    missing = [k for k in REQUIRED_KEYS if get(cfg, k, None) is None]
    if missing:
        raise ConfigError(f"missing config keys: {missing}")
    if cfg["run"]["split"] not in ("dev", "test"):
        raise ConfigError("run.split must be 'dev' or 'test'")
    systems = set(cfg["run"]["systems"])
    if systems - set(KNOWN_SYSTEMS):
        raise ConfigError(f"unknown systems: {sorted(systems - set(KNOWN_SYSTEMS))}")
    if "gate" in systems and not {"vlm_all", "jev"} <= systems:
        raise ConfigError("the gate system needs both 'vlm_all' and 'jev'")
    if "gate_small" in systems and not {"vlm_all", "small_vlm"} <= systems:
        raise ConfigError("the gate_small system needs both 'vlm_all' and 'small_vlm'")
    if not set(cfg["endpoints"]["primary_pair"]) <= systems:
        raise ConfigError("endpoints.primary_pair must be run systems")
    if BACKGROUND in cfg["classes"]:
        raise ConfigError("classes are the six animals; background is the result of rejecting a track")
    if not cfg["label"]["background_iou"] <= cfg["label"]["match_iou"]:
        raise ConfigError("label.background_iou must be <= label.match_iou")
    if "{cls}" not in cfg["prompts"]["question"]:
        raise ConfigError("prompts.question must contain {cls}")
    if cfg["run"]["split"] == "test" and not cfg.get("thresholds_file"):
        raise ConfigError("test runs need thresholds_file fitted on dev")


def class_names(cfg: dict) -> list[str]:
    """The six animal classes in config order (YOLO class ids 0-5)."""
    return list(cfg["classes"])


def output_classes(cfg: dict) -> list[str]:
    """Box-level outcome classes: six animals, then background (a rejected or untracked box)."""
    return class_names(cfg) + [BACKGROUND]


def load_plain_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """Load a config (following `inherit`) and apply overrides, without the Brackish experiment validation."""
    return deep_merge(_load_with_inherit(Path(path)), parse_overrides(overrides or []))


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """Load a config file (following `inherit`), apply overrides, validate, and return it."""
    cfg = deep_merge(_load_with_inherit(Path(path)), parse_overrides(overrides or []))
    validate(cfg)
    return cfg


def config_hash(cfg: dict) -> str:
    """Short stable hash of the resolved config, for metrics_history rows."""
    blob = json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(blob).hexdigest()[:10]
