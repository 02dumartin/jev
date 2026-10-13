"""YAML config loading: `inherit` chains, `--set a.b=v` overrides, validation and hashing."""

import copy
import hashlib
import json
from pathlib import Path

import yaml

REQUIRED_KEYS = [
    "run.name", "run.out_root", "run.split", "run.systems",
    "data.root", "data.images_dir", "data.aau_csvs", "data.split_lists", "data.frame_wh",
    "clips.dev", "clips.test",
    "detector.model", "detector.prompt", "detector.box_threshold", "detector.nms_iou",
    "label.real_iou", "label.fake_iou",
    "evidence.history_frames", "evidence.match_iou",
    "prompts.context", "prompts.question",
    "jev.model", "jev.endpoint", "jev.batching",
    "vlm.model",
    "gate.max_auto_error",
    "metrics.ece_bins", "metrics.bootstrap",
    "endpoints.sequential",
    "budget.max_jev_requests", "budget.max_jev_tokens", "budget.max_vlm_calls",
]
KNOWN_SYSTEMS = {"threshold", "vlm", "jev", "gate"}


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
    unknown = set(cfg["run"]["systems"]) - KNOWN_SYSTEMS
    if unknown:
        raise ConfigError(f"unknown systems: {sorted(unknown)}")
    if "gate" in cfg["run"]["systems"] and not {"vlm", "jev"} <= set(cfg["run"]["systems"]):
        raise ConfigError("the gate system needs both 'vlm' and 'jev'")
    if not cfg["label"]["fake_iou"] <= cfg["label"]["real_iou"]:
        raise ConfigError("label.fake_iou must be <= label.real_iou")
    if cfg["jev"]["batching"] not in ("per_frame", "per_box"):
        raise ConfigError("jev.batching must be per_frame or per_box")
    overlap = set(cfg["clips"]["dev"]) & set(cfg["clips"]["test"])
    if overlap:
        raise ConfigError(f"clips appear in both dev and test: {sorted(overlap)}")
    if cfg["run"]["split"] == "test" and not cfg.get("thresholds_file"):
        raise ConfigError("test runs need thresholds_file fitted on dev")


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """Load a config file (following `inherit`), apply overrides, validate, and return it."""
    cfg = deep_merge(_load_with_inherit(Path(path)), parse_overrides(overrides or []))
    validate(cfg)
    return cfg


def config_hash(cfg: dict) -> str:
    """Short stable hash of the resolved config, for metrics_history rows."""
    blob = json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(blob).hexdigest()[:10]
