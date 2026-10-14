"""YAML config loading: `inherit` chains, `--set a.b=v` overrides, validation and hashing."""

import copy
import hashlib
import json
from pathlib import Path

import yaml

REQUIRED_KEYS = [
    "run.name", "run.out_root", "run.split", "run.systems", "run.cache_root",
    "data.root", "data.images_dir", "data.aau_csvs", "data.frame_wh",
    "splits.dev", "splits.test", "classes", "candidates.source",
    "detector.model", "detector.prompt", "detector.phrase_to_class", "detector.box_threshold", "detector.nms_iou",
    "label.match_iou", "label.background_iou",
    "evidence.history_frames", "evidence.match_iou",
    "prompts.context", "prompts.question", "prompts.need_vlm_question", "prompts.need_vlm_levels",
    "jev.model", "jev.endpoint", "jev.batching",
    "vlm.model",
    "gate.primary", "gate.max_auto_error", "gate.need_vlm_grid_step", "gate.conf_grid_step",
    "metrics.ece_bins", "metrics.bootstrap", "metrics.min_class_n",
    "endpoints.sequential",
    "budget.max_jev_requests", "budget.max_jev_tokens", "budget.max_vlm_calls",
]
KNOWN_SYSTEMS = {"threshold", "vlm", "jev", "gate", "gate_conf", "gate_either"}
GATE_SYSTEMS = {"gate", "gate_conf", "gate_either"}
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
    if systems - KNOWN_SYSTEMS:
        raise ConfigError(f"unknown systems: {sorted(systems - KNOWN_SYSTEMS)}")
    if systems & GATE_SYSTEMS and not {"vlm", "jev"} <= systems:
        raise ConfigError("gate systems need both 'vlm' and 'jev'")
    if cfg["gate"]["primary"] not in GATE_SYSTEMS:
        raise ConfigError(f"gate.primary must be one of {sorted(GATE_SYSTEMS)}")
    if BACKGROUND not in cfg["classes"]:
        raise ConfigError("classes must include 'background'")
    if cfg["candidates"]["source"] not in ("dino", "gt"):
        raise ConfigError("candidates.source must be 'dino' or 'gt'")
    if cfg["candidates"]["source"] == "gt" and "threshold" in systems:
        raise ConfigError("the detector baseline needs DINO scores; remove 'threshold' when candidates.source=gt")
    unmapped = set(cfg["detector"]["phrase_to_class"].values()) - set(cfg["classes"])
    if unmapped:
        raise ConfigError(f"detector.phrase_to_class maps to unknown classes: {sorted(unmapped)}")
    if not cfg["label"]["background_iou"] <= cfg["label"]["match_iou"]:
        raise ConfigError("label.background_iou must be <= label.match_iou")
    if cfg["jev"]["batching"] not in ("per_frame", "per_box"):
        raise ConfigError("jev.batching must be per_frame or per_box")
    overlap = set(cfg["splits"]["dev"]) & set(cfg["splits"]["test"])
    if overlap:
        raise ConfigError(f"split lists appear in both dev and test: {sorted(overlap)}")
    if cfg["run"]["split"] == "test" and not cfg.get("thresholds_file"):
        raise ConfigError("test runs need thresholds_file fitted on dev")


def class_names(cfg: dict) -> list[str]:
    """Ordered judgment classes: six animals, then background (dropped for GT boxes, which are all animals)."""
    return [c for c in cfg["classes"] if not (c == BACKGROUND and cfg["candidates"]["source"] == "gt")]


def class_defs(cfg: dict) -> dict[str, str]:
    """Descriptions of the judgment classes, shared by the VLM options and Jev's class definitions."""
    return {c: cfg["classes"][c] for c in class_names(cfg)}


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """Load a config file (following `inherit`), apply overrides, validate, and return it."""
    cfg = deep_merge(_load_with_inherit(Path(path)), parse_overrides(overrides or []))
    validate(cfg)
    return cfg


def config_hash(cfg: dict) -> str:
    """Short stable hash of the resolved config, for metrics_history rows."""
    blob = json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(blob).hexdigest()[:10]
