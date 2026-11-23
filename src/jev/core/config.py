"""YAML config loading: `inherit` chains, `--set a.b=v` overrides, validation."""

import copy
from pathlib import Path

import yaml


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


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """Load a config (following `inherit`) and apply overrides, validate Retail fields."""
    cfg = deep_merge(_load_with_inherit(Path(path)), parse_overrides(overrides or []))
    validate(cfg)
    return cfg


def validate(cfg: dict) -> None:
    """Validate Retail stage configuration before allocating models or creating a run."""
    required = [
        "run.name",
        "run.out_root",
        "data.root",
        "data.clips_dir",
        "data.meta",
        "data.license_files",
        "rotation",
        "events.enabled",
        "vlm.model",
        "jev.model",
        "jev.confident",
        "questions",
        "actions",
        "render",
        "vlm_only.enabled",
        "budget",
    ]
    missing = [key for key in required if get(cfg, key, None) is None]
    if missing:
        raise ConfigError(f"missing config keys: {missing}")
    if not 0.5 <= cfg["jev"]["confident"] <= 1:
        raise ConfigError("jev.confident must be in [0.5,1]")
    if not cfg["questions"]:
        raise ConfigError("questions must not be empty")
    for key, q in cfg["questions"].items():
        if q.get("type") not in ("noul", "choice", "score") or not q.get("title") or not q.get("instructions"):
            raise ConfigError(f"invalid question: {key}")
        if q["type"] != "noul" and not q.get("criteria"):
            raise ConfigError(f"question {key} requires criteria")
    if cfg["questions"].get("product_taken", {}).get("type") != "noul":
        raise ConfigError("Retail evaluation requires product_taken as a noul question")
    n = cfg["run"].get("max_clips")
    if n is not None and (type(n) is not int or n <= 0):
        raise ConfigError("run.max_clips must be null or a positive integer")
