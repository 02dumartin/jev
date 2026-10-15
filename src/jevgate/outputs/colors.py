"""One colour per label, read from `viz.class_colors`, shared by overlays, videos and figures."""

from jevgate.core.config import output_classes


def hex_to_rgb(code: str) -> tuple[int, int, int]:
    code = code.lstrip("#")
    return tuple(int(code[i:i + 2], 16) for i in (0, 2, 4))


def class_colors(cfg: dict) -> dict[str, str]:
    """label -> '#RRGGBB' for the six animals and background; fails if any label has no colour."""
    colors = cfg["viz"]["class_colors"]
    missing = [c for c in output_classes(cfg) if c not in colors]
    if missing:
        raise ValueError(f"viz.class_colors has no colour for {missing}")
    return {c: colors[c] for c in output_classes(cfg)}


def class_colors_bgr(cfg: dict) -> dict[str, tuple[int, int, int]]:
    """Same colours in OpenCV's BGR order."""
    return {c: hex_to_rgb(h)[::-1] for c, h in class_colors(cfg).items()}
