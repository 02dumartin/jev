"""Dashboard frames for the demo video (PIL drawing, H.264 via outputs.viz.h264_writer).

Per clip: the upright clip plays with the known events circled; when it ends the VLM STATE appears, then the Jev
cards (all from one request) and the action badge, and the frame is held for `render.hold_s`.
"""

import textwrap
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from jevgate.demo.clip import active_events, mark_events

FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
BG, PANEL, INK, MUTED, LINE = (14, 17, 23), (27, 32, 41), (235, 238, 243), (150, 158, 170), (52, 60, 74)
SURE, UNSURE, BAR = (78, 211, 107), (255, 225, 53), (76, 120, 168)
EVENT_COLORS = {"take": (255, 140, 26), "put": (60, 200, 255), "touch": (211, 91, 255)}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_DIR / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")), size)


def hex_rgb(code: str) -> tuple[int, int, int]:
    code = code.lstrip("#")
    return tuple(int(code[i:i + 2], 16) for i in (0, 2, 4))


class Dashboard:
    """Draws one canvas per video frame. Layout is fixed for a 1600x900 canvas."""

    def __init__(self, cfg: dict):
        self.cfg, self.r = cfg, cfg["render"]
        self.W, self.H = self.r["canvas"]
        self.f = {k: font(s, b) for k, (s, b) in {"title": (22, True), "h": (17, True), "b": (15, False),
                                                  "s": (13, False), "big": (30, True), "card": (14, True)}.items()}

    # pieces ---------------------------------------------------------------------------------
    def header(self, d: ImageDraw.ImageDraw, clip_no: int, n_clips: int, sample_id: str) -> None:
        d.rectangle([0, 0, self.W, 46], fill=PANEL)
        d.text((18, 12), self.r["title"], font=self.f["title"], fill=INK)
        d.text((self.W - 300, 15), f"camera clip {clip_no}/{n_clips}  ·  #{sample_id}", font=self.f["b"], fill=MUTED)

    def timeline(self, d, x, y, w, n_frames: int, events: list[dict], cur: int) -> None:
        """Annotated events as coloured spans over the clip's frames, with a playhead."""
        d.text((x, y), "Known events (dataset annotation, stands in for an action detector)", font=self.f["s"],
               fill=MUTED)
        y += 20
        d.rectangle([x, y, x + w, y + 16], fill=PANEL)
        for e in events:
            a, b = (e["frames"][0] - 1) / n_frames, e["frames"][1] / n_frames
            d.rectangle([x + a * w, y, x + b * w, y + 16], fill=EVENT_COLORS.get(e["label"], MUTED))
            d.text((x + a * w + 3, y + 1), e["label"], font=self.f["s"], fill=BG)
        px = x + min(cur, n_frames) / n_frames * w
        d.line([px, y - 3, px, y + 19], fill=INK, width=2)

    def state_box(self, d, x, y, w, h, obs: dict | None) -> None:
        d.rounded_rectangle([x, y, x + w, y + h], 10, fill=PANEL, outline=LINE)
        d.text((x + 14, y + 10), "① VLM observes  (Qwen3-VL-8B)", font=self.f["h"], fill=INK)
        if obs is None:
            d.text((x + 14, y + 42), "watching the clip ...", font=self.f["b"], fill=MUTED)
            return
        d.text((x + w - 120, y + 12), f"{obs['vlm_s']:.1f} s", font=self.f["h"], fill=MUTED)
        lines, yy = [], y + 40
        for k, v in obs["state"].items():
            v = ", ".join(map(str, v)) if isinstance(v, list) else str(v)
            lines += textwrap.wrap(f"{k}: {v}", 64) or [""]
        for line in lines[: (h - 50) // 19]:
            d.text((x + 14, yy), line, font=self.f["s"], fill=INK)
            yy += 19

    def cards(self, d, x, y, w, dec: dict | None, single: dict | None, reveal: float) -> None:
        d.text((x, y), "② Jev decides: one request, all questions at once", font=self.f["h"], fill=INK)
        if dec is None:
            d.text((x, y + 30), "waiting for the observation ...", font=self.f["b"], fill=MUTED)
            return
        n_q = len(dec["cards"])
        tag = f"1 request · {n_q} questions · {dec['latency_ms'] / 1000:.2f} s"
        if single:
            tag += f"   (asked one by one: {n_q} requests · {single['wall_ms'] / 1000:.2f} s wall)"
        d.text((x, y + 24), tag, font=self.f["s"], fill=MUTED)
        cw, ch, gap = (w - 12) // 2, 88, 8
        shown = dec["cards"][: max(0, int(np.ceil(reveal * n_q)))]
        for i, c in enumerate(shown):
            cx, cy = x + (i % 2) * (cw + 12), y + 50 + (i // 2) * (ch + gap)
            d.rounded_rectangle([cx, cy, cx + cw, cy + ch], 9, fill=PANEL, outline=SURE if c["sure"] else UNSURE,
                                width=2 if c["sure"] else 3)
            d.text((cx + 10, cy + 7), c["title"], font=self.f["card"], fill=MUTED)
            d.text((cx + 10, cy + 26), str(c["value"]).replace("_", " "), font=self.f["h"], fill=INK)
            d.text((cx + cw - 70, cy + 8), {"noul": "Boolean", "choice": "Choice", "score": "Score"}[c["type"]],
                   font=self.f["s"], fill=MUTED)
            if not c["sure"]:
                d.text((cx + cw - 70, cy + 28), "unsure", font=self.f["s"], fill=UNSURE)
            items = list(c["probs"].items())
            top = items if c["type"] == "score" else sorted(items, key=lambda kv: -kv[1])[:3]
            bw = (cw - 20) / len(top)
            for j, (lab, p) in enumerate(top):
                bx = cx + 10 + j * bw
                d.rectangle([bx, cy + 50, bx + bw - 6, cy + 62], fill=LINE)
                d.rectangle([bx, cy + 50, bx + (bw - 6) * p, cy + 62], fill=BAR)
                text = self.fit(d, f"{str(lab).replace('_', ' ')} {p:.2f}", bw - 8, self.f["s"])
                d.text((bx, cy + 66), text, font=self.f["s"], fill=MUTED)

    @staticmethod
    def fit(d, text: str, width: float, f) -> str:
        """Shorten text with an ellipsis until it fits in width pixels."""
        if d.textlength(text, font=f) <= width:
            return text
        while text and d.textlength(text + "…", font=f) > width:
            text = text[:-1]
        return text + "…"

    def badge(self, d, x, y, w, act: dict | None) -> None:
        d.text((x, y), "③ Code acts on the answers", font=self.f["h"], fill=INK)
        if act is None:
            return
        color = hex_rgb(act["color"])
        d.rounded_rectangle([x, y + 26, x + w, y + 80], 12, fill=color)
        d.text((x + 18, y + 36), act["badge"], font=self.f["big"], fill=BG)
        if not act.get("sure", True):
            d.text((x + w - 260, y + 42), f"staff action unsure ({act['p']:.2f})", font=self.f["b"], fill=BG)

    # whole frame ----------------------------------------------------------------------------
    def frame(self, img: np.ndarray, meta: dict, cur: int, clip_no: int, n_clips: int, obs, dec, single, act,
              reveal: float) -> np.ndarray:
        canvas = Image.new("RGB", (self.W, self.H), BG)
        d = ImageDraw.Draw(canvas)
        self.header(d, clip_no, n_clips, meta["sample_id"])
        side = 700
        ev = self.cfg["events"]
        shown = mark_events(img, active_events(meta["events"], cur), ev["marker_radius_frac"], ev["marker_color"])
        canvas.paste(Image.fromarray(shown).resize((side, side)), (20, 60))
        self.timeline(d, 20, 770, side, meta["n_frames"], meta["events"], cur)
        d.text((20, self.H - 26), self.r["credit"], font=self.f["s"], fill=MUTED)
        rx, rw = 740, self.W - 760
        self.state_box(d, rx, 60, rw, 200, obs)
        self.cards(d, rx, 274, rw, dec, single, reveal)
        self.badge(d, rx, 742, rw, act)
        return np.asarray(canvas)


def render_clip(dash: Dashboard, writer, frames: list[np.ndarray], meta: dict, clip_no: int, n_clips: int, obs: dict,
                dec: dict, single: dict | None, act: dict) -> None:
    """Play the clip with an empty panel, then reveal STATE, cards and badge over the hold."""
    for i, img in enumerate(frames):
        writer.send(dash.frame(img, meta, i + 1, clip_no, n_clips, None, None, None, None, 0.0))
    fps, last = dash.r["fps"], frames[-1]
    hold = int(dash.r["hold_s"] * fps)
    steps = [(obs, None, None, 0.0)] * max(1, fps // 2)
    steps += [(obs, dec, single, (k + 1) / max(1, fps // 2)) for k in range(max(1, fps // 2))]
    steps += [(obs, dec, single, 1.0)] * hold
    for o, de, si, rv in steps:
        writer.send(dash.frame(last, meta, len(frames), clip_no, n_clips, o, de, si,
                               act if rv >= 1.0 else None, rv))
