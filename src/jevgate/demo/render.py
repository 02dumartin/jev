"""Dashboard frames for the demo video ("Store and Next acts"), light theme, 1920x1080.

Left:  the upright clip with the known events circled, the event timeline, and the VLM's description (STATE).
Right, top:    VLM only: the VLM answers the questions itself -> time, plain answers (no probabilities), act.
Right, bottom: VLM + Jev: the VLM describes, Jev answers every question in one request -> time, 3x2 cards, act.
Text never gets cut: every label is wrapped to its box width.
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from jevgate.demo.clip import active_events, mark_events

EVENT_COLORS = {"take": "#FF8C1A", "put": "#3CA0DC", "touch": "#B05CE0"}
TYPE_NAMES = {"noul": "Boolean", "choice": "Choice", "score": "Score"}


def rgb(code: str) -> tuple[int, int, int]:
    code = code.lstrip("#")
    return tuple(int(code[i:i + 2], 16) for i in (0, 2, 4))


def pretty(value) -> str:
    return str(value).replace("_", " ")


class Dashboard:
    """Draws one canvas per video frame."""

    def __init__(self, cfg: dict, repo: Path):
        self.cfg, self.r = cfg, cfg["render"]
        self.W, self.H = self.r["canvas"]
        self.c = {k: rgb(v) for k, v in self.r["colors"].items()}
        fdir = repo / self.r["font_dir"]
        f = self.r["fonts"]

        def font(weight, size):
            return ImageFont.truetype(str(fdir / f[weight]), size)

        self.f = {"title": font("bold", 34), "h": font("bold", 21), "h2": font("medium", 18), "b": font("regular", 16),
                  "s": font("regular", 14), "value": font("bold", 22), "badge": font("bold", 30),
                  "card_t": font("medium", 16), "total": font("bold", 28)}

    # text helpers ---------------------------------------------------------------------------
    def wrap(self, d: ImageDraw.ImageDraw, text: str, width: float, font) -> list[str]:
        """Greedy word wrap by pixel width; a single over-long word is split by characters."""
        lines, cur = [], ""
        for word in str(text).split():
            trial = f"{cur} {word}".strip()
            if d.textlength(trial, font=font) <= width:
                cur = trial
                continue
            if cur:
                lines.append(cur)
            while d.textlength(word, font=font) > width:
                cut = len(word)
                while cut > 1 and d.textlength(word[:cut], font=font) > width:
                    cut -= 1
                lines.append(word[:cut])
                word = word[cut:]
            cur = word
        return lines + ([cur] if cur else [])

    def text_block(self, d, x, y, width, text, font, fill, gap=4) -> int:
        """Draw wrapped text; return the y below it."""
        h = font.size + gap
        for line in self.wrap(d, text, width, font):
            d.text((x, y), line, font=font, fill=fill)
            y += h
        return y

    def panel(self, d, box, title: str, right: str | None = None, total_s: float | None = None) -> int:
        """Panel with a title; `total_s` is drawn large in a red box at the top right, `right` beside it."""
        x0, y0, x1, _ = box
        d.rounded_rectangle(box, 14, fill=self.c["panel"], outline=self.c["line"], width=2)
        d.text((x0 + 18, y0 + 12), title, font=self.f["h"], fill=self.c["title"])
        edge = x1 - 18
        if total_s is not None:
            text = f"Time {total_s:.1f} s" if total_s >= 1 else f"Time {total_s:.2f} s"
            tw = d.textlength(text, font=self.f["total"])
            bx0 = edge - tw - 28
            d.rounded_rectangle([bx0, y0 + 8, edge, y0 + 54], 10, fill=self.c["panel"], outline=self.c["total"],
                                width=3)
            d.text(((bx0 + edge) / 2, y0 + 31), text, font=self.f["total"], fill=self.c["total"], anchor="mm")
            edge = bx0 - 14
        if right:
            d.text((edge, y0 + 31), right, font=self.f["h2"], fill=self.c["muted"], anchor="rm")
        return y0 + (64 if total_s is not None else 50)

    def badge(self, d, x, y, w, act: dict | None, note: str | None = None) -> None:
        if act is None:
            d.rounded_rectangle([x, y, x + w, y + 56], 12, outline=self.c["line"], width=2)
            d.text((x + 18, y + 28), "deciding …", font=self.f["h2"], fill=self.c["muted"], anchor="lm")
            return
        d.rounded_rectangle([x, y, x + w, y + 56], 12, fill=rgb(act["color"]))
        d.text((x + 18, y + 28), act["badge"], font=self.f["badge"], fill=self.c["title"], anchor="lm")
        note = note or act.get("reason")
        if note:
            tw = w - 40 - d.textlength(act["badge"], font=self.f["badge"])
            lines = self.wrap(d, note, max(120, tw), self.f["s"])[:2]
            top = y + 28 - 9 * len(lines)
            for i, line in enumerate(lines):
                d.text((x + w - 16, top + 9 + i * 18), line, font=self.f["s"], fill=self.c["title"], anchor="rm")

    # left column ----------------------------------------------------------------------------
    def left(self, canvas, d, img, meta, cur, state) -> None:
        side, x, y = 640, 30, 84
        ev = self.cfg["events"]
        shown = mark_events(img, active_events(meta["events"], cur), ev["marker_radius_frac"], ev["marker_color"])
        canvas.paste(Image.fromarray(shown).resize((side, side)), (x, y))
        ty = y + side + 14
        d.text((x, ty), "Known events (dataset annotation, stands in for an action detector)", font=self.f["s"],
               fill=self.c["muted"])
        ty += 22
        n = meta["n_frames"]
        d.rectangle([x, ty, x + side, ty + 18], fill=self.c["bar_bg"])
        for e in meta["events"]:
            a, b = (e["frames"][0] - 1) / n * side, e["frames"][1] / n * side
            d.rectangle([x + a, ty, x + b, ty + 18], fill=rgb(EVENT_COLORS.get(e["label"], "#9A9A9A")))
            d.text((x + a + 4, ty), e["label"], font=self.f["s"], fill=(255, 255, 255))
        px = x + min(cur, n) / n * side
        d.line([px, ty - 4, px, ty + 22], fill=self.c["title"], width=2)
        sy = ty + 34
        y_end = self.panel(d, [x, sy, x + side, self.H - 40], "Scene description (VLM)",
                           f"{state['vlm_s']:.1f} s" if state else None)
        if state is None:
            d.text((x + 18, y_end), "watching the clip …", font=self.f["b"], fill=self.c["muted"])
        else:
            yy = y_end
            for k, v in state["state"].items():
                v = ", ".join(map(str, v)) if isinstance(v, list) else str(v)
                yy = self.text_block(d, x + 18, yy, side - 36, f"{pretty(k)}: {v}", self.f["s"], self.c["ink"], 2)
                if yy > self.H - 60:
                    break
        d.text((x, self.H - 30), self.r["credit"], font=self.f["s"], fill=self.c["muted"])

    # right column ---------------------------------------------------------------------------
    def vlm_only(self, d, box, vo: dict | None) -> None:
        x0, y0, x1, y1 = box
        n_q = len(self.cfg["questions"])
        right = f"{n_q} questions, one generation" if vo else None
        y = self.panel(d, box, "VLM only  ·  the VLM looks and answers every question itself", right,
                       vo["vlm_s"] if vo else None)
        if vo is None:
            d.text((x0 + 18, y), "waiting …", font=self.f["b"], fill=self.c["muted"])
            self.badge(d, x0 + 18, y1 - 72, x1 - x0 - 36, None)
            return
        cols, colw = 3, (x1 - x0 - 36) / 3
        for i, c in enumerate(vo["cards"]):
            cx, cy = x0 + 18 + (i % cols) * colw, y + (i // cols) * 62
            d.text((cx, cy), c["title"], font=self.f["s"], fill=self.c["muted"])
            d.text((cx, cy + 20), pretty(c["value"]), font=self.f["h2"], fill=self.c["ink"])
        d.text((x0 + 18, y + 130), "No probabilities: the VLM cannot say which answers it is unsure about.",
               font=self.f["s"], fill=self.c["muted"])
        self.badge(d, x0 + 18, y1 - 72, x1 - x0 - 36, vo["action"])

    def card(self, d, x, y, w, h, c: dict) -> None:
        sure = c["sure"]
        d.rounded_rectangle([x, y, x + w, y + h], 12, fill=self.c["panel"],
                            outline=self.c["sure"] if sure else self.c["unsure"], width=2 if sure else 4)
        tag = TYPE_NAMES[c["type"]] + ("" if sure else "  ·  unsure")
        tw = d.textlength(tag, font=self.f["s"])
        d.text((x + w - 14 - tw, y + 10), tag, font=self.f["s"], fill=self.c["muted"] if sure else self.c["unsure"])
        yy = self.text_block(d, x + 14, y + 8, w - 40 - tw, c["title"], self.f["card_t"], self.c["muted"], 2)
        yy = self.text_block(d, x + 14, yy + 2, w - 28, pretty(c["value"]), self.f["value"], self.c["ink"], 2) + 4
        items = list(c["probs"].items())
        rows = items if c["type"] == "score" else sorted(items, key=lambda kv: -kv[1])[:3]
        lab_w = w * 0.46
        for lab, p in rows:
            lines = self.wrap(d, pretty(lab), lab_w, self.f["s"])
            for k, line in enumerate(lines):
                d.text((x + 14, yy + k * 17), line, font=self.f["s"], fill=self.c["ink"])
            bx0, bx1 = x + 18 + lab_w, x + w - 58
            d.rounded_rectangle([bx0, yy + 3, bx1, yy + 13], 4, fill=self.c["bar_bg"])
            if p > 0:
                d.rounded_rectangle([bx0, yy + 3, bx0 + max(6, (bx1 - bx0) * p), yy + 13], 4, fill=self.c["bar"])
            d.text((bx1 + 8, yy - 1), f"{p:.2f}", font=self.f["s"], fill=self.c["ink"])
            yy += 17 * len(lines) + 5

    def vlm_jev(self, d, box, dec: dict | None, state: dict | None, single: dict | None, act, reveal: float) -> None:
        x0, y0, x1, y1 = box
        right, total = None, None
        if dec is not None:
            right = f"Jev only (description {state['vlm_s']:.1f} s not included)"
            total = dec["latency_ms"] / 1000
        y = self.panel(d, box, "VLM + Jev  ·  the VLM describes, Jev answers all in one request", right, total)
        if dec is None:
            d.text((x0 + 18, y), "waiting for the description …", font=self.f["b"], fill=self.c["muted"])
            self.badge(d, x0 + 18, y1 - 72, x1 - x0 - 36, None)
            return
        if single:
            n_q = len(dec["cards"])
            d.text((x0 + 18, y - 10), f"asked one by one instead: {n_q} requests, {single['tokens']} tokens "
                   f"(vs {sum(dec['usage'].values())} tokens in one request)", font=self.f["s"], fill=self.c["muted"])
        cols, gap = 3, 14
        cw = (x1 - x0 - 36 - gap * (cols - 1)) / cols
        ch = (y1 - 72 - 16 - (y + 16) - gap) / 2
        shown = dec["cards"][: max(0, int(np.ceil(reveal * len(dec["cards"]))))]
        for i, c in enumerate(shown):
            self.card(d, x0 + 18 + (i % cols) * (cw + gap), y + 16 + (i // cols) * (ch + gap), cw, ch, c)
        self.badge(d, x0 + 18, y1 - 72, x1 - x0 - 36, act if reveal >= 1.0 else None)

    # whole frame ----------------------------------------------------------------------------
    def frame(self, img, meta, cur, clip_no, n_clips, state, dec_all: dict | None, reveal: float) -> np.ndarray:
        canvas = Image.new("RGB", (self.W, self.H), self.c["background"])
        d = ImageDraw.Draw(canvas)
        d.text((30, 18), self.r["title"], font=self.f["title"], fill=self.c["title"])
        tag = f"camera clip {clip_no}/{n_clips}  ·  #{meta['sample_id']}"
        d.text((self.W - 30 - d.textlength(tag, font=self.f["h2"]), 30), tag, font=self.f["h2"], fill=self.c["muted"])
        self.left(canvas, d, img, meta, cur, state)
        rx, rw = 700, self.W - 730
        done = dec_all is not None
        self.vlm_only(d, [rx, 84, rx + rw, 84 + 290], dec_all.get("vlm_only") if done else None)
        self.vlm_jev(d, [rx, 392, rx + rw, self.H - 40], dec_all["dec"] if done else None, state,
                     dec_all.get("single") if done else None, dec_all["action"] if done else None, reveal)
        return np.asarray(canvas)


def render_clip(dash: Dashboard, writer, frames: list[np.ndarray], meta: dict, clip_no: int, n_clips: int,
                state: dict, dec_all: dict) -> None:
    """Play the clip with empty panels, then reveal the description, both answer panels, and hold."""
    for i, img in enumerate(frames):
        writer.send(dash.frame(img, meta, i + 1, clip_no, n_clips, None, None, 0.0))
    fps, last = dash.r["fps"], frames[-1]
    half = max(1, fps // 2)
    steps = [(state, None, 0.0)] * half + [(state, dec_all, (k + 1) / half) for k in range(half)]
    steps += [(state, dec_all, 1.0)] * int(dash.r["hold_s"] * fps)
    for st, de, rv in steps:
        writer.send(dash.frame(last, meta, len(frames), clip_no, n_clips, st, de, rv))
