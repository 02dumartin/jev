"""Qwen3-VL judges of a track (PLAN_brackish.md v7 §1, §4).

Both see the same two images of a track's representative box (highest detector conf): the full frame with
the box drawn in red and a zoomed crop (view settings in `vlm:`).
- Describer (small VLM, `describer:`): writes what is in the box (text for Jev; not told the detector's
  class) and gives its own p_yes for the common question.
- Verifier (large VLM, `vlm:`): p_yes for the common question.
p_yes = softmax over the next-token logits of "Yes" and "No".
"""

import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from jevgate.data.brackish import image_path
from jevgate.judges.base import Budget


def crop_view(img: Image.Image, box, context: float, long_side: int) -> Image.Image:
    """Box expanded by `context` x its size on each side, resized so the longer side = long_side."""
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    region = (max(0, x1 - context * bw), max(0, y1 - context * bh),
              min(img.width, x2 + context * bw), min(img.height, y2 + context * bh))
    crop = img.crop(tuple(int(round(v)) for v in region))
    scale = long_side / max(crop.width, crop.height, 1)
    return crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))), Image.BICUBIC)


def frame_view(img: Image.Image, box, color, width: int) -> Image.Image:
    """The full frame with the box outlined."""
    out = img.copy()
    ImageDraw.Draw(out).rectangle(box, outline=tuple(color), width=width)
    return out


def image_stem(track_uid: str) -> str:
    return track_uid.replace("#", "_")


def track_images(cfg: dict, box: dict, image_dir: Path, track_uid: str) -> list[Image.Image]:
    """The two input images of a track; written once to image_dir and re-read by later judges."""
    paths = [image_dir / f"{image_stem(track_uid)}_frame.jpg", image_dir / f"{image_stem(track_uid)}_crop.jpg"]
    if all(p.exists() for p in paths):
        return [Image.open(p).convert("RGB") for p in paths]
    v = cfg["vlm"]
    img = Image.open(image_path(cfg, box["file"])).convert("RGB")
    views = [frame_view(img, box["box_xyxy"], v["box_color"], v["box_width"]),
             crop_view(img, box["box_xyxy"], v["crop_context"], v["crop_long_side"])]
    for view, p in zip(views, paths, strict=True):
        view.save(p, quality=95)
    return [Image.open(p).convert("RGB") for p in paths]  # judges always see the saved JPEGs


def class_lines(cfg: dict) -> str:
    return "\n".join(f"- {k}: {v}" for k, v in cfg["classes"].items())


IMAGES_INTRO = ("Image 1 is one frame of the track with the track's box drawn in red. "
                "Image 2 is a zoomed crop around that box.\n")


def question_prompt(cfg: dict, cls: str) -> str:
    """The common question (same wording Jev gets), answered Yes/No."""
    return (f"{cfg['prompts']['context']}\nClass definitions:\n{class_lines(cfg)}\n{IMAGES_INTRO}"
            f"{cfg['prompts']['question'].format(cls=cls)} Answer Yes or No.")


def describe_prompt(cfg: dict) -> str:
    """Free description of the box; the detector's class is not mentioned."""
    return f"{cfg['prompts']['context']}\n{IMAGES_INTRO}{cfg['prompts']['describe']}"


class QwenVL:
    """One loaded Qwen3-VL model: Yes probability and short generation over (prompt, two images)."""

    def __init__(self, section: dict):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.device = section["device"]
        self.processor = AutoProcessor.from_pretrained(section["model"])
        self.processor.tokenizer.padding_side = "left"
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            section["model"], dtype=getattr(torch, section["dtype"]), device_map=self.device).eval()
        tok = self.processor.tokenizer
        self.yes_id = tok.encode("Yes", add_special_tokens=False)[0]
        self.no_id = tok.encode("No", add_special_tokens=False)[0]

    def _inputs(self, prompts: list[str], images: list[list[Image.Image]]):
        texts = [self.processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": p}]}],
            tokenize=False, add_generation_prompt=True) for p in prompts]
        return self.processor(text=texts, images=images, padding=True, return_tensors="pt").to(self.device)

    @torch.inference_mode()
    def p_yes(self, prompts: list[str], images: list[list[Image.Image]]) -> list[float]:
        logits = self.model(**self._inputs(prompts, images)).logits[:, -1, :].float()
        return torch.softmax(logits[:, [self.yes_id, self.no_id]], dim=1)[:, 0].tolist()

    @torch.inference_mode()
    def generate(self, prompts: list[str], images: list[list[Image.Image]], max_new_tokens: int) -> list[str]:
        inputs = self._inputs(prompts, images)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return [t.strip() for t in self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                                               skip_special_tokens=True)]

    def sync(self) -> None:
        if self.device.startswith("cuda"):
            torch.cuda.synchronize(self.device)


class TrackVLM:
    """Runs a QwenVL over tracks in batches. describe=True also writes a description per track."""

    def __init__(self, cfg: dict, section: str, budget: Budget | None, image_dir: Path, describe: bool, log=None):
        self.cfg, self.section, self.budget, self.image_dir, self.describe, self.log = (
            cfg, cfg[section], budget, image_dir, describe, log)
        self.name = section
        self.vl = QwenVL(self.section)

    def judge(self, tracks: list[dict], boxes: dict[str, dict]) -> tuple[list[dict], list[dict]]:
        """(decision rows, prompt rows), one per track in track order."""
        rows, prompt_rows, bs = [], [], self.section["batch_size"]
        for s in range(0, len(tracks), bs):
            batch = tracks[s:s + bs]
            if self.budget is not None:
                self.budget.reserve_vlm_calls(len(batch))
            reps = [boxes[t["rep_box_id"]] for t in batch]
            images = [track_images(self.cfg, b, self.image_dir, t["track_uid"]) for t, b in zip(batch, reps,
                                                                                                strict=True)]
            prompts = [question_prompt(self.cfg, t["majority_class"]) for t in batch]
            self.vl.sync()
            started = time.perf_counter()
            probs = self.vl.p_yes(prompts, images)
            texts = (self.vl.generate([describe_prompt(self.cfg)] * len(batch), images,
                                      self.section["max_new_tokens"]) if self.describe else [None] * len(batch))
            self.vl.sync()
            per_track = (time.perf_counter() - started) * 1000 / len(batch)
            for t, b, p, text, prompt in zip(batch, reps, probs, texts, prompts, strict=True):
                row = {"track_uid": t["track_uid"], "p_yes": round(p, 6), "latency_ms": round(per_track, 3),
                       "rep_box_id": b["box_id"], "batch_size": len(batch)}
                if self.describe:
                    row["description"] = text
                rows.append(row)
                stem = image_stem(t["track_uid"])
                prompt_rows.append({"track_uid": t["track_uid"], "rep_box_id": b["box_id"], "file": b["file"],
                                    "box_xyxy": b["box_xyxy"], "question_prompt": prompt,
                                    "describe_prompt": describe_prompt(self.cfg) if self.describe else None,
                                    "images": [f"images/{stem}_frame.jpg", f"images/{stem}_crop.jpg"]})
            if self.log and (s // bs) % 20 == 0:
                self.log(f"  {self.name} {s + len(batch)}/{len(tracks)} tracks")
        return rows, prompt_rows
