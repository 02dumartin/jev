"""③ VLM: Qwen3-VL answers the same question as Jev from pixels (PLAN.md v6 §5).

Input per track: its representative box (highest detector conf) as two images, the full frame with the
box drawn and a zoomed crop. p_yes = softmax over the next-token logits of "Yes" and "No". The images
and prompt actually used are saved under vlm/input/.
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


def build_prompt(cfg: dict, cls: str) -> str:
    """Shared context, the class definitions and the same question Jev gets."""
    defs = "\n".join(f"- {k}: {v}" for k, v in cfg["classes"].items())
    return (f"{cfg['prompts']['context']}\nClass definitions:\n{defs}\n"
            "Image 1 is one frame of the track with the track's box drawn in red. "
            "Image 2 is a zoomed crop around that box.\n"
            f"{cfg['prompts']['question'].format(cls=cls)} Answer Yes or No.")


class VLMJudge:
    """Loads Qwen3-VL once and judges tracks in batches."""

    def __init__(self, cfg: dict, budget: Budget, image_dir: Path, log=None):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.cfg, self.vcfg, self.budget, self.image_dir, self.log = cfg, cfg["vlm"], budget, image_dir, log
        self.device = self.vcfg["device"]
        self.processor = AutoProcessor.from_pretrained(self.vcfg["model"])
        self.processor.tokenizer.padding_side = "left"
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.vcfg["model"], dtype=getattr(torch, self.vcfg["dtype"]), device_map=self.device).eval()
        tok = self.processor.tokenizer
        self.yes_id = tok.encode("Yes", add_special_tokens=False)[0]
        self.no_id = tok.encode("No", add_special_tokens=False)[0]

    def _views(self, track: dict, box: dict) -> list[Image.Image]:
        img = Image.open(image_path(self.cfg, box["file"])).convert("RGB")
        views = [frame_view(img, box["box_xyxy"], self.vcfg["box_color"], self.vcfg["box_width"]),
                 crop_view(img, box["box_xyxy"], self.vcfg["crop_context"], self.vcfg["crop_long_side"])]
        stem = track["track_uid"].replace("#", "_")
        views[0].save(self.image_dir / f"{stem}_frame.jpg", quality=90)
        views[1].save(self.image_dir / f"{stem}_crop.jpg", quality=90)
        return views

    @torch.inference_mode()
    def _p_yes(self, prompts: list[str], images: list[list[Image.Image]]) -> list[float]:
        texts = [self.processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": p}]}],
            tokenize=False, add_generation_prompt=True) for p in prompts]
        inputs = self.processor(text=texts, images=images, padding=True, return_tensors="pt").to(self.device)
        logits = self.model(**inputs).logits[:, -1, :].float()
        return torch.softmax(logits[:, [self.yes_id, self.no_id]], dim=1)[:, 0].tolist()

    def judge(self, tracks: list[dict], boxes: dict[str, dict]) -> tuple[list[dict], list[dict]]:
        """(decision rows, prompt rows), one per track in track order."""
        rows, prompt_rows, bs = [], [], self.vcfg["batch_size"]
        for s in range(0, len(tracks), bs):
            batch = tracks[s:s + bs]
            self.budget.reserve_vlm_calls(len(batch))
            reps = [boxes[t["rep_box_id"]] for t in batch]
            prompts = [build_prompt(self.cfg, t["majority_class"]) for t in batch]
            images = [self._views(t, b) for t, b in zip(batch, reps, strict=True)]
            if self.device.startswith("cuda"):
                torch.cuda.synchronize(self.device)
            started = time.perf_counter()
            probs = self._p_yes(prompts, images)
            if self.device.startswith("cuda"):
                torch.cuda.synchronize(self.device)
            per_track = (time.perf_counter() - started) * 1000 / len(batch)
            for t, b, p, prompt in zip(batch, reps, probs, prompts, strict=True):
                rows.append({"track_uid": t["track_uid"], "p_yes": round(p, 6), "latency_ms": round(per_track, 3),
                             "rep_box_id": b["box_id"], "batch_size": len(batch)})
                stem = t["track_uid"].replace("#", "_")
                prompt_rows.append({"track_uid": t["track_uid"], "rep_box_id": b["box_id"], "file": b["file"],
                                    "box_xyxy": b["box_xyxy"], "prompt": prompt,
                                    "images": [f"images/{stem}_frame.jpg", f"images/{stem}_crop.jpg"]})
            if self.log and (s // bs) % 20 == 0:
                self.log(f"  VLM {s + len(batch)}/{len(tracks)} tracks")
        return rows, prompt_rows
