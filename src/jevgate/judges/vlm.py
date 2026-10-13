"""② VLM: Qwen3-VL answers the same question from pixels (full frame with the box drawn + zoomed crop).

The decision is the next-token comparison of "Yes" vs "No" after the prompt; p_yes is stored for
the appendix only (the plan does not report ECE for ②).
"""

import time

import torch
from PIL import Image, ImageDraw

from ..data import image_path
from .base import Budget, Judge, decision


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
    out = img.copy()
    ImageDraw.Draw(out).rectangle(box, outline=tuple(color), width=width)
    return out


class VLMJudge(Judge):
    name = "vlm"

    def __init__(self, cfg: dict, budget: Budget):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.cfg = cfg
        self.vcfg = cfg["vlm"]
        self.budget = budget
        self.device = self.vcfg["device"]
        self.processor = AutoProcessor.from_pretrained(self.vcfg["model"])
        self.processor.tokenizer.padding_side = "left"
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.vcfg["model"], dtype=getattr(torch, self.vcfg["dtype"]), device_map=self.device
        ).eval()
        tok = self.processor.tokenizer
        self.yes_id = tok.encode("Yes", add_special_tokens=False)[0]
        self.no_id = tok.encode("No", add_special_tokens=False)[0]
        self.prompt = (
            f"{cfg['prompts']['context']}\n"
            "Image 1 is the full frame with the candidate box drawn in red. "
            "Image 2 is a zoomed crop around the candidate box.\n"
            f"{cfg['prompts']['question']} Answer with Yes or No."
        )

    def _inputs(self, items):
        texts, images = [], []
        for img, box in items:
            messages = [{"role": "user", "content": [
                {"type": "image"}, {"type": "image"}, {"type": "text", "text": self.prompt}]}]
            texts.append(self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            images.append([
                frame_view(img, box, self.vcfg["box_color"], self.vcfg["box_width"]),
                crop_view(img, box, self.vcfg["crop_context"], self.vcfg["crop_long_side"]),
            ])
        return self.processor(text=texts, images=images, padding=True, return_tensors="pt").to(self.device)

    @torch.inference_mode()
    def _p_yes(self, items) -> list[float]:
        logits = self.model(**self._inputs(items)).logits[:, -1, :].float()
        pair = torch.stack([logits[:, self.yes_id], logits[:, self.no_id]], dim=1)
        return torch.softmax(pair, dim=1)[:, 0].tolist()

    def judge(self, frames, candidates, evidence):
        file_of = {(f["clip"], f["frame_idx"]): f["file"] for f in frames}
        cache: dict[str, Image.Image] = {}
        rows = []
        bs = self.vcfg["batch_size"]
        for s in range(0, len(candidates), bs):
            batch = candidates[s:s + bs]
            self.budget.reserve_vlm_calls(len(batch))
            items = []
            for c in batch:
                file = file_of[(c["clip"], c["frame_idx"])]
                if file not in cache:
                    cache = {file: Image.open(image_path(self.cfg, file)).convert("RGB")}
                items.append((cache[file], c["box_xyxy"]))
            if self.device.startswith("cuda"):
                torch.cuda.synchronize(self.device)
            started = time.perf_counter()
            probs = self._p_yes(items)
            if self.device.startswith("cuda"):
                torch.cuda.synchronize(self.device)
            per_box = (time.perf_counter() - started) * 1000 / len(batch)
            rows += [decision(c["cand_id"], p >= 0.5, None, per_box, p_yes=round(p, 4), batch_size=len(batch))
                     for c, p in zip(batch, probs, strict=True)]
        return rows
