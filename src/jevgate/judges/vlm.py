"""② VLM: Qwen3-VL picks one of the judgment classes from pixels (full frame with the box drawn + crop).

The classes are listed as options A, B, C, ...; one forward pass gives the next-token logits of
those letters, and their softmax is the class distribution (used for accuracy, ROC and ECE).
"""

import time

import torch
from PIL import Image, ImageDraw

from ..config import class_names
from ..data import image_path
from .base import Budget, Judge, decision

LETTERS = "ABCDEFGHIJ"


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

    def __init__(self, cfg: dict, budget: Budget, log=None):
        self.log = log
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
        self.classes = class_names(cfg)
        self.letters = LETTERS[: len(self.classes)]
        self.letter_ids = [tok.encode(letter, add_special_tokens=False)[0] for letter in self.letters]
        options = "\n".join(f"{letter}. {c}: {cfg['classes'][c]}" for letter, c in zip(self.letters, self.classes,
                                                                                   strict=True))
        self.prompt = (
            f"{cfg['prompts']['context']}\n"
            "Image 1 is the full frame with the candidate box drawn in red. "
            "Image 2 is a zoomed crop around the candidate box.\n"
            "{hint}"
            f"{cfg['prompts']['question']}\n{options}\n"
            f"Answer with a single letter ({', '.join(self.letters)})."
        )

    def _prompt(self, cand: dict) -> str:
        """Prompt for one box; with show_detector_hint the detector's class and score are stated."""
        hint = (f"Detector proposal for this box: {cand['dino_label']} (score {cand['dino_score']:.2f}).\n"
                if self.vcfg.get("show_detector_hint") else "")
        return self.prompt.replace("{hint}", hint)

    def _inputs(self, items):
        texts, images = [], []
        for img, box, prompt in items:
            messages = [{"role": "user", "content": [
                {"type": "image"}, {"type": "image"}, {"type": "text", "text": prompt}]}]
            texts.append(self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            images.append([
                frame_view(img, box, self.vcfg["box_color"], self.vcfg["box_width"]),
                crop_view(img, box, self.vcfg["crop_context"], self.vcfg["crop_long_side"]),
            ])
        return self.processor(text=texts, images=images, padding=True, return_tensors="pt").to(self.device)

    @torch.inference_mode()
    def _class_probs(self, items) -> list[dict[str, float]]:
        logits = self.model(**self._inputs(items)).logits[:, -1, :].float()
        probs = torch.softmax(logits[:, self.letter_ids], dim=1).tolist()
        return [dict(zip(self.classes, p, strict=True)) for p in probs]

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
                    # candidates arrive frame by frame, so one cached image is enough
                    cache = {file: Image.open(image_path(self.cfg, file)).convert("RGB")}
                items.append((cache[file], c["box_xyxy"], self._prompt(c)))
            if self.device.startswith("cuda"):
                torch.cuda.synchronize(self.device)
            started = time.perf_counter()
            probs = self._class_probs(items)
            if self.device.startswith("cuda"):
                torch.cuda.synchronize(self.device)
            per_box = (time.perf_counter() - started) * 1000 / len(batch)
            rows += [decision(c["cand_id"], p, per_box, batch_size=len(batch))
                     for c, p in zip(batch, probs, strict=True)]
            if self.log and (s // bs) % 200 == 0:
                self.log(f"  VLM {s + len(batch)}/{len(candidates)} boxes")
        return rows
