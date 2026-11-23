"""VLM step: look at a clip and write an objective STATE (what is observable; no decisions)."""

import json
import re
import time

import numpy as np
import torch
from PIL import Image

from jev.preprocess.frames import active_events, mark_events, sample_indices


def events_sentence(events: list[dict], idx: list[int]) -> str:
    """Known events stated in the numbering of the frames the VLM sees (frame 1..k)."""
    if not events:
        return "An in-store action detector found no product interaction in this clip. "
    parts = []
    for e in events:
        shown = [k + 1 for k, i in enumerate(idx) if e["frames"][0] <= i + 1 <= e["frames"][1]]
        where = (
            f"frames {shown[0]}-{shown[-1]}"
            if len(shown) > 1
            else f"frame {shown[0]}"
            if shown
            else "between the shown frames"
        )
        parts.append(f"{e['label']} ({where})")
    return (
        "Known events from an in-store action detector, in time order (a yellow circle marks where the hand "
        f"touched the product): {'; '.join(parts)}. "
    )


def parse_state(text: str) -> dict:
    """The JSON object in the model's answer, or {'raw': text} when it does not parse."""
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {"raw": text.strip()}


class Observer:
    """Qwen3-VL that turns a clip into a STATE dict."""

    def __init__(self, cfg: dict):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.cfg, self.v = cfg, cfg["vlm"]
        self.processor = AutoProcessor.from_pretrained(self.v["model"])
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.v["model"], dtype=getattr(torch, self.v["dtype"]), device_map=self.v["device"]
        ).eval()

    def inputs_for(self, frames: list[np.ndarray], events: list[dict]) -> tuple[list[Image.Image], str, list[int]]:
        """The k frames the VLM sees (events marked when enabled) and its prompt."""
        idx = sample_indices(len(frames), self.v["frames"])
        ev_cfg = self.cfg["events"]
        imgs = []
        for i in idx:
            img = frames[i]
            if ev_cfg["enabled"]:
                img = mark_events(
                    img, active_events(events, i + 1), ev_cfg["marker_radius_frac"], ev_cfg["marker_color"], label=False
                )
            px = self.v["frame_px"]
            imgs.append(Image.fromarray(img).resize((px, px)))
        ev = events_sentence(events, idx) if ev_cfg["enabled"] else ""
        return imgs, self.v["prompt"].format(n=len(imgs), events=ev), idx

    @torch.inference_mode()
    def answer(self, frames: list[np.ndarray], events: list[dict]) -> dict:
        """VLM-only path: the same frames, the configured questions answered directly as JSON (no probabilities)."""
        from jev.decision.policy import option_values

        imgs, _, idx = self.inputs_for(frames, events)
        qs = "\n".join(
            f"- {k}: {q['instructions']} One of: {', '.join(option_values(q))}"
            for k, q in self.cfg["questions"].items()
        )
        ev = events_sentence(events, idx) if self.cfg["events"]["enabled"] else ""
        prompt = self.cfg["vlm_only"]["prompt"].format(n=len(imgs), events=ev, questions=qs)
        raw, seconds = self._generate(imgs, prompt, self.cfg["vlm_only"]["max_new_tokens"])
        return {"answers": parse_state(raw), "raw": raw, "prompt": prompt, "vlm_s": round(seconds, 3)}

    def _generate(self, imgs, prompt: str, max_new_tokens: int) -> tuple[str, float]:
        msgs = [{"role": "user", "content": [{"type": "image"} for _ in imgs] + [{"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[imgs], return_tensors="pt").to(self.v["device"])
        torch.cuda.synchronize()
        started = time.perf_counter()
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        torch.cuda.synchronize()
        raw = self.processor.batch_decode(out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)[0]
        return raw, time.perf_counter() - started

    @torch.inference_mode()
    def observe(self, frames: list[np.ndarray], events: list[dict]) -> dict:
        imgs, prompt, idx = self.inputs_for(frames, events)
        msgs = [{"role": "user", "content": [{"type": "image"} for _ in imgs] + [{"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[imgs], return_tensors="pt").to(self.v["device"])
        torch.cuda.synchronize()
        started = time.perf_counter()
        out = self.model.generate(**inputs, max_new_tokens=self.v["max_new_tokens"], do_sample=False)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        raw = self.processor.batch_decode(out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)[0]
        return {
            "state": parse_state(raw),
            "raw": raw,
            "prompt": prompt,
            "frame_indices": idx,
            "vlm_s": round(seconds, 3),
            "images": imgs,
        }
