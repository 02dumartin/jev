"""S2: Grounding DINO candidate boxes, shared by every system (PLAN.md §2.2).

Detections are cached per frame under run.cache_root, keyed by the detector + data config, so dev,
test and smoke runs never re-detect the same frame.
"""

import time
from pathlib import Path

import torch
from PIL import Image

from .boxes import area, clip_box, nms
from .config import config_hash
from .data import frame_file, image_path
from .runio import read_jsonl


def normalize_label(raw: str, phrase_to_class: dict[str, str]) -> str | None:
    """Map a DINO phrase (may contain word pieces like '##fish' or merged phrases) to one class."""
    text = " ".join(raw.replace("##", "").lower().split())
    for phrase in sorted(phrase_to_class, key=len, reverse=True):
        if phrase in text:
            return phrase_to_class[phrase]
    for phrase in phrase_to_class:
        if text and text in phrase:
            return phrase_to_class[phrase]
    return None


def cand_id(clip: str, idx: int, k: int) -> str:
    return f"{clip[-25:]}-{idx:04d}-{k:02d}"


class GroundingDinoDetector:
    """Zero-shot detector with a low box threshold so that both real and false candidates appear."""

    def __init__(self, cfg: dict):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.dcfg = cfg["detector"]
        self.wh = cfg["data"]["frame_wh"]
        self.device = self.dcfg["device"]
        self.processor = AutoProcessor.from_pretrained(self.dcfg["model"])
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.dcfg["model"]).to(self.device).eval()

    @torch.inference_mode()
    def _detect_batch(self, images: list[Image.Image]) -> list[dict]:
        prompt = self.dcfg["prompt"]
        inputs = self.processor(images=images, text=[prompt] * len(images), return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        return self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            threshold=self.dcfg["box_threshold"], text_threshold=self.dcfg["text_threshold"],
            target_sizes=[(img.height, img.width) for img in images],
        )

    def _postprocess(self, result: dict) -> list[dict]:
        """Clip to the frame, drop degenerate/huge boxes, class-agnostic NMS, keep top-N by score."""
        frame_area = self.wh[0] * self.wh[1]
        boxes, scores, labels = [], [], []
        text_labels = result.get("text_labels", result.get("labels"))
        for box, score, label in zip(result["boxes"].tolist(), result["scores"].tolist(), text_labels, strict=True):
            box = clip_box(box, self.wh)
            if area(box) < 4 or area(box) / frame_area > self.dcfg["max_box_area_frac"]:
                continue
            cls = normalize_label(str(label), self.dcfg["phrase_to_class"])
            if cls is None:  # phrase that maps to no class (e.g. an empty word piece): not a candidate
                continue
            boxes.append(box)
            scores.append(float(score))
            labels.append(cls)
        keep = nms(boxes, scores, self.dcfg["nms_iou"])[: self.dcfg["max_candidates_per_frame"]]
        return [{"box_xyxy": [round(v, 1) for v in boxes[i]], "dino_label": labels[i],
                 "dino_score": round(scores[i], 4)} for i in keep]

    def detect_frames(self, cfg: dict, keys: list[tuple[str, int]], log=None) -> list[dict]:
        """One cache row per frame: {clip, frame_idx, detect_ms, boxes: [...]}."""
        rows = []
        bs = self.dcfg["batch_size"]
        for s in range(0, len(keys), bs):
            chunk = keys[s:s + bs]
            images = [Image.open(image_path(cfg, frame_file(c, i))).convert("RGB") for c, i in chunk]
            if self.device.startswith("cuda"):
                torch.cuda.synchronize(self.device)
            started = time.perf_counter()
            results = self._detect_batch(images)
            if self.device.startswith("cuda"):
                torch.cuda.synchronize(self.device)
            per_frame = (time.perf_counter() - started) * 1000 / len(chunk)
            for (clip, idx), result in zip(chunk, results, strict=True):
                rows.append({"clip": clip, "frame_idx": idx, "detect_ms": round(per_frame, 2),
                             "boxes": self._postprocess(result)})
            if log and (s // bs) % 100 == 0:
                log(f"  DINO {s + len(chunk)}/{len(keys)} frames")
        return rows


class DetectionCache:
    """Per-frame detector outputs in one JSONL file, keyed by everything that changes them."""

    def __init__(self, cfg: dict, repo: Path):
        key = config_hash({"detector": {k: v for k, v in cfg["detector"].items() if k not in ("device", "batch_size")},
                           "data": cfg["data"]})
        self.path = repo / cfg["run"]["cache_root"] / "dino" / f"{key}.jsonl"
        self.rows = {(r["clip"], r["frame_idx"]): r for r in read_jsonl(self.path)} if self.path.exists() else {}

    def missing(self, keys) -> list[tuple[str, int]]:
        return [k for k in keys if k not in self.rows]

    def add(self, rows: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            import json

            for r in rows:
                f.write(json.dumps(r) + "\n")
                self.rows[(r["clip"], r["frame_idx"])] = r

    def candidates(self, keys) -> list[dict]:
        """Flatten cached frames into candidate rows (ordered by clip, frame, score desc)."""
        out = []
        for clip, idx in keys:
            for k, b in enumerate(self.rows[(clip, idx)]["boxes"]):
                out.append({"cand_id": cand_id(clip, idx, k), "clip": clip, "frame_idx": idx, **b})
        return out

    def detect_ms(self, keys) -> dict[tuple, float]:
        return {k: self.rows[k]["detect_ms"] for k in keys}
