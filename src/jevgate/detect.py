"""S2: Grounding DINO candidate boxes, shared by every system (PLAN.md §2.2)."""

import torch
from PIL import Image

from .boxes import area, clip_box, nms
from .data import image_path


def normalize_label(raw: str, classes: list[str]) -> str:
    """Map a DINO phrase (may contain word-piece fragments like '##fish') to one prompt class."""
    text = raw.replace("##", "").strip().lower()
    for cls in sorted(classes, key=len, reverse=True):
        if cls in text:
            return cls
    for cls in classes:
        if text and text in cls:
            return cls
    return "object"


class GroundingDinoDetector:
    """Zero-shot detector with a low box threshold so that both real and false candidates appear."""

    def __init__(self, cfg: dict):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.dcfg = cfg["detector"]
        self.wh = cfg["data"]["frame_wh"]
        self.classes = [c.strip() for c in self.dcfg["prompt"].split(".") if c.strip()]
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
            boxes.append(box)
            scores.append(float(score))
            labels.append(normalize_label(str(label), self.classes))
        keep = nms(boxes, scores, self.dcfg["nms_iou"])[: self.dcfg["max_candidates_per_frame"]]
        return [{"box_xyxy": [round(v, 1) for v in boxes[i]], "label": labels[i], "dino_score": round(scores[i], 4)}
                for i in keep]

    def detect(self, cfg: dict, frames: list[dict]) -> list[dict]:
        """Candidates for all frames, ordered by (clip, frame_idx, score desc), with stable cand_ids."""
        rows = []
        bs = self.dcfg["batch_size"]
        for s in range(0, len(frames), bs):
            chunk = frames[s:s + bs]
            images = [Image.open(image_path(cfg, f["file"])).convert("RGB") for f in chunk]
            for f, result in zip(chunk, self._detect_batch(images), strict=True):
                for k, cand in enumerate(self._postprocess(result)):
                    rows.append({"cand_id": f"{f['clip'][-25:]}-{f['frame_idx']:04d}-{k:02d}",
                                 "clip": f["clip"], "frame_idx": f["frame_idx"], **cand})
        return rows
