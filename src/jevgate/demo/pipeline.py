"""Demo run: meta -> VLM states -> Jev decisions (multi + single) -> actions -> summary -> dashboard video.

runs/<YYMMDD>_retail-demo/
├── README.md, run.yaml
├── data/                 links: clips, meta.jsonl, LICENSE, NOTICE.txt
├── vlm/input/            prompts.jsonl, frames/<sample>.jpg (the frames the VLM saw, events circled)
├── vlm/output/           states.jsonl
├── jev/input/            requests.jsonl (one request per clip, every question)
├── jev/output/           responses.jsonl, decisions.jsonl (cards), single.jsonl (one request per question)
├── actions/actions.jsonl
└── evaluation/           summary.json, summary.md, dashboard.mp4, stills/<n>_<sample>.png
"""

import datetime as dt
import json
import random
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv
from PIL import Image

from jevgate.core.runio import git_info, link, next_run_dir, read_jsonl, write_jsonl
from jevgate.demo.clip import read_frames
from jevgate.demo.decide import Decider, action
from jevgate.judges.base import Budget


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def contact_sheet(imgs: list[Image.Image], cols: int = 4) -> Image.Image:
    w, h = imgs[0].size
    rows = (len(imgs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, rows * h))
    for i, im in enumerate(imgs):
        sheet.paste(im, ((i % cols) * w, (i // cols) * h))
    return sheet


def took_product(events: list[dict]) -> bool:
    """Annotation-derived truth: more takes than put-backs."""
    return sum(e["label"] == "take" for e in events) > sum(e["label"] == "put" for e in events)


def pick_for_video(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Round-robin over action badges so the video shows every kind of outcome."""
    groups = defaultdict(list)
    for r in rows:
        groups[r["action"]["badge"]].append(r)
    rng = random.Random(seed)
    for g in groups.values():
        rng.shuffle(g)
    out, keys = [], sorted(groups)
    while len(out) < min(n, len(rows)):
        for k in keys:
            if groups[k] and len(out) < n:
                out.append(groups[k].pop())
    return out


class DemoRun:
    def __init__(self, cfg: dict, repo: Path):
        self.cfg, self.repo = cfg, repo
        self.root = repo / cfg["data"]["root"]
        load_dotenv(repo / ".env")
        self.path = next_run_dir(repo / cfg["run"]["out_root"], cfg["run"]["name"])
        self.path.mkdir(parents=True)
        git = git_info(repo) | {"time": dt.datetime.now().isoformat(timespec="seconds")}
        (self.path / "run.yaml").write_text(yaml.safe_dump({"run_id": self.path.name, "git": git, "config": cfg},
                                                           sort_keys=False, allow_unicode=True))
        for name in (cfg["data"]["clips_dir"], cfg["data"]["meta"], *cfg["data"]["license_files"]):
            link(self.root / name, self.path / "data" / name)

    def p(self, rel: str) -> Path:
        out = self.path / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        return out

    def observe(self, metas: list[dict]) -> dict[str, dict]:
        from jevgate.demo.observe import Observer

        obs_model, states, prompts = Observer(self.cfg), {}, []
        for k, m in enumerate(metas, 1):
            frames = read_frames(self.root / self.cfg["data"]["clips_dir"] / f"{m['sample_id']}.mp4", m["rotation_cw"])
            o = obs_model.observe(frames, m["events"])
            contact_sheet(o.pop("images")).save(self.p(f"vlm/input/frames/{m['sample_id']}.jpg"), quality=88)
            prompts.append({"sample_id": m["sample_id"], "prompt": o["prompt"], "frame_indices": o["frame_indices"]})
            states[m["sample_id"]] = {"sample_id": m["sample_id"], **{k2: o[k2] for k2 in ("state", "raw", "vlm_s")}}
            if k % 20 == 0:
                log(f"  VLM {k}/{len(metas)}")
        write_jsonl(self.p("vlm/input/prompts.jsonl"), prompts)
        write_jsonl(self.p("vlm/output/states.jsonl"), states.values())
        del obs_model
        import gc

        import torch

        gc.collect()
        torch.cuda.empty_cache()
        return states

    def decide(self, metas: list[dict], states: dict[str, dict]) -> dict[str, dict]:
        decider, cfg = Decider(self.cfg, Budget(self.cfg)), self.cfg

        def one(m):
            st = states[m["sample_id"]]["state"]
            dec = decider.multi(st)
            single = decider.single(st) if cfg["jev"]["single_question_compare"] else None
            return m["sample_id"], dec, single

        out = {}
        with ThreadPoolExecutor(cfg["jev"]["max_concurrency"]) as pool:
            for sid, dec, single in pool.map(one, metas):
                out[sid] = {"dec": dec, "single": single, "action": action(cfg, dec["cards"])}
        order = [m["sample_id"] for m in metas]
        write_jsonl(self.p("jev/input/requests.jsonl"), ({"sample_id": s, "request": out[s]["dec"]["request"]}
                                                           for s in order))
        write_jsonl(self.p("jev/output/responses.jsonl"),
                    ({"sample_id": s, "response": out[s]["dec"]["response"], "latency_ms": out[s]["dec"]["latency_ms"]}
                     for s in order))
        write_jsonl(self.p("jev/output/decisions.jsonl"), ({"sample_id": s, "cards": out[s]["dec"]["cards"]}
                                                             for s in order))
        if cfg["jev"]["single_question_compare"]:
            write_jsonl(self.p("jev/output/single.jsonl"), ({"sample_id": s, **out[s]["single"]} for s in order))
        write_jsonl(self.p("actions/actions.jsonl"), ({"sample_id": s, **out[s]["action"]} for s in order))
        return out

    def summary(self, metas, states, decs) -> dict:
        ids = [m["sample_id"] for m in metas]
        multi = [decs[s]["dec"]["latency_ms"] for s in ids]
        single = [decs[s]["single"]["wall_ms"] for s in ids if decs[s]["single"]]
        tok_m = [sum(decs[s]["dec"]["usage"].get(k, 0) for k in ("input_tokens", "output_tokens")) for s in ids]
        tok_s = [decs[s]["single"]["tokens"] for s in ids if decs[s]["single"]]
        sure = defaultdict(list)
        for s in ids:
            for c in decs[s]["dec"]["cards"]:
                sure[c["title"]].append(c["sure"])
        cards = {s: {c["key"]: c for c in decs[s]["dec"]["cards"]} for s in ids}
        meta = {m["sample_id"]: m for m in metas}
        agree = [(cards[s]["product_taken"]["p"] >= 0.5) == took_product(meta[s]["events"]) for s in ids
                 if "product_taken" in cards[s]]
        n_q = len(self.cfg["questions"])
        return {
            "clips": len(ids), "questions_per_clip": n_q,
            "vlm_s_median": float(np.median([states[s]["vlm_s"] for s in ids])),
            "jev_multi_ms_median": float(np.median(multi)), "jev_multi_ms_p90": float(np.percentile(multi, 90)),
            "jev_single_wall_ms_median": float(np.median(single)) if single else None,
            "requests_per_clip": {"multi": 1, "single": n_q},
            "tokens_per_clip_median": {"multi": float(np.median(tok_m)),
                                       "single": float(np.median(tok_s)) if tok_s else None},
            "sure_rate_per_question": {k: round(float(np.mean(v)), 3) for k, v in sure.items()},
            "actions": dict(Counter(decs[s]["action"]["badge"] for s in ids).most_common()),
            "product_taken_agrees_with_annotation": round(float(np.mean(agree)), 3) if agree else None,
            "rotated_clips": sum(m["rotation_cw"] != 0 for m in metas),
            "clips_without_pose_axis": sum(m["pose_axis"] is None for m in metas),
            "unparsed_states": sum("raw" in states[s]["state"] for s in ids),
            "jev_models": sorted({decs[s]["dec"]["model"] for s in ids}),
        }

    def render(self, metas, states, decs) -> Path:
        from jevgate.demo.render import Dashboard, render_clip
        from jevgate.outputs.viz import h264_writer

        rows = [m | {"action": decs[m["sample_id"]]["action"]} for m in metas]
        chosen = pick_for_video(rows, self.cfg["render"]["clips_in_video"], self.cfg["render"]["seed"])
        dash = Dashboard(self.cfg)
        path = self.p("evaluation/dashboard.mp4")
        writer = h264_writer(path, dash.W, dash.H, self.cfg["render"]["fps"])
        for k, m in enumerate(chosen, 1):
            sid = m["sample_id"]
            frames = read_frames(self.root / self.cfg["data"]["clips_dir"] / f"{sid}.mp4", m["rotation_cw"])
            render_clip(dash, writer, frames, m, k, len(chosen), states[sid], decs[sid]["dec"], decs[sid]["single"],
                        decs[sid]["action"])
            last = dash.frame(frames[-1], m, len(frames), k, len(chosen), states[sid], decs[sid]["dec"],
                              decs[sid]["single"], decs[sid]["action"], 1.0)
            Image.fromarray(last).save(self.p(f"evaluation/stills/{k:02d}_{sid}.png"))
        writer.close()
        return path

    def run(self) -> Path:
        metas = read_jsonl(self.root / self.cfg["data"]["meta"])
        if self.cfg["run"].get("max_clips"):
            metas = metas[: self.cfg["run"]["max_clips"]]
        log(f"run folder {self.path}: {len(metas)} clips")
        states = self.observe(metas)
        log(f"VLM done: median {np.median([s['vlm_s'] for s in states.values()]):.2f} s per clip")
        decs = self.decide(metas, states)
        log("Jev done")
        summ = self.summary(metas, states, decs)
        (self.p("evaluation/summary.json")).write_text(json.dumps(summ, indent=2))
        log(f"summary: {summ}")
        video = self.render(metas, states, decs)
        log(f"video: {video}")
        return self.path
