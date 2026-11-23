"""Demo run: meta -> VLM states -> Jev decisions (multi + single) -> actions -> summary -> dashboard video.

runs/<YYMMDD>_retail-demo/
├── run.yaml
├── data/                 links: clips, meta.jsonl, LICENSE, NOTICE.txt
├── vlm/input/            prompts.jsonl, frames/<sample>.jpg (the frames the VLM saw, events circled)
├── vlm/output/           states.jsonl (or reused from reuse_states_from)
├── vlm_only/             input/prompts.jsonl, output/answers.jsonl (the VLM answers the questions itself)
├── jev/input/            requests.jsonl (one request per clip, every question)
├── jev/output/           responses.jsonl, decisions.jsonl (cards), single.jsonl (one request per question)
├── actions/actions.jsonl
└── evaluation/           summary.json, calibration/, dashboard.mp4, stills/<n>_<sample>.png
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

from jev.core.runio import git_info, link, next_run_dir, read_jsonl, write_jsonl
from jev.decision.policy import Decider, action, vlm_cards
from jev.evaluation.metrics import took_product
from jev.models.budget import Budget
from jev.preprocess.frames import read_frames


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def contact_sheet(imgs: list[Image.Image], cols: int = 4) -> Image.Image:
    w, h = imgs[0].size
    rows = (len(imgs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, rows * h))
    for i, im in enumerate(imgs):
        sheet.paste(im, ((i % cols) * w, (i // cols) * h))
    return sheet


def pick_for_video(
    rows: list[dict], n: int, seed: int, include: list[str] = (), exclude: list[str] = (), fill_others: bool = True
) -> list[dict]:
    """`include` clips first (in the given order), then round-robin over action badges for variety; `exclude` never."""
    by_id = {r["sample_id"]: r for r in rows}
    missing = [s for s in include if s not in by_id]
    if missing:
        raise ValueError(f"render.include clips not in this run: {missing}")
    out = [by_id[s] for s in include if s not in exclude][:n]
    if not fill_others:
        return out
    groups = defaultdict(list)
    for r in rows:
        if r["sample_id"] not in exclude and r["sample_id"] not in include:
            groups[r["action"]["badge"]].append(r)
    rng = random.Random(seed)
    for g in groups.values():
        rng.shuffle(g)
    keys = sorted(groups)
    while len(out) < n and any(groups.values()):
        for k in keys:
            if groups[k] and len(out) < n:
                out.append(groups[k].pop())
    return out


class RetailRun:
    def __init__(self, cfg: dict, repo: Path):
        self.cfg, self.repo = cfg, repo
        self.root = repo / cfg["data"]["root"]
        load_dotenv(repo / ".env")
        self.path = next_run_dir(repo / cfg["run"]["out_root"], cfg["run"]["name"])
        self.path.mkdir(parents=True)
        git = git_info(repo) | {"time": dt.datetime.now().isoformat(timespec="seconds")}
        (self.path / "run.yaml").write_text(
            yaml.safe_dump({"run_id": self.path.name, "git": git, "config": cfg}, sort_keys=False, allow_unicode=True)
        )
        for name in (cfg["data"]["clips_dir"], cfg["data"]["meta"], *cfg["data"]["license_files"]):
            link(self.root / name, self.path / "data" / name)
        self.answers: dict[str, dict] = {}

    def p(self, rel: str) -> Path:
        out = self.path / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        return out

    def observe(self, metas: list[dict]) -> tuple[dict[str, dict], dict[str, dict]]:
        """VLM states (generated, or reused from an earlier run) and the VLM-only answers."""
        reuse = self.cfg.get("reuse_states_from")
        states = {r["sample_id"]: r for r in read_jsonl(self.repo / reuse / "vlm/output/states.jsonl")} if reuse else {}
        if reuse:
            log(f"  VLM states reused from {reuse}")
        need_answers = self.cfg["vlm_only"]["enabled"]
        answers, prompts, answer_prompts = {}, [], []
        if not reuse or need_answers:
            from jev.models.observer import Observer

            vlm = Observer(self.cfg)
            for k, m in enumerate(metas, 1):
                sid = m["sample_id"]
                frames = read_frames(self.root / self.cfg["data"]["clips_dir"] / f"{sid}.mp4", m["rotation_cw"])
                if not reuse:
                    o = vlm.observe(frames, m["events"])
                    contact_sheet(o.pop("images")).save(self.p(f"vlm/input/frames/{sid}.jpg"), quality=88)
                    prompts.append({"sample_id": sid, "prompt": o["prompt"], "frame_indices": o["frame_indices"]})
                    states[sid] = {"sample_id": sid, **{k2: o[k2] for k2 in ("state", "raw", "vlm_s")}}
                if need_answers:
                    a = vlm.answer(frames, m["events"])
                    answer_prompts.append({"sample_id": sid, "prompt": a.pop("prompt")})
                    answers[sid] = {"sample_id": sid, **a}
                if k % 20 == 0:
                    log(f"  VLM {k}/{len(metas)}")
            del vlm
            import gc

            import torch

            gc.collect()
            torch.cuda.empty_cache()
        if prompts:
            write_jsonl(self.p("vlm/input/prompts.jsonl"), prompts)
        write_jsonl(self.p("vlm/output/states.jsonl"), (states[m["sample_id"]] for m in metas))
        if answers:
            write_jsonl(self.p("vlm_only/input/prompts.jsonl"), answer_prompts)
            write_jsonl(self.p("vlm_only/output/answers.jsonl"), (answers[m["sample_id"]] for m in metas))
        return states, answers

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
        for m in metas:
            a = self.answers.get(m["sample_id"])
            if a:
                cards = vlm_cards(cfg, a["answers"])
                out[m["sample_id"]]["vlm_only"] = {"cards": cards, "action": action(cfg, cards), "vlm_s": a["vlm_s"]}
        order = [m["sample_id"] for m in metas]
        write_jsonl(
            self.p("jev/input/requests.jsonl"), ({"sample_id": s, "request": out[s]["dec"]["request"]} for s in order)
        )
        write_jsonl(
            self.p("jev/output/responses.jsonl"),
            (
                {"sample_id": s, "response": out[s]["dec"]["response"], "latency_ms": out[s]["dec"]["latency_ms"]}
                for s in order
            ),
        )
        write_jsonl(
            self.p("jev/output/decisions.jsonl"), ({"sample_id": s, "cards": out[s]["dec"]["cards"]} for s in order)
        )
        if cfg["jev"]["single_question_compare"]:
            write_jsonl(self.p("jev/output/single.jsonl"), ({"sample_id": s, **out[s]["single"]} for s in order))
        write_jsonl(
            self.p("actions/actions.jsonl"),
            (
                {"sample_id": s, "jev": out[s]["action"], "vlm_only": out[s].get("vlm_only", {}).get("action")}
                for s in order
            ),
        )
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
        agree = [
            (cards[s]["product_taken"]["p"] >= 0.5) == took_product(meta[s]["events"])
            for s in ids
            if "product_taken" in cards[s]
        ]
        n_q = len(self.cfg["questions"])
        return {
            "clips": len(ids),
            "questions_per_clip": n_q,
            "vlm_s_median": float(np.median([states[s]["vlm_s"] for s in ids])),
            "jev_multi_ms_median": float(np.median(multi)),
            "jev_multi_ms_p90": float(np.percentile(multi, 90)),
            "jev_single_wall_ms_median": float(np.median(single)) if single else None,
            "requests_per_clip": {"multi": 1, "single": n_q},
            "tokens_per_clip_median": {
                "multi": float(np.median(tok_m)),
                "single": float(np.median(tok_s)) if tok_s else None,
            },
            "sure_rate_per_question": {k: round(float(np.mean(v)), 3) for k, v in sure.items()},
            "actions": dict(Counter(decs[s]["action"]["badge"] for s in ids).most_common()),
            "product_taken_agrees_with_annotation": round(float(np.mean(agree)), 3) if agree else None,
            "rotated_clips": sum(m["rotation_cw"] != 0 for m in metas),
            "clips_without_pose_axis": sum(m["pose_axis"] is None for m in metas),
            "unparsed_states": sum("raw" in states[s]["state"] for s in ids),
            "jev_models": sorted({decs[s]["dec"]["model"] for s in ids}),
            "vlm_only": self.vlm_only_summary(ids, decs),
        }

    @staticmethod
    def vlm_only_summary(ids, decs) -> dict | None:
        rows = [decs[s]["vlm_only"] for s in ids if "vlm_only" in decs[s]]
        if not rows:
            return None
        return {
            "vlm_s_median": float(np.median([r["vlm_s"] for r in rows])),
            "invalid_answers": sum(not c["valid"] for r in rows for c in r["cards"]),
            "actions": dict(Counter(r["action"]["badge"] for r in rows).most_common()),
            "same_action_as_jev_path": round(
                float(
                    np.mean(
                        [
                            decs[s]["vlm_only"]["action"]["badge"] == decs[s]["action"]["badge"]
                            for s in ids
                            if "vlm_only" in decs[s]
                        ]
                    )
                ),
                3,
            ),
        }

    def render(self, metas, states, decs) -> Path:
        from jev.visualization.dashboard import Dashboard, render_clip
        from jev.visualization.video import h264_writer

        rows = [m | {"action": decs[m["sample_id"]]["action"]} for m in metas]
        r = self.cfg["render"]
        chosen = pick_for_video(
            rows, r["clips_in_video"], r["seed"], r.get("include", []), r.get("exclude", []), r.get("fill_others", True)
        )
        dash = Dashboard(self.cfg, self.repo)
        path = self.p("evaluation/dashboard.mp4")
        writer = h264_writer(path, dash.W, dash.H, self.cfg["render"]["video_fps"])
        for k, m in enumerate(chosen, 1):
            sid = m["sample_id"]
            frames = read_frames(self.root / self.cfg["data"]["clips_dir"] / f"{sid}.mp4", m["rotation_cw"])
            render_clip(dash, writer, frames, m, k, len(chosen), states[sid], decs[sid])
            last = dash.frame(
                frames[-1],
                m,
                len(frames),
                k,
                len(chosen),
                states[sid],
                decs[sid],
                dash.finish_s(states[sid], decs[sid]) + 1,
            )
            Image.fromarray(last).save(self.p(f"evaluation/stills/{k:02d}_{sid}.png"))
        writer.close()
        return path

    def run(self) -> Path:
        metas = read_jsonl(self.root / self.cfg["data"]["meta"])
        if self.cfg["run"].get("max_clips"):
            metas = metas[: self.cfg["run"]["max_clips"]]
        log(f"run folder {self.path}: {len(metas)} clips")
        states, self.answers = self.observe(metas)
        log(f"VLM done: median {np.median([states[m['sample_id']]['vlm_s'] for m in metas]):.2f} s per clip")
        decs = self.decide(metas, states)
        log("Jev done")
        summ = self.summary(metas, states, decs)
        (self.p("evaluation/summary.json")).write_text(json.dumps(summ, indent=2))
        log(f"summary: {summ}")
        from jev.evaluation.report import evaluate_run

        evaluate_run(self.path, self.path / "evaluation/calibration")
        video = self.render(metas, states, decs)
        log(f"video: {video}")
        return self.path


def load_results(run_dir: Path, cfg: dict) -> tuple[list[dict], dict, dict]:
    """(metas, states, decs) rebuilt from a finished run folder, without calling any model."""
    metas = read_jsonl(run_dir / "data" / cfg["data"]["meta"])
    keep = {r["sample_id"] for r in read_jsonl(run_dir / "vlm/output/states.jsonl")}
    metas = [m for m in metas if m["sample_id"] in keep]
    states = {r["sample_id"]: r for r in read_jsonl(run_dir / "vlm/output/states.jsonl")}
    cards = {r["sample_id"]: r["cards"] for r in read_jsonl(run_dir / "jev/output/decisions.jsonl")}
    resp = {r["sample_id"]: r for r in read_jsonl(run_dir / "jev/output/responses.jsonl")}
    single_path = run_dir / "jev/output/single.jsonl"
    single = {r["sample_id"]: r for r in read_jsonl(single_path)} if single_path.exists() else {}
    vo_path = run_dir / "vlm_only/output/answers.jsonl"
    answers = {r["sample_id"]: r for r in read_jsonl(vo_path)} if vo_path.exists() else {}
    decs = {}
    for m in metas:
        sid = m["sample_id"]
        response = resp[sid]["response"]
        dec = {
            "cards": cards[sid],
            "latency_ms": resp[sid]["latency_ms"],
            "usage": response.get("usage", {}),
            "model": response.get("model"),
        }
        decs[sid] = {"dec": dec, "single": single.get(sid), "action": action(cfg, cards[sid])}
        if sid in answers:
            vc = vlm_cards(cfg, answers[sid]["answers"])
            decs[sid]["vlm_only"] = {"cards": vc, "action": action(cfg, vc), "vlm_s": answers[sid]["vlm_s"]}
    return metas, states, decs


def rerender(run_dir: Path, cfg: dict, repo: Path) -> Path:
    """Redraw evaluation/dashboard.mp4 and stills/ of an existing run with the given (render) config."""
    import shutil

    metas, states, decs = load_results(run_dir, cfg)
    shutil.rmtree(run_dir / "evaluation" / "stills", ignore_errors=True)
    shim = RetailRun.__new__(RetailRun)
    shim.cfg, shim.repo, shim.path, shim.root = cfg, repo, run_dir, repo / cfg["data"]["root"]
    return shim.render(metas, states, decs)
