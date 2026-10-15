"""evaluation/tables/metrics.md, errors.csv and the run README.md (PLAN.md v6 §6-7)."""

import csv
from pathlib import Path

import numpy as np

from jevgate.outputs.plots import TITLES


def _f(v, digits=3):
    return "–" if v is None else f"{v:.{digits}f}" if isinstance(v, float) else str(v)


def macro(per_class: dict, key: str) -> float | None:
    vals = [v[key] for v in per_class.values() if v["support"] > 0 and v[key] is not None]
    return float(np.mean(vals)) if vals else None


def results_table(m: dict) -> list[str]:
    """Columns: accuracy, macro F1, macro P, macro R, ECE, VLM calls, latency, total time, Jev tokens/cost."""
    lines = ["| system | accuracy | macro F1 | macro P | macro R | detection recall | ECE (track) | VLM calls | "
             "judge ms/track (median) | total time s | Jev tokens | cost |", "|" + "---|" * 12]
    ece = m["track_level"].get("jev_ece")
    for s, r in m["systems"].items():
        lines.append(f"| {TITLES.get(s, s)} | **{_f(r['accuracy'])}** | {_f(r['macro_f1'])} | "
                     f"{_f(macro(r['per_class'], 'precision'))} | {_f(macro(r['per_class'], 'recall'))} | "
                     f"{_f(r['detection_recall'])} | {_f(ece) if s == 'jev' else '–'} | {r['vlm_calls']} | "
                     f"{_f(r['latency_ms_median'], 1)} | {_f(r['total_time_s'], 1)} | {r['jev_tokens']} | "
                     f"{_f(r['jev_cost_usd'], 2)} |")
    return lines


def metrics_md(m: dict) -> str:
    n, p, tl = m["n"], m["primary"], m["track_level"]
    lines = [f"# Results ({m['split']})", "",
             f"videos {n['videos']} · frames {n['frames']} · detections {n['detections']} "
             f"(scored {n['boxes_scored']}, ambiguous {n['boxes_ambiguous']}, unknown GT {n['boxes_unknown_gt']}) · "
             f"tracks {n['tracks']} (scored {n['tracks_scored']}: {n['track_types']}) · "
             f"untracked boxes {n['untracked_boxes']} (real animals among them {n['untracked_real_animals']})", "",
             f"Thresholds (dev): t = {_f(m['thresholds'].get('t'))}, v = {_f(m['thresholds'].get('v'), 5)}, "
             f"tau = {_f(m['thresholds'].get('tau'), 2)}", "", "## Box level (7-way: six animals + background)", ""]
    lines += results_table(m)
    mo = m["missed_objects"]
    lines += ["", f"Detector recall upper bound: {_f(mo['recall_upper_bound'])} "
              f"({mo['n_covered']}/{mo['n_gt_objects']} GT objects have a detection with IoU ≥ 0.5); "
              "no judge can recover the rest.", ""]
    if p.get("delta_acc"):
        d = p["delta_acc"]
        lines += ["## Primary endpoint", "",
                  f"Δacc ({p['pair'][0]} − {p['pair'][1]}) = {d['delta']:+.4f} "
                  f"[95% CI {_f(d['ci_low'], 4)}, {_f(d['ci_high'], 4)}], {d['n_clusters']} clusters · "
                  f"VLM call ratio {_f(p['call_ratio'])}", ""]
        lines += [f"- {e['name']}: CI low ≥ {e['min_delta_acc_ci_low']}, ratio ≤ {e['max_vlm_call_ratio']} → "
                  f"**{e['result']}**" for e in m["endpoints"]]
        g = m["systems"].get("gate", {})
        lines += ["", f"Gate: Jev handled {_f(g.get('auto_rate'))} of tracks, "
                  f"error on them {_f(g.get('auto_error'))}", ""]
    lines += ["## Track level", ""]
    for k in ("auroc_jev_p_ok", "auroc_vlm_p_yes", "auroc_mean_conf"):
        if k in tl:
            a = tl[k]
            lines.append(f"- {k[6:]}: AUROC {_f(a['auc'])} [{_f(a['ci_low'])}, {_f(a['ci_high'])}]")
    lines += [f"- Jev ECE {_f(tl.get('jev_ece'))}, Brier {_f(tl.get('jev_brier'))}",
              f"- class_fix (exploratory): {tl.get('class_fix', {})}", "",
              "## Flicker (frame-to-frame change in shown boxes, mean over videos)", "",
              " · ".join(f"{s} {_f(r['flicker'])}" for s, r in m["systems"].items()), "",
              f"Sensitivity, ambiguous as background: {m['sensitivity']['ambiguous_as_background']}", "",
              f"Timing (s): {m['timing']} · budget {m['budget_used']} · Jev model {m['jev_models']}"]
    return "\n".join(lines) + "\n"


def write_errors(path: Path, tracks: list[dict], track_labels: dict, jev: dict, vlm: dict, gate_rows: list[dict]):
    """Tracks the primary gate got wrong, with every judge's score and the VLM images."""
    gate = {r["track_uid"]: r for r in gate_rows}
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["track_uid", "video", "first_frame", "last_frame", "majority_class", "type", "y_votes",
                    "jev_p_ok", "vlm_p_yes", "routed_to", "gate_accept", "vlm_frame_image"])
        for t in tracks:
            lab, g = track_labels[t["track_uid"]], gate.get(t["track_uid"])
            if lab["y_track"] is None or g is None or g["accept"] == bool(lab["y_track"]):
                continue
            uid = t["track_uid"]
            w.writerow([uid, t["video"], t["first_frame"], t["last_frame"], t["majority_class"], lab["type"],
                        lab["y_votes"], jev[uid]["p_ok"], vlm[uid]["p_yes"], g["routed_to"], g["accept"],
                        f"vlm/input/images/{uid.replace('#', '_')}_frame.jpg"])


def readme(run_id: str, m: dict) -> str:
    files = ["", "## Files", "",
             "| path | content |", "|---|---|",
             "| `run.yaml` | resolved config, git, thresholds, stage times, usage |",
             "| `data/` | links to the images, AAU annotations and the video split |",
             "| `yolo/input/` | `weights.pt` (link), `videos.txt` · `yolo/output/` detections and track summaries |",
             "| `jev/input/requests.jsonl` | exact request bodies · `jev/output/` raw responses and per-track p_ok |",
             "| `vlm/input/` | prompts and the two images per track · `vlm/output/` per-track p_yes |",
             "| `evaluation/labels/` | y per box and per track |",
             "| `evaluation/tables/` | metrics.json, this table, errors.csv, thresholds.yaml / endpoints.json |",
             "| `evaluation/confusion/`, `plots/` | confusion matrices, ROC, reliability, tau curve |",
             "| `evaluation/viz/`, `videos/` | sample frames (①②③⑤), per video ① vs ⑤ |",
             "", "Box colours: green = right class, yellow = wrong animal, red = kept without an animal, "
             "orange = animal dropped, white thin = GT."]
    return f"# {run_id}\n\n" + metrics_md(m).split("\n", 1)[1] + "\n".join(files) + "\n"
