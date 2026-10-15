"""evaluation/tables/metrics.md, errors.csv and the run README.md (PLAN_brackish.md v7 §5-6)."""

import csv
from pathlib import Path

import numpy as np

from jevgate.outputs.plots import TITLES

THRESHOLD_KEYS = ("t", "v", "v_small", "tau", "small_lo", "small_hi")


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


def limits(m: dict) -> list[str]:
    """Plain statements of what the numbers cannot show, derived from the run itself."""
    th, tl, n = m["thresholds"], m["track_level"], m["n"]
    out = ["## Limits", ""]
    if th.get("tau", 0) > 1:
        out.append(f"- No tau met the dev target (error ≤ {th.get('max_auto_error')} on Jev-handled tracks), so tau = "
                   f"{th['tau']:.2f}: the Jev gate routes every track to the large VLM and equals ③ by construction.")
    j, s = tl.get("auroc_jev_p_ok", {}), tl.get("auroc_small_p_yes", {})
    if j.get("auc") is not None and s.get("auc") is not None:
        out.append(f"- Jev reads the small VLM's description; its track AUROC {j['auc']:.3f} vs the small VLM's own "
                   f"p_yes {s['auc']:.3f}: Jev adds no separating information beyond its input.")
    if tl.get("jev_ece") is not None:
        out.append(f"- Jev calibration on this domain: ECE {tl['jev_ece']:.3f} (Jev is not trained on these videos and "
                   "cannot be fine-tuned through the API).")
    out.append(f"- n: {n['tracks_scored']} scored tracks, of which {n['track_types'].get('false', 0)} false and "
               f"{n['track_types'].get('misclassified', 0)} misclassified; CIs are wide.")
    mo = m["missed_objects"]
    out.append(f"- Detector recall upper bound {mo['recall_upper_bound']:.3f}: no judge recovers missed animals.")
    return out + [""]


def metrics_md(m: dict) -> str:
    n, p, tl = m["n"], m["primary"], m["track_level"]
    lines = [f"# Results ({m['split']})", "",
             f"videos {n['videos']} · frames {n['frames']} · detections {n['detections']} "
             f"(scored {n['boxes_scored']}, ambiguous {n['boxes_ambiguous']}, unknown GT {n['boxes_unknown_gt']}) · "
             f"tracks {n['tracks']} (scored {n['tracks_scored']}: {n['track_types']}) · "
             f"untracked boxes {n['untracked_boxes']} (real animals among them {n['untracked_real_animals']})", "",
             "Thresholds (dev): " + ", ".join(f"{k} = {m['thresholds'][k]:.5g}" for k in THRESHOLD_KEYS
                                             if k in m["thresholds"]),
             "", "## Box level (7-way: six animals + background)", ""]
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
        for name in ("gate", "gate_small"):
            g = m["systems"].get(name)
            if g:
                lines.append(f"- {TITLES[name]}: handled alone {_f(g.get('auto_rate'))} of tracks, "
                             f"error on them {_f(g.get('auto_error'))}, large-VLM calls {g['vlm_calls']}")
        sg = m.get("gate_secondary")
        if sg and sg.get("delta_acc"):
            d = sg["delta_acc"]
            lines.append(f"- S1 small-VLM gate: Δacc vs ③ {d['delta']:+.4f} [{_f(d['ci_low'], 4)}, "
                         f"{_f(d['ci_high'], 4)}], call ratio {_f(sg['call_ratio'])}")
        lines.append("")
    lines += limits(m)
    lines += ["## Track level", ""]
    for k in ("auroc_jev_p_ok", "auroc_vlm_p_yes", "auroc_small_p_yes", "auroc_mean_conf"):
        if k in tl:
            a = tl[k]
            lines.append(f"- {k[6:]}: AUROC {_f(a['auc'])} [{_f(a['ci_low'])}, {_f(a['ci_high'])}]")
    lines += [f"- Jev ECE {_f(tl.get('jev_ece'))}, Brier {_f(tl.get('jev_brier'))}", "",
              "## Flicker (frame-to-frame change in shown boxes, mean over videos)", "",
              " · ".join(f"{s} {_f(r['flicker'])}" for s, r in m["systems"].items()), "",
              f"Sensitivity, ambiguous as background: {m['sensitivity']['ambiguous_as_background']}", "",
              f"Timing (s): {m['timing']} · budget {m['budget_used']} · Jev model {m['jev_models']}"]
    return "\n".join(lines) + "\n"


def write_errors(path: Path, tracks: list[dict], track_labels: dict, data) -> None:
    """Tracks the primary gate got wrong, with every judge's score, the description and the input images."""
    gate = {r["track_uid"]: r for r in data.gate_rows}
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["track_uid", "video", "first_frame", "last_frame", "majority_class", "type", "y_votes",
                    "jev_p_ok", "large_vlm_p_yes", "small_vlm_p_yes", "routed_to", "gate_accept", "description",
                    "image"])
        for t in tracks:
            uid = t["track_uid"]
            lab, g = track_labels[uid], gate.get(uid)
            if lab["y_track"] is None or g is None or g["accept"] == bool(lab["y_track"]):
                continue
            small = data.small.get(uid, {})
            w.writerow([uid, t["video"], t["first_frame"], t["last_frame"], t["majority_class"], lab["type"],
                        lab["y_votes"], data.jev[uid]["p_ok"], data.vlm[uid]["p_yes"], small.get("p_yes"),
                        g["routed_to"], g["accept"], small.get("description"),
                        f"describer/input/images/{uid.replace('#', '_')}_frame.jpg"])


def readme(run_id: str, m: dict) -> str:
    files = ["", "## Files", "",
             "| path | content |", "|---|---|",
             "| `run.yaml` | resolved config, git, thresholds, stage times, usage |",
             "| `data/` | links to the images, AAU annotations and the video split |",
             "| `yolo/input/` | `weights.pt` (link), `videos.txt` · `yolo/output/` detections and track summaries |",
             "| `jev/input/requests.jsonl` | exact request bodies · `jev/output/` raw responses and per-track p_ok |",
             "| `describer/input/` | prompts and the two images per track (shared with vlm) · "
             "`describer/output/` description + small-VLM p_yes |",
             "| `vlm/input/prompts.jsonl` | large-VLM prompts · `vlm/output/` per-track p_yes |",
             "| `evaluation/labels/` | y per box and per track |",
             "| `evaluation/tables/` | metrics.json, this table, errors.csv, thresholds.yaml / endpoints.json |",
             "| `evaluation/confusion/`, `plots/` | confusion matrices, ROC, reliability, tau curve |",
             "| `evaluation/viz/`, `videos/` | sample frames (①②③⑥), per video GT vs ⑥ |",
             "", "Box colour = label (viz.class_colors); solid = right, x = wrong class, dashed = no animal, "
             "dotted = missed animal."]
    return f"# {run_id}\n\n" + metrics_md(m).split("\n", 1)[1] + "\n".join(files) + "\n"
