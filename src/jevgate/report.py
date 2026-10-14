"""README.md of a run: the one file a person reads first (results, n flow, time and cost, figure guide)."""

from pathlib import Path

from .evaluate import SYSTEM_ORDER, SYSTEM_TITLE
from .runio import RunDir

FIGURE_GUIDE = {
    "fig1_tradeoff.png": "**Primary**: 시스템마다 점 하나. x = VLM으로 보낸 박스 비율, "
                         "y = 정확도(95% CI). 초록 음영 안에 ④가 들어가면 primary 성공.",
    "fig2_per_class.png": "클래스별 성능. A·B = Jev·VLM의 클래스별 ROC와 AUC [95% CI] (마커 = 실제 판정 지점), "
                          "C·D = Jev·게이트의 클래스별 2×2 (sens/spec). 회색 패널은 n < min_class_n.",
    "fig3_jev_knows.png": "왼쪽: Jev가 매긴 need_vlm 단계별 실제 Jev 오류율. 단계가 높을수록 오류율도 높으면 "
                          "Jev가 자기가 틀릴 때를 아는 것. 오른쪽: τ_e에 따른 자동 처리 비율과 그 오류율, "
                          "점선 = dev에서 고른 τ_e.",
    "fig4_calibration.png": "Jev·VLM의 reliability diagram. 막대가 대각선에 가까울수록 확률을 믿을 수 있음 "
                            "(ECE가 작음).",
}
FILE_GUIDE = [
    ("run.yaml", "설정 전체, git commit, 환경, 임계값, 단계별 시간, Jev·VLM 사용량"),
    ("frames.jsonl", "프레임 한 줄씩 (gt_status, 검출 시간)"),
    ("boxes.jsonl", "박스 한 줄씩: 위치, 정답 y, Jev가 본 증거"),
    ("judgments.jsonl", "박스 한 줄씩: 모든 시스템의 pred·confidence, need_vlm, routed_to"),
    ("metrics.json", "모든 숫자와 그림용 곡선 데이터"),
    ("errors.csv", "②·③·④ 중 하나라도 틀린 박스 목록"),
    ("cache/", "비싼 단계 원본 (vlm.jsonl, jev.jsonl, jev_responses.jsonl) — 재사용·재개용"),
    ("figures/", "아래 그림 4장"),
    ("samples/", "무작위 프레임의 비교 이미지 (GT | 시스템별 판정)"),
]


def _f(v, d=3):
    return "–" if v is None else f"{v:.{d}f}"


def _pct(v):
    return "–" if v is None else f"{v:.0%}"


def render(run: RunDir, m: dict, figures: list[Path]) -> str:
    meta = run.meta()
    cfg = meta["config"]
    classes = m["classes"]
    flow = m["n_flow"]
    cov = flow["gt_objects"]
    L = [f"# {m['run_id']} — {m['split']} ({'GT boxes' if m['candidate_source'] == 'gt' else 'DINO candidates'})", "",
         f"git `{meta['git'].get('commit')}`{' (dirty)' if meta['git'].get('dirty') else ''} · "
         f"{meta['git'].get('time')} · Jev `{', '.join(m['jev_models']) or '–'}` · VLM `{cfg['vlm']['model']}`", ""]

    L += ["## n", "",
          f"- 프레임 {m['n_frame']} · 영상 {m['n_clip']} · 후보 박스 {m['n_candidates']} "
          f"(채점 {m['n_box']}, ambiguous {flow['ambiguous']})"]
    if m["candidate_source"] == "dino":
        L.append(f"- GT 물체 {cov['gt']} 중 DINO가 후보로 잡은 것 {cov['covered']} "
                 f"(recall 상한 {_f(cov['recall_upper_bound'])}), 놓친 것 {cov['missed']}")
    first = next(iter(m["systems"].values()))
    L += ["", "| 클래스 | " + " | ".join(classes) + " |", "|---|" + "---|" * len(classes),
          "| 채점 박스 (y) | " + " | ".join(str(first["per_class"][c]["support"]) for c in classes) + " |", ""]

    L += ["## 결과", "",
          "| 시스템 | 정확도 [95% CI] | macro F1 | macro precision | macro recall | ECE | VLM 호출 | "
          "박스당 ms | 프레임당 ms | fps |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for s in SYSTEM_ORDER:
        if s not in m["systems"]:
            continue
        r = m["systems"][s]
        lo, hi = r.get("accuracy_ci", [None, None])
        L.append(f"| {SYSTEM_TITLE[s]} | {_f(r['accuracy'])} [{_f(lo)}, {_f(hi)}] | {_f(r['macro_f1'])} | "
                 f"{_f(r['macro_precision'])} | {_f(r['macro_recall'])} | {_f(r['ece'])} | "
                 f"{r['vlm_calls']} ({_pct(r['vlm_calls'] / m['n_candidates'])}) | {_f(r['latency_ms_median'], 1)} | "
                 f"{_f(r['frame_ms_mean'], 1)} | {_f(r['fps'], 2)} |")

    p = m["primary"]
    if p.get("delta_acc"):
        d = p["delta_acc"]
        L += ["", f"**Primary ({p['gate']} vs ②)**: Δacc {_f(d['delta'])} [95% CI {_f(d['ci_low'])}, "
              f"{_f(d['ci_high'])}], VLM 호출 비율 {_f(p['call_ratio'])} → "
              + ", ".join(f"{e['name']} **{e['result']}**" for e in m["endpoints"])]
        if "gate" in m["systems"]:
            g = m["systems"]["gate"]
            L.append(f"\n게이트 자동 처리 {_pct(g['auto_rate'])}, 그 오류율 {_f(g['auto_error'])} "
                     f"(dev 목표 ≤ {cfg['gate']['max_auto_error']})")

    L += ["", "### 클래스별 recall / precision", "", "| 시스템 | " + " | ".join(classes) + " |",
          "|---|" + "---|" * len(classes)]
    for s in SYSTEM_ORDER:
        if s in m["systems"]:
            pc = m["systems"][s]["per_class"]
            L.append(f"| {s} | " + " | ".join(f"{_f(pc[c]['recall'], 2)} / {_f(pc[c]['precision'], 2)}"
                                               for c in classes) + " |")

    k = m.get("jev_self_knowledge")
    if k:
        L += ["", "### Jev는 자기가 틀릴 때를 아는가", "",
              f"AUROC(need_vlm → Jev 오답) {_f(k['auroc_need_vlm_for_error'])}, "
              f"AUROC(1 − confidence → Jev 오답) {_f(k['auroc_unconfidence_for_error'])}", "",
              "| need_vlm | " + " | ".join(k["jev_error_by_need_vlm_level"]) + " |",
              "|---|" + "---|" * len(k["jev_error_by_need_vlm_level"]),
              "| 박스 수 | " + " | ".join(str(v["n"]) for v in k["jev_error_by_need_vlm_level"].values()) + " |",
              "| Jev 오류율 | " + " | ".join(_f(v["jev_error"]) for v in k["jev_error_by_need_vlm_level"].values())
              + " |"]

    th = meta.get("thresholds") or {}
    usage = meta.get("usage", {})
    L += ["", "## 시간·비용", "",
          "- 임계값: " + ", ".join(f"{k}={v}" for k, v in th.items() if k == "t" or k.startswith("tau"))
          + f" (fit_run `{th.get('fit_run')}`)",
          "- 단계별 시간(초): " + ", ".join(f"{k} {v}" for k, v in meta.get("timing_s", {}).items()),
          f"- Jev 요청 {usage.get('jev_requests')}, 토큰 {usage.get('jev_tokens')} · VLM 호출 {usage.get('vlm_calls')}"]

    L += ["", "## 그림", ""]
    for path in figures:
        rel = path.relative_to(run.path)
        if rel.parent.name == "figures":
            L += [f"### {rel.name}", FIGURE_GUIDE.get(rel.name, ""), "", f"![{rel.name}]({rel.as_posix()})", ""]
    L += ["## 파일", "", "| 파일 | 내용 |", "|---|---|"] + [f"| `{f}` | {d} |" for f, d in FILE_GUIDE]
    L += ["", "## 주의", "",
          "- 원본 분할은 프레임 단위 무작위라 모든 영상이 dev와 test 양쪽에 있다 (영상 단위로 분리된 test가 아님).",
          "- 연속 프레임은 독립이 아니므로 CI는 10프레임 블록 bootstrap으로 계산했다."]
    if m["candidate_source"] == "gt":
        L.append("- GT 박스 평가: 모든 박스가 실제 동물이라 background 선택지가 없는 6-way 분류이고, "
                 "① detector는 없다.")
    return "\n".join(L) + "\n"
