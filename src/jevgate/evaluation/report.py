"""Recompute a Retail report from stored responses; never calls a model or fits on test data."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from jevgate.core.runio import read_jsonl
from jevgate.evaluation.metrics import binary_metrics, took_product
from jevgate.pipeline import RetailRun, load_results


def evaluate_run(run_dir: Path, output: Path, bins: int = 10) -> dict:
    cfg = yaml.safe_load((run_dir / "run.yaml").read_text())["config"]
    # Check the complete join rather than silently evaluating the intersection.
    sources = ["vlm/output/states.jsonl", "jev/output/decisions.jsonl", "jev/output/responses.jsonl"]
    if cfg["vlm_only"]["enabled"]:
        sources.append("vlm_only/output/answers.jsonl")
    if cfg["jev"]["single_question_compare"]:
        sources.append("jev/output/single.jsonl")
    ids = None
    for source in sources:
        rows = read_jsonl(run_dir / source)
        current = {r["sample_id"] for r in rows}
        if len(current) != len(rows) or not rows:
            raise ValueError(f"empty or duplicate sample IDs: {source}")
        if ids is not None and current != ids:
            raise ValueError(f"sample ID mismatch: {source}")
        ids = current
    meta_path = run_dir / "data" / cfg["data"]["meta"]
    metadata = read_jsonl(meta_path)
    meta_ids = [r["sample_id"] for r in metadata]
    if len(set(meta_ids)) != len(meta_ids) or not ids <= set(meta_ids):
        raise ValueError("missing or duplicate metadata")
    metas, states, decs = load_results(run_dir, cfg)
    records = []
    for m in metas:
        sid = m["sample_id"]
        cards = {c["key"]: c for c in decs[sid]["dec"]["cards"]}
        c = cards["product_taken"]
        vo = {c["key"]: c for c in decs[sid].get("vlm_only", {}).get("cards", [])}.get("product_taken")
        records.append(
            {
                "sample_id": sid,
                "target": int(took_product(m["events"])),
                "p_yes": c["p"],
                "prediction": int(c["p"] >= 0.5),
                "event_count": len(m["events"]),
                "vlm_prediction": int(vo["p"]) if vo and vo["valid"] else None,
            }
        )
    y, p = [r["target"] for r in records], [r["p_yes"] for r in records]
    metrics = binary_metrics(y, p, bins)
    shim = RetailRun.__new__(RetailRun)
    shim.cfg = cfg
    summary = shim.summary(metas, states, decs)
    summary["end_to_end_s_median"] = float(
        np.median([states[m["sample_id"]]["vlm_s"] + decs[m["sample_id"]]["dec"]["latency_ms"] / 1000 for m in metas])
    )
    summary["tokens_total_multi"] = sum(
        sum(decs[m["sample_id"]]["dec"]["usage"].get(k, 0) for k in ["input_tokens", "output_tokens"]) for m in metas
    )
    summary["tokens_total_single"] = sum(
        decs[m["sample_id"]]["single"]["tokens"] for m in metas if decs[m["sample_id"]]["single"]
    )
    valid = [r for r in records if r["vlm_prediction"] is not None]
    baseline = {
        "vlm_valid": len(valid),
        "vlm_invalid_or_missing": len(records) - len(valid),
        "vlm_accuracy": float(np.mean([r["vlm_prediction"] == r["target"] for r in valid])) if valid else None,
        "always_no_accuracy": float(1 - np.mean(y)),
        "always_yes_accuracy": float(np.mean(y)),
    }
    subsets = {}
    for name, selected in [
        ("with_events", [r for r in records if r["event_count"]]),
        ("without_events", [r for r in records if not r["event_count"]]),
    ]:
        subsets[name] = (
            binary_metrics([r["target"] for r in selected], [r["p_yes"] for r in selected], bins) if selected else None
        )
    source_paths = [run_dir / source for source in sources] + [meta_path, run_dir / "run.yaml"]
    result = {
        "run": run_dir.name,
        "target": "count(take) > count(put)",
        "evaluation_kind": "annotation-conditioned proxy agreement; not independent ground truth",
        "events_in_model_input": cfg["events"]["enabled"],
        "bins": bins,
        "summary": summary,
        "product_taken": metrics,
        "baselines": baseline,
        "subsets": subsets,
        "source_sha256": {
            str(p.relative_to(run_dir)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    with (output / "predictions.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    write_report(result, output)
    plot_reliability(metrics, output / "calibration.png")
    return result


def write_report(result: dict, output: Path) -> None:
    m, s, b = result["product_taken"], result["summary"], result["baselines"]
    vlm_accuracy = f"{b['vlm_accuracy']:.4f}" if b["vlm_accuracy"] is not None else "N/A"
    lines = [
        f"# Retail 결과 및 calibration — {result['run']}",
        "",
        f"대상: {m['n']}개 클립, 질문 {s['questions_per_clip']}개. "
        "저장된 응답을 오프라인으로 재계산했으며 모델 재호출·확률 보정 학습은 하지 않았다.",
        "",
        "## 평가 범위",
        "",
        "**아래 수치는 독립적인 정답 평가가 아니라 행동 주석과의 일치도 및 proxy calibration이다.**",
        "`product_taken`의 대리 정답은 `count(take) > count(put)`이다. 주석이 없으면 음성으로 취급한다. "
        "상품 식별·최종 소지·구매를 확인하는 정답은 아니다. take/put/touch 주석은 VLM 입력에도 제공되었다. "
        "나머지 5개 질문과 직원 행동에는 독립적인 정답이 없어 정확도·ECE를 산출하지 않았다.",
        "",
        f"대리 양성 {m['positives']} / 음성 {m['n'] - m['positives']}. 예측 기준 P(yes) ≥ 0.5.",
        "",
        "| 지표 | Jev product_taken |",
        "|---|---:|",
    ]
    for name in ["accuracy", "precision", "recall", "f1", "brier", "nll", "ece", "mce", "confidence_ece"]:
        value = m[name]
        lines.append(f"| {name} | {value:.4f} |" if value is not None else f"| {name} | N/A |")
    lines += [
        "",
        f"혼동행렬: TP={m['tp']}, FP={m['fp']}, FN={m['fn']}, TN={m['tn']}.",
        "",
        f"VLM-only 일치율: {vlm_accuracy} ({b['vlm_valid']}개 유효 응답). "
        f"항상 NO: {b['always_no_accuracy']:.4f}, 항상 YES: {b['always_yes_accuracy']:.4f}.",
        "VLM-only는 확률을 출력하지 않으므로 Brier/NLL/ECE를 계산하지 않았다. 답을 0/1 확률로 간주하지 않는다.",
        "",
        f"ECE = Σ (bin 표본 수 / N) × |평균 P(yes) − 양성 비율|. [0,1]을 동일 폭 {result['bins']}개 bin으로 나누며, "
        "구간은 [하한, 상한), 마지막 구간만 1을 포함한다. MCE는 비어 있지 않은 bin의 최대 오차. "
        "Confidence ECE는 max(p,1−p)와 예측의 정답 여부를 비교한다. Brier는 평균 (p−y)², "
        "NLL은 자연로그 이진 log loss이며 로그 계산만 [1e−15,1−1e−15]로 clip한다. 낮을수록 좋다.",
        "",
        "## Reliability bins",
        "",
        "| P(yes) 구간 | n | 평균 확률 | 양성 비율 |",
        "|---|---:|---:|---:|",
    ]
    for r in m["reliability"]:
        lines.append(
            f"| {r['lower']:.1f}–{r['upper']:.1f} | {r['n']} | "
            + (f"{r['mean_probability']:.4f} | {r['observed_rate']:.4f} |" if r["n"] else "— | — |")
        )
    lines += [
        "",
        "![Reliability and selective risk](calibration.png)",
        "",
        "## 확신 임계값별 coverage / risk",
        "",
        "product_taken 한 질문에만 적용한 참고 분석이다. "
        "전체 질문을 사용하는 MANAGER CHECK 정책과 다르며 임계값을 이 데이터에 맞춰 선택하지 않았다.",
        "",
        "| 확신 기준 | 자동 판정 수 | Coverage | 대리 정답 오류율 |",
        "|---|---:|---:|---:|",
    ]
    for r in m["selective"]:
        risk = f"{r['risk']:.4f}" if r["risk"] is not None else "N/A"
        lines.append(f"| {r['threshold']:.2f} | {r['n']} | {r['coverage']:.4f} | {risk} |")
    lines += ["", "## 주석 유무별 결과", "", "| 그룹 | n | Accuracy | Brier | ECE |", "|---|---:|---:|---:|---:|"]
    for name, r in result["subsets"].items():
        if r:
            lines.append(f"| {name} | {r['n']} | {r['accuracy']:.4f} | {r['brier']:.4f} | {r['ece']:.4f} |")
    lines += [
        "",
        "## 속도·토큰",
        "",
        "| 항목 | 중앙값 |",
        "|---|---:|",
        f"| VLM-only | {s['vlm_only']['vlm_s_median']:.3f} s |" if s["vlm_only"] else "| VLM-only | N/A |",
        f"| VLM 관찰 | {s['vlm_s_median']:.3f} s |",
        f"| Jev multi | {s['jev_multi_ms_median']:.1f} ms |",
        f"| VLM 관찰 + Jev (클립별 합의 중앙값) | {s['end_to_end_s_median']:.3f} s |",
        f"| Jev single 병렬 요청 | {s['jev_single_wall_ms_median']} ms |",
        f"| Jev multi 토큰/클립 | {s['tokens_per_clip_median']['multi']} |",
        f"| Jev single 토큰/클립 | {s['tokens_per_clip_median']['single']} |",
        "",
        f"총 토큰: multi {s['tokens_total_multi']:,}, single {s['tokens_total_single']:,}. "
        "API 단가 정보가 없어 금액으로 환산하지 않았다. 저장된 처리 시간으로 모델 로딩·전처리·렌더링은 제외한다. "
        "VLM state 재사용 시 관찰 시간은 최초 실행의 기록이다. "
        "Jev 시간은 성공한 요청의 기록이며 재시도 전체 시간이 아니다.",
        "",
        "## 질문별 확신 비율 (정확도 아님)",
        "",
        "| 질문 | 확신 비율 |",
        "|---|---:|",
    ]
    lines += [f"| {k} | {v:.3f} |" for k, v in s["sure_rate_per_question"].items()]
    lines += ["", "## 직원 행동 분포", "", "| 행동 | 클립 수 |", "|---|---:|"]
    lines += [f"| {k} | {v} |" for k, v in s["actions"].items()]
    lines += [
        "",
        "## 해석 및 재현",
        "",
        "확신이 높다는 사실만으로 확률이 보정되었다고 볼 수 없다. 이 결과는 선택된 test 클립의 기술 통계이며 "
        "전체 데이터셋·새 매장으로의 일반화나 직원 행동의 유효성을 입증하지 않는다. 주석 조건화, 대리 정답의 한계, "
        "작은 bin 표본 수 및 클립 간 종속 가능성이 있다. "
        "독립적으로 라벨링한 평가셋과 별도의 calibration split이 필요하다.",
        "",
        f"`uv run python scripts/evaluate_retail.py --run-dir runs/{result['run']} --output docs/retail-results`",
        "",
        "`metrics.json`: 전체 지표·원본 SHA-256, `predictions.csv`: 클립별 대조값, "
        "`calibration.png`: reliability/risk 그림.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines))


def plot_reliability(metrics: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    rows = [r for r in metrics["reliability"] if r["n"]]
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].scatter(
        [r["mean_probability"] for r in rows],
        [r["observed_rate"] for r in rows],
        s=[r["n"] * 5 for r in rows],
        color="#336699",
    )
    for r in rows:
        axes[0].annotate(
            str(r["n"]), (r["mean_probability"], r["observed_rate"]), xytext=(5, 5), textcoords="offset points"
        )
    axes[0].set(
        xlim=(-0.05, 1.05),
        ylim=(-0.05, 1.05),
        xlabel="Mean P(yes)",
        ylabel="Proxy positive rate",
        title="Reliability (labels: bin n)",
    )
    rows = [r for r in metrics["selective"] if r["n"]]
    axes[1].plot([r["coverage"] for r in rows], [r["risk"] for r in rows], "o-", color="#336699")
    axes[1].set(
        xlim=(0, 1.05), ylim=(0, 1), xlabel="Coverage", ylabel="Proxy error rate", title="Product-taken selective risk"
    )
    fig.suptitle("Retail: annotation-conditioned proxy evaluation")
    fig.savefig(path, dpi=160)
    plt.close(fig)
