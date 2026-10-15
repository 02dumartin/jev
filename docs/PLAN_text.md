# Jev 텍스트 실험 계획 (초안 v1, 2026-10-15)

> 질문: **판단 근거가 처음부터 텍스트에 있을 때**, Jev가 한 state에 여러 질문을 한 번에 던지는 방식(Multiple questions)으로
> 빠르고 calibrated한 1차 판정을 내리고, **확실한 것은 바로 확정하고 애매한 것만 큰 LLM에 넘기면**,
> 모든 판정을 큰 LLM이 하는 것과 같은 정확도를 큰 LLM 호출을 줄이면서 얻을 수 있는가?
> 그리고 그 게이트가 **작은 LLM의 자기 확률로 만든 게이트보다 나은가?**
>
> 배경: Brackish 실험(`docs/PLAN_brackish.md`, `docs/RESULTS_brackish.md`)에서 판단 근거가 픽셀에 있으면 Jev가 게이트 역할을 못했다.
> 이번에는 Jev의 설계 의도에 맞는 조건을 만든다: ① 답 후보가 정해져 있고, ② 같은 판단을 대량으로 반복하며, ③ 정보가 텍스트 안에 있다.

---

## 1. 데이터: Indiana University 흉부 X-ray 판독문 (OpenI)

| 항목 | 값 |
|---|---|
| 출처 | NLM OpenI IU X-ray 판독문 (`indiana_reports.csv`, HF 미러 `sasi2004/chest-xrays-indiana-university`) → `data/iu-xray/` |
| 크기 | 판독문 3,851건, findings 또는 impression이 있는 것 3,826건 |
| 텍스트 | `findings` + `impression` (중앙값 34단어). `indication`은 판단 대상과 섞이지 않게 따로 표기 |
| 정답 | `Problems` 열: NLM 인덱서가 **판독문을 읽고 사람이 붙인** MeSH 기반 소견 태그 |

- 정답이 같은 텍스트에서 사람이 뽑은 것이라, **정보가 텍스트 안에 있다**는 조건을 만족한다. 어려운 부분은 부정 표현("No pleural effusion"), 애매한 표현("borderline"), 비식별 토큰(`XXXX`)이다.
- 이미지는 쓰지 않는다. ICH DaR과 연결해 보면 "VLM이 판독문을 썼다고 치고, 그 텍스트로 수용·확인을 정하는" 단계에 해당한다.

### 1.1 질문 = 소견 9개 (Noul, 정답 있음)
| 키 | 정답 = `Problems`에 다음 중 하나라도 있음 | 양성 수 (대략) |
|---|---|---|
| `abnormal` | `normal` 이외의 태그 | 2,470 |
| `cardiomegaly` | Cardiomegaly | 345 |
| `pleural_effusion` | Pleural Effusion | 149 |
| `opacity` | Opacity, Airspace Disease, Consolidation, Infiltrate, Pneumonia | ~600 |
| `atelectasis` | Pulmonary Atelectasis | 315 |
| `granuloma_calcification` | Calcified Granuloma, Granulomatous Disease, Calcinosis | ~550 |
| `nodule_mass` | Nodule, Mass | ~130 |
| `device` | Catheters, Indwelling, Surgical Instruments, Implanted Medical Device, Medical Device | ~300 |
| `fracture` | Fractures, Bone | 89 |

- 태그와 질문의 대응표는 config(`questions:`)에 두고, 결과를 보기 전에 고정한다.
- (탐색, 정답 없음) Jev의 다른 질문 형식 시연: `acuity` Score(1~5), `main_system` Choice(cardiac / pleural / parenchymal / skeletal / device / none). 표에는 넣지 않는다.

### 1.2 분할
- 판독문 단위 무작위 분할(seed 고정): **dev 800건 / test 3,026건**. 판독문끼리는 독립이다 (영상 프레임처럼 이웃이 없음).
- 모든 임계값은 dev에서만 정하고, test는 한 번만 돌린다.

## 2. 비교할 시스템 (판정 단위 = 판독문 × 질문)

| # | 시스템 | 판정 | 큰 LLM 호출 | 정할 값 (dev) |
|---|---|---|---|---|
| T1 | `rules` | 키워드 + 부정 규칙 (NegEx 방식). "정확한 규칙 → Code" 기준선 | 0 | – |
| T2 | `small_llm` | 작은 LLM의 Yes 확률 ≥ `v_s` (질문마다 forward 1번) | 0 | `v_s` (질문별) |
| T3 | **`large_llm_all` (기준)** | 큰 LLM의 Yes 확률 ≥ `v` | 판정 수 | `v` (질문별) |
| T4 | `jev_multi` | **Jev, 판독문 하나 = 요청 하나, 질문 9개를 한 번에** (p ≥ 0.5) | 0 | – |
| T4′ | `jev_single` | Jev, 질문 하나 = 요청 하나 (같은 문장) | 0 | – |
| T5 | **`gate` (primary)** | T4가 확실하면(max(p, 1−p) ≥ τ) 확정, 아니면 T3 | 넘긴 판정 수 | `τ` |
| T6 | `gate_small` | T2 자기 확률로 게이트: ≥ `hi`면 양성, ≤ `lo`면 음성, 그 사이는 T3 | 넘긴 판정 수 | `lo`, `hi` |
| T7 | `gate_rules` (탐색) | 규칙이 확실한 문장 패턴이면 확정, 아니면 Jev → 애매하면 큰 LLM (3단 라우팅) | 넘긴 판정 수 | `τ` |

- 모델: 작은 LLM = Qwen3-VL-4B-Instruct (텍스트만 입력), 큰 LLM = Qwen3.5-27B (캐시 있음; transformers 지원이 안 되면 Qwen3-VL-8B로 바꾸고 **M0에서 확정**).
- 모든 판정기는 **같은 질문 문장**을 받는다: "Does this chest X-ray report describe <소견 정의>?" (부정·불확실 표현 처리 기준도 같은 문장으로).
- T6은 Brackish에서 Jev를 이긴 대조군이다. 이번에도 반드시 비교한다.

## 3. Endpoint (결과를 보기 전에 고정)

**Primary: T5 `gate`는 T3 대비 정확도를 유지하면서 큰 LLM 호출을 줄이는가**
| 지표 | 정의 | 성공 기준 |
|---|---|---|
| 정확도 차이 | Δacc = acc(T5) − acc(T3), 판정(판독문×질문) 단위 micro 정확도, 95% CI는 **판독문 cluster bootstrap** | CI 하한 ≥ −0.02 |
| 큰 LLM 호출 비율 | T5가 넘긴 판정 수 / 전체 판정 수 | ≤ 0.5 |

순차 검정: ① −0.02 & 0.5 → ② −0.01 & 0.5 → ③ −0.01 & 0.3. 실제 값은 항상 보고한다.

**Secondary**
| # | 질문 | 지표 |
|---|---|---|
| S1 | Jev 게이트가 작은 LLM 자기 확률 게이트보다 나은가 | T5 vs T6: 같은 호출 비율에서의 정확도 (호출 비율–정확도 곡선 전체도 같이) |
| S2 | Multiple questions가 Single보다 나은가 | T4 vs T4′: 정확도·AUROC, 요청 수, 토큰, wall time |
| S3 | Jev p를 믿을 수 있나 | 질문별·전체 ECE, Brier, reliability diagram |
| S4 | dev 게이트 기준이 test에서 유지되나 | test 자동 처리분 오류율 vs 0.05 |
| S5 | 질문별로 어디서 갈리나 | 질문별 AUROC [CI], 민감도·특이도 (특히 희귀 소견: fracture, pleural_effusion) |
| S6 | 속도·비용 | 판독문당 지연, 전체 처리 시간, Jev 토큰, GPU-초 |

지표 순서: 정확도 → macro F1 (질문 평균) → Precision → Recall → ECE → 큰 LLM 호출 → 지연 → 총 시간 → 비용.

## 4. 공정성 규칙
1. 질문 정의, 태그 대응표, 분할, 모델은 결과를 보기 전에 고정한다. 임계값은 dev에서만 정한다.
2. 모든 판정기는 같은 텍스트(findings + impression)와 같은 질문 문장을 받는다. `Problems`·`MeSH` 열은 어떤 입력에도 넣지 않는다.
3. 큰 LLM과 작은 LLM에는 Brackish와 같은 방식(Yes/No 다음 토큰 확률)을 쓴다.
4. Jev 요청은 같은 본문이면 이전 응답을 재사용한다 (비용 절약, 재현).

## 5. run 폴더 (Brackish와 같은 원칙)
```text
runs/<YYMMDD>_text-<이름>/
├── README.md, run.yaml
├── data/          (링크) indiana_reports.csv, split.yaml
├── rules/         output/decisions.jsonl
├── small_llm/     input/prompts.jsonl   output/decisions.jsonl
├── large_llm/     input/prompts.jsonl   output/decisions.jsonl
├── jev_multi/     input/requests.jsonl  output/responses.jsonl, decisions.jsonl
├── jev_single/    input/requests.jsonl  output/responses.jsonl, decisions.jsonl
└── evaluation/    labels.jsonl, judgments.jsonl, tables/ (metrics.json·md, errors.csv, thresholds.yaml, endpoints.json),
                   plots/ (roc, reliability, 호출 비율–정확도 곡선, 질문별 막대), cases/ (틀린 판정 예시)
```

## 6. 코드 구조 (기존 `jevgate` 재사용)
| 재사용 | 새로 |
|---|---|
| `core/config.py`, `core/runio.py`(경로표만 추가), `evaluation/metrics.py`(AUROC, ECE, cluster bootstrap, endpoint), `judges/jev.py`의 `JevClient`·검증·재사용, `judges/gate.py`의 `fit_tau`·`fit_band`, `outputs/colors.py` | `text/data.py`(로드·라벨·분할), `text/rules.py`, `text/llm.py`(텍스트 전용 Yes 확률), `text/jev_text.py`(multi/single 요청), `text/evaluate.py`, `text/pipeline.py`, `configs/text*.yaml`, `scripts/run_text.py` |

## 7. 마일스톤
| M | 내용 | 완료 기준 |
|---|---|---|
| M0 | 데이터 로드·라벨·분할 고정, 질문 문장 고정, 큰 LLM 모델 확정 (Qwen3.5-27B 로딩 확인) | ruff + pytest, `configs/text_split.yaml` 커밋 |
| M1 | smoke: dev 30건, 모든 시스템 | 끝까지 실행, 계약 통과, 예산 안 |
| M2 | dev 800건 → 임계값 고정 | `thresholds_text.yaml` 커밋 |
| M3 | test 3,026건 1회 | 결과 표·그림·`endpoints.json` |
| M4 | 발표 정리: Brackish(이미지 근거) vs IU(텍스트 근거) 대비 | 한 장 비교표 |

## 8. 예상 비용·시간 (dev 기준, 확인 필요)
- Jev multi: 800 요청. Brackish 기준 요청당 약 0.3초, 동시 8개 → 수십 초. 토큰은 판독문 + 질문 9개 → 요청당 약 1~1.5천 (smoke에서 측정).
- Jev single: 800 × 9 = 7,200 요청 → multi의 약 9배 요청 수. S2의 비교 대상이므로 dev와 test 모두에서 돌린다 (예산 상한으로 보호).
- 큰 LLM: 800 × 9 판정, 질문마다 forward 1번 (batch). 27B는 GPU 1장(B200)에 bf16으로 올라간다.

## 9. 열린 이슈
- `Problems` 태그는 인덱서 1명의 판단이라 노이즈가 있다 (예: "borderline cardiomegaly"를 양성으로 볼지). 질문 문장의 기준을 태그 규칙에 맞춰 적고, 틀린 판정 사례를 `cases/`에서 따로 본다.
- `abnormal`은 척추 퇴행 같은 사소한 태그까지 양성이라 질문 문장으로 범위를 분명히 해야 한다.
- 희귀 소견(fracture 89, pleural_effusion 149)은 test에서도 양성이 수십~백여 개뿐이라 질문별 CI가 넓다.
- Jev 단가 미확인 (토큰 수로만 보고). 공식 Vercel AI Gateway 경로(`typesafe-ai/jev`)도 있지만, 현재 키로 직접 호출이 되므로 그대로 쓴다.
