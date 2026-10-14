# jev-gate 실험 계획 (v2, 2026-10-14)

> 질문: Grounding DINO가 낸 후보 박스가 **무엇인가**(6종 생물 또는 background)를 판정할 때,
> Jev(calibrated 확률)를 게이트로 쓰면 VLM 호출을 줄이면서 정확도를 유지할 수 있는가?
>
> v1 → v2 변경 (결과를 보기 전, 2026-10-14): 이진 판정(real/fake) → 7-way 클래스 판정, 클립 재선정, 단계별 산출물 정의. §14 참고.

---

## 1. 클래스와 비교할 시스템

### 1.1 클래스 (7개)
`fish · small_fish · crab · shrimp · jellyfish · starfish · background`

- 앞의 6개는 AAU GT 태그를 그대로 쓴다. `background`는 "후보 박스 안에 생물이 없음"이다.
- `background`가 없으면 탁도·부유물에 잡힌 가짜 후보에도 생물 클래스를 억지로 붙이게 되므로 반드시 둔다.
- DINO 프롬프트: `fish . small fish . crab . shrimp . jellyfish . starfish .` (`small fish` → `small_fish`로 매핑)

### 1.2 시스템 (표의 행)

모든 시스템은 **같은 DINO 후보 박스**를 받아 박스마다 7개 클래스 중 하나와 클래스별 확률을 낸다.

| # | 시스템 | 판정 방식 | 클래스별 점수 (ROC·ECE용) | 정할 임계값 |
|---|---|---|---|---|
| ① | 기존 detector | DINO score ≥ `t`면 DINO 라벨, 아니면 background | 라벨 클래스 = score, background = 1 − score, 나머지 0 | `t` (dev, 정확도 최대) |
| ② | DINO + VLM | Qwen3-VL-8B가 crop을 보고 선택지 A~G 중 하나 | 선택지 글자 토큰 7개의 softmax (forward 1번) | 없음 |
| ③ | DINO + Jev | Jev `choice` 질문 (선택지 7개) | Jev `probabilities` | 없음 (argmax 고정) |
| ④ | Jev 게이트 + VLM | Jev 최고 확률 ≥ `τ`면 Jev 답, 아니면 VLM 답 | 경로별 원래 점수 (ROC는 점으로만) | `τ` (dev) |

- ④는 ②와 ③의 판정 파일을 조합해서 계산한다. 추가 실행이 필요 없다.
- **Jev는 텍스트만 받는다** ("Jev accepts text only"). ③은 픽셀이 아니라 **구조화된 증거**(§4)로 판정한다. 클래스 구분 근거가 DINO 라벨·크기·위치·움직임뿐이라 사실상 DINO 라벨을 재보정하는 수준일 수 있다. 발표에서 반드시 밝힌다.
- ②와 ③은 같은 질문 문장과 같은 선택지 설명(`classes.*.description`)을 쓴다.
- (탐색, P2) ③′: 같은 텍스트 증거를 Qwen3-VL에 이미지 없이 넣는 대조군.

## 2. 데이터와 정답

### 2.1 정답 소스
- **AAU CSV** (`annotations/annotations_AAU/{train,valid,test}.csv`)를 GT로 쓴다. 12,444장, 박스 35,565개.
  - COCO json에는 valid 분할이 없고, YOLO는 빈 파일 1,449장에 실제 박스가 누락돼 있다.
- 분할 목록에 있지만 AAU에 없는 프레임 → `gt_status = empty_by_absence` (생물 없음). 기본 포함, 민감도 분석으로 제외.
- 분할 목록 밖 프레임(410장, 한 영상) → `gt_status = unknown`, 채점 제외. 해당 영상은 클립 후보에서 뺀다.
- AAU `Object ID`는 채점·분석에만 쓴다. Jev와 VLM 입력에는 넣지 않는다.

### 2.2 박스 정답 y
| 후보 박스와 GT의 최대 IoU | y |
|---|---|
| ≥ 0.5 | 매칭된 GT 물체의 클래스 |
| < 0.3 | `background` |
| 0.3 ~ 0.5 | `ambiguous` → 채점 제외, 개수 보고 (민감도: background로 넣기) |

- IoU는 클래스 무관으로 계산하고, 가장 많이 겹친 GT의 클래스를 y로 쓴다.
- GT 박스는 프레임 경계로 자른 뒤 IoU를 계산한다.
- 후보는 클래스 무관 NMS(IoU 0.5) 후 라벨링한다.

### 2.3 클립 분할 (영상 단위, v2 재선정)
v1 클립 3개는 shrimp·jellyfish가 없고 dev에 fish·crab이 없어서, **dev와 test 각각이 6종을 모두 포함**하도록 다시 골랐다. v1 test 클립 2개는 그대로 유지했다.

| 용도 | 영상 | 프레임 | GT 박스 (AAU) |
|---|---|---|---|
| dev | `2019-03-19_17-02-30to2019-03-19_17-02-42_1` | 183 | small_fish 233, starfish 173 |
| dev | `2019-02-21_06-36-31to2019-02-21_06-36-39_1` | 120 | crab 240, shrimp 77, starfish 120 |
| dev | `2019-02-20_19-43-37to2019-02-20_19-43-46_1` | 130 | fish 89, crab 130 |
| dev | `2019-03-20_05-40-25to2019-03-20_05-40-33_1` | 118 | jellyfish 57 |
| test | `2019-03-21_05-01-24to2019-03-21_05-01-31_1` | 110 | small_fish 74, crab 440, starfish 110 |
| test | `2019-03-19_19-30-33to2019-03-19_19-30-45_1` | 185 | fish 97, starfish 182 |
| test | `2019-02-22_22-18-13to2019-02-22_22-18-20_1` | 111 | shrimp 71, starfish 111 |
| test | `2019-03-20_15-15-12to2019-03-20_15-15-19_1` | 111 | jellyfish 87 |

- 합계: dev 551프레임 (fish 89 · small_fish 233 · crab 370 · shrimp 77 · jellyfish 57 · starfish 293), test 517프레임 (fish 97 · small_fish 74 · crab 440 · shrimp 71 · jellyfish 87 · starfish 403).
- 바꾸려면 §7의 변경 의도 질문을 거친다.

## 3. 파이프라인: 단계별 데이터와 남길 산출물

각 단계는 앞 단계의 **파일만** 읽는다. "데이터"는 다음 단계가 읽는 파일, "보고용"은 사람이 보는 파일이다.

| 단계 | 데이터 (다음 단계 입력) | 보고용 산출물 | 계약 (`check_contracts.py`) |
|---|---|---|---|
| S0 meta | `config.resolved.yaml`, `git.json`, `env.json` | `run.log`, `timing.json` | 파일 존재 |
| S1 frames | `frames.jsonl` | – | 클립별 프레임 수 = 이미지 수 |
| S2 detect | `candidates.jsonl` | `detect_summary.json`, `viz/frames/dino/`, `viz/videos/*_dino.*` | 박스가 프레임 안, score ∈ [0,1], cand_id 고유, label ∈ 6종 |
| S3 label | `labels.jsonl` | `label_summary.json` + `.md`, `plots/dino_vs_gt_confusion.png`, `viz/frames/gt/` | candidates와 1:1, y ∈ 7종 ∪ {ambiguous, null} |
| S4 evidence | `evidence.jsonl` | – | 이전 프레임만 참조 (미래 누설 0) |
| S5 judge | `decisions/{threshold,vlm,jev}.jsonl` | `raw/jev_responses.jsonl`, `raw/vlm_outputs.jsonl`, `usage.json` | 시스템마다 후보 수 동일, 확률 합 = 1 ± 0.02 |
| S6 gate | `decisions/gate.jsonl` | `tau_curve.csv`, (dev) `thresholds.yaml` | `routed_to=vlm`인 행은 ② 판정과 일치 |
| S7 eval | `metrics.json` | `metrics.md`, `errors.csv`, `plots/*`, (test) `endpoints.json` | 스키마 검사 |
| S8 viz | – | `viz/frames/<system>/`, `viz/frames/compare/`, `viz/videos/`, `viz/index.html` | – |
| ⑤ 육안 | – | `review_notes.md` (사람이 작성) | – |

### 3.1 단계별 필수 필드
| 파일 | 필드 |
|---|---|
| `frames.jsonl` | clip, split, frame_idx, file, gt_status |
| `candidates.jsonl` | cand_id, clip, frame_idx, box_xyxy, dino_label, dino_score |
| `labels.jsonl` | cand_id, max_iou, y, matched_object_id, matched_gt_label, scored |
| `evidence.jsonl` | cand_id, history_frame_idxs, detector, geometry, track, scene |
| `decisions/*.jsonl` | cand_id, pred, probs{7종}, confidence, latency_ms, (jev) tokens_in/out, (gate) routed_to |

### 3.2 보고용 산출물 내용
| 파일 | 내용 |
|---|---|
| `env.json` | 패키지 버전, GPU 이름, DINO·VLM 모델 revision, Jev 응답의 `model` 버전 |
| `timing.json` | 단계별 wall-clock |
| `detect_summary.json` | NMS 전/후 후보 수, 프레임당 후보 수 분포, DINO 라벨별 개수, score 히스토그램 |
| `label_summary.json` / `.md` | **개수 흐름표**(아래), GT 클래스별·DINO 라벨별로 나눈 같은 표 |
| `usage.json` | Jev 요청 수·토큰·추정 비용, VLM 호출 수·GPU-초 |
| `metrics.md` | 결과 표(PPT·PR에 그대로 붙이는 형식) |
| `errors.csv` | 틀린 박스 목록: cand_id, file, y, 시스템별 pred·confidence, crop 경로 |
| `endpoints.json` | (test만) 순차 검정 단계별 통과 여부와 CI·비율 실제 값 |

**개수 흐름표** (`label_summary.md`, 매 실행):
```text
GT 물체 N ── DINO가 IoU≥0.5로 잡음 n (recall 상한 n/N) ── 놓침 N−n
DINO 후보 M ── 클래스 y별 개수 (6종 + background) · ambiguous a · unknown u
              └──────── 채점 대상 M−a−u ────────┘   └─ 제외 a+u (사유별) ─┘
```

### 3.3 그림 (`plots/`)
| 파일 | 내용 | 시스템 |
|---|---|---|
| `dino_vs_gt_confusion.png` | DINO 라벨 × y, 7×7 (DINO가 클래스를 맞게 붙였나) | S3 |
| `confusion_<system>.png` | 예측 × y, 7×7, 행 정규화 % 병기 | ①②③④ |
| `ovr_2x2_<system>.png` | 클래스별 one-vs-rest 2×2 (sens/spec 병기), 7패널 | ①②③④ |
| `roc_<system>.png` | 클래스별 one-vs-rest ROC + AUC [95% CI], 운영점 마커 | ①②③ (④는 ①②③ 그림에 점으로) |
| `reliability_<system>.png` | top-label confidence reliability diagram + ECE | ②③ |
| `tau_curve.png` | τ별 자동 처리 비율 · 자동 처리분 오류율 · VLM 호출 수 | ④ |

- 박스가 20개 미만인 클래스 패널은 n을 표시하고 회색 처리한다.
- AUC CI는 프레임 블록 bootstrap(§6)으로 구한다.

### 3.4 이미지와 영상 (`viz/`)
| 경로 | 내용 |
|---|---|
| `viz/frames/gt/<clip>/<frame>.jpg` | GT 박스 + 클래스 |
| `viz/frames/dino/<clip>/<frame>.jpg` | DINO 후보 + 라벨 + score, 테두리 색 = y |
| `viz/frames/<system>/<clip>/<frame>.jpg` | 시스템별 판정 (①②③④) |
| `viz/frames/compare/<clip>/<frame>.jpg` | ①②③④ 2×2 한 장 |
| `viz/videos/<clip>_<view>.webm` / `.mp4` | 위 view별 영상. webm(VP9)은 에디터용, mp4(H.264)는 PPT용 |
| `viz/crops/errors/` | `errors.csv`의 박스 crop |
| `viz/index.html` | 프레임 넘겨 보기, view 전환 (VS Code/Cursor의 Live Preview 확장으로 열기) |

- 색 규칙(모든 그림 공통): 맞음 = 초록 실선, 생물을 다른 생물로 = 노랑, background를 생물로(FP) = 빨강, 생물을 background로(FN) = 주황 점선, background를 background로 = 회색 점선. 라벨에 `pred (conf)`를 쓴다.
- 영상 인코딩은 `imageio-ffmpeg`(ffmpeg 내장)로 한다. 현재 OpenCV는 H.264를 못 쓴다.
- smoke 실행은 영상을 만들지 않는다 (`viz.videos: false`).

### 3.5 runs 폴더 전체 모습
```text
runs/261014_dev/
├── config.resolved.yaml      git.json      env.json      timing.json      run.log
├── frames.jsonl              candidates.jsonl      labels.jsonl      evidence.jsonl
├── decisions/  threshold.jsonl  vlm.jsonl  jev.jsonl  gate.jsonl
├── raw/        jev_responses.jsonl  vlm_outputs.jsonl
├── detect_summary.json       label_summary.json  label_summary.md   usage.json
├── metrics.json  metrics.md  errors.csv  tau_curve.csv
├── thresholds.yaml           (dev: t, τ, fit_split, git commit)
├── endpoints.json            (test만)
├── plots/      dino_vs_gt_confusion · confusion_* · ovr_2x2_* · roc_* · reliability_* · tau_curve
├── viz/        frames/{gt,dino,threshold,vlm,jev,gate,compare}/  videos/  crops/errors/  index.html
└── review_notes.md           (⑤ 사람이 작성)
```

- `run_id = <YYMMDD>_<run.name>`. 같은 날 같은 이름이면 `_2`, `_3`을 붙인다. 코드 버전은 `git.json`에서 본다.
- `raw/`에는 API 키를 쓰지 않는다. Jev 요청 본문은 저장한다(재현용).
- dev 결과는 레포 루트 `metrics_history.jsonl`에 한 줄씩 추가한다 (git 추적).
- 예상 용량: 프레임 551장 × view 7종 JPG + 영상 → run 하나에 수백 MB.

## 4. Jev 입력: 구조화된 증거 (S4)

**인과적**으로 만든다. 현재 프레임과 **이전 K 프레임의 DINO 후보만** 쓰고, GT와 미래 프레임은 쓰지 않는다.

| 그룹 | 필드 |
|---|---|
| 검출기 | DINO 라벨, dino_score |
| 기하 | 면적 비율, 가로·세로 px, 가로세로비, 정규화 중심 (x, y), 프레임 경계 접촉 |
| 시간 (이전 K프레임) | 연속 등장 횟수 (IoU ≥ 0.3), 직전 프레임과의 최대 IoU, 이동량/대각선, 면적 변화율, 트랙 평균 score |
| 장면 | 이 프레임의 후보 수 |

- Jev 질문: `type: choice`, `criteria` = 7개 클래스의 설명(`classes.*.description`), `instructions` = `prompts.question`.
- 요청 단위 `jev.batching`: `per_frame`(기본, 프레임 하나에 박스별 질문 여러 개) | `per_box`(지연 비교용).

## 5. Endpoint (결과를 보기 전에 고정. v1 2026-10-13, v2 2026-10-14에 다중 클래스로 갱신)

모든 endpoint는 **test 클립 4개**에서, dev에서 고정한 `t`·`τ`로 **한 번만** 측정한다.

### 5.1 Primary: ④는 ② 대비 정확도를 유지하면서 VLM 호출을 줄이는가
| 지표 | 정의 | 성공 기준 |
|---|---|---|
| 정확도 차이 | Δacc = acc(④) − acc(②), 7-way 정확도, 95% CI는 프레임 블록 bootstrap | CI 하한 ≥ **−0.02** |
| VLM 호출 비율 | ④ 호출 수 / ② 호출 수 | ≤ **0.5** |

- 두 조건을 **동시에** 만족해야 성공이다.
- 순차 검정 (처음 실패한 단계에서 멈춤, 재실행 없음):
  1. Δacc CI 하한 ≥ −0.02, 호출 비율 ≤ 0.5 (primary)
  2. Δacc CI 하한 ≥ −0.01, 호출 비율 ≤ 0.5
  3. Δacc CI 하한 ≥ −0.01, 호출 비율 ≤ 0.3
- 실제 CI와 비율은 항상 보고한다. 결과를 본 뒤 기준을 느슨하게 바꾸지 않는다.

### 5.2 Secondary
| # | 질문 | 지표 |
|---|---|---|
| S1 | Jev 확률을 믿을 수 있나 (③) | top-label ECE (15 bin), 다중 클래스 Brier, reliability diagram |
| S2 | dev 게이트 기준이 test에서도 지켜지나 (④) | test 자동 처리분 오류율 vs ε = 0.05, 자동 처리 비율 |
| S3 | Jev 단독이 기존 detector보다 나은가 (③ vs ①) | 정확도 → macro F1 → 클래스별 Precision·Recall |
| S4 | 클래스별로 어디서 갈리나 (①~④) | 클래스별 one-vs-rest AUC [CI], sens/spec |
| S5 | 실제 속도와 비용 (①~④) | 박스당 지연 중앙값, 총 시간, 비용 |
| S6 | 영상에서 보이는 결과 (①~④) | 프레임 단위 "생물 보임?"(pred ≠ background인 박스 ≥ 1) 정확도 |

### 5.3 탐색적 분석 (결론에 쓰지 않음)
③′ 텍스트 증거 대조군, 민감도 분석(ambiguous를 background로, 생물 없는 프레임 제외, fish/small_fish 병합), τ 곡선.

## 6. 지표 정의

| 지표 | 정의 |
|---|---|
| 정확도 | pred = y인 비율 (ambiguous·unknown 제외) |
| macro F1 | 7개 클래스 F1의 평균 (정확도 다음에 보고) |
| 클래스별 Precision / Recall | one-vs-rest |
| ECE | top-label confidence, 15 bin. ②③ 전체, ④는 자동 처리분 |
| Brier | 다중 클래스 (one-hot과 확률 벡터의 제곱 오차 합 평균) |
| AUC | 클래스별 one-vs-rest, 프레임 블록 bootstrap CI |
| VLM 호출 | ② = 후보 수, ③ = 0, ④ = 넘긴 수 |
| 지연 | 박스당 판정 시간 중앙값. ④는 경로별로도 |
| 비용 | Jev 토큰 × `cost.jev_usd_per_1m_tokens`, VLM GPU-초 |

- 지표 순서는 항상 정확도 → macro F1 → Precision → Recall → ECE → VLM 호출 → 지연 → 비용.
- 각주 1 (recall 상한): DINO 후보 recall 상한 = x%. 어느 시스템도 이를 넘을 수 없다.
- 각주 2 (독립성): 박스 n, 프레임 n, 클립 n을 표기한다. 연속 프레임은 독립이 아니다.
- 프레임 블록 bootstrap: 클립 안에서 연속 10프레임 블록을 단위로 2,000회 재표본.

## 7. 공정성 규칙

1. `t`와 `τ`는 **dev에서만** 정하고 `thresholds.yaml`에 고정한다 (`fit_split: dev`, commit 기록). test는 이 파일만 읽고, `fit_split ≠ dev`면 실행을 거부한다.
   - `τ`: dev 자동 처리분 오류율 ≤ `gate.max_auto_error`(0.05)인 가장 작은 τ.
2. 네 시스템 모두 같은 `candidates.jsonl`을 쓴다.
3. VLM과 Jev는 같은 질문 문장과 같은 클래스 설명을 쓴다.
4. 표에 n_box, n_frame, n_clip을 함께 쓴다.
5. config 변경 의도 질문은 `/jev-loop` ⓪이 묻는다.

## 8. 개발 검증 루프 (`/jev-loop`)

```
⓪ 변경 의도 → ① 정적 검사 → ② 데이터 계약 → ③ smoke E2E → ④ 지표 회귀 → ⑤ 사람 육안 → ⑥ PR 기록
```
세부 기준은 `.claude/skills/jev-loop/SKILL.md`. ⑤는 `viz/index.html`과 `compare` 영상으로 본다.

## 9. 레포 구조

```
jev-gate/
├── configs/            base.yaml, smoke.yaml, test.yaml
├── src/jevgate/
│   ├── config.py       YAML 로드·상속·검증
│   ├── runio.py        run 폴더, JSONL 입출력, git·env 기록
│   ├── boxes.py        IoU, NMS, clip
│   ├── data.py         클립·프레임·AAU GT
│   ├── detect.py       S2 Grounding DINO
│   ├── label.py        S3 IoU 매칭 → y, 개수 흐름표
│   ├── evidence.py     S4 인과적 특징
│   ├── judges/         base.py, threshold.py, vlm.py, jev.py (S5)
│   ├── gate.py         S6
│   ├── metrics.py      S7 정확도·F1·ECE·Brier·AUC·bootstrap·endpoint
│   ├── plots.py        S7 그림
│   ├── viz.py          S8 overlay·영상·index.html
│   └── contracts.py    단계별 스키마 검사
├── scripts/            run_pipeline.py, fit_thresholds.py, check_contracts.py,
│                       compare_metrics.py, config_guard.py, render_samples.py
├── tests/   docs/PLAN.md   metrics_history.jsonl
└── data/ runs/ .env    (gitignore)
```

## 10. git 운영

- `main`: 검증 루프를 통과한 코드만.
- `feat/<단계>`: 코드 작업. `exp/<YYMMDD>-<실험 이름>`: config만 바꾸는 실험.
- 원격은 팀 공유 시점에 만든다.

## 11. 마일스톤

| M | 내용 | 완료 기준 |
|---|---|---|
| M0 | config(7클래스·새 클립), runio, boxes, data, label, evidence, contracts, metrics, 테스트 | ruff + pytest 통과 |
| M1 | S1~S3 dev 실행 + `label_summary`, `dino_vs_gt_confusion`, `viz/frames/{gt,dino}` | 계약 통과, recall 상한 산출, 사람이 DINO overlay 확인 |
| M2 | S4~S5 ①·③ + smoke E2E | smoke가 예산 안에서 끝까지 실행 |
| M3 | S5 ② VLM (선택지 logit) | dev 전 후보 판정 |
| M4 | S6~S7 + plots, dev에서 t·τ 고정 → test 1회 | 결과 표·그림·`endpoints.json` |
| M5 | S8 영상·index.html, 결과 카드 PPT | ⑤ 육안 확인 |
| M6 (선택) | autoresearch식 증거 형식 탐색 (별도 탐색 클립, test 차단, 비용 상한) | M4 이후에만 |

## 12. 열린 이슈

- Jev 토큰 단가와 rate limit 미확인 → smoke에서 사용량을 재고 콘솔에서 확인.
- DINO `box_threshold`(후보를 넉넉히 내도록 낮게)는 dev에서만 정한다.
- shrimp(dev 77·test 71)와 jellyfish(dev 57·test 87)는 박스가 적어 클래스별 CI가 넓다.
- fish vs small_fish는 종이 아니라 크기·무리 기준이라, 텍스트만 보는 Jev가 구분할 근거가 약하다 → 병합 민감도 분석.
- AAU 육안 검수는 254장뿐이다. 흐릿한 실루엣이 background로 채점될 수 있다 → ⑤와 `errors.csv`에서 표시.

## 13. 산출물 → 발표 매핑

| 발표 요소 | 출처 |
|---|---|
| 결과 표 | test `metrics.md` |
| 그림 A·C 형태 (ROC, 2×2) | `plots/roc_*`, `plots/ovr_2x2_*` |
| calibration 슬라이드 | `plots/reliability_jev.png`, `tau_curve.png` |
| Before → After 영상 | `viz/videos/<clip>_dino.mp4` → `<clip>_gate.mp4` |
| 오류 사례 | `errors.csv`, `viz/crops/errors/` |

## 14. 변경 기록

- v1 (2026-10-13): 이진 판정, 클립 3개, endpoint 사전 등록.
- v2 (2026-10-14, 결과 보기 전): 7-way 클래스 판정 (6종 + background), VLM은 선택지 logit, Jev는 `choice`, 클립 8개로 재선정 (dev·test 각 6종 포함), 단계별 산출물 정의.
