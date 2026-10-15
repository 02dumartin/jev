# jev-gate 실험 계획 (v6, 2026-10-15)

> 질문: 학습한 detector(YOLO12s)와 tracker가 만든 **track이 맞는지**(진짜 그 클래스의 동물인지)를 판정할 때,
> Jev(calibrated 확률)를 게이트로 써서 **확실한 track은 Jev가 처리하고 애매한 track만 VLM에 넘기면**,
> 모든 track을 VLM이 보는 것과 같은 정확도를 VLM 호출을 줄이면서 얻을 수 있는가?
>
> v6에서 바뀐 것: 후보는 Grounding DINO가 아니라 **영상 단위 분할로 학습한 YOLO12s + ByteTrack**,
> 판정 단위는 박스가 아니라 **track**, Jev 질문은 Noul 하나(`track_ok`)가 게이트를 정한다. 이전 버전은 §14.

---

## 1. 구성 요소

| 구성 | 무엇 | 입력 | 출력 |
|---|---|---|---|
| Detector | YOLO12s `train/yolo12s_video/weights/best.pt` (영상 단위 분할로 학습, best epoch 35) | 프레임 이미지 | 프레임별 박스, 클래스(6종), conf |
| Tracker | ByteTrack (Ultralytics), conf 0.1 | detector 박스 | 박스마다 track id |
| Jev | TypeSafe API, `jev-latest`. **텍스트만 받는다** | track 요약 증거 (§4) | `track_ok` p (Noul), `class_fix` 분포 (Choice, 탐색용) |
| VLM | Qwen3-VL-8B-Instruct (로컬) | track 대표 프레임: 전체 프레임 + 빨간 박스, 확대 crop | "진짜 <클래스>인가?" Yes/No 확률 p_yes |

### 1.1 클래스
`fish · small_fish · crab · shrimp · jellyfish · starfish` (AAU 태그 그대로). 판정 결과로 박스를 지우면 `background`(동물 아님)가 된다.

### 1.2 track과 다수결 클래스
- YOLO는 프레임마다 따로 클래스를 붙인다. track의 **다수결 클래스** = track 안 박스들에서 가장 많이 나온 클래스 (동점이면 conf 합이 큰 쪽).
- **판단은 track 단위, 결과는 박스에 물려준다.** track이 수용되면 그 track의 모든 박스가 다수결 클래스가 되고, 거부되면 모든 박스가 background가 된다.
- ByteTrack이 track으로 확정하지 않은 박스(한두 프레임만 나온 탐지)는 **제거**(background)한다. 그중 진짜 동물 수는 따로 보고한다.

## 2. 데이터와 정답

### 2.1 분할 (`configs/splits/video_split.yaml`, 고정)
| 용도 | 영상 | 프레임 | 쓰임 |
|---|---|---|---|
| train | 63 | 10,593 | YOLO 학습만 |
| **dev** | 13 (= YOLO val 영상) | 2,030 | `t`, `v`, `τ` 결정, 프롬프트 점검 |
| **test** | 13 | 2,051 | 설정 고정 후 한 번만 평가 |

- 한 영상은 한 분할에만 들어간다. 원본 프레임 무작위 분할(`train.txt` 등)은 같은 영상의 이웃 프레임이 train/test에 섞여 성능이 부풀려지므로 쓰지 않는다 (`yolo12s_orig` test mAP50 0.99 vs `yolo12s_video` 0.55).
- dev 영상은 YOLO의 best epoch를 고를 때도 쓰였다. 판정 단계의 임계값을 정하는 데 다시 쓰는 것은 허용하되, 각주로 밝힌다.

### 2.2 박스 정답 y_box
- 정답 소스: AAU CSV. 같은 프레임의 GT 박스와 클래스 무관 IoU로 매칭한다 (최대 IoU 기준).

| 최대 IoU | y_box |
|---|---|
| ≥ 0.5 | 매칭된 GT의 클래스 |
| < 0.3 | `background` |
| 0.3 ~ 0.5 | `ambiguous` → 채점 제외, 개수 보고 |

- GT 박스는 프레임 경계로 자른 뒤 IoU를 계산한다. 어떤 YOLO 박스와도 IoU ≥ 0.5로 맞지 않은 GT는 **놓친 물체**로 세고, 어떤 판정기도 되살릴 수 없다 (recall 상한, 각주).

### 2.3 track 정답 y_track (calibration·τ 결정용)
- y_track = 1: track의 채점 대상 박스 중 **절반 이상이 다수결 클래스의 GT**와 맞음 (= 수용이 맞는 track).
- y_track = 0: 그렇지 않음. 유형을 `false`(대부분 background), `misclassified`(다른 동물)로 나눠 기록한다.
- 채점 대상 박스가 없는 track(전부 ambiguous)은 y_track = null, 채점 제외.

## 3. 비교할 시스템 (표의 행)

| # | 시스템 | track 판정 | 정할 값 (dev) |
|---|---|---|---|
| ① | `yolo` | track 없이 프레임별: conf ≥ `t`면 YOLO 클래스, 아니면 background | `t` (박스 정확도 최대) |
| ② | `track_vote` | 모든 track 수용 (판정기 없음, tracking 효과만) | – |
| ③ | `vlm_all` | 모든 track을 VLM에: p_yes ≥ `v`면 수용 | `v` (track 정확도 최대) |
| ④ | `jev` | Jev p ≥ 0.5면 수용 | – (0.5 고정: calibration을 믿는 설정) |
| ⑤ | **`gate` (primary)** | p ≥ τ면 수용, p ≤ 1−τ면 거부, 그 사이는 ③의 VLM 판정 | `τ` |

- ⑤는 ③과 ④의 판정 파일을 조합한다 (추가 실행 없음). VLM 지연은 넘긴 track만 더한다.
- `τ`: dev에서 "Jev가 혼자 처리한 track의 오류율 ≤ 0.05"인 설정 중 혼자 처리한 track이 가장 많은 값. 없으면 τ > 1(전부 VLM).
- ②~⑤에서 untracked 박스는 모두 background로 같다. 시스템 사이 차이는 track 판정에서만 생긴다.
- ③과 ④는 **같은 질문 문장**을 받는다: "Is this track a real `<다수결 클래스>`?" (§4, §5)

## 4. Jev 입력과 질문

### 4.1 state (track 하나, 텍스트)
판정 시점은 **track이 끝난 뒤**(오프라인 영상 검수 시나리오)이므로 track 전체 구간을 쓴다. GT, 영상 이름, track id는 넣지 않는다.

| 그룹 | 필드 |
|---|---|
| 길이 | length_frames, span_frames, gap_frames, starts_at_video_start, ends_at_video_end |
| 클래스 | class_votes(클래스별 횟수), majority_class, majority_share |
| conf | mean, min, max, trend (앞 절반 vs 뒤 절반) |
| 박스 | median 가로·세로 px, median 중심 (x, y) 정규화, 경계 접촉 비율 |
| 움직임 | 전체 이동량 / 박스 대각선, 프레임당 평균 이동 px |
| 장면 | 같은 시간대의 다른 track 수 (클래스별) |
| (탐색) detector 신뢰도 | YOLO의 클래스별 dev(val) precision·recall. `jev.include_detector_reliability`, 기본 false |

### 4.2 질문 (track마다 2개)
| 키 | 형식 | 문장 | 쓰임 |
|---|---|---|---|
| `track_ok` | Noul | "Is this track a real `<majority_class>`?" (true = 그 동물이 맞음 / false = 다른 동물이거나 동물이 아님) | **게이트 결정. ④·⑤의 p** |
| `class_fix` | Choice (6종 + `not_an_animal`) | "What is this track?" | 탐색: 거부한 track을 어떤 클래스로 고칠지 |

- 요청 단위: 영상 하나의 track들을 `jev.max_tracks_per_request`개씩 묶어 한 요청으로 보낸다 (track마다 질문 2개).
- 클래스 정의(`classes.*`)는 state에 한 번만 쓴다. VLM 프롬프트와 같은 문장이다.
- v4의 `need_vlm` Score 질문은 뺐다: Noul p 자체가 확신도이고, v4 dev에서 need_vlm 게이트는 박스 99.9%를 VLM으로 넘겼다.

## 5. VLM 입력
- track의 대표 프레임 = conf가 가장 높은 박스의 프레임. 이미지 2장: 전체 프레임 + 빨간 박스, 박스 주변 확대 crop (박스 크기만큼 여유, 긴 변 336px).
- 프롬프트: 공통 맥락 + 클래스 정의 + "Is this track a real `<majority_class>`? Answer Yes or No." → 다음 토큰의 Yes/No logit softmax = p_yes.
- Qwen3-VL은 이 탁한 영상에서 Yes를 거의 안 한다(v1 dev: real 박스에도 p_yes ≈ 0.001). 그래서 0.5가 아니라 dev에서 정한 `v`로 자른다.

## 6. 지표

### 6.1 Endpoint (결과를 보기 전에 고정, 2026-10-15)
test 13개 영상에서, dev에서 고정한 `t`·`v`·`τ`로 **한 번만** 측정한다.

**Primary: ⑤ `gate`는 ③ `vlm_all` 대비 정확도를 유지하면서 VLM 호출을 줄이는가**
| 지표 | 정의 | 성공 기준 |
|---|---|---|
| 정확도 차이 | Δacc = acc(⑤) − acc(③), **박스 단위** 7-way 정확도 (6종 + background), 95% CI는 cluster bootstrap (§6.3) | CI 하한 ≥ **−0.02** |
| VLM 호출 비율 | ⑤가 VLM에 넘긴 track 수 / 전체 track 수 | ≤ **0.5** |

- 두 조건을 **동시에** 만족해야 성공이다. 순차 검정 (처음 실패에서 멈춤, 재실행 없음):
  1. CI 하한 ≥ −0.02, 호출 비율 ≤ 0.5 (primary)
  2. CI 하한 ≥ −0.01, 호출 비율 ≤ 0.5
  3. CI 하한 ≥ −0.01, 호출 비율 ≤ 0.3
- 실제 CI와 비율은 항상 보고한다. 결과를 본 뒤 기준을 느슨하게 바꾸지 않는다.

**Secondary**
| # | 질문 | 지표 |
|---|---|---|
| S1 | Jev p를 믿을 수 있나 (④) | track 단위 ECE (15 bin), Brier, reliability diagram |
| S2 | dev 게이트 기준이 test에서도 지켜지나 (⑤) | test 자동 처리분 오류율 vs 0.05, 자동 처리 비율 |
| S3 | 누가 맞는 track과 틀린 track을 잘 가르나 | track 단위 ROC + AUROC [95% CI]: Jev p, VLM p_yes, track 평균 YOLO conf |
| S4 | tracking과 판정기가 detector를 얼마나 고치나 (①→②→⑤) | 박스 정확도 → macro F1 → 클래스별 P/R, confusion matrix |
| S5 | 속도와 비용 | 단계별 시간, track당 지연 중앙값, 영상 1개 처리 시간, Jev 토큰·비용, VLM GPU-초 |
| S6 | 영상에서 보이는 안정성 | 프레임 간 박스 수 변화(깜빡임), track 길이 |

**탐색 (결론에 쓰지 않음)**: `class_fix` 교정 정확도, detector 신뢰도 표를 넣은 Jev, ambiguous를 background로 넣은 민감도, τ 곡선.

### 6.2 지표 정의
| 지표 | 정의 |
|---|---|
| 정확도 | 박스 단위 pred = y_box 비율 (ambiguous 제외). pred ∈ 6종 ∪ {background} |
| macro F1 | 7개 클래스 F1 평균 (정확도 다음에 보고) |
| Precision / Recall | 클래스별 one-vs-rest |
| detection recall | GT 물체 중, 최종적으로 올바른 클래스로 남은 박스와 맞는 비율 (놓친 GT 포함 분모) |
| ECE / Brier | track 단위, Jev p와 y_track |
| AUROC | track 단위, y_track 기준 |
| VLM 호출 | ③ = track 수, ⑤ = 넘긴 track 수 |
| 시간 | `run.yaml` timing: yolo_track, jev, vlm(로드 제외), evaluate. ⑤ 시간 = Jev + 넘긴 track의 VLM 시간 |
| 비용 | Jev 토큰 × `cost.jev_usd_per_1m_tokens` (단가 미확인 시 토큰만) |

- 표의 열 순서: 정확도 → macro F1 → Precision → Recall → ECE → VLM 호출 → 지연 → 총 시간 → 비용.
- 모든 표에 n_box, n_track, n_frame, n_video를 쓴다.

### 6.3 신뢰구간
- 판정이 track 단위라서 bootstrap 단위도 **track**이다. 각 track(그 박스 전체)과, 영상별 untracked 박스 묶음을 하나의 cluster로 보고 1,000회 재표본한다.
- AUROC CI도 track 재표본으로 구한다.

## 7. run 폴더 (`runs/<YYMMDD>_<이름>/`)

```text
runs/261015_dev/
├── README.md          결과 표, n 흐름, 시간·비용, 그림 설명
├── run.yaml           config + git + 환경 + 임계값 + 단계별 시간 + 사용량
├── data/              (심볼릭 링크) images → 원본 이미지, annotations → AAU CSV 폴더, video_split.yaml
├── yolo/
│   ├── input/         weights.pt (링크), tracker.yaml (링크), videos.txt
│   └── output/        detections.jsonl (박스: video, frame, box, class, conf, track_id)
│                      tracks.jsonl (track: 박스 목록, 투표, 다수결 클래스, 요약 통계)
├── jev/
│   ├── input/         requests.jsonl (보낸 요청 본문 그대로, 키 없음)
│   └── output/        responses.jsonl (원본 응답 + 지연), decisions.jsonl (track별 p, class_fix)
├── vlm/
│   ├── input/         prompts.jsonl, images/<track>_frame.jpg, images/<track>_crop.jpg (실제로 넣은 이미지)
│   └── output/        decisions.jsonl (track별 p_yes, 지연)
└── evaluation/
    ├── labels/        boxes.jsonl (y_box, 매칭 GT), tracks.jsonl (y_track, 유형)
    ├── judgments.jsonl  박스마다 시스템별 pred
    ├── tables/        metrics.json, metrics.md, errors.csv, thresholds.yaml (dev), endpoints.json (test)
    ├── confusion/     <system>.png
    ├── plots/         roc.png, reliability.png, tau_curve.png
    ├── viz/           샘플 프레임: ① yolo / ② track_vote / ③ vlm_all / ⑤ gate 비교
    └── videos/        <video>.mp4 (왼쪽 ① yolo, 오른쪽 ⑤ gate)
```
- 원본 데이터·가중치·설정처럼 이미 있는 것은 **심볼릭 링크**로 건다.
- Jev 응답은 요청 본문이 완전히 같으면 이전 run의 `responses.jsonl`에서 재사용한다 (비용 절약, 재현).
- smoke는 영상·viz를 만들지 않는다.

## 8. 공정성 규칙
1. `t`, `v`, `τ`는 **dev에서만** 정하고 `thresholds.yaml`에 고정한다 (`fit_split: dev`, run, commit 기록). test는 이 파일만 읽고, dev 영상 목록이 다르면 실행을 거부한다.
2. 다섯 시스템 모두 같은 `detections.jsonl`·`tracks.jsonl`을 쓴다.
3. VLM과 Jev는 같은 질문 문장과 같은 클래스 정의를 쓴다.
4. config 변경 의도 질문은 `/jev-loop` ⓪이 묻는다.

## 9. 개발 검증 루프 (`/jev-loop`)
```
⓪ 변경 의도 → ① 정적 검사 → ② 데이터 계약 → ③ smoke E2E → ④ 지표 회귀 → ⑤ 사람 육안 → ⑥ PR 기록
```
`/jev-loop`은 호출할 때 한 번 도는 스킬이다 (백그라운드 프로세스 아님). 세부는 `.claude/skills/jev-loop/SKILL.md`.

## 10. 레포 구조
```
jev-gate/
├── configs/        base.yaml (전부), smoke.yaml, test.yaml, yolo*.yaml (YOLO 학습 변형)
├── splits/         video_split.yaml
├── scripts/        run_pipeline · fit_thresholds · check_contracts · compare_metrics · config_guard
│                   make_video_split · prepare_yolo · train_yolo · eval_yolo · yolo_progress · track_yolo · diagnose_yolo_class
├── src/jevgate/
│   ├── pipeline.py   단계 실행 순서
│   ├── core/         config.py, runio.py (run 폴더·링크)
│   ├── data/         brackish.py (프레임·AAU), splits.py, boxes.py (IoU), labels.py (y_box, y_track)
│   ├── detectors/    yolo.py (데이터·학습·평가), track.py (YOLO + ByteTrack, track 요약)
│   ├── judges/       jev.py, vlm.py, gate.py
│   ├── evaluation/   metrics.py, evaluate.py, contracts.py
│   └── outputs/      plots.py, viz.py, report.py
├── src/tests/
├── docs/           PLAN.md, DATA.md
└── train/ data/ runs/ .env   (gitignore)
```

## 11. git 운영
- `main`: 검증 루프 통과 코드. `feat/<기능>`: 코드 작업. `exp/<YYMMDD>-<이름>`: config만 바꾸는 실험.

## 12. 마일스톤
| M | 내용 | 완료 기준 |
|---|---|---|
| M0 | DINO·GT 경로 제거, 새 config, track 요약, 라벨, Jev/VLM 질문, gate, 평가, 테스트 | ruff + pytest 통과 |
| M1 | smoke (dev 2개 영상) | 끝까지 실행, 계약 통과, 예산 안 |
| M2 | dev 전체 → `t`·`v`·`τ` 고정 | `thresholds.yaml` 커밋 |
| M3 | test 1회 | 결과 표·그림·`endpoints.json` |
| M4 | 영상, viz, 결과 카드 | ⑤ 육안 확인 |
| M5 (탐색) | detector 신뢰도 표 넣은 Jev, `class_fix` 분석 | M3 이후에만 |

## 13. 열린 이슈
- Jev 토큰 단가 미확인 → 토큰 수만 보고.
- test track 수가 수백 개 수준이라 CI가 넓을 수 있다 (conf 0.1로 track 수를 늘렸다).
- shrimp·jellyfish는 test 박스가 68·97개로 적다.
- ByteTrack ID switch(물고기 떼)로 한 track에 두 물체가 섞이면 track 판정이 일부 박스에서 틀린다 → 오류 분석에서 따로 센다.
- Jev는 YOLO가 놓친 물체를 되살리지 못한다. 고칠 수 있는 것은 거짓 track과 오분류뿐이다.

## 14. 변경 기록
- v1 (2026-10-13): Grounding DINO 후보 박스 이진 판정(real/fake), 클립 3개, endpoint 사전 등록.
- v2–v3 (2026-10-14): 7-way 클래스 판정, 원본 분할 전체 사용, Jev `need_vlm` 질문.
- v4 (2026-10-14): GT 박스 6-way 판정. test primary 실패 (호출 비율 0.999). 결과 폴더는 2026-10-15 삭제.
- **v6 (2026-10-15, 결과 보기 전)**: DINO·GT 방식 폐기. YOLO12s(영상 단위 분할) + ByteTrack track을 판정 단위로, Jev `track_ok` Noul 게이트, VLM은 track 대표 프레임 Yes/No(`v`는 dev에서 결정), dev = val 영상 13개 / test = test 영상 13개, run 폴더를 yolo·jev·vlm·evaluation으로 재구성.
