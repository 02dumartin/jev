# jev-gate 실험 계획 (v7, 2026-10-15)

> 질문: 작은 VLM이 track을 **보고 텍스트로 서술**하면, Jev가 그 서술과 track 요약으로 "진짜 그 동물인가"를
> calibrated 확률로 판단해서 **확실한 track은 바로 확정하고 애매한 track만 큰 VLM에 넘길 수 있는가?**
> 즉, 모든 track을 큰 VLM이 보는 것과 같은 정확도를 **큰 VLM 호출을 줄이면서** 얻을 수 있는가?
>
> v7에서 바뀐 것 (v6 dev 결과로 결정, test는 보지 않음, §14):
> - v6의 Jev는 텍스트 요약 숫자만 봐서 맞는/틀린 track을 거의 못 갈랐다 (track AUROC 0.66 < YOLO conf 평균 0.71).
>   판단 근거가 픽셀에 있으므로 **작은 VLM(Qwen3-VL-4B)이 본 것을 텍스트로 바꿔 Jev에 넣는다.**
> - track에 들지 못한 박스는 지우지 않고 ①과 같은 규칙(conf ≥ t면 YOLO 클래스)으로 남긴다 (v6 dev에서 진짜 동물 596개 손실).

---

## 1. 구성 요소

| 구성 | 무엇 | 입력 | 출력 |
|---|---|---|---|
| Detector | YOLO12s `train/yolo12s_video/weights/best.pt` (영상 단위 분할로 학습) | 프레임 | 프레임별 박스, 클래스(6종), conf |
| Tracker | ByteTrack (Ultralytics), conf 0.1 | detector 박스 | 박스마다 track id (또는 null) |
| **Describer (작은 VLM)** | Qwen3-VL-4B-Instruct | track 대표 프레임: 전체 프레임 + 빨간 박스, 확대 crop | ① 박스 안을 서술한 짧은 텍스트, ② 같은 질문의 Yes 확률 `p_small` |
| Jev | TypeSafe API `jev-latest`, **텍스트만 받는다** | track 요약 (§4) + **describer 서술** | `track_ok` p (Noul) |
| Verifier (큰 VLM) | Qwen3-VL-8B-Instruct | describer와 같은 이미지 2장 | 같은 질문의 Yes 확률 `p_big` |

### 1.1 클래스
`fish · small_fish · crab · shrimp · jellyfish · starfish` (AAU 태그). 박스를 지우면 `background`.

### 1.2 track, 다수결 클래스, track 밖 박스
- track의 **다수결 클래스** = track 박스들에서 가장 많이 나온 YOLO 클래스 (동점이면 conf 합이 큰 쪽).
- **판단은 track 단위, 결과는 박스에 물려준다.** 수용된 track의 모든 박스 = 다수결 클래스, 거부된 track = background.
- **track에 들지 못한 박스(`track_uid: null`)**는 ByteTrack이 정한다 (conf 0.25 미만이라 새 track을 못 열었거나, 첫 프레임이라 미확정, 물고기 떼에서 연결 실패). 판정기에는 가지 않고 ①과 같은 규칙으로 처리한다: conf ≥ `t`면 YOLO 클래스, 아니면 background. 모든 시스템(②~⑦)에 똑같이 적용한다.

## 2. 데이터와 정답

### 2.1 분할 (`configs/splits/video_split.yaml`, 고정)
| 용도 | 영상 | 프레임 | 쓰임 |
|---|---|---|---|
| train | 63 | 10,593 | YOLO 학습만 |
| **dev** | 13 (= YOLO val 영상) | 2,030 | `t`, `v`, `τ` 등 결정, 프롬프트 점검 |
| **test** | 13 | 2,051 | 설정 고정 후 한 번만 평가 |

- 한 영상은 한 분할에만 들어간다. 원본 프레임 무작위 분할은 쓰지 않는다 (`yolo12s_orig` test mAP50 0.99 vs `yolo12s_video` 0.55).
- dev 영상은 YOLO best epoch 선택에도 쓰였다 → 각주로 밝힌다.
- v6 dev 결과(`runs/261015_dev`)를 보고 v7을 설계했다. 그래서 v7 dev 숫자는 낙관적일 수 있고, 결론은 test로만 낸다.

### 2.2 박스 정답 y_box
AAU CSV와 클래스 무관 최대 IoU: ≥ 0.5면 그 GT 클래스, < 0.3이면 `background`, 그 사이는 `ambiguous`(채점 제외, 개수 보고). 어떤 박스와도 IoU ≥ 0.5로 맞지 않은 GT는 **놓친 물체**(recall 상한, 각주).

### 2.3 track 정답 y_track (calibration·임계값용)
y_track = 1: 채점 대상 박스의 절반 이상이 다수결 클래스의 GT와 맞음. 0이면 유형을 `false`(대부분 background) / `misclassified`로 기록. 채점 박스가 없으면 null.

## 3. 비교할 시스템 (표의 행)

| # | 시스템 | track 판정 | 큰 VLM 호출 | 정할 값 (dev) |
|---|---|---|---|---|
| ① | `yolo` | track 없이 프레임별: conf ≥ `t`면 YOLO 클래스 | 0 | `t` (박스 정확도 최대) |
| ② | `track_vote` | 모든 track 수용 | 0 | – |
| ③ | **`vlm_all` (기준)** | 모든 track을 큰 VLM에: `p_big` ≥ `v` | track 수 | `v` (track 정확도 최대) |
| ④ | `small_vlm` | 작은 VLM 단독: `p_small` ≥ `v_s` | 0 | `v_s` |
| ⑤ | `jev` | Jev 단독: p ≥ 0.5 | 0 | – (0.5 고정) |
| ⑥ | **`gate` (primary)** | Jev가 확실하면(max(p, 1−p) ≥ τ) Jev 판정, 아니면 ③의 판정 | 넘긴 track 수 | `τ` |
| ⑦ | `gate_small` | 작은 VLM 자기 확률로 게이트: `p_small` ≥ `hi`면 수용, ≤ `lo`면 거부, 그 사이는 ③ | 넘긴 track 수 | `lo`, `hi` |

- **⑦은 "Jev가 꼭 필요한가"를 보는 대조군**이다. 작은 VLM의 확률만으로 같은 게이트를 만들 수 있다면 Jev의 기여는 없다.
- 게이트 임계값(τ, lo/hi)은 dev에서 "자동 처리한 track의 오류율 ≤ 0.05" 중 자동 처리가 가장 많은 값으로 정한다. 없으면 전부 큰 VLM.
- ⑥⑦은 ③④⑤의 판정 파일을 조합한다 (추가 실행 없음). 작은 VLM은 ④⑥⑦에서 모든 track에 대해 실행된 것으로 시간을 센다.
- describer, Jev, 큰 VLM은 **같은 질문 문장**을 쓴다: "Is this track a real `<다수결 클래스>`?"

## 4. 입력

### 4.1 Describer (작은 VLM)
- 대표 프레임 = conf가 가장 높은 박스의 프레임. 이미지 2장: 전체 프레임 + 빨간 박스, 박스 주변 확대 crop (긴 변 336px). 큰 VLM도 같은 이미지를 받는다.
- **서술 프롬프트**: 다수결 클래스를 알려주지 않는다 (서술이 detector 판단에 끌려가지 않게). "빨간 박스 안에 무엇이 보이는지 1~2문장으로: 동물이 있는가, 있다면 모양·크기·색·질감, 흐리거나 불분명하면 그렇다고." greedy, 최대 60토큰.
- **Yes 확률 `p_small`**: 같은 이미지에 공통 질문(+ Answer Yes or No)으로 forward 1번.

### 4.2 Jev state (track 하나)
| 그룹 | 필드 |
|---|---|
| **본 것** | `visual_description` (describer 서술, 그대로) |
| 길이 | frames_detected, span, gap, 영상 시작·끝에 걸림 |
| 클래스 | 클래스별 투표, 다수결 클래스, 비율 |
| conf | 평균, 최소, 최대, 추세 |
| 박스 | median 가로·세로 px, 중심 (x, y), 경계 접촉 비율 |
| 움직임 | 전체 이동량 / 박스 대각선, 프레임당 이동 px |
| 장면 | 같은 시간대의 다른 track 수 (클래스별) |

- 질문: `track_ok` Noul 하나 ("Is this track a real `<cls>`?", true = 그 동물 / false = 다른 동물이거나 동물 아님). v6의 `class_fix`는 뺐다 (dev 23개 중 0개 적중).
- 요청 단위: 영상 하나의 track을 `jev.max_tracks_per_request`개씩.

### 4.3 큰 VLM
v6와 같다. 다음 토큰 Yes/No logit softmax = `p_big`. Qwen3-VL은 이 영상에서 Yes를 거의 안 하므로 0.5가 아니라 dev에서 정한 `v`로 자른다.

## 5. 지표

### 5.1 Endpoint (결과를 보기 전에 고정, 2026-10-15)
test 13개 영상에서, dev에서 고정한 임계값으로 **한 번만** 측정한다.

**Primary: ⑥ `gate`는 ③ `vlm_all` 대비 정확도를 유지하면서 큰 VLM 호출을 줄이는가**
| 지표 | 정의 | 성공 기준 |
|---|---|---|
| 정확도 차이 | Δacc = acc(⑥) − acc(③), 박스 단위 7-way, 95% CI는 track cluster bootstrap | CI 하한 ≥ **−0.02** |
| 큰 VLM 호출 비율 | ⑥이 큰 VLM에 넘긴 track 수 / 전체 track 수 | ≤ **0.5** |

순차 검정 (처음 실패에서 멈춤, 재실행 없음): ① −0.02 & 0.5 → ② −0.01 & 0.5 → ③ −0.01 & 0.3. 실제 값은 항상 보고한다.

**Secondary**
| # | 질문 | 지표 |
|---|---|---|
| S1 | **Jev가 작은 VLM보다 나은 게이트인가** (⑥ vs ⑦) | 두 게이트의 큰 VLM 호출 비율과 Δacc, track AUROC (Jev p vs `p_small`) |
| S2 | Jev p를 믿을 수 있나 | track ECE (15 bin), Brier, reliability diagram |
| S3 | dev 게이트 기준이 test에서도 지켜지나 | test 자동 처리분 오류율 vs 0.05 |
| S4 | 누가 맞는/틀린 track을 잘 가르나 | track ROC + AUROC [95% CI]: Jev p, `p_big`, `p_small`, YOLO conf 평균 |
| S5 | detector를 얼마나 고치나 (①→②→⑥) | 박스 정확도 → macro F1 → 클래스별 P/R, confusion |
| S6 | 속도·비용 | 단계별 시간, track당 지연 중앙값, Jev 토큰, GPU-초 |
| S7 | 영상 안정성 | 프레임 간 박스 수 변화 (깜빡임) |

**탐색**: Jev에 서술을 빼고 넣은 차이 (v6 dev vs v7 dev), ambiguous를 background로 넣은 민감도.

### 5.2 지표 정의 · 신뢰구간
- 정확도: 박스 단위 pred = y_box (ambiguous 제외), pred ∈ 6종 ∪ {background}. macro F1은 정확도 다음.
- detection recall: GT 물체 중 최종적으로 올바른 클래스로 남은 박스와 맞는 비율 (놓친 GT 포함).
- 시간: ⑥ = 작은 VLM(서술 + p_small) + Jev + 넘긴 track의 큰 VLM. ③ = 큰 VLM 전부. 모델 로딩 시간은 따로 보고.
- 열 순서: 정확도 → macro F1 → Precision → Recall → ECE → 큰 VLM 호출 → 지연 → 총 시간 → 비용.
- CI: track(그 박스 전체)과 영상별 track 밖 박스 묶음을 cluster로 1,000회 bootstrap.

## 6. run 폴더 (`runs/<YYMMDD>_<이름>/`)
```text
├── README.md, run.yaml
├── data/              (링크) images, annotations, video_split.yaml
├── yolo/        input/ weights.pt (링크), tracker.yaml (링크), videos.txt   output/ detections.jsonl, tracks.jsonl
├── describer/   input/ prompts.jsonl, images/<track>_{frame,crop}.jpg       output/ decisions.jsonl (서술, p_small)
├── jev/         input/ requests.jsonl                                       output/ responses.jsonl, decisions.jsonl
├── vlm/         input/ prompts.jsonl (이미지는 describer/input/images 공유) output/ decisions.jsonl (p_big)
└── evaluation/  labels/, judgments.jsonl, gate_decisions.jsonl, tables/, confusion/, plots/, viz/, videos/
```
- 영상: 왼쪽 GT, 오른쪽 ⑥ gate. 박스 라벨에 `[Jev]`·`[VLM]`·`[untracked]`. 색 = 라벨(`viz.class_colors`), 선 모양 = 정답 여부.

## 7. 공정성 규칙
1. 모든 임계값은 **dev에서만** 정하고 `thresholds.yaml`에 고정한다. test는 이 파일만 읽고, dev 영상 목록이 다르면 실행을 거부한다.
2. 모든 시스템은 같은 `detections.jsonl`·`tracks.jsonl`을 쓴다.
3. describer, Jev, 큰 VLM은 같은 질문 문장과 같은 클래스 정의를 쓴다.
4. config 변경 의도 질문은 `/jev-loop` ⓪이 묻는다.

## 8. 개발 검증 루프 (`/jev-loop`)
```
⓪ 변경 의도 → ① 정적 검사 → ② 데이터 계약 → ③ smoke E2E → ④ 지표 회귀 → ⑤ 사람 육안 → ⑥ PR 기록
```
호출할 때 한 번 도는 스킬이다 (백그라운드 프로세스 아님). 세부는 `.claude/skills/jev-loop/SKILL.md`.

## 9. 레포 구조
```
jev-gate/
├── configs/        base.yaml, smoke.yaml, test.yaml, yolo*.yaml, splits/video_split.yaml
├── scripts/        run_pipeline · fit_thresholds · check_contracts · compare_metrics · config_guard · render_videos
│                   make_video_split · prepare_yolo · train_yolo · eval_yolo · yolo_progress · diagnose_yolo_class
├── src/jevgate/
│   ├── pipeline.py   단계 실행 순서
│   ├── core/         config.py, runio.py (run 폴더·링크)
│   ├── data/         brackish.py, splits.py, boxes.py, labels.py
│   ├── detectors/    yolo.py (학습·평가), track.py (YOLO + ByteTrack, track 요약)
│   ├── judges/       vlm.py (describer + verifier), jev.py, gate.py
│   ├── evaluation/   metrics.py, evaluate.py, contracts.py
│   └── outputs/      colors.py, plots.py, viz.py, report.py
├── src/tests/
├── docs/           PLAN.md, DATA.md
└── train/ data/ runs/ .env   (gitignore)
```

## 10. git 운영
`main`: 검증 루프 통과 코드. `feat/<기능>`: 코드 작업. `exp/<YYMMDD>-<이름>`: config만 바꾸는 실험.

## 11. 마일스톤
| M | 내용 | 완료 기준 |
|---|---|---|
| M0 | describer, Jev state에 서술, track 밖 박스 규칙, ④⑦ 시스템, 테스트 | ruff + pytest |
| M1 | smoke (dev 2개 영상) | 끝까지 실행, 계약 통과 |
| M2 | dev 전체 → 임계값 고정 | `thresholds.yaml` 커밋 |
| M3 | test 1회 | 결과 표·그림·`endpoints.json` |
| M4 | 영상, viz, 결과 카드 | 사람 육안 확인 |

## 12. 열린 이슈
- Jev 토큰 단가 미확인 → 토큰 수만 보고.
- track 수가 dev 127개 수준이라 CI가 넓다. 틀린 track은 dev 23개뿐이다.
- 작은 VLM(4B)과 큰 VLM(8B)이 같은 계열이라 같은 실수를 공유할 수 있다 → 게이트 이득이 작을 수 있다.
- describer 서술은 작은 VLM의 판단을 이미 담고 있다 → ⑦과의 비교(S1)로 Jev의 추가 기여를 따로 본다.
- Jev는 YOLO가 놓친 물체를 되살리지 못한다.

## 13. 산출물 → 발표
| 발표 요소 | 출처 |
|---|---|
| 결과 표 | test `evaluation/tables/metrics.md` |
| "텍스트 요약만으로는 안 된다" | v6 dev (`runs/261015_dev`): Jev track AUROC 0.66 |
| calibration | `evaluation/plots/reliability.png`, `tau_curve.png` |
| Jev vs 작은 VLM 게이트 | ⑥ vs ⑦ 행, `roc.png` |
| 영상 | `evaluation/videos/<video>.mp4` (GT | gate) |

## 14. 변경 기록
- v1 (2026-10-13): Grounding DINO 후보 이진 판정, endpoint 사전 등록.
- v2–v3 (2026-10-14): 7-way 판정, 원본 분할, Jev `need_vlm`.
- v4 (2026-10-14): GT 박스 6-way. test primary 실패 (호출 비율 0.999). 결과 삭제.
- v6 (2026-10-15): YOLO12s + ByteTrack track 단위, Jev는 텍스트 요약만. **dev만 실행, test 생략.** dev: Jev track AUROC 0.66 (YOLO conf 평균 0.71, 큰 VLM 0.80), τ = 0.81에서 Jev 자동 처리 2.4%, 호출 비율 0.976. track 밖 박스 1,242개 중 진짜 동물 596개 손실.
- **v7 (2026-10-15, v6 dev만 보고 test 전)**: 작은 VLM 서술을 Jev state에 추가, 큰 VLM 호출 절약을 primary로, 작은 VLM 자기 확률 게이트(⑦) 대조군, track 밖 박스는 ① 규칙으로 유지. `class_fix` 제거.
