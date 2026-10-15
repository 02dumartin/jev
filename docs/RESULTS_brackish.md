# Brackish × Jev: 발표용 결과 정리 (2026-10-15)

> 계획: `docs/PLAN_brackish.md` (v7). 결과 폴더: v6 dev `runs/261015_dev`, v7 dev `runs/261015_dev_2`, v7 test `runs/261015_test`.
> 모든 test 수치는 dev에서 고정한 임계값(`thresholds.yaml`, commit d961a79)으로 **한 번만** 잰 값이다.

## 0. 한 줄 결론
**판단 근거가 이미지에 있는 태스크에서, 텍스트만 받는 Jev는 게이트 역할을 하지 못했다.** Jev가 받은 텍스트(YOLO 요약 숫자, 작은 VLM의 서술)보다 나은 판단을 만들지 못했고, 이 도메인에서는 calibration도 맞지 않아(ECE 0.19~0.28) "확신하면 넘기지 않는다"는 전제가 성립하지 않았다.

## 1. 데이터 분할이 detector 성능을 얼마나 부풀리나 (YOLO12s, test mAP50)

| 분할 | test mAP50 | test mAP50-95 |
|---|---|---|
| 원본 프레임 무작위 분할 (같은 영상의 이웃 프레임이 train/test에 섞임) | **0.990** | 0.840 |
| 영상 단위 분할 (test 영상은 학습에서 한 프레임도 안 봄) | **0.553** | 0.315 |

→ 이후 실험은 전부 영상 단위 분할 (train 63 / dev 13 / test 13 영상).

## 2. 실험 구조
```
YOLO12s (영상 단위 학습) → ByteTrack → track마다 "진짜 <클래스>인가?"
   v6: Jev ← track 요약 숫자 (길이, conf, 크기, 움직임)
   v7: 작은 VLM(Qwen3-VL-4B)이 보고 서술 → Jev ← 서술 + 요약
   게이트: Jev가 확실하면 확정, 애매하면 큰 VLM(Qwen3-VL-8B)
```
Primary endpoint (사전 등록): 게이트 vs 큰 VLM 전수 판정, 정확도 차이 CI 하한 ≥ −0.02 **그리고** 큰 VLM 호출 ≤ 50%.

## 3. 핵심 수치: 누가 맞는/틀린 track을 가르나 (track 단위 AUROC)

| 판단 근거 | v6 dev | v7 dev | **v7 test** |
|---|---|---|---|
| Jev (요약 숫자만) | 0.656 | – | – |
| Jev (요약 + 작은 VLM 서술) | – | 0.790 | **0.700** [0.57, 0.82] |
| 작은 VLM 자기 확률 (Jev에게 서술을 준 모델) | – | 0.829 | **0.713** [0.59, 0.83] |
| 큰 VLM 확률 | 0.798 | 0.811 | **0.696** [0.57, 0.82] |
| YOLO conf 평균 (Jev 입력에 들어 있음) | 0.710 | 0.710 | **0.787** [0.67, 0.89] |

- 서술을 넣자 Jev가 좋아졌다 (0.66 → 0.79), 하지만 **서술을 쓴 모델 자신의 확률을 넘지 못했다** (dev 0.79 < 0.83, test 0.70 < 0.71).
- test에서는 판정기 셋 다 YOLO conf 평균 하나보다 못했다.

## 4. 게이트 결과 (v7 test, track 70개)

| | Jev 게이트 (primary) | 작은 VLM 자기 확률 게이트 (대조군) |
|---|---|---|
| dev에서 정한 임계값 | τ = 1.01 (조건을 만족하는 τ 없음) | p ≥ 0.060 수용, ≤ 0.00055 거부 |
| 큰 VLM 호출 비율 | **1.00** (전부 넘김) | 0.40 |
| 혼자 처리한 track의 오류율 (목표 ≤ 0.05) | – (혼자 처리 0%) | dev 0.043 → **test 0.244** |
| 정확도 차이 vs 큰 VLM 전수 | +0.000 | +0.001 [0.000, 0.003] |
| Primary 판정 | **실패** | (대조군) |

- Jev는 거짓 track에 p 0.92, 맞는 track에 p 0.09를 주기도 해서, dev에서 "혼자 처리분 오류 ≤ 5%"를 만족하는 확신 구간이 없었다.
- 작은 VLM 게이트는 호출을 60% 줄였지만, **dev에서 정한 기준이 test로 넘어가지 않았다** (오류 4% → 24%). 정확도 차이가 0인 건 큰 VLM 자체가 약하기 때문이다.

## 5. Calibration (Jev p_ok vs 실제로 맞는 track 비율)
| | v6 dev | v7 dev | v7 test |
|---|---|---|---|
| ECE | 0.197 | 0.282 | 0.187 |
| Brier | 0.195 | 0.206 | 0.242 |

그림: `runs/261015_test/evaluation/plots/reliability.png`

## 6. 박스 단위 정확도 (v7 test, 7-way: 6종 + background, 채점 박스 3,203개)

| 시스템 | 정확도 | macro F1 | 큰 VLM 호출 |
|---|---|---|---|
| ① YOLO 프레임별 | 0.731 | 0.632 | 0 |
| ② track 다수결 | **0.738** | 0.609 | 0 |
| ③ 큰 VLM 전수 | 0.699 | 0.587 | 70 |
| ④ 작은 VLM 단독 | 0.602 | 0.572 | 0 |
| ⑤ Jev 단독 | 0.462 | 0.357 | 0 |
| ⑥ Jev 게이트 | 0.699 | 0.587 | 70 |
| ⑦ 작은 VLM 게이트 | 0.700 | 0.588 | 28 |

- 판정기를 붙인 모든 시스템이 YOLO + tracking만 쓴 것(②)보다 낮다.
- detector recall 상한 0.575: GT 동물의 43%는 YOLO가 아예 못 잡아서 어떤 판정기로도 되살릴 수 없다.

## 7. 속도와 비용 (B200 GPU, Jev 동시 요청 8개)
| 단계 | 시간 |
|---|---|
| YOLO + ByteTrack | 프레임당 중앙값 25 ms (test 2,051프레임 63초) |
| Jev | 요청 1개 중앙값 0.37초 (track 20개씩), **test track 70개 전체 0.66초**, track당 약 530 토큰 |
| 작은 VLM (4B) 서술 + 확률 | track 70개 22초 (+ 모델 로딩 27초) |
| 큰 VLM (8B) 확률 | track 70개 4.9초 (+ 모델 로딩 25초) |

→ **Jev는 실제로 빠르고 싸다** (병렬 요청 기준). 문제는 속도가 아니라 이 태스크에서의 판정 품질이었다.

## 8. dev → test 차이 (과적합/분포 차이 신호)
| | dev | test |
|---|---|---|
| 작은 VLM 단독 정확도 | 0.812 | 0.602 |
| 작은 VLM 게이트 자동 처리분 오류 | 0.043 | 0.244 |
| Jev AUROC | 0.790 | 0.700 |

v7은 v6 dev 결과를 보고 설계했고, dev 영상은 YOLO best epoch 선택에도 쓰였다. dev 숫자는 낙관적이고 결론은 test로만 낸다.

## 9. 발표 메시지
1. **Calibrated 결정 모델은 입력에 없는 정보를 만들지 못한다.** 게이트 성능의 상한은 증거가 정한다.
2. **Jev를 게이트로 쓸 때는 앞단 모델의 자기 확률과 반드시 비교한다.** (이번에는 대조군과 같거나 낮았다)
3. **RLCD calibration은 학습 분포 안의 이야기다.** 새 도메인에서는 재보정(재학습)이 필요하지만 API로는 할 수 없다.
4. Jev가 맞는 곳: 판단 근거가 처음부터 텍스트에 있고, 같은 판단을 대량으로 빠르게 반복하는 곳 → 다음 실험 (`docs/PLAN_text.md`).

## 10. 기록할 한계
- track 수가 적다 (test 채점 68개, 그중 틀린 track 28개). CI가 넓다.
- small_fish가 track의 대부분이다 (dev 92/117).
- test run의 데이터 계약 검사에서 YOLO가 낸 높이 0짜리 박스 1개(`0222-232724-0048-03`)가 걸렸다. 결과에 주는 영향은 무시할 만하고, test는 다시 돌리지 않았다.
- Jev 토큰 단가를 몰라 비용은 토큰 수로만 보고했다.

## 11. 그림·영상 위치
| 슬라이드 | 파일 |
|---|---|
| 분할 효과 | `train/yolo12s_orig/README.md`, `train/yolo12s_video/README.md` |
| ROC (Jev vs 작은/큰 VLM vs conf) | `runs/261015_test/evaluation/plots/roc.png` |
| calibration | `runs/261015_test/evaluation/plots/reliability.png` |
| τ 곡선 | `runs/261015_dev_2/evaluation/plots/tau_curve.png` |
| confusion | `runs/261015_test/evaluation/confusion/*.png` |
| 영상 (GT vs Jev 게이트) | `runs/261015_test/evaluation/videos/*.mp4` |
| 틀린 track 사례 (서술·이미지 포함) | `runs/261015_test/evaluation/tables/errors.csv` |
