# Retail demo: 편의점 카메라 → VLM 관찰 → Jev 다중 판단 → 행동 (2026-10-15)

> 목적은 **실험이 아니라 데모**다. Jev의 강점인 "한 state에 여러 typed question을 한 번에, 빠르게, 확률과 함께"를
> 눈으로 보여준다. 정량 수치(속도, 확신 비율)는 부차적인 참고용이다.

## 1. 역할 분리
| 단계 | 담당 | 하는 일 | 하지 않는 일 |
|---|---|---|---|
| ① 관찰 | VLM (Qwen3-VL-8B) | 프레임 8장을 보고 **관찰한 사실만** STATE(JSON)로 적는다 | 판단·조언 |
| ② 판단 | Jev (요청 1번) | 같은 STATE에 질문 6개(Boolean 3, Choice 2, Score 1)를 동시에 판단, 확률 포함 | 이미지 보기 |
| ③ 실행 | 코드 (`configs/retail.yaml` `actions`) | 답과 확률로 행동 배지 하나를 정한다 | 의미 판단 |


## 2. 데이터
- RetailAction (Standard AI, ICCV 2025 워크숍): 실제 편의점 천장 카메라, 얼굴 블러. test split에서 188개 클립 (`data/retail-demo/retailaction/selection.json`). 라이선스는 Standard.AI Dataset License (출처 표기, 라이선스 사본 동봉).
- **회전**: 천장 어안 카메라라 사람이 옆으로 눕거나 뒤집혀 보인다. 데이터셋의 자세 키포인트(엉덩이 → 머리 방향)로 90° 단위 회전을 정해 사람이 서 있는 화면으로 만든다 (`scripts/prepare_retail.py` → `meta.jsonl`). 자세가 없는 클립은 회전하지 않는다.
- **알려진 행동**: 데이터셋의 take / put / touch 주석(시점, 손이 닿은 위치)을 VLM 프롬프트와 화면에 넣는다. 매장의 행동 검출기 역할을 대신하는 것이고, 화면에 "dataset annotation"이라고 표시한다. (30개 클립 시험에서 VLM 혼자서는 "상품을 가져갔나"를 0.53~0.60 정확도로밖에 못 봤다.)

## 3. 질문 (`configs/retail.yaml` `questions`)
| 카드 | 형식 |
|---|---|
| Shopper needs help? | Boolean |
| Product taken? | Boolean |
| Fridge door open? | Boolean |
| Crowding | Score (empty … blocked) |
| Shopper activity | Choice (browsing / picking / returning / passing / no shopper) |
| Staff action | Choice (none / offer help / close fridge / tidy / restock check / keep watching) |

- 카드의 가장 높은 확률이 0.75 미만이면 노란 테두리 "unsure".
- 행동 규칙: 냉장고 문 → CLOSE FRIDGE, 도움 필요 → OFFER HELP, 주요 질문 5개 중 하나라도 확신 0.75 미만 → MANAGER CHECK, 그 외에는 Jev의 Staff action 선택을 배지로 (애매하면 배지에 표시).
- 같은 질문 6개를 따로 6번 요청한 결과도 함께 재서 요청 수·토큰·시간을 화면에 나란히 보여준다.

## 4. 결과물 (`runs/<YYMMDD>_retail-demo/`)
- `evaluation/dashboard.mp4`: 현재 설정에서 지정한 클립 7개. 왼쪽 영상(알려진 행동 원·타임라인), 오른쪽 STATE → Jev 카드 → 배지 순서로 나타남. H.264.
- `evaluation/calibration/`: Brier, NLL, ECE, reliability bins, coverage/risk, 클립별 CSV 및 보고서.
- `evaluation/stills/`: 클립별 마지막 화면 PNG (슬라이드용).
- `evaluation/summary.json`: 속도(VLM, Jev 1요청 vs 6요청), 토큰, 질문별 확신 비율, 행동 분포, "Product taken?"과 주석의 일치율 (주석을 입력에 줬으므로 참고용).
- `vlm/`, `jev/`, `actions/`: 입력과 출력 원본 (재현용).

## 5. 실행
```bash
uv run python scripts/prepare_retail.py          # meta.jsonl (회전, 행동)
uv run python scripts/run_retail.py              # 전체 188개 + 영상
uv run python scripts/run_retail.py --set run.max_clips=5 run.name=retail-demo-smoke render.clips_in_video=3 render.include=[] render.fill_others=true
```

## 6. 한계 (화면과 발표에서 밝힐 것)
- 행동(take/put/touch)은 데이터셋 주석을 넣은 것이다. 실제 시스템에서는 행동 검출기가 필요하다.
- Jev의 판단은 VLM이 적은 STATE를 넘지 못한다 (VLM이 구역을 잘못 보면 Jev도 따라간다).
- 회전은 자세 정보가 있는 클립에서만 정확하다.
