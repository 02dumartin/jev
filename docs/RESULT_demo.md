> 최신 재계산 결과와 calibration: [Retail 결과](retail-results/report.md). 아래는 기존 데모 설명이며 확신 비율은 정확도가 아니다.

# Store and Next acts: 편의점 카메라 × Jev 데모 결과 (2026-10-15)

> 계획: `docs/PLAN_demo.md` · 결과 폴더: `runs/261015_retail-demo-v2/` (188개 클립 처리, 영상은 7개 클립)
> 영상: `runs/261015_retail-demo-v2/evaluation/dashboard.mp4` (1분 9초, H.264) · 스틸: `evaluation/stills/*.png`
> 이 문서의 목적은 **데모 설명**이다. 수치는 속도·비용·확신 비율 같은 참고값이고, 판단의 정답률을 잰 실험이 아니다.

---

## 0. 한 장 요약
```
편의점 천장 카메라 클립
   │
   ├─ VLM only:  Qwen3-VL-8B가 프레임을 보고 질문 6개에 바로 답함          → 1.5 초, 확률 없음
   │
   └─ VLM + Jev: Qwen3-VL-8B가 장면을 "관찰한 사실"로만 서술 (2.8 초)
                  → Jev가 그 서술 하나에 질문 6개를 한 번에 판단 (0.28 초, 확률 포함)
                  → 코드가 답과 확률로 행동을 정함 (애매하면 MANAGER CHECK)
```
- **Jev가 보여주는 것**: 한 state에 Boolean·Choice·Score 질문을 섞어서 요청 1번으로 0.28초에 판단하고, 질문마다 **얼마나 확신하는지**를 준다.
- **VLM only가 못하는 것**: 확률이 없어서 어떤 답이 애매한지 모른다. 188개 클립 중 173개를 그냥 "NO ACTION"으로 처리했다 (Jev 경로는 83개를 사람 확인으로 넘김).

## 1. 사용한 데이터셋
| 항목 | 내용 |
|---|---|
| 이름 | **RetailAction** (Standard AI, ICCV 2025 Retail Vision Workshop) |
| 내용 | 실제 편의점 여러 곳의 **천장 어안 카메라** 영상. 얼굴은 데이터셋에서 이미 블러 처리됨 |
| 규모 (전체) | 약 41시간, 샘플 21,000개 × 카메라 2대. 샘플마다 손님이 상품을 **take / put / touch** 한 시점과 위치 주석 |
| 이번에 쓴 것 | test split에서 장면이 다양하게 섞이도록 고른 **188개 클립** (클립당 최대 32프레임, 6fps 재생, 약 5초) |
| 영상에 넣은 것 | 그중 **7개**: 001404 · 002429 · 002491 · 000421 · 000134 · 000951 · 000953 |
| 라이선스 | Standard.AI Dataset License: 연구·시연 사용 가능, 출처 표기와 라이선스 사본 동봉 (`data/retail-demo/retailaction/LICENSE`, `NOTICE.txt`) |
| 받은 곳 | Hugging Face 미러 `Voxel51/RetailAction` (원본 `standard-cognition/RetailAction`) |

## 2. 방식

### 2.1 전처리
- **회전**: 천장 어안 카메라라 사람이 옆으로 눕거나 뒤집혀 보인다. 데이터셋의 자세 키포인트로 엉덩이 → 머리 방향을 구해, 사람이 서 있도록 90° 단위로 돌렸다. 188개 중 139개를 회전했고, 자세 정보가 없는 8개는 원래 방향 그대로 두었다.
- **알려진 행동**: 데이터셋의 take/put/touch 주석(시점, 손이 닿은 위치)을 VLM 입력 프레임에 노란 원으로 표시하고 프롬프트에도 적었다. **매장의 행동 검출기를 대신하는 입력**이며, 화면에 "dataset annotation"이라고 밝혔다.
  - 이유: 30개 클립으로 시험해 보니 VLM 혼자서는 "상품을 가져갔나"를 정확도 0.53~0.60(전부 "가져감"으로 찍은 기준 0.60)으로밖에 보지 못했다. 손 동작이 어안 화면에서 몇 픽셀에 불과하기 때문이다.

### 2.2 두 경로 (같은 프레임 8장, 같은 행동 정보)
| | VLM only | VLM + Jev |
|---|---|---|
| ① 보기 | Qwen3-VL-8B가 프레임 8장을 봄 | Qwen3-VL-8B가 프레임 8장을 봄 |
| ② 출력 | 질문 6개의 답을 JSON으로 바로 생성 | **판단 없이 관찰한 사실만** 장면 설명(STATE, JSON)으로 생성: 구역, 인원, 아이 동반, 냉장고 문, 손님 움직임, 들고 있는 것, 다룬 상품, 진열 상태, 바닥 장애물, 한 줄 메모 |
| ③ 판단 | (②에서 끝) | **Jev**가 STATE 하나에 질문 6개를 **요청 1번**으로 판단, 질문마다 확률 |
| ④ 행동 | 같은 코드 규칙 (확률이 없으므로 "애매함" 규칙은 적용되지 않음) | 코드 규칙 (아래 3.2) |

- 역할 분리: **VLM = 무엇이 보이나**, **Jev = 그래서 무엇을 판단하나**, **코드 = 무엇을 실행하나**.

### 2.3 장면 설명(STATE) 예시 (클립 002491)
```json
{"area": "other", "people_count": 2, "children_present": false, "fridge_door_open": null,
 "shopper_movement": "standing still", "carrying": "products in hands", "products_handled": [],
 "shelf_condition": "tidy", "floor_obstruction": "none",
 "notes": "A person in a grey jacket stands over a shopping cart, looking at their phone, while another person in a yellow shirt is partially visible near a fridge."}
```

## 3. Jev에 넘긴 질문 (Multiple questions, 요청 1번)

### 3.1 질문 6개 (`configs/retail.yaml` `questions`)
| 카드 | 형식 | Jev에 보낸 문장 | 선택지 |
|---|---|---|---|
| Shopper needs help? | **Boolean** (noul) | Does a shopper appear to need staff help right now (searching, hesitating for long, struggling)? | yes / no |
| Product taken? | **Boolean** | Did the shopper end up keeping a product (picked up and not put back)? | yes / no |
| Fridge door open? | **Boolean** | Is a fridge door open, or left open without anyone using it? | yes / no |
| Crowding | **Score** | How crowded is this area? | empty · one shopper · a few shoppers · busy · blocked |
| Shopper activity | **Choice** | What is the main shopper doing? | browsing · picking products · returning product · passing through · no shopper |
| Staff action | **Choice** | What should a staff member do next? | none · offer help · close fridge · tidy shelf · restock check · keep watching |

- state: `{"camera": "convenience store ceiling camera", "observation": <장면 설명 JSON>}`
- 가장 높은 확률이 **0.75 미만인 카드는 "unsure"**(노란 테두리)로 표시한다.

### 3.2 코드가 정하는 행동 (위에서부터 처음 맞는 규칙)
| 순서 | 조건 | 행동 배지 |
|---|---|---|
| 1 | Fridge door open의 yes 확률 ≥ 0.75 | CLOSE FRIDGE |
| 2 | Shopper needs help의 yes 확률 ≥ 0.75 | OFFER HELP |
| 3 | 주요 카드 5개(Staff action 제외) 중 하나라도 unsure | **MANAGER CHECK** (Jev의 확률이 사람 확인 여부를 정함) |
| 4 | 그 외 | Staff action에서 Jev가 고른 행동 (NO ACTION / KEEP WATCHING / OFFER HELP / RESTOCK CHECK …) |

처음에는 "사람이 봐야 하나?"를 질문으로 따로 물었지만, 그 질문 자체가 애매하게 답해져서(확신 55%) 뺐다. 대신 **Jev가 이미 주는 확률**로 규칙을 만들었다.

## 4. 토큰 · 비용 · 시간

### 4.1 188개 클립 전체 (중앙값)
| 항목 | VLM only | VLM + Jev |
|---|---|---|
| VLM 생성 | **1.5 초** (답 6개) | **2.8 초** (장면 설명) |
| Jev 판단 | – | **0.28 초** (상위 10% 0.36 초), 요청 **1번** |
| 클립 하나 전체 | **1.5 초** | **3.1 초** |
| 확률(확신도) | 없음 | 질문마다 있음 |
| 사람에게 넘긴 클립 (MANAGER CHECK) | 0 / 188 | 83 / 188 |

- 시간은 모두 B200 GPU 한 장 기준이고, **모델 로딩 시간(약 25초, 한 번만 듦)은 제외**했다. Jev는 TypeSafe API까지의 왕복 시간이다.
- 영상의 빨간 박스 `Time`은 VLM only = 1.5 초 안팎, VLM + Jev = **Jev 시간만**이다. 장면 설명 시간은 왼쪽 "Scene description" 패널에 따로 적었다.

### 4.2 공정한 비교: VLM only도 설명을 쓰고 판단하면 (클립 10개 측정)
| VLM only 방식 | 시간 (중앙값) |
|---|---|
| 바로 답하기 | 1.7 초 |
| 장면 설명을 쓰고 이어서 답하기 (생성 1번) | **6.1 초** |
| (참고) VLM 설명 + Jev | 2.8 + 0.28 ≈ **3.1 초** |

VLM은 글자를 하나씩 생성하므로 쓸 글이 길수록 느리다. **설명을 쓰고 판단한다는 조건을 맞추면 VLM + Jev가 약 2배 빠르다.** 그리고 설명은 한 번만 쓰면 되므로, 질문을 더 붙여도 **추가 비용은 Jev 요청 시간(약 0.3초)뿐**이다.

### 4.3 Multiple questions vs 질문을 하나씩 (같은 질문 6개)
| | 요청 1번에 6개 | 6번 따로 (동시에 전송) |
|---|---|---|
| 요청 수 | **1** | 6 |
| 토큰 (클립당 중앙값) | **873** (입력 665 + 출력 208) | 2,941 (**3.4배**) |
| 시간 (중앙값) | **0.28 초** | 0.39 초 |
| 188개 전체 토큰 | **164,177** | 553,394 |

**Multiple questions의 이득은 주로 토큰과 요청 수다.** state를 한 번만 보내기 때문이다. 시간도 짧지만, 따로 보낸 요청도 동시에 처리되므로 차이는 0.1초 정도다.

### 4.4 비용
- **Jev**: TypeSafe 단가를 확인하지 못해 **토큰 수로만** 보고한다. 클립당 873토큰, 188개 전체 16만 토큰. 단가가 확인되면 총 토큰 수에 적용해 별도로 환산해야 한다.
- **VLM**: 로컬 GPU(B200)라 API 비용은 없다. GPU 시간으로 보면 188개 기준 장면 설명 525초, VLM only 답변 272초다.

## 5. 영상에 넣은 7개 클립
| # | 클립 | 구역 (VLM) | 알려진 행동 | Jev 판단 (주요 답, 확률) | Jev 경로 행동 | VLM only 행동 |
|---|---|---|---|---|---|---|
| 1 | 001404 | drinks fridge | take | 가져감 YES 0.88, picking products 0.99, restock check 0.67 | **RESTOCK CHECK** | NO ACTION |
| 2 | 002429 | snack shelf | take | 가져감 YES 0.94, browsing 0.79, none 0.55 | NO ACTION | NO ACTION |
| 3 | 002491 | other | – | 도움 NO 0.87, keep watching 0.34 (애매) | KEEP WATCHING | CLOSE FRIDGE (냉장고 문 열림으로 잘못 답함) |
| 4 | 000421 | drinks fridge | – | 가져감 YES 0.68 (애매), browsing 0.71 (애매) | **MANAGER CHECK** | NO ACTION |
| 5 | 000134 | snack shelf | – | 도움 NO 0.84, offer help 0.49 (애매) | OFFER HELP | NO ACTION |
| 6 | 000951 | other | touch | 가져감 YES 0.56 (애매), 냉장고 NO 0.68 (애매) | **MANAGER CHECK** | NO ACTION |
| 7 | 000953 | snack shelf | – | **도움 YES 0.78**, offer help 1.00 | **OFFER HELP** | NO ACTION |

클립당 Jev 0.20~0.35 초, 토큰 861~877. 같은 질문을 따로 보내면 2,870~2,971 토큰이다.

## 6. 188개 전체에서 본 질문별 확신 비율 (top 확률 ≥ 0.75)
| 질문 | 확신 비율 |
|---|---|
| Shopper needs help? | 1.00 |
| Crowding | 0.99 |
| Fridge door open? | 0.89 |
| Shopper activity | 0.77 |
| Product taken? | 0.75 |
| Staff action | 0.06 |

- 사실에 가까운 질문(혼잡도, 냉장고)은 확신이 높고, **행동을 고르는 질문(Staff action)은 거의 항상 애매하다.** STATE만으로는 "지켜볼지, 다가갈지"를 가를 근거가 부족하기 때문이고, 이것이 "애매하면 사람에게"를 보여주는 장면이 된다.
- 행동 분포 (Jev 경로): MANAGER CHECK 83 · NO ACTION 57 · CLOSE FRIDGE 23 · OFFER HELP 15 · KEEP WATCHING 9 · RESTOCK CHECK 1.
- 두 경로가 같은 행동을 고른 비율: 36%.

## 7. 발표에서 밝힐 한계
1. **take/put/touch는 데이터셋 주석을 넣은 것이다.** 실제 시스템에서는 행동 검출기가 필요하다. 그래서 "Product taken?"과 주석의 일치율 0.88은 참고값이다.
2. **판단의 정답은 없다.** "도움 제안", "매니저 확인"이 맞는지는 아무도 채점하지 않았다. 예를 들어 002491(휴대폰을 보며 물건을 찾는 장면)에서 Jev는 "도움 필요 NO"로 봤다. 판단 기준을 바꾸려면 질문 문장에 기준을 써야 한다.
3. **Jev의 판단은 VLM의 장면 설명을 넘지 못한다.** VLM이 구역을 "other"로 적거나 상황을 놓치면 Jev도 따라간다.
4. MANAGER CHECK 비율(44%)은 확신 기준 0.75와 대상 카드 선택에 따라 크게 달라진다. 운영에 쓰려면 기준을 정하는 별도 검증이 필요하다.
5. 회전은 자세 정보가 있는 클립에서만 정확하고, 시간은 B200 GPU 한 장 기준이다.

## 8. 재현
```bash
uv run python scripts/prepare_retail.py                       # 회전·행동 메타 (meta.jsonl)
uv run python scripts/run_retail.py --set run.name=retail-demo-v2 reuse_states_from=runs/261015_retail-demo
uv run python scripts/render_retail.py --run-dir runs/261015_retail-demo-v2   # 영상만 다시 그리기
```
- 설정: `configs/retail.yaml` (질문, 행동 규칙, 화면, 포함·제외 클립)
- 코드: `src/jev/` (meta, clip, observe, decide, render, pipeline)
- 입출력 원본: `runs/261015_retail-demo-v2/{vlm,vlm_only,jev,actions}/`
- 장면 설명은 `runs/261015_retail-demo`에서 만든 것을 재사용했다 (같은 모델, 같은 프레임, 같은 프롬프트).
