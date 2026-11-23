# Retail 코드 안내

참고 구조: [tomato-robot/src](https://github.com/NVA-KIST/tomato-robot/tree/main/src),
[TomatoVision3D](https://github.com/NVA-KIST/TomatoVision3D)의 실행 스크립트 / 기능별 모듈 분리.
패키지 import는 `jevgate`로 유지하고 모든 실행 명령은 저장소 루트에서 수행한다.

| 실행 파일 | 역할 | 핵심 모듈 |
|---|---|---|
| `scripts/prepare_retail.py` | 데이터셋 주석·회전 메타 생성 | `preprocess/metadata.py`, `preprocess/frames.py` |
| `scripts/run_retail.py` | 관찰 → 판단 → 행동 → 평가 → 영상 | `pipeline.py` |
| `scripts/evaluate_retail.py` | 저장된 결과로 지표·calibration·보고서 재계산 | `evaluation/metrics.py`, `evaluation/report.py` |
| `scripts/render_retail.py` | 저장된 결과로 영상만 재생성 | `visualization/dashboard.py`, `visualization/video.py` |

`src/jevgate/` 아래 나머지 구성:

- `models/observer.py`: Qwen VLM 장면 관찰 및 VLM-only 비교 경로.
- `models/jev.py`, `models/budget.py`: Jev HTTP 클라이언트와 요청 예산.
- `decision/policy.py`: 질문 구성, 응답 카드 변환, 확률에 따른 직원 행동 규칙.
- `core/config.py`, `core/runio.py`: YAML 설정·override, 실행 폴더·JSONL·provenance.
- `tests/`는 저장소 루트에 위치하며 서비스 접속 없이 실행한다.

```bash
uv sync --group dev
uv run python scripts/prepare_retail.py
uv run python scripts/run_retail.py
uv run python scripts/evaluate_retail.py --run-dir runs/261015_retail-demo-v2 --output docs/retail-results
uv run python scripts/render_retail.py --run-dir runs/261015_retail-demo-v2
uv run pytest
```

설정은 `configs/retail.yaml` 하나로 모았다. 작은 실행은 다음과 같다.

```bash
uv run python scripts/run_retail.py --set run.max_clips=5 render.include=[] render.fill_others=true render.clips_in_video=3
```

전처리는 이미 내려받은 `data/retail-demo/retailaction/{selection,samples,frames}.json`과 `clips/`를 사용한다.
실제 추론에는 설정에 맞는 GPU와 `.env`의 `TYPESAFE_API_KEY`가 필요하다.
평가·렌더링은 저장된 실행 결과를 사용하며 API를 호출하지 않는다.
평가는 해당 실행의 `run.yaml`을 사용한다. 렌더링은 현재 설정을 사용하므로 당시 조건을 재현하려면 설정을 맞춰야 한다.
`run_retail.py`는 앞으로 생성되는 실행에도 `evaluation/calibration/` 보고서를 자동 저장한다.

기존 `*_retail_demo.py` 명령은 `*_retail.py`, `configs/retail_demo.yaml`은 `configs/retail.yaml`로 변경했다.
기존 Retail 데이터 및 실행 폴더 이름은 원본 결과 참조를 위해 유지했다.

결과: [188개 클립 평가와 calibration](retail-results/report.md).
