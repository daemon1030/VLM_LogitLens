# Qwen3-VL Logit Lens 분석

VLM이 단계별 설명(CoT)과 답변을 생성할 때, **각 레이어에서 어떤 토큰이 높은 점수를 받는지** 살펴보는 연구 도구입니다. 생성 토큰별 내부 벡터(히든 스테이트)를 저장하고, Logit Lens로 토큰 후보·확률·순위의 변화를 비교합니다.

```text
텍스트 + 이미지(선택) → 답변 생성 및 내부 벡터 저장 → 레이어별 토큰 예측 비교
```

기본 지원은 **dense Qwen3-VL**입니다. 다른 MLX 모델은 [모델별 어댑터를 구현해 연결](docs/ADAPTERS.md)할 수 있습니다.

## 빠른 시작

**Apple Silicon Mac, Python 3.11 이상, Metal GPU 접근**이 필요합니다. 기본 모델은 `mlx-community/Qwen3-VL-4B-Instruct-4bit`이며 첫 실행 시 다운로드됩니다.

프로젝트 폴더에서 환경을 준비합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

짧은 질문으로 벡터를 수집하고 분석합니다.

```bash
# 1. 답변 생성 및 벡터 저장
python -m src.capture.save_vectors \
  --prompt '17 + 25를 단계별로 계산한 후 답을 말해줘.' \
  --max-tokens 20 \
  --output runs/demo

# 2. 레이어별 상위 5개 토큰과 실제 생성 토큰의 점수 분석
python -m src.analysis.logit_lens --run runs/demo --top-k 5
```

이미지를 사용하려면 첫 명령에 `--image '/절대/경로/image.png'`를 추가합니다. 한 번에 질문 하나와 이미지 최대 한 장을 처리합니다. `--max-tokens 20`은 빠른 확인을 위한 생성 길이 상한입니다.

기존 출력 폴더는 덮어쓰지 않습니다. 재실행할 때는 수집·분석 명령의 실행 경로를 모두 `runs/demo_002`처럼 바꾸세요.

## 확인할 수 있는 결과

아래 경로는 `runs/demo/` 기준입니다.

| 보고 싶은 것 | 파일 |
| --- | --- |
| 모델이 생성한 설명과 답변 | `output.txt` |
| 실제 생성 토큰과 생성 당시 점수 | `tokens.jsonl` |
| 레이어별 상위 토큰 후보 | `logit_lens/top_tokens.jsonl` |
| 실제 생성·관심 토큰의 레이어별 점수·확률·순위 | `logit_lens/target_scores.csv` |
| 생성 토큰별·레이어별 내부 벡터 | `hidden_states.npy` |
| 입력·모델·생성 설정 | `metadata.json` |

분석 시 터미널에도 첫 생성 토큰에 대한 레이어별 후보와 확률·순위가 표시됩니다. 예를 들어 답변 토큰이 얕은 레이어에서 낮은 순위였다가 깊은 레이어에서 상위로 올라오는지 확인할 수 있습니다.

Logit Lens는 내부 표현을 해석하기 위한 단서입니다. 모델의 사고 과정을 직접 증명하지 않으며, CoT와 최종 답변 구간을 자동으로 구분하지는 않습니다.

## 문서

| 필요한 내용 | 문서 |
| --- | --- |
| CLI 옵션, 결과 필드, 분석 방법과 한계 | [GUIDE](docs/GUIDE.md) |
| 지원 모델, 어댑터 선택과 새 모델 추가 | [ADAPTERS](docs/ADAPTERS.md) |
| Qwen3-VL 구조와 벡터 수집 지점 | [QWEN_ARCHITECTURE](docs/QWEN_ARCHITECTURE.md) |
