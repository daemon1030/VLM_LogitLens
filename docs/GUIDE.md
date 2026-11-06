# CLI 사용법과 Logit Lens 분석 가이드

생성·분석 옵션과 결과 필드를 찾아보는 문서입니다. 명령은 프로젝트 루트에서 가상환경을 활성화한 뒤 실행합니다.

[설치·빠른 시작](../README.md) · [모델 지원·어댑터 구현](ADAPTERS.md) · [Qwen 구조](QWEN_ARCHITECTURE.md)

## 목차

- [1. 명령과 옵션 사용법](#1-명령과-옵션-사용법)
- [2. 생성 및 벡터 수집 옵션](#2-생성-및-벡터-수집-옵션)
- [3. Logit Lens 분석 옵션](#3-logit-lens-분석-옵션)
- [4. 결과 파일과 필드](#4-결과-파일과-필드)
- [5. Logit Lens 계산과 해석](#5-logit-lens-계산과-해석)
- [6. 한계와 비교 시 주의점](#6-한계와-비교-시-주의점)
- [7. 검증 및 오류 대응](#7-검증-및-오류-대응)

## 1. 명령과 옵션 사용법

`--옵션 값` 형태로 기본 설정을 바꿉니다. 아래 명령은 최대 100개 토큰을 생성하고 `runs/demo`에 저장합니다.

```bash
python -m src.capture.save_vectors \
  --prompt '17 + 25를 단계별로 계산한 후 답을 말해줘.' \
  --max-tokens 100 \
  --output runs/demo

python -m src.analysis.logit_lens --run runs/demo --top-k 5
```

옵션은 `--이름 값`으로 지정하며, 생략하면 기본값을 사용합니다. 공백이 있는 값은 따옴표로 감싸고, 줄 끝의 `\`는 명령을 다음 줄로 이어 쓸 때 사용합니다. 전체 옵션은 `--help`로 확인합니다.

```bash
python -m src.capture.save_vectors --help
python -m src.analysis.logit_lens --help
```

벡터 수집과 분석은 별도 실행입니다. 분석할 토큰·레이어·상위 후보 수만 바꾸려면 저장된 실행을 다시 분석하면 됩니다. **두 명령 모두 기존 출력 폴더를 덮어쓰지 않습니다.**

## 2. 생성 및 벡터 수집 옵션

### 입력·모델·저장

| 옵션 | 기본값 | 의미와 사용 조건 |
| --- | --- | --- |
| `--prompt '질문'` | 없음 | 직접 입력하는 프롬프트. `--prompt-file`과 둘 중 하나만 필수 |
| `--prompt-file prompts/question.txt` | 없음 | UTF-8 프롬프트 파일. `--prompt`와 동시 사용 불가 |
| `--image images/example.png` | 없음 | 선택 이미지 한 장. 생략하면 텍스트만 입력 |
| `--model ID_또는_경로` | `mlx-community/Qwen3-VL-4B-Instruct-4bit` | Hugging Face ID 또는 MLX 체크포인트 경로. 해당 모델의 어댑터 필요 |
| `--revision COMMIT_또는_TAG` | 미지정 | 모델 로딩에 전달할 Hugging Face revision |
| `--adapter qwen3_vl` | `auto` | 등록된 어댑터 이름 또는 `package.module:AdapterClass`. `auto`는 등록된 구현 중 지원 조건이 맞는 하나를 선택 |
| `--output runs/demo` | 없음, 필수 | 새 결과 폴더 |
| `--max-tokens 200` | `200` | 생성 토큰 상한. 1 이상. EOS가 나오면 먼저 종료 |
| `--max-input-tokens 2048` | `2048` | 이미지·템플릿 토큰을 포함한 입력 상한. 1 이상 |
| `--check-atol 0.001` | `0.001` | 수집 시 최종 레이어 로짓 재현 검사의 절대 오차 허용치. 양수 |

기본 제공 어댑터는 dense Qwen3-VL이며 Ollama 태그·GGUF는 지원하지 않습니다. 한 번에 질문 하나·이미지 최대 한 장을 처리합니다. Qwen 어댑터는 패딩 입력을 거부하고 마지막 입력 위치가 텍스트인지 확인합니다.

### 생성 방식·반복 억제

| 옵션 | 기본값 | 의미와 사용 조건 |
| --- | --- | --- |
| `--temperature 0.7` | `0.0` | 0이면 greedy 선택, 양수이면 샘플링. 현재 구현은 양수 중 0.01 미만을 실효값 0.01로 처리 |
| `--top-p 0.8` | `1.0` | 샘플링 후보의 누적 확률 범위. `0 < p <= 1`, 1이면 제한 없음 |
| `--top-k 20` | `0` | 샘플링 후보 수 제한. 0이면 제한 없음, 모델 어휘 수 이하의 정수 |
| `--presence-penalty 1.5` | `0.0` | 최근 생성 이력에 등장한 토큰의 점수를 낮춤. 0 이상, 0이면 비활성 |
| `--repetition-penalty 1.1` | `1.0` | 부호를 고려한 곱셈 방식의 반복 패널티. 양수, 1이면 비활성. 1보다 큰 값을 반복 억제에 사용 |
| `--penalty-context 256` | `256` | 패널티에 사용할 최근 생성 토큰 수. 1 이상. 프롬프트 토큰은 제외 |
| `--seed 42` | `42` | 난수 시드. 샘플링 재현을 위한 설정 |

생성의 `--top-p` 또는 `--top-k`를 기본값에서 바꾸려면 `--temperature`를 양수로 지정해야 합니다. 반복 패널티는 temperature=0에서도 사용할 수 있습니다.

이미지 입력에 샘플링과 반복 억제를 적용하는 예시:

```bash
python -m src.capture.save_vectors \
  --prompt '이미지에서 확인한 근거를 설명하고 원이 몇 개인지 답해줘.' \
  --image images/example.png \
  --temperature 0.7 --top-p 0.8 --top-k 20 \
  --presence-penalty 1.5 --penalty-context 256 --seed 42 \
  --max-tokens 200 --output runs/image_demo
```

`images/example.png`는 직접 준비한 파일로 바꿉니다. 패널티는 가중치나 수집 벡터를 직접 바꾸지 않고 다음 토큰 선택에 적용됩니다. 선택이 바뀌면 이후 문맥과 벡터도 바뀝니다. 반복 억제가 EOS 도달이나 답변 품질을 보장하지는 않습니다.

## 3. Logit Lens 분석 옵션

| 옵션 | 기본값 | 의미와 사용 조건 |
| --- | --- | --- |
| `--run runs/demo` | 없음, 필수 | 벡터 수집 결과 폴더. 정상 완료된 실행만 허용 |
| `--output runs/demo/lens_selected` | `RUN/logit_lens` | 새 분석 결과 폴더. 재분석 시 다른 폴더를 지정 |
| `--steps 0 1 2` | 모든 생성 위치 | 분석할 생성 토큰 위치 목록. **0부터** 시작 |
| `--layers 1 12 24 36` | 모든 디코더 레이어 | 분석할 레이어 번호 목록. **1부터** 시작 |
| `--target-id 18` | 추가 대상 없음 | 관심 토큰 ID. 옵션을 반복해 여러 개 지정 |
| `--target-text '3'` | 추가 대상 없음 | 단일 토큰으로 인코딩되는 문자열. 옵션 반복 가능. 여러 토큰이면 오류와 ID 목록 출력 |
| `--top-k 5` | `5` | 결과에 저장할 레이어별 상위 후보 수. 1 이상. 어휘 수를 넘으면 어휘 수로 제한 |
| `--preview-step 0` | `0` | 터미널에 미리 볼 생성 위치. `--steps`에 포함된 위치여야 표시됨 |
| `--model ID_또는_경로` | 수집 메타데이터의 체크포인트 경로 | 같은 체크포인트를 다른 위치에서 불러올 때 지정. 파일 내용 해시가 다르면 중단 |
| `--revision COMMIT_또는_TAG` | 미지정 | `--model`에 Hugging Face ID를 지정했을 때 사용할 revision |
| `--adapter qwen3_vl` | `auto` | 수집과 동일한 어댑터. 사용자 어댑터는 수집 때 사용한 `package.module:AdapterClass`를 다시 명시 |

`--adapter auto`는 등록된 구현 중 하나를 선택합니다. 새 모델 연결과 사용자 어댑터 지정은 [어댑터 가이드](ADAPTERS.md)를 참고하세요.

**생성 명령의 `--top-k`는 다음 토큰 선택을 제한하지만, 분석 명령의 `--top-k`는 저장할 후보 수만 바꿉니다.** 분석의 확률·순위는 항상 전체 어휘 기준입니다. 실제 생성 토큰은 상위 K개 안에 없어도 `target_scores.csv`에 기록됩니다.

특정 위치·레이어와 관심 토큰만 분석하는 예시:

```bash
python -m src.analysis.logit_lens \
  --run runs/demo \
  --steps 0 1 2 \
  --layers 1 12 24 36 \
  --target-text '3' --target-id 18 \
  --top-k 10 --preview-step 1 \
  --output runs/demo/lens_selected
```

이 예시는 실행에 생성 토큰이 3개 이상, 레이어가 36개 이상 있을 때 유효합니다. 토큰 ID와 문자열의 대응은 토크나이저마다 다르며, 문자열 앞 공백도 토큰화에 영향을 줍니다. 여러 토큰으로 나뉘는 단어의 확률을 구하는 기능은 없습니다.

## 4. 결과 파일과 필드

결과는 수집의 `--output` 아래에 생성됩니다. 분석 폴더 기본값은 그 아래의 `logit_lens/`입니다.

| 파일 | 기록 단위 | 용도 |
| --- | --- | --- |
| `output.txt` | 전체 생성 텍스트 | 설명·답변 읽기 |
| `tokens.jsonl` | 생성 토큰 하나 | 실제 생성 토큰과 step 찾기 |
| `logit_lens/target_scores.csv` | step × 레이어 × 대상 토큰 | 관심 토큰의 점수 변화 비교 |
| `logit_lens/top_tokens.jsonl` | step × 레이어 | 경쟁하는 상위 후보 확인 |
| `hidden_states.npy` | `[T, L, H]` 배열 | 내부 벡터를 직접 분석 |
| `metadata.json` | 수집 실행 하나 | 입력·설정·종료 상태 확인 |
| `logit_lens/analysis.json` | 분석 실행 하나 | 분석 범위·완료 상태·검증 확인 |

JSONL은 한 줄마다 독립적인 JSON 객체가 있는 형식입니다. **`tokens.jsonl`은 실제 생성 기록이고, `top_tokens.jsonl`은 레이어별 읽기 결과**입니다.

### `output.txt`와 `hidden_states.npy`

`output.txt`는 특수 토큰을 제외하고 전체 생성 토큰을 디코딩한 텍스트입니다. `decoded_token`을 단순히 이어 붙인 문자열과 다를 수 있습니다.

`hidden_states.npy`는 float32 배열이며 형태는 `[생성 토큰 수 T, 레이어 수 L, 히든 차원 H]`입니다. `hidden_states[i, l-1]`은 **생성 토큰 i를 예측하기 직전의 레이어 l 벡터**입니다. 첫 행은 프롬프트의 마지막 텍스트 위치, 다음 행은 첫 생성 토큰 위치에서 수집됩니다. 프롬프트 전체나 이미지 위치의 벡터는 저장하지 않습니다.

원래 fp16/bf16 값을 float32로 저장하고, 분석할 때 원래 dtype으로 복구합니다. EOS가 생성되면 그 토큰과 예측 벡터도 포함합니다. 실제 길이 T는 `--max-tokens`보다 작을 수 있습니다. 저장 공간은 순수 배열 기준 대략 `T × L × H × 4`바이트입니다.

### `tokens.jsonl`: 실제 생성 토큰별 기록

생성 토큰당 한 줄, 총 T줄입니다.

| 필드 | 의미 |
| --- | --- |
| `step` | 생성 위치, 0부터 시작. 배열의 첫 축 인덱스와 대응 |
| `token_id` | 실제 선택된 토큰 ID |
| `token_piece` | 토크나이저 내부 토큰 표기. 읽기 어려운 문자로 보일 수 있음 |
| `decoded_token` | 해당 토큰 하나를 디코딩한 문자열. 일부 문자는 단일 토큰 디코딩에서 불완전하게 보일 수 있음 |
| `is_eos` | 종료 토큰 여부 |
| `actual_output_score` | 실제 모델 출력에서 선택된 토큰의 원래 `logit`, `probability`, `rank`와 `token_id` |
| `actual_top1_token_id` | 패널티·샘플링 전 실제 모델 로짓의 최상위 토큰 ID |

<details>
<summary>예측 위치·dtype·점수 재현 필드</summary>

| 필드 | 의미 |
| --- | --- |
| `source_position` | 예측에 사용한 위치의 논리적 인덱스. `입력 토큰 수 - 1 + step`, 0부터 시작 |
| `source_token_id` | 그 위치의 입력 토큰. step=0에서는 마지막 프롬프트 토큰, 이후에는 직전 생성 토큰 |
| `activation_dtype` | 수집 당시 활성값 dtype |
| `single_position_readout_score` | 마지막 레이어 벡터 한 위치만 다시 읽었을 때 선택된 토큰의 점수 객체 |
| `single_position_top1_token_id` | 단일 위치 재계산의 최상위 토큰 ID |
| `selection_logit_before_sampling` | 선택된 토큰의 패널티 적용 후, 후보 필터링·temperature 적용 전 점수. 이전 실행에는 없을 수 있음 |
| `final_readout_max_abs_error` | 원래 입력과 같은 계산 크기로 최종 레이어를 재계산한 로짓과 실제 로짓의 전체 어휘 최대 절대 차이 |
| `single_position_vs_actual_max_abs_error` | 단일 위치 재계산 로짓과 실제 로짓의 전체 어휘 최대 절대 차이 |

</details>

`actual_output_score.probability`는 패널티나 샘플링 변환을 적용하지 않은 확률입니다. 실제로 선택된 토큰도 샘플링·패널티를 사용했다면 원래 점수에서 rank=1이 아닐 수 있습니다.

### `metadata.json`: 실행 조건과 완료 상태

먼저 `status`, `stop_reason`, `count`를 확인합니다. `completed`는 정상 종료를 뜻하며, `stop_reason=max_tokens`이면 답변이 길이 상한에서 잘렸을 수 있습니다. `eos`는 종료 토큰에 도달했다는 뜻입니다.

<details>
<summary>수집 메타데이터 전체 필드</summary>

| 필드 또는 필드 묶음 | 의미 |
| --- | --- |
| `schema_version`, `created_at` | 저장 형식 버전, UTC 생성 시각 |
| `status`, `stop_reason`, `error` | 실행 상태(`running`, `completed`, `interrupted`, `failed`), 종료 이유, 오류 내용 |
| `count`, `shape`, `generated_token_ids` | 실제 기록 개수, 배열 형태, 생성 ID 목록 |
| `model`, `revision_requested`, `checkpoint` | 요청 모델·revision, 확인된 경로와 파일별·전체 SHA-256 |
| `packages`, `quantization` | 패키지 버전과 양자화 설정 |
| `prompt`, `formatted_prompt`, `image` | 원본·템플릿 적용 프롬프트, 이미지 경로·SHA-256. 이미지 미사용 시 null |
| `input_token_ids`, `input_token_count` | 템플릿·이미지 토큰을 포함한 입력 토큰 |
| `num_layers`, `hidden_size`, `layer_numbers`, `storage_dtype` | 배열 축과 저장 형식 해석에 필요한 값 |
| `vector_site`, `alignment`, `readout_check` | 수집 위치·토큰 대응·검증 방식 설명 |
| `adapter` | 어댑터의 `name`, `contract_version`, `vector_site`, `readout`. 분석 시 현재 어댑터와 모두 일치해야 함. 기존 Qwen 결과에는 없을 수 있음 |
| `generation` | 토큰 상한, 요청·실효 temperature, seed, top-p/k, 반복 패널티, EOS ID |
| `final_readout_max_abs_error`, `single_position_vs_actual_max_abs_error` | 각 단계의 해당 오차 중 실행 전체 최댓값 |
| `elapsed_seconds` | 수집 루프 경과 시간. 초기 모델 로딩·해시 계산과 결과 마무리 시간은 제외 |

</details>

### `top_tokens.jsonl`: 생성 위치 × 레이어별 후보

한 줄은 **생성 위치 하나와 레이어 하나의 조합**입니다. 기본 전체 분석에서 T개 생성 토큰과 L개 레이어를 분석하면 T×L줄입니다.

| 필드 | 의미 |
| --- | --- |
| `step`, `layer` | 생성 위치(0부터), 레이어 번호(1부터) |
| `generated_token_id`, `generated_token_text` | 이 위치에서 실제로 생성된 토큰 |
| `top_tokens` | 이 레이어를 Logit Lens로 읽은 상위 K개 후보의 배열 |
| `top_tokens` 각 원소 | `token_id`, `token_text`, `logit`, `probability`, `rank` |

`tokens.jsonl`이 “실제로 무엇을 생성했나”를 기록한다면, 이 파일은 “그 토큰을 예측할 때 각 레이어에서 어떤 후보가 높게 읽혔나”를 보여줍니다. 후보 목록의 확률 합은 일반적으로 1보다 작습니다. 전체 어휘 확률에서 상위 K개만 저장하기 때문입니다.

### `target_scores.csv`: 관심 토큰의 레이어별 변화

한 행은 생성 위치 × 레이어 × 대상 토큰입니다. 실제 생성 토큰을 항상 포함하고, `--target-id`·`--target-text`로 지정한 토큰을 추가합니다.

| 열 | 의미 |
| --- | --- |
| `step`, `layer` | 생성 위치(0부터), 레이어(1부터) |
| `generated_token_id`, `generated_token_text` | 실제 생성 토큰 |
| `target_kind` | `generated`는 실제 생성 토큰, `requested`는 추가 관심 토큰 |
| `target_token_id`, `target_token_text` | 이번 행에서 점수를 보는 대상 |
| `logit`, `probability`, `rank` | 해당 위치·레이어에서 대상 토큰의 점수·확률·순위 |

같은 ID를 실제 생성 토큰과 관심 대상으로 모두 지정하면 두 종류의 행이 나올 수 있습니다. CSV는 UTF-8 BOM 포함 형식입니다.

### `analysis.json`: 분석 조건과 검증 결과

먼저 `status`와 `last_layer_verified`를 확인합니다. 마지막 레이어를 분석에서 제외하면 재검증 값은 false입니다.

<details>
<summary>분석 메타데이터 전체 필드</summary>

| 필드 또는 필드 묶음 | 의미 |
| --- | --- |
| `status`, `error` | 분석 상태와 오류. 정상 결과인지 먼저 확인 |
| `run`, `checkpoint_sha256` | 입력 실행 경로와 모델 식별값 |
| `capture_packages`, `analysis_packages` | 수집·분석 시 패키지 버전 |
| `steps`, `layers`, `extra_target_ids` | 분석 범위와 추가 대상 |
| `rank_definition`, `probability_definition`, `readout` | 순위·확률·어댑터별 읽기 방식 정의 |
| `adapter` | 분석에 사용한 어댑터의 이름·계약 버전·수집 지점·읽기 방식 |
| `last_layer_verified` | 마지막 레이어 재검증 수행 여부. 마지막 레이어를 제외하면 false |
| `max_last_layer_target_logit_error` | 마지막 레이어에서 실제 생성 토큰의 저장 당시 단일 위치 점수와 재분석 점수의 최대 절대 차이. 검증하지 않으면 null |

</details>

## 5. Logit Lens 계산과 해석

각 위치·레이어의 벡터 h를 어댑터의 `project_logits()`로 읽어 전체 어휘 로짓을 계산합니다. Qwen은 final RMSNorm을 한 번 적용한 뒤 원래 출력 헤드를 사용합니다. 아래 `FinalNorm`은 모델별 최종 정규화를 뜻하며, 학습한 보정 변환은 추가하지 않습니다.

```text
z = OutputHead(FinalNorm(h))
p(v) = exp(z[v]) / Σ_u exp(z[u])
rank(v) = 1 + #{u : z[u] > z[v]}
```

- **logit**: softmax 이전의 점수. 같은 위치·레이어에서 값이 클수록 유력한 후보입니다.
- **probability**: 전체 어휘에 대한 temperature=1 softmax. 생성 때 사용한 temperature·top-p/k·반복 패널티를 적용하지 않습니다.
- **rank**: 대상보다 로짓이 엄격히 큰 토큰 수에 1을 더한 값. 동점이면 같은 순위이며, 후보 배열에서 동점 토큰은 ID 순으로 정렬됩니다.

예를 들어 실제 생성된 답변 토큰의 rank가 레이어를 거치며 500 → 30 → 1로 변한다면, **이 읽기 방식에서 해당 토큰의 상대적 순위가 올라갔다**고 해석할 수 있습니다. 그 레이어에서 답을 처음 이해하거나 결정했다고 단정할 수는 없습니다.

### 결과 읽는 순서

1. `output.txt`에서 생성된 설명·답변을 읽습니다.
2. `tokens.jsonl`에서 관심 토큰의 `step`을 찾습니다.
3. `target_scores.csv`에서 해당 step의 레이어별 확률·순위를 비교합니다.
4. `top_tokens.jsonl`에서 같은 step·레이어의 경쟁 후보를 확인합니다.

다른 토큰이 궁금하면 관심 토큰을 추가하고 새 분석 출력 폴더로 재분석합니다. 분석은 한 위치·레이어씩 계산하므로 전체 `[토큰, 레이어, 어휘]` 배열을 메모리에 만들지 않습니다.

## 6. 한계와 비교 시 주의점

### 내부 사고와 인과관계를 직접 보여주지는 않음

중간 벡터를 최종 출력 헤드로 읽은 결과는 해당 레이어가 실제로 생성한 텍스트가 아닙니다. 높은 점수는 특정 의미 피처의 존재나 답을 내는 인과적 기여를 직접 증명하지 않습니다. 그런 주장은 별도의 개입 실험이 필요합니다. 생성된 CoT도 관찰 대상 텍스트이며, 이 도구는 CoT와 최종 답변을 자동 구분하지 않습니다.

### 중간 레이어와 출력 헤드의 불일치

최종 출력용 읽기 방식을 모든 레이어에 그대로 적용하므로 중간 표현을 잘못 읽을 수 있습니다. 레이어마다 벡터와 로짓의 스케일이 달라질 수 있어 원시 logit만 비교하지 말고 확률·순위·후보 변화도 함께 보세요. 확률 역시 모델 정답률이나 보정된 신뢰도를 뜻하지 않습니다.

Tuned Lens는 레이어별 affine probe를 학습해 이런 읽기의 불안정성을 개선하는 방법입니다. 이 프로젝트는 해당 probe를 학습하거나 사용하지 않는 Logit Lens 구현입니다. [Tuned Lens 논문](https://arxiv.org/abs/2303.08112)

### 단일 토큰 분석의 범위

단어·숫자·한글 음절이 여러 토큰으로 나뉠 수 있고 공백에 따라 ID가 달라집니다. 한 토큰의 확률을 단어나 문장 전체의 확률로 해석할 수 없습니다. 전체 시퀀스 확률이나 의미 단위별 기여도는 계산하지 않습니다.

### 수치 정밀도와 재현성

BF16·양자화 출력 헤드에서는 전체 입력 크기와 한 위치만 재계산할 때 사용하는 커널·누적 순서 차이로 로짓이 달라질 수 있습니다. 매우 가까운 후보의 순위도 바뀔 수 있습니다. 따라서 실제 생성 점수와 단일 위치 Lens 점수는 별도 필드로 보존합니다.

seed만 같다고 다른 모델·패키지·하드웨어에서 동일 결과가 보장되지는 않습니다. 입력·모델 해시·패키지 버전·생성 설정을 함께 비교하세요. 패널티나 샘플링 설정이 달라지면 이후 문맥 자체도 달라집니다.

### 지원 범위

| 구분 | 범위 |
| --- | --- |
| 실행 기반 | MLX·`mlx-vlm==0.7.8`. 버전은 실행 시 검사 |
| 모델 | 기본 dense Qwen3-VL. 다른 MLX 모델의 확장은 [어댑터 가이드](ADAPTERS.md) 참고 |
| 입력·실행 | 질문 한 건·이미지 최대 한 장. 배치·영상·동시 수집·컴파일 미지원. Qwen은 패딩 입력 미지원 |
| 벡터 | 모든 레이어의 히든 차원과 dtype이 동일해야 함. 분석 dtype은 float16·bfloat16·float32 |

모델 자체의 기능과 이 도구의 지원 범위는 다릅니다. 기본 Instruct 모델에 단계별 설명을 요청해도 Thinking 체크포인트로 바뀌지는 않습니다.

## 7. 검증 및 오류 대응

| 검사 시점 | 확인 내용 | 기준 |
| --- | --- | --- |
| 수집 매 단계 | 원래 입력과 같은 계산 크기로 재계산한 최종 레이어 로짓 ↔ 실제 로짓 | 전체 어휘 최대 절대 차이. `--check-atol` 기본 0.001, 상대 허용치 0 |
| 분석 시작 전 | 체크포인트 해시, 어댑터 정보, 레이어 수·히든 차원 | 수집 당시 값과 일치 |
| 분석 중 마지막 레이어 | 실제 생성 토큰의 단일 위치 logit ↔ 저장 당시 단일 위치 logit | 절대 차이 0.05 이하. `--check-atol`과 별개 |

마지막 레이어를 제외하면 분석 중 재검증은 수행하지 않습니다. 이전 기록에 단일 위치 점수가 없으면 실제 출력 점수로 대신 비교합니다. `adapter` 정보가 없는 기존 결과는 같은 수집 지점·크기의 Qwen 어댑터로만 분석합니다.

최종 로짓 재현만으로 중간 레이어의 수집 지점과 해석까지 보장되지는 않습니다. 새 어댑터는 [구조 검토와 실행 검증](ADAPTERS.md#공통-수집-계약과-검증)이 필요합니다.

| 상황 | 확인할 내용 |
| --- | --- |
| 어댑터를 자동 선택하지 못함 | 해당 구조의 구현 여부 확인. 구현된 사용자 어댑터는 `--adapter package.module:AdapterClass`로 지정 |
| 수집 당시와 어댑터 계약이 다름 | 이름·계약 버전·수집 지점·읽기 방식과 레이어 수·히든 차원 확인. 수집과 동일한 어댑터로 분석 |
| 출력 폴더가 이미 존재함 | 수집은 새 `--output`, 재분석은 별도의 분석 `--output` 지정 |
| 입력 토큰 상한 초과 | 이미지 해상도를 줄이거나 메모리를 고려해 `--max-input-tokens` 조정 |
| 생성이 반복되거나 잘림 | `stop_reason` 확인. 필요하면 샘플링·반복 패널티·생성 상한을 바꾸고 새 실행으로 수집 |
| 체크포인트 경로를 찾지 못함 | 동일 파일 내용의 모델을 분석의 `--model`로 지정. 다른 해시는 허용하지 않음 |
| `--target-text`가 여러 토큰으로 분리됨 | 오류에 나온 ID를 개별 `--target-id`로 지정. 합친 문자열 확률은 제공하지 않음 |
| 수집이 중단되거나 실패함 | 확보한 결과는 보존되지만 분석기는 `status=completed`만 허용 |
| `hidden_states.partial.npy`가 남음 | 강제 종료 등으로 마무리되지 않은 실행일 수 있음. 완료된 결과로 취급하지 않음 |

GPU나 모델 다운로드 없이 핵심 로직을 검증하려면:

```bash
python -m unittest discover -s tests -v
```

<details>
<summary>구현 코드 찾아보기</summary>

| 역할 | 코드 |
| --- | --- |
| 수집·분석 CLI | [save_vectors.py](../src/capture/save_vectors.py), [logit_lens.py](../src/analysis/logit_lens.py) |
| 모델 로딩·블록 수집 | [model.py](../src/common/model.py) |
| 모델별 처리 | [어댑터 계약](../src/adapters/base.py), [Qwen 구현](../src/adapters/qwen3_vl.py) |
| 저장·점수 계산 | [storage.py](../src/common/storage.py), [scoring.py](../src/common/scoring.py) |

</details>
