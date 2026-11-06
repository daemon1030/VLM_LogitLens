# 모델 지원 범위와 어댑터 추가

[README](../README.md) · [CLI와 결과 해석](GUIDE.md) · [Qwen3-VL 구조](QWEN_ARCHITECTURE.md)

모델을 추가할 때 읽는 개발 가이드입니다.

- [지원 범위](#현재-가능한-것)
- [어댑터 선택](#자동-선택과-직접-지정)
- [새 모델 연결](#새-모델-연결-절차)
- [구현 항목](#어댑터가-구현하는-항목)
- [수집 계약·검증](#공통-수집-계약과-검증)
- [결과 호환성](#저장-결과의-호환성)

## 현재 가능한 것

수집·저장·Logit Lens 분석은 공통 파이프라인을 사용하고, 모델별 차이는 `ModelAdapter`에 모읍니다. 실행 기반은 MLX·`mlx-vlm==0.7.8`입니다.

| 모델 또는 상황 | 현재 동작 |
| --- | --- |
| 기본 dense Qwen3-VL | 제공된 `qwen3_vl` 어댑터를 자동 선택하고 구조 검사 |
| 같은 구조의 다른 크기·체크포인트 | 기존 어댑터를 재사용할 수 있으나 호환성과 로짓 재현 확인 필요 |
| 다른 구조의 MLX 모델 | 해당 구조의 어댑터를 직접 구현하고 지정 |
| PyTorch 등 다른 실행 기반 | 현재 지원하지 않음 |

**`auto`는 등록된 어댑터를 선택합니다.** 새 모델 구조를 조사해 어댑터를 자동으로 생성하지는 않습니다.

## 자동 선택과 직접 지정

두 CLI 모두 `--adapter`를 받습니다.

| 지정 방식 | 동작 |
| --- | --- |
| 생략 또는 `--adapter auto` | 등록된 클래스의 `supports(model)`이 일치하는 하나를 선택. 없거나 여러 개면 오류 |
| `--adapter qwen3_vl` | 등록된 이름으로 선택한 뒤 모델 구조 검사 |
| `--adapter package.module:AdapterClass` | import 가능한 모듈의 `ModelAdapter` 하위 클래스를 직접 지정. 등록 불필요 |

사용자 어댑터는 **수집과 분석 양쪽에서 직접 지정**합니다. 메타데이터의 이름만 보고 사용자 모듈을 자동 import하지 않습니다.

## 새 모델 연결 절차

1. 체크포인트가 현재 MLX-VLM에서 로드되는지 확인합니다. 입력 준비·캐시·위치 정보와 디코더·출력 헤드 구조를 조사합니다.
2. `src/adapters/my_model.py` 등에 `ModelAdapter` 하위 클래스를 구현합니다. 기존 Qwen 파일을 덮어쓰지 않고 추가합니다.
3. 짧은 텍스트 생성으로 토큰 정렬과 최종 레이어 로짓 재현을 확인합니다. 이미지 입력을 지원한다면 이미지도 따로 검증합니다.
4. 같은 어댑터로 재분석해 마지막 레이어 재검증과 후보·확률·순위를 확인합니다.

다음은 **`src.adapters.my_model.MyModelAdapter`를 구현한 뒤** 사용할 명령 형식입니다. 해당 예제 어댑터는 저장소에 기본 제공되지 않습니다.

```bash
python -m src.capture.save_vectors \
  --model /path/to/mlx-checkpoint \
  --adapter src.adapters.my_model:MyModelAdapter \
  --prompt '17 + 25는 얼마야?' \
  --max-tokens 5 --output runs/new_model

python -m src.analysis.logit_lens \
  --run runs/new_model \
  --adapter src.adapters.my_model:MyModelAdapter
```

자동 선택까지 제공하려면 구현 클래스를 [어댑터 등록부](../src/adapters/__init__.py)의 `BUILTIN_ADAPTERS`에 고유 이름으로 등록합니다. 이름만 등록하지 말고, 다른 어댑터와 지원 조건이 겹치지 않는지도 확인합니다.

## 어댑터가 구현하는 항목

기준은 [ModelAdapter](../src/adapters/base.py), 예시는 [Qwen3VLAdapter](../src/adapters/qwen3_vl.py)입니다. 아래에서 S는 시퀀스 길이, H는 히든 차원, V는 어휘 크기입니다.

| 항목 | 구현 내용 |
| --- | --- |
| `name`, `contract_version` | 고유 이름과 수집·읽기 방식의 계약 버전 |
| `vector_site`, `readout` | 벡터 수집 지점과 출력 점수 계산 방식 설명 |
| `supports(model)` | 로드한 모델이 이 어댑터의 구조인지 판별 |
| `validate()` | 디코더, 차원, 출력 헤드 등 구현의 전제조건 확인 |
| `layers` | 실제 실행 순서의 서로 다른 디코더 블록 목록 |
| `hidden_size`, `vocab_size` | 모델에서 읽은 히든 차원과 어휘 크기 |
| `quantization` | 양자화 설정. 없는 경우 None |
| `prepare(prompt, image, max_input_tokens)` | 입력 검증과 `PreparedInput(input_ids, formatted_prompt, state)` 생성 |
| `forward(input_ids, state)` | prefill/decode, 캐시·위치 정보 갱신, 패널티 전 `[1, S, V]` 로짓 반환 |
| `project_logits(hidden)` | `[1, S, H]`에서 원래 최종 정규화·출력 헤드를 거쳐 `[1, S, V]` 계산 |

`tokenizer`와 `eos_ids`에는 기본 구현이 있습니다. 토크나이저는 `encode()`, `decode()`, `convert_ids_to_tokens()`를 제공해야 합니다. 모델에는 체크포인트 해시 계산에 사용할 `model_path`가 필요합니다.

블록이 튜플·객체를 반환하면 `extract_hidden(block, output)`에서 `[1, S, H]` 텐서를 꺼냅니다. 기본 `record()`는 서로 다른 블록 클래스도 지원하고 원래 반환값을 유지합니다.

블록 반환값 외의 위치에서 수집해야 한다면 `record()`를 재정의합니다. 이때도 아래 수집 계약을 따르는 컨텍스트 매니저를 제공해야 합니다.

## 공통 수집 계약과 검증

- 각 생성 단계에서 레이어별 마지막 위치의 벡터를 수집합니다. 첫 행은 프롬프트 마지막 텍스트 위치에서 첫 생성 토큰을 예측하는 값입니다.
- 기본 수집기는 단계당 각 블록이 한 번 호출되어야 합니다. 블록 재사용과 중첩·동시 수집은 거부합니다.
- 수집 결과는 레이어 수만큼의 `[H]` 벡터와, 같은 입력 크기의 로짓 재현에 사용할 `final_hidden`(`[1, S, H]`)을 제공해야 합니다. 기본 수집기는 `RecordedLayers`로 이 계약을 구현합니다.
- 레이어별 히든 차원과 활성값 dtype은 동일해야 합니다. 분석은 float16·bfloat16·float32를 지원합니다.
- 마지막 블록의 수집 값에 `project_logits()`를 적용했을 때 원래 모델 로짓이 재현되어야 합니다. 공통 수집기가 매 단계 이를 검사합니다.

**최종 로짓 재현은 중간 레이어의 수집 지점과 의미 해석까지 보장하지 않습니다.** 잔차 연결·정규화·이미지 특징 주입 전후를 구조에 맞게 검토해야 합니다. 검증 허용치와 수치 차이는 [GUIDE의 검증 설명](GUIDE.md#7-검증-및-오류-대응)을 참고하세요.

## 저장 결과의 호환성

새 수집의 `metadata.json`과 분석의 `analysis.json`에는 어댑터의 `name`, `contract_version`, `vector_site`, `readout`이 저장됩니다. 분석 전에 현재 어댑터와 수집 당시 값이 모두 일치하는지 확인하고, 체크포인트 해시·레이어 수·히든 차원도 확인합니다.

벡터 의미나 읽기 방식을 바꾸면 `contract_version`을 올립니다. 기존 버전의 결과는 당시 계약을 구현한 어댑터로 분석해야 합니다. 어댑터 정보가 없던 기존 결과는 같은 수집 지점·차원의 Qwen 어댑터로만 읽습니다.

검증 코드는 [Qwen 파이프라인 테스트](../tests/test_pipeline.py)와 [다른 내부 구조의 모의 모델 테스트](../tests/test_adapters.py)에 있습니다. 모의 모델 검증과 별도로 새 실제 모델의 실행 검증이 필요합니다.
