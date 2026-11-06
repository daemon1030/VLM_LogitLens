"""모델 계열별 입력·생성·수집·readout 계약. MLX 없이도 import할 수 있다."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class PreparedInput:
    """input_ids는 [1, S], state는 어댑터가 소유하는 캐시·위치 정보."""

    input_ids: Any
    formatted_prompt: str
    state: Any


class ModelAdapter(ABC):
    """MLX 실행용 확장점. 사용자 어댑터는 추측 없이 정확한 수집 지점을 정의한다.

    forward()는 [1, S, V] 원래 로짓을 반환하고 state를 갱신한다.
    project_logits()는 정규화 전 [1, S, H]를 받아 같은 형태의 로짓을 반환한다.
    마지막 블록의 extract_hidden() 결과는 project_logits()로 원래 로짓을
    재현할 수 있어야 한다. 다른 프레임워크의 텐서는 아직 지원하지 않는다.
    """

    # model은 체크포인트 파일을 읽을 수 있는 model_path를 제공해야 한다.
    # 이름은 구현별로 고유하게 정하며 같은 이름의 수집/readout 변경은 버전을 올린다.
    name: str
    vector_site: str
    readout: str
    contract_version = 1

    def __init__(self, mx, model, processor):
        self.mx = mx
        self.model = model
        self.processor = processor

    @classmethod
    @abstractmethod
    def supports(cls, model):
        """로드된 모델이 이 어댑터의 구조인지 판단한다."""

    @abstractmethod
    def validate(self):
        """레이어, 출력 형식, 헤드 등 이 구현의 전제조건을 검사한다."""

    @property
    @abstractmethod
    def layers(self):
        """실행 순서대로 정렬한 디코더 블록 목록."""

    @property
    @abstractmethod
    def hidden_size(self):
        pass

    @property
    @abstractmethod
    def vocab_size(self):
        pass

    @property
    def tokenizer(self):
        return getattr(self.processor, "tokenizer", self.processor)

    @property
    def eos_ids(self):
        tokenizer = self.tokenizer
        ids = set(getattr(getattr(tokenizer, "stopping_criteria", None), "eos_token_ids", []) or [])
        eos = getattr(tokenizer, "eos_token_id", None)
        if eos is not None:
            ids.add(eos)
        return ids

    @property
    @abstractmethod
    def quantization(self):
        pass

    @abstractmethod
    def prepare(self, prompt, image, max_input_tokens):
        """입력을 검증하고 PreparedInput을 반환한다. image는 Path 또는 None."""

    @abstractmethod
    def forward(self, input_ids, state):
        """prefill/decode를 실행하고 패널티 전 모든 위치의 로짓을 반환한다."""

    @abstractmethod
    def project_logits(self, hidden):
        """원래 모델의 최종 정규화·출력 헤드를 한 번 적용한다."""

    def extract_hidden(self, block, output):
        """블록 출력이 텐서가 아니면 어댑터에서 명시적으로 재정의한다."""
        return output

    def record(self):
        from src.common.model import record_last_position

        return record_last_position(self.layers, self.extract_hidden)

    def identity(self):
        """벡터 의미가 바뀌는 수정에는 contract_version을 올린다."""
        return {
            "name": self.name, "contract_version": self.contract_version,
            "vector_site": self.vector_site, "readout": self.readout,
        }

    def check_capture(self, metadata):
        """분석 전에 모델 크기와 수집/readout 계약의 일치를 확인한다."""
        saved = metadata.get("adapter")
        if saved is None:
            # schema v1의 어댑터 정보 없는 결과는 기존 Qwen 수집만 허용한다.
            if self.name != "qwen3_vl" or metadata.get("vector_site") != self.vector_site:
                raise ValueError("어댑터 정보 없는 기존 결과는 Qwen3-VL로만 분석할 수 있습니다")
        elif saved != self.identity():
            raise ValueError("수집 당시와 모델 어댑터 또는 벡터/readout 계약이 다릅니다")
        if len(self.layers) != metadata["num_layers"] or self.hidden_size != metadata["hidden_size"]:
            raise ValueError("모델의 레이어 수 또는 히든 차원이 저장 데이터와 다릅니다")
