"""기본 어댑터 등록과 명시적인 Python 확장 로딩.

새 MLX 모델은 ModelAdapter를 구현하고 --adapter package.module:AdapterClass로
연결한다. CLI에서 명시한 코드만 import하며 실행 메타데이터로 코드를 로드하지 않는다.
자동 감지를 추가하려면 BUILTIN_ADAPTERS에 클래스를 등록한다.
"""

from importlib import import_module

from .base import ModelAdapter, PreparedInput
from .qwen3_vl import Qwen3VLAdapter

BUILTIN_ADAPTERS = {Qwen3VLAdapter.name: Qwen3VLAdapter}


def select_adapter(mx, model, processor, selection="auto"):
    if selection == "auto":
        matches = [cls for cls in BUILTIN_ADAPTERS.values() if cls.supports(model)]
        if len(matches) != 1:
            kind = getattr(model.config, "model_type", "unknown")
            raise ValueError(
                f"모델 {kind!r}에 대응하는 어댑터를 하나로 결정할 수 없습니다. "
                "--adapter 이름 또는 package.module:AdapterClass를 지정하세요"
            )
        cls = matches[0]
    elif selection in BUILTIN_ADAPTERS:
        cls = BUILTIN_ADAPTERS[selection]
    else:
        module, separator, name = selection.partition(":")
        if not separator or not module or not name:
            raise ValueError(f"알 수 없는 어댑터: {selection!r}. 이름 또는 package.module:AdapterClass를 사용하세요")
        try:
            cls = getattr(import_module(module), name)
        except (ImportError, AttributeError) as exc:
            raise ValueError(f"어댑터를 불러올 수 없습니다: {selection}") from exc
        if not isinstance(cls, type) or not issubclass(cls, ModelAdapter):
            raise ValueError("사용자 어댑터는 ModelAdapter의 하위 클래스여야 합니다")
    if not cls.supports(model):
        raise ValueError(f"선택한 어댑터 {selection!r}가 이 모델을 지원하지 않습니다")
    instance = cls(mx, model, processor)
    instance.validate()
    for field in ("name", "vector_site", "readout"):
        if not isinstance(getattr(instance, field, None), str) or not getattr(instance, field):
            raise ValueError(f"어댑터의 {field} 정의가 필요합니다")
    return instance
