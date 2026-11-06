"""MLX 런타임, 어댑터 로딩, 체크포인트 식별과 공통 블록 수집."""

from contextlib import contextmanager
from functools import wraps
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import sys
from threading import Lock

SUPPORTED_VLM_VERSION = "0.7.8"
DEFAULT_MODEL = "mlx-community/Qwen3-VL-4B-Instruct-4bit"


def package_versions():
    result = {}
    for name in ("mlx", "mlx-metal", "mlx-vlm", "numpy", "transformers"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def runtime():
    versions = package_versions()
    if versions["mlx-vlm"] != SUPPORTED_VLM_VERSION:
        detected = versions["mlx-vlm"] or "미설치"
        raise RuntimeError(
            f"mlx-vlm=={SUPPORTED_VLM_VERSION}이 필요합니다. 감지한 버전: {detected}\n"
            f"실행 중인 Python: {sys.executable}\n"
            "설치한 가상환경의 Python으로 실행하세요. 이 Python에 설치하려면:\n"
            f'"{sys.executable}" -m pip install -r requirements.txt'
        )
    try:
        import mlx.core as mx
        from mlx_vlm import load
    except ImportError as exc:
        raise RuntimeError("MLX를 로드하지 못했습니다. Metal GPU 접근과 설치 환경을 확인하세요.") from exc
    return mx, load


def load_adapter(model_ref, revision=None, adapter="auto"):
    """모델은 MLX-VLM으로 로드하고 계열별 차이는 어댑터에 위임한다."""
    from src.adapters import select_adapter

    mx, load = runtime()
    model, processor = load(model_ref, revision=revision)
    model.eval()
    return select_adapter(mx, model, processor, adapter)


def checkpoint_identity(model):
    """다른 가중치·토크나이저로 분석하는 것을 막기 위한 파일 내용 해시."""
    folder = Path(model.model_path).resolve()
    paths = sorted({*folder.glob("*.safetensors"), *folder.glob("*.json"),
                    *folder.glob("*.jinja"), *folder.glob("*.model")})
    manifest = {}
    for path in paths:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        manifest[path.name] = digest.hexdigest()
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return {"path": str(folder), "sha256": fingerprint, "files": manifest}


class RecordedLayers(list):
    """저장할 마지막 위치들 + 같은 계산 크기의 검증에만 쓸 마지막 블록 출력."""

    def __init__(self, size):
        super().__init__([None] * size)
        self.final_hidden = None


_record_lock = Lock()


@contextmanager
def record_last_position(layers, extract_hidden=None):
    """블록별 출력에서 [1, S, H]를 읽으며 원래 반환값은 그대로 유지한다.

    서로 다른 블록 클래스와 명시적인 출력 추출 함수를 지원한다.
    class __call__을 임시 교체하므로 중첩·동시 수집은 허용하지 않는다.
    """
    layers = list(layers)
    if not layers or len({id(layer) for layer in layers}) != len(layers):
        raise ValueError("서로 다른 디코더 블록이 하나 이상 필요합니다")
    if not _record_lock.acquire(blocking=False):
        raise RuntimeError("동시 또는 중첩 벡터 수집은 지원하지 않습니다")
    extract = extract_hidden or (lambda block, output: output)
    indices = {id(layer): i for i, layer in enumerate(layers)}
    captured = RecordedLayers(len(layers))
    originals = {}

    def wrap(original):
        @wraps(original)
        def traced(block, *args, **kwargs):
            output = original(block, *args, **kwargs)
            index = indices.get(id(block))
            if index is not None:
                if captured[index] is not None:
                    raise ValueError("한 단계에서 같은 디코더 블록의 재사용은 지원하지 않습니다")
                hidden = extract(block, output)
                if getattr(hidden, "ndim", None) != 3 or hidden.shape[0] != 1 or hidden.shape[1] < 1:
                    raise ValueError("블록 히든 스테이트는 [1, sequence, hidden]이어야 합니다")
                captured[index] = hidden[0, -1, :]
                if index == len(layers) - 1:
                    captured.final_hidden = hidden
            return output
        return traced

    try:
        # 상속된 __call__도 정확히 복원할 수 있도록 원래 클래스의 소유 여부를 보존한다.
        classes = list(dict.fromkeys(type(layer) for layer in layers))
        calls = {cls: cls.__call__ for cls in classes}
        for cls in classes:
            originals[cls] = ("__call__" in cls.__dict__, cls.__dict__.get("__call__"))
            cls.__call__ = wrap(calls[cls])
        yield captured
        if any(value is None for value in captured):
            raise RuntimeError("Not all decoder layers were captured")
    finally:
        for cls, (owned, original) in originals.items():
            if owned:
                cls.__call__ = original
            else:
                del cls.__call__
        _record_lock.release()
