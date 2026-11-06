"""NumPy 배열과 JSON만 사용하는, pickle 없는 실험 저장 형식."""

import json
from pathlib import Path

import numpy as np


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


class RunWriter:
    """한 토큰씩 디스크에 기록하고, 종료 시 유효한 길이로 배열을 확정한다."""

    def __init__(self, folder, max_tokens, layers, width, metadata):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=False)
        self.metadata = dict(metadata, schema_version=1, status="running", count=0)
        self.records = []
        self.partial = self.folder / "hidden_states.partial.npy"
        self.array = np.lib.format.open_memmap(
            self.partial, mode="w+", dtype=np.float32,
            shape=(max_tokens, layers, width),
        )
        write_json(self.folder / "metadata.json", self.metadata)

    def append(self, vectors, record):
        index = len(self.records)
        if record["step"] != index or vectors.shape != self.array.shape[1:]:
            raise ValueError("Hidden-state shape or generated-token alignment mismatch")
        if not np.all(np.isfinite(vectors)):
            raise ValueError("Non-finite hidden states")
        self.array[index] = vectors
        self.array.flush()
        with (self.folder / "tokens.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.records.append(record)
        self.metadata["count"] = len(self.records)
        write_json(self.folder / "metadata.json", self.metadata)

    def finish(self, status, text, **details):
        count = len(self.records)
        self.array.flush()
        final = np.lib.format.open_memmap(
            self.folder / "hidden_states.npy", mode="w+", dtype=np.float32,
            shape=(count, *self.array.shape[1:]),
        )
        for index in range(count):
            final[index] = self.array[index]
        final.flush()
        del final
        del self.array
        self.partial.unlink()
        (self.folder / "output.txt").write_text(text, encoding="utf-8")
        self.metadata.update(status=status, count=count, **details)
        self.metadata["shape"] = [count, self.metadata["num_layers"], self.metadata["hidden_size"]]
        write_json(self.folder / "metadata.json", self.metadata)


def read_run(folder):
    folder = Path(folder)
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1:
        raise ValueError("Unsupported capture format")
    if metadata["status"] != "completed":
        raise ValueError(f"Capture is not complete: {metadata['status']}")
    vectors = np.load(folder / "hidden_states.npy", mmap_mode="r", allow_pickle=False)
    records = [json.loads(line) for line in
               (folder / "tokens.jsonl").read_text(encoding="utf-8").splitlines()]
    expected = (metadata["count"], metadata["num_layers"], metadata["hidden_size"])
    if vectors.shape != expected or len(records) != expected[0]:
        raise ValueError("Capture dimensions and token records do not match")
    if any(record["step"] != i for i, record in enumerate(records)):
        raise ValueError("Token steps are not contiguous")
    return metadata, vectors, records
