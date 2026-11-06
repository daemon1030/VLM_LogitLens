"""MLX 없이도 검증할 수 있는 어휘 점수·순위 계산."""

import numpy as np


def token_scores(logits, token_id, top_k=5):
    """순위는 1부터, 동점은 competition rank (1 + 더 큰 점수의 수)."""
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("logits must be a finite one-dimensional array")
    if not 0 <= token_id < len(values) or not 1 <= top_k <= len(values):
        raise ValueError("token_id or top_k is outside the vocabulary")
    maximum = values.max()
    log_z = maximum + np.log(np.exp(values - maximum).sum())
    # Stable sorting makes ties deterministic by token ID.
    top_ids = np.argsort(-values, kind="stable")[:top_k]

    def entry(index):
        logit = float(values[index])
        return {
            "token_id": int(index),
            "logit": logit,
            "probability": float(np.exp(logit - log_z)),
            "rank": int(1 + np.count_nonzero(values > logit)),
        }

    return {"target": entry(token_id), "top_tokens": [entry(i) for i in top_ids]}
