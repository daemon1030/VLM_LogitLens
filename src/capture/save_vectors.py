"""프롬프트 → 생성 토큰별 모든 디코더 블록의 예측 직전 벡터 저장.

실행: python -m src.capture.save_vectors --prompt '17 + 25 = ?' --output runs/demo
"""

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys
import time

import numpy as np

from src.common.model import (
    DEFAULT_MODEL, checkpoint_identity, load_adapter, package_versions,
)
from src.common.scoring import token_scores
from src.common.storage import RunWriter


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    prompts = result.add_mutually_exclusive_group(required=True)
    prompts.add_argument("--prompt", help="질문 또는 CoT를 요청하는 프롬프트")
    prompts.add_argument("--prompt-file", type=Path, help="UTF-8 프롬프트 파일")
    result.add_argument("--image", type=Path, help="선택 사항: 로컬 이미지 한 장")
    result.add_argument("--model", default=DEFAULT_MODEL, help="MLX 모델 경로 또는 HF ID; Ollama 태그 아님")
    result.add_argument("--revision", help="HF 체크포인트 commit/tag")
    result.add_argument("--adapter", default="auto", help="auto, 등록 이름 또는 package.module:AdapterClass")
    result.add_argument("--output", type=Path, required=True, help="새 실행 폴더; 기존 폴더 덮어쓰기 금지")
    result.add_argument("--max-tokens", type=int, default=200)
    result.add_argument("--max-input-tokens", type=int, default=2048, help="이미지 토큰을 포함한 입력 상한")
    result.add_argument("--temperature", type=float, default=0.0, help="기본 0: greedy; 양수: temperature sampling")
    result.add_argument("--top-p", type=float, default=1.0, help="샘플링 후보 누적 확률; 1이면 제한 없음")
    result.add_argument("--top-k", type=int, default=0, help="샘플링 후보 개수; 0이면 제한 없음")
    result.add_argument("--presence-penalty", type=float, default=0.0, help="최근 생성된 토큰의 점수에서 빼는 값; 0이면 비활성")
    result.add_argument("--repetition-penalty", type=float, default=1.0, help="부호를 고려하는 반복 패널티; 1이면 비활성")
    result.add_argument("--penalty-context", type=int, default=256, help="패널티를 적용할 최근 생성 토큰 수")
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--check-atol", type=float, default=0.001, help="동일 계산 크기의 마지막 레이어 검증 절대 오차")
    return result


def capture(args):
    if args.max_tokens < 1 or args.max_input_tokens < 1:
        raise ValueError("토큰 수는 1 이상이어야 합니다")
    if not np.isfinite(args.temperature) or args.temperature < 0 or args.check_atol <= 0:
        raise ValueError("temperature는 0 이상, check-atol은 양수여야 합니다")
    if not 0 < args.top_p <= 1 or args.top_k < 0:
        raise ValueError("top-p는 (0, 1], top-k는 0 이상이어야 합니다")
    if not np.isfinite(args.presence_penalty) or args.presence_penalty < 0:
        raise ValueError("presence-penalty는 유한한 0 이상의 값이어야 합니다")
    if not np.isfinite(args.repetition_penalty) or args.repetition_penalty <= 0 or args.penalty_context < 1:
        raise ValueError("repetition-penalty와 penalty-context는 양수여야 합니다")
    if args.temperature == 0 and (args.top_p != 1 or args.top_k != 0):
        raise ValueError("top-p/top-k를 사용하려면 temperature를 0보다 크게 지정하세요")
    if args.output.exists():
        raise FileExistsError(f"출력 폴더가 이미 있습니다: {args.output}")
    prompt = args.prompt if args.prompt is not None else args.prompt_file.read_text(encoding="utf-8")
    if not prompt.strip():
        raise ValueError("프롬프트가 비어 있습니다")
    image = None
    if args.image is not None:
        path = args.image.expanduser().resolve(strict=True)
        image = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    adapter = load_adapter(args.model, args.revision, args.adapter)
    mx, model, tokenizer = adapter.mx, adapter.model, adapter.tokenizer
    sampler, processors = None, []
    if args.temperature > 0 or args.presence_penalty != 0 or args.repetition_penalty != 1:
        from mlx_vlm.sample_utils import make_logits_processors, make_sampler

        sampler = make_sampler(temp=args.temperature, top_p=args.top_p, top_k=args.top_k)
        processors = make_logits_processors(
            presence_penalty=args.presence_penalty or None,
            presence_context_size=args.penalty_context,
            repetition_penalty=args.repetition_penalty if args.repetition_penalty != 1 else None,
            repetition_context_size=args.penalty_context,
        )
    if args.top_k > adapter.vocab_size:
        raise ValueError("top-k가 모델 어휘 수보다 큽니다")
    prepared = adapter.prepare(prompt, Path(image["path"]) if image else None, args.max_input_tokens)
    input_ids = prepared.input_ids
    # 어댑터 구현과 무관하게 공통 저장 포맷의 입력 전제조건을 검사한다.
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or not 0 < input_ids.shape[1] <= args.max_input_tokens:
        raise ValueError("어댑터 입력은 토큰 상한 이내의 [1, sequence]여야 합니다")
    last_input = int(input_ids[0, -1].item())
    print("체크포인트 식별값을 계산합니다…", flush=True)
    identity = checkpoint_identity(model)
    layers = adapter.layers
    mx.random.seed(args.seed)
    eos_ids = adapter.eos_ids
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(), "model": args.model,
        "revision_requested": args.revision, "checkpoint": identity,
        "packages": package_versions(), "prompt": prompt, "formatted_prompt": prepared.formatted_prompt,
        "image": image, "input_token_ids": input_ids[0].tolist(),
        "input_token_count": input_ids.shape[1], "num_layers": len(layers),
        "hidden_size": adapter.hidden_size,
        "adapter": adapter.identity(),
        "storage_dtype": "float32", "layer_numbers": list(range(1, len(layers) + 1)),
        "vector_site": adapter.vector_site,
        "alignment": "hidden_states[i] predicts generated_token_ids[i]; first row uses last prompt position",
        "generation": {"max_tokens": args.max_tokens, "temperature": args.temperature,
                       "effective_temperature": max(0.01, args.temperature) if args.temperature else 0,
                       "seed": args.seed, "penalties": {
                           "presence": args.presence_penalty, "repetition": args.repetition_penalty,
                           "context_size": args.penalty_context, "history": "generated tokens only",
                       }, "top_k": args.top_k, "top_p": args.top_p,
                       "eos_token_ids": sorted(eos_ids)},
        "quantization": adapter.quantization,
        "readout_check": "full sequence shape preserved; single-position lens recorded separately",
    }
    writer = RunWriter(args.output, args.max_tokens, len(layers), metadata["hidden_size"], metadata)
    token_ids = []
    max_error = 0.0
    max_single_error = 0.0
    status, stop_reason, error = "completed", "max_tokens", None
    started = time.monotonic()
    y = input_ids
    try:
        for step in range(args.max_tokens):
            with adapter.record() as captured:
                outputs = adapter.forward(y, prepared.state)
            if outputs.ndim != 3 or outputs.shape[0] != 1 or outputs.shape[1] < 1 or outputs.shape[-1] != adapter.vocab_size:
                raise ValueError("어댑터 로짓은 [1, sequence, vocab_size]여야 합니다")
            if any(h.shape != (adapter.hidden_size,) for h in captured):
                raise ValueError("어댑터 hidden_size와 수집 벡터 차원이 다릅니다")
            if any(h.dtype != captured[-1].dtype for h in captured):
                raise ValueError("레이어별 활성값 dtype이 다른 모델은 현재 저장 형식으로 지원하지 않습니다")
            logits = outputs[0, -1, :]
            mx.eval(logits, *captured)
            # No half-precision conversion: float32 preserves fp16/bf16 values exactly.
            vectors = np.stack([np.array(h.astype(mx.float32)) for h in captured])
            # Quantized BF16 matmul may use different kernels for N prompt rows
            # and one row. A faithful extraction check must keep the same shape.
            projected = adapter.project_logits(captured.final_hidden)[0, -1, :]
            single_projected = adapter.project_logits(captured[-1][None, None, :])[0, 0]
            mx.eval(projected, single_projected)
            actual = np.array(logits.astype(mx.float32))
            reread = np.array(projected.astype(mx.float32))
            single_readout = np.array(single_projected.astype(mx.float32))
            difference = float(np.max(np.abs(actual - reread)))
            max_error = max(max_error, difference)
            if not np.allclose(actual, reread, atol=args.check_atol, rtol=0):
                raise RuntimeError(f"마지막 레이어 readout 검증 실패: max_abs_error={difference}")
            single_error = float(np.max(np.abs(actual - single_readout)))
            max_single_error = max(max_single_error, single_error)
            # Penalties affect selection only. Lens scores always use raw model
            # logits and unmodified hidden states; do not confuse the two.
            selection_logits = logits.astype(mx.float32)[None, :]
            if processors and token_ids:
                history = mx.array(token_ids[-args.penalty_context:], dtype=input_ids.dtype)
                for processor_fn in processors:
                    selection_logits = processor_fn(history, selection_logits)
            if sampler is None:
                token_id = int(mx.argmax(selection_logits).item())
            else:
                logprobs = selection_logits - mx.logsumexp(selection_logits, axis=-1, keepdims=True)
                token_id = int(sampler(logprobs).item())
            generated_score = token_scores(actual, token_id, top_k=1)["target"]
            record = {
                "step": step, "token_id": token_id,
                "token_piece": tokenizer.convert_ids_to_tokens(token_id),
                "decoded_token": tokenizer.decode([token_id], skip_special_tokens=False),
                "is_eos": token_id in eos_ids,
                "source_position": input_ids.shape[1] - 1 + step,
                "source_token_id": last_input if step == 0 else token_ids[-1],
                "activation_dtype": str(captured[-1].dtype).removeprefix("mlx.core."),
                "final_readout_max_abs_error": difference,
                "single_position_vs_actual_max_abs_error": single_error,
                "single_position_readout_score": token_scores(single_readout, token_id, top_k=1)["target"],
                "single_position_top1_token_id": int(np.argmax(single_readout)),
                "actual_output_score": generated_score,
                "actual_top1_token_id": int(np.argmax(actual)),
                "selection_logit_before_sampling": float(selection_logits[0, token_id].item()),
            }
            writer.append(vectors, record)
            token_ids.append(token_id)
            print(f"[{step + 1:03d}] {record['decoded_token']!r}", flush=True)
            if token_id in eos_ids:
                stop_reason = "eos"
                break
            y = mx.array([[token_id]], dtype=input_ids.dtype)
            del outputs, logits, projected, single_projected, captured
            if (step + 1) % 16 == 0:
                mx.clear_cache()
    except KeyboardInterrupt:
        status, stop_reason = "interrupted", "keyboard_interrupt"
    except Exception as exc:
        status, stop_reason, error = "failed", "error", exc
    finally:
        writer.finish(
            status, tokenizer.decode(token_ids, skip_special_tokens=True),
            stop_reason=stop_reason, generated_token_ids=token_ids,
            elapsed_seconds=time.monotonic() - started,
            final_readout_max_abs_error=max_error, error=str(error) if error else None,
            single_position_vs_actual_max_abs_error=max_single_error,
        )
    if error is not None:
        raise error
    print(f"저장 완료: {args.output.resolve()} / {len(token_ids)}개 생성 토큰")
    return 0 if status == "completed" else 130


def main():
    args = parser().parse_args()
    try:
        return capture(args)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
