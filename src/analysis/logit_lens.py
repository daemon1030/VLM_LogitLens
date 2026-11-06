"""저장된 생성 벡터 → 레이어별 Logit Lens 점수·확률·순위.

실행: python -m src.analysis.logit_lens --run runs/demo
"""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

from src.common.model import checkpoint_identity, load_adapter, package_versions
from src.common.scoring import token_scores
from src.common.storage import read_run, write_json


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--run", type=Path, required=True, help="save_vectors의 출력 폴더")
    result.add_argument("--output", type=Path, help="새 분석 폴더; 기본 RUN/logit_lens")
    result.add_argument("--model", help="이동한 동일 체크포인트의 경로 또는 HF ID")
    result.add_argument("--revision", help="--model에 HF ID를 지정할 때 사용할 commit/tag")
    result.add_argument("--adapter", default="auto", help="수집과 동일한 어댑터: auto, 이름 또는 package.module:AdapterClass")
    result.add_argument("--steps", type=int, nargs="+", help="분석할 생성 위치; 0부터")
    result.add_argument("--layers", type=int, nargs="+", help="분석할 레이어; 1부터")
    result.add_argument("--target-id", type=int, action="append", default=[], help="추가 관심 토큰 ID; 반복 가능")
    result.add_argument("--target-text", action="append", default=[], help="추가 관심 문자열; 단일 토큰만 허용")
    result.add_argument("--top-k", type=int, default=5)
    result.add_argument("--preview-step", type=int, default=0, help="터미널에 출력할 생성 위치; 0부터")
    return result


def select_indices(values, size, first, label):
    chosen = list(range(first, size + first)) if values is None else list(dict.fromkeys(values))
    if not chosen or any(index < first or index >= size + first for index in chosen):
        raise ValueError(f"{label} 범위는 {first}..{size + first - 1}입니다")
    return chosen


def resolve_targets(tokenizer, ids, texts, vocab_size):
    result = list(ids)
    for text in texts:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(
                f"{text!r}은 단일 토큰이 아닙니다: {encoded}. "
                "토큰별 --target-id를 사용하세요. 앞 공백도 토큰화에 영향을 줍니다."
            )
        result.append(encoded[0])
    if any(token_id < 0 or token_id >= vocab_size for token_id in result):
        raise ValueError("관심 토큰 ID가 어휘 범위를 벗어났습니다")
    return list(dict.fromkeys(result))


def analyze(args):
    metadata, vectors, records = read_run(args.run)
    steps = select_indices(args.steps, len(records), 0, "step")
    layers = select_indices(args.layers, metadata["num_layers"], 1, "layer")
    if args.top_k < 1:
        raise ValueError("top-k는 1 이상이어야 합니다")
    output = args.output or args.run / "logit_lens"
    if output.exists():
        raise FileExistsError(f"출력 폴더가 이미 있습니다: {output}")
    model_ref = args.model or metadata["checkpoint"]["path"]
    adapter = load_adapter(model_ref, args.revision, args.adapter)
    mx, model, tokenizer = adapter.mx, adapter.model, adapter.tokenizer
    if checkpoint_identity(model)["sha256"] != metadata["checkpoint"]["sha256"]:
        raise ValueError("수집 당시와 가중치·설정·토크나이저가 다른 체크포인트입니다")
    adapter.check_capture(metadata)
    extra_targets = resolve_targets(tokenizer, args.target_id, args.target_text,
                                    adapter.vocab_size)
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running", "run": str(args.run.resolve()),
        "checkpoint_sha256": metadata["checkpoint"]["sha256"],
        "capture_packages": metadata["packages"], "analysis_packages": package_versions(),
        "steps": steps, "layers": layers, "extra_target_ids": extra_targets,
        "rank_definition": "1 + number of vocabulary logits strictly greater than target logit",
        "probability_definition": "full-vocabulary softmax; temperature=1; no sampling penalties",
        "readout": adapter.readout, "adapter": adapter.identity(),
    }
    write_json(output / "analysis.json", report)
    fields = ["step", "layer", "generated_token_id", "generated_token_text", "target_kind",
              "target_token_id", "target_token_text", "logit", "probability", "rank"]
    last_layer_errors = []
    try:
        with (output / "target_scores.csv").open("w", encoding="utf-8-sig", newline="") as csv_stream, \
                (output / "top_tokens.jsonl").open("w", encoding="utf-8") as top_stream:
            writer = csv.DictWriter(csv_stream, fieldnames=fields)
            writer.writeheader()
            for step in steps:
                record = records[step]
                dtype_name = record["activation_dtype"]
                if dtype_name not in ("float16", "bfloat16", "float32"):
                    raise ValueError(f"지원하지 않는 원본 활성값 dtype: {dtype_name}")
                activation_dtype = getattr(mx, dtype_name)
                for layer in layers:
                    # One position/layer at a time: never allocate [T,L,V] logits.
                    h = mx.array(np.asarray(vectors[step, layer - 1]), dtype=activation_dtype)
                    scores = adapter.project_logits(h[None, None, :])[0, 0]
                    mx.eval(scores)
                    values = np.array(scores.astype(mx.float32))
                    generated_id = record["token_id"]
                    summary = token_scores(values, generated_id, min(args.top_k, len(values)))
                    generated = summary["target"]
                    candidates = summary["top_tokens"]
                    for candidate in candidates:
                        candidate["token_text"] = tokenizer.decode(
                            [candidate["token_id"]], skip_special_tokens=False,
                        )
                    top_stream.write(json.dumps({
                        "step": step, "layer": layer, "generated_token_id": generated_id,
                        "generated_token_text": record["decoded_token"], "top_tokens": candidates,
                    }, ensure_ascii=False) + "\n")
                    targets = [(generated_id, "generated")]
                    targets += [(token_id, "requested") for token_id in extra_targets]
                    for token_id, kind in targets:
                        target = (generated if token_id == generated_id else
                                  token_scores(values, token_id, top_k=1)["target"])
                        writer.writerow({
                            "step": step, "layer": layer, "generated_token_id": generated_id,
                            "generated_token_text": record["decoded_token"], "target_kind": kind,
                            "target_token_id": token_id,
                            "target_token_text": tokenizer.decode([token_id], skip_special_tokens=False),
                            "logit": target["logit"], "probability": target["probability"],
                            "rank": target["rank"],
                        })
                    if layer == metadata["num_layers"]:
                        # Compare like-for-like: a single-position readout can
                        # differ from prompt-wide BF16 matmul despite identical h.
                        reference = record.get("single_position_readout_score", record["actual_output_score"])
                        difference = abs(generated["logit"] - reference["logit"])
                        last_layer_errors.append(difference)
                        if difference > 0.05:
                            raise RuntimeError(f"저장 벡터의 단일 위치 readout 재검증 실패: step={step}, error={difference}")
                    if step == args.preview_step:
                        print(f"layer {layer:02d} | top1={candidates[0]['token_text']!r} | "
                              f"생성 토큰={record['decoded_token']!r} | "
                              f"p={generated['probability']:.6f} | rank={generated['rank']}")
                    del h, scores
                mx.clear_cache()
        report.update(status="completed", max_last_layer_target_logit_error=
                      max(last_layer_errors) if last_layer_errors else None,
                      last_layer_verified=bool(last_layer_errors))
    except BaseException as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        write_json(output / "analysis.json", report)
    print(f"분석 저장 완료: {output.resolve()}")
    return 0


def main():
    args = parser().parse_args()
    try:
        return analyze(args)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
