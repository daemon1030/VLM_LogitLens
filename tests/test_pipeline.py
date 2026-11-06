"""실제 GPU 대신 작은 결정적 모델로 저장→재분석의 토큰 정렬을 검증한다."""

from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from src.analysis import logit_lens
from src.capture import save_vectors
from src.adapters.qwen3_vl import Qwen3VLAdapter
from src.common.model import record_last_position
from src.common.scoring import token_scores
from src.common.storage import RunWriter, read_run


class ScoringTests(unittest.TestCase):
    def test_stable_full_vocabulary_probability_and_ties(self):
        scores = token_scores([10000, 10000, 9999, -10000], 1, 3)
        self.assertEqual(scores["target"]["rank"], 1)
        self.assertAlmostEqual(scores["target"]["probability"], 1 / (2 + np.exp(-1)))
        self.assertEqual([item["token_id"] for item in scores["top_tokens"]], [0, 1, 2])
        self.assertEqual(token_scores([2, 2, 1], 2, 1)["target"]["rank"], 3)

    def test_multitoken_target_rejected(self):
        tokenizer = SimpleNamespace(encode=lambda *args, **kwargs: [1, 2])
        with self.assertRaisesRegex(ValueError, "단일 토큰"):
            logit_lens.resolve_targets(tokenizer, [], ["two tokens"], 6)


class Block:
    def __init__(self, offset):
        self.offset = offset

    def __call__(self, value):
        return value + self.offset


class Embedding:
    def __call__(self, ids):
        return np.eye(6, dtype=np.float32)[ids]

    def as_linear(self, hidden):
        # The largest input coordinate at i predicts i+1.
        result = np.roll(hidden, 1, axis=-1)
        # Model a BF16 kernel whose rounding depends on row count. This must
        # not be mistaken for capturing the wrong layer or normalizing twice.
        if hidden.ndim >= 2 and hidden.shape[-2] > 1:
            result[..., 3] += 0.125
        return result


class ToyLanguage:
    def __init__(self):
        self.args = SimpleNamespace(tie_word_embeddings=True, hidden_size=6, vocab_size=6)
        self.model = SimpleNamespace(
            layers=[Block(0.1), Block(0.2)], norm=lambda h: h * 0.5,
            embed_tokens=Embedding(),
        )
        self.layers = self.model.layers

    def __call__(self, inputs, cache=None, inputs_embeds=None, **kwargs):
        hidden = inputs_embeds if inputs_embeds is not None else self.model.embed_tokens(inputs)
        for layer in self.layers:
            hidden = layer(hidden)
        return SimpleNamespace(logits=self.model.embed_tokens.as_linear(self.model.norm(hidden)))


class ToyTokenizer:
    eos_token_id = 5
    stopping_criteria = SimpleNamespace(eos_token_ids=[5])

    def decode(self, ids, skip_special_tokens=False):
        return "".join(str(i) for i in ids if not (skip_special_tokens and i == 5))

    def convert_ids_to_tokens(self, index):
        return f"token_{index}"

    def encode(self, text, add_special_tokens=False):
        return [int(text)]


class CaptureTests(unittest.TestCase):
    def test_recorder_restores_original_call_on_error(self):
        blocks = [Block(1), Block(2)]
        original = Block.__call__
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with record_last_position(blocks):
                raise RuntimeError("intentional")
        self.assertIs(Block.__call__, original)

    def test_projection_applies_norm_once_and_handles_untied_head(self):
        model = SimpleNamespace(language_model=ToyLanguage())
        hidden = np.eye(6, dtype=np.float32)[2]
        adapter = Qwen3VLAdapter(None, model, None)
        result = adapter.project_logits(hidden)
        self.assertEqual(result[3], 0.5)
        model.language_model.args.tie_word_embeddings = False
        model.language_model.lm_head = lambda h: h * 3
        np.testing.assert_array_equal(adapter.project_logits(hidden), hidden * 1.5)

    def test_capture_to_analysis_preserves_prediction_alignment(self):
        mx = SimpleNamespace(
            array=np.array, float32=np.float32, eval=lambda *args: None,
            all=np.all, argmax=np.argmax, clear_cache=lambda: None,
            random=SimpleNamespace(seed=lambda seed: None),
        )
        language = ToyLanguage()
        model = SimpleNamespace(
            language_model=language, model_path="toy",
            config=SimpleNamespace(image_token_index=99, image_token_id=99, video_token_id=98),
        )

        def features(ids, **kwargs):
            result = {"inputs_embeds": language.model.embed_tokens(ids),
                      "rope_deltas": np.array([[0]])}
            return SimpleNamespace(to_dict=lambda: result, rope_deltas=result["rope_deltas"])

        model.get_input_embeddings = features
        tokenizer = ToyTokenizer()
        processor = SimpleNamespace(tokenizer=tokenizer)
        utils = ModuleType("mlx_vlm.utils")
        utils.load_config = lambda path: {"model_type": "qwen3_vl"}
        utils.prepare_inputs = lambda *args, **kwargs: {
            "input_ids": np.array([[1, 2]]), "attention_mask": np.ones((1, 2)),
        }
        cache = ModuleType("mlx_vlm.models.cache")
        cache.make_prompt_cache = lambda lm: []
        prompts = ModuleType("mlx_vlm.prompt_utils")
        prompts.apply_chat_template = lambda *args, **kwargs: "formatted"
        identity = {"path": "toy", "sha256": "test-model", "files": {}}

        with tempfile.TemporaryDirectory() as temporary, \
                patch.dict("sys.modules", {"mlx_vlm.models.cache": cache,
                                          "mlx_vlm.prompt_utils": prompts,
                                          "mlx_vlm.utils": utils}), \
                patch.object(save_vectors, "load_adapter", return_value=Qwen3VLAdapter(mx, model, processor)), \
                patch.object(save_vectors, "checkpoint_identity", return_value=identity), \
                patch.object(logit_lens, "load_adapter", return_value=Qwen3VLAdapter(mx, model, processor)), \
                patch.object(logit_lens, "checkpoint_identity", return_value=identity), \
                redirect_stdout(io.StringIO()):
            run = Path(temporary) / "run"
            args = save_vectors.parser().parse_args([
                "--prompt", "test", "--output", str(run), "--max-tokens", "10",
            ])
            self.assertEqual(save_vectors.capture(args), 0)
            metadata, vectors, records = read_run(run)
            self.assertEqual(metadata["generated_token_ids"], [3, 4, 5])
            self.assertEqual(vectors.shape, (3, 2, 6))
            self.assertEqual(metadata["stop_reason"], "eos")
            self.assertEqual(metadata["final_readout_max_abs_error"], 0)
            self.assertAlmostEqual(metadata["single_position_vs_actual_max_abs_error"], 0.125)
            self.assertAlmostEqual(records[0]["actual_output_score"]["logit"] -
                                   records[0]["single_position_readout_score"]["logit"], 0.125)
            self.assertEqual([r["source_token_id"] for r in records], [2, 3, 4])
            self.assertEqual([r["source_position"] for r in records], [1, 2, 3])
            np.testing.assert_allclose(vectors[0, 0], np.eye(6)[2] + 0.1)
            np.testing.assert_allclose(vectors[0, 1], np.eye(6)[2] + 0.3)
            self.assertTrue(records[-1]["is_eos"])
            self.assertEqual((run / "output.txt").read_text(), "34")
            lens_args = logit_lens.parser().parse_args([
                "--run", str(run), "--target-text", "2",
            ])
            self.assertEqual(logit_lens.analyze(lens_args), 0)
            report = json.loads((run / "logit_lens" / "analysis.json").read_text())
            self.assertEqual(report["status"], "completed")
            self.assertTrue(report["last_layer_verified"])
            self.assertEqual(report["max_last_layer_target_logit_error"], 0)
            with (run / "logit_lens" / "target_scores.csv").open(encoding="utf-8-sig") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 3 * 2 * 2)
            self.assertTrue(all(row["rank"] == "1" for row in rows if row["target_kind"] == "generated"))
            with self.assertRaises(FileExistsError):
                save_vectors.capture(args)

    def test_failed_capture_not_silently_analyzed(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder) / "failed"
            writer = RunWriter(run, 2, 2, 3, {"num_layers": 2, "hidden_size": 3})
            writer.finish("failed", "", error="test error")
            with self.assertRaisesRegex(ValueError, "not complete"):
                read_run(run)


if __name__ == "__main__":
    unittest.main()
