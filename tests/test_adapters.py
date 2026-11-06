"""Qwen과 다른 경로·출력 형식의 모델로 공통 파이프라인과 확장 로딩을 검증."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from src.adapters import select_adapter
from src.adapters.base import ModelAdapter, PreparedInput
from src.adapters.qwen3_vl import Qwen3VLAdapter
from src.analysis import logit_lens
from src.capture import save_vectors
from src.common import model as model_utils
from src.common.storage import read_run


class ArrayBlock:
    def __call__(self, value):
        return value + 0.1


class TupleBlock:
    def __call__(self, value):
        return value + 0.2, {"unused": True}


class ToyAdapter(ModelAdapter):
    """언어 모델 속성 경로도, 블록 출력 형식도 Qwen과 다른 시험용 어댑터."""

    name = "toy_tuple"
    vector_site = "toy_block_output_before_norm"
    readout = "toy final scale once, then rolling head"

    @classmethod
    def supports(cls, model):
        return model.config.model_type == "toy"

    def validate(self):
        if not self.layers:
            raise ValueError("missing toy layers")

    @property
    def layers(self):
        return self.model.blocks

    @property
    def hidden_size(self):
        return 4

    @property
    def vocab_size(self):
        return 4

    @property
    def quantization(self):
        return None

    def extract_hidden(self, block, output):
        return output[0] if isinstance(output, tuple) else output

    def prepare(self, prompt, image, max_input_tokens):
        if image is not None:
            raise ValueError("toy images unsupported")
        return PreparedInput(np.array([[0, 1]]), prompt, {"calls": []})

    def forward(self, input_ids, state):
        state["calls"].append(input_ids.tolist())
        value = np.eye(4, dtype=np.float32)[input_ids]
        for block in self.layers:
            value = self.extract_hidden(block, block(value))
        return self.project_logits(value)

    def project_logits(self, hidden):
        return np.roll(hidden * 0.5, 1, axis=-1)


class AdapterTests(unittest.TestCase):
    def make_model(self, folder="unused"):
        return SimpleNamespace(
            config=SimpleNamespace(model_type="toy"),
            blocks=[ArrayBlock(), TupleBlock()], model_path=folder, eval=lambda: None,
        )

    def test_explicit_extension_and_unknown_auto_selection(self):
        model = self.make_model()
        adapter = select_adapter(None, model, None, f"{__name__}:ToyAdapter")
        self.assertIsInstance(adapter, ToyAdapter)
        with self.assertRaisesRegex(ValueError, "--adapter"):
            select_adapter(None, model, None)
        with self.assertRaisesRegex(ValueError, "지원하지"):
            select_adapter(None, model, None, "qwen3_vl")
        for invalid in ("missing", f"{__name__}:ArrayBlock", "missing_package:Adapter"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                select_adapter(None, model, None, invalid)

    def test_builtin_auto_selection_validates_qwen(self):
        block_type = type("Qwen3VLDecoderLayer", (ArrayBlock,), {})
        embedding = SimpleNamespace(as_linear=lambda h: h)
        model = SimpleNamespace(
            config=SimpleNamespace(model_type="qwen3_vl"),
            language_model=SimpleNamespace(
                model=SimpleNamespace(layers=[block_type()], embed_tokens=embedding),
                args=SimpleNamespace(hidden_size=4, vocab_size=4, tie_word_embeddings=True),
            ),
        )
        self.assertIsInstance(select_adapter(None, model, None), Qwen3VLAdapter)
        model.language_model.model.layers = [ArrayBlock()]
        with self.assertRaisesRegex(ValueError, "decoder"):
            select_adapter(None, model, None)

    def test_recorder_handles_mixed_outputs_and_restores_inherited_call(self):
        class Inherited(ArrayBlock):
            pass

        blocks = [Inherited(), TupleBlock()]
        original = TupleBlock.__call__
        adapter = ToyAdapter(None, SimpleNamespace(blocks=blocks), None)
        with adapter.record() as recorded:
            value = blocks[0](np.zeros((1, 2, 4)))
            output = blocks[1](value)
            self.assertIsInstance(output, tuple)
        np.testing.assert_allclose(recorded[0], 0.1)
        np.testing.assert_allclose(recorded[1], 0.3)
        self.assertEqual(recorded.final_hidden.shape, (1, 2, 4))
        self.assertNotIn("__call__", Inherited.__dict__)
        self.assertIs(TupleBlock.__call__, original)

    def test_recorder_restores_on_invalid_output_and_rejects_nested_capture(self):
        block = ArrayBlock()
        original = ArrayBlock.__call__
        with self.assertRaisesRegex(ValueError, "sequence"):
            with model_utils.record_last_position([block]):
                block(np.zeros((2, 4)))
        self.assertIs(ArrayBlock.__call__, original)
        with model_utils.record_last_position([block]):
            with self.assertRaisesRegex(RuntimeError, "중첩"):
                with model_utils.record_last_position([block]):
                    pass
            block(np.zeros((1, 1, 4)))
        self.assertIs(ArrayBlock.__call__, original)

    def test_adapter_contract_and_legacy_result_checks(self):
        adapter = ToyAdapter(None, self.make_model(), None)
        metadata = {"adapter": adapter.identity(), "num_layers": 2, "hidden_size": 4}
        adapter.check_capture(metadata)
        for changes in ({"contract_version": 2}, {"vector_site": "after_norm"}, {"name": "other"}):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "계약"):
                adapter.check_capture(dict(metadata, adapter=dict(adapter.identity(), **changes)))
        with self.assertRaisesRegex(ValueError, "히든"):
            adapter.check_capture(dict(metadata, hidden_size=8))
        with self.assertRaisesRegex(ValueError, "기존 결과"):
            adapter.check_capture({"num_layers": 2, "hidden_size": 4})
        # 어댑터 필드가 없던 기존 Qwen 결과도 동일한 수집 지점/차원이면 읽는다.
        qwen_model = SimpleNamespace(language_model=SimpleNamespace(
            model=SimpleNamespace(layers=[object(), object()]), args=SimpleNamespace(hidden_size=4),
        ))
        qwen = Qwen3VLAdapter(None, qwen_model, None)
        qwen.check_capture({"num_layers": 2, "hidden_size": 4, "vector_site": qwen.vector_site})

    def test_extension_capture_and_analysis_without_qwen_model_attributes(self):
        mx = SimpleNamespace(
            array=np.array, float32=np.float32, eval=lambda *args: None,
            argmax=np.argmax, clear_cache=lambda: None,
            random=SimpleNamespace(seed=lambda seed: None),
        )
        tokenizer = SimpleNamespace(
            eos_token_id=3, encode=lambda text, **kwargs: [int(text)],
            decode=lambda ids, skip_special_tokens=False: "".join(
                str(i) for i in ids if not (skip_special_tokens and i == 3)),
            convert_ids_to_tokens=lambda i: str(i),
        )
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            checkpoint = folder / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "config.json").write_text('{"model_type": "toy"}')
            model = self.make_model(checkpoint)
            processor = SimpleNamespace(tokenizer=tokenizer)
            selection = f"{__name__}:ToyAdapter"
            run = folder / "run"
            with patch.object(model_utils, "runtime", return_value=(mx, lambda *args, **kwargs: (model, processor))), \
                    redirect_stdout(io.StringIO()):
                capture_args = save_vectors.parser().parse_args([
                    "--prompt", "test", "--model", str(checkpoint), "--adapter", selection,
                    "--max-tokens", "10", "--output", str(run),
                ])
                self.assertEqual(save_vectors.capture(capture_args), 0)
                metadata, vectors, records = read_run(run)
                self.assertEqual(metadata["adapter"]["name"], "toy_tuple")
                self.assertEqual(metadata["generated_token_ids"], [2, 3])
                self.assertEqual(vectors.shape, (2, 2, 4))
                self.assertEqual([r["source_token_id"] for r in records], [1, 2])
                self.assertTrue(records[-1]["is_eos"])
                self.assertEqual(metadata["final_readout_max_abs_error"], 0)
                np.testing.assert_allclose(vectors[0, 1], np.eye(4)[1] + 0.3)
                args = logit_lens.parser().parse_args(["--run", str(run), "--adapter", selection])
                self.assertEqual(logit_lens.analyze(args), 0)
                report = json.loads((run / "logit_lens/analysis.json").read_text())
                self.assertTrue(report["last_layer_verified"])
                self.assertEqual(report["max_last_layer_target_logit_error"], 0)
                candidates = (run / "logit_lens/top_tokens.jsonl").read_text().splitlines()
                self.assertEqual(len(candidates), 4)
                # 체크포인트가 같아도 수집 계약이 다르면 결과를 만들기 전에 거부한다.
                metadata["adapter"]["contract_version"] = 2
                (run / "metadata.json").write_text(json.dumps(metadata))
                rejected = run / "rejected"
                args.output = rejected
                with self.assertRaisesRegex(ValueError, "계약"):
                    logit_lens.analyze(args)
                self.assertFalse(rejected.exists())


if __name__ == "__main__":
    unittest.main()
