"""MLX-VLM 0.7.8 dense Qwen3-VL 어댑터."""

from dataclasses import dataclass

from .base import ModelAdapter, PreparedInput


@dataclass
class QwenState:
    cache: object
    kwargs: dict
    rope_deltas: object


class Qwen3VLAdapter(ModelAdapter):
    name = "qwen3_vl"
    vector_site = "decoder_block_output_pre_deepstack_pre_final_norm_last_text_position"
    readout = "original final RMSNorm once, then original tied/untied output head"

    @classmethod
    def supports(cls, model):
        return getattr(model.config, "model_type", None) == "qwen3_vl"

    def validate(self):
        if not self.supports(self.model):
            raise ValueError("Qwen3VLAdapter requires dense Qwen3-VL")
        if not self.layers or any(type(layer).__name__ != "Qwen3VLDecoderLayer" for layer in self.layers):
            raise ValueError("Unexpected Qwen3-VL decoder implementation")
        language = self.model.language_model
        if self.hidden_size < 1 or self.vocab_size < 1:
            raise ValueError("Invalid Qwen3-VL dimensions")
        if language.args.tie_word_embeddings:
            if not callable(getattr(language.model.embed_tokens, "as_linear", None)):
                raise ValueError("Qwen3-VL tied output head is unavailable")
        elif not callable(getattr(language, "lm_head", None)):
            raise ValueError("Qwen3-VL output head is unavailable")

    @property
    def layers(self):
        return self.model.language_model.model.layers

    @property
    def hidden_size(self):
        return self.model.language_model.args.hidden_size

    @property
    def vocab_size(self):
        return self.model.language_model.args.vocab_size

    @property
    def config(self):
        if not hasattr(self, "_config"):
            from mlx_vlm.utils import load_config

            self._config = load_config(self.model.model_path)
        return self._config

    @property
    def quantization(self):
        return self.config.get("quantization", self.config.get("quantization_config"))

    def prepare(self, prompt, image, max_input_tokens):
        from mlx_vlm.models.cache import make_prompt_cache
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import prepare_inputs

        formatted = apply_chat_template(self.processor, self.config, prompt, num_images=int(image is not None))
        inputs = dict(prepare_inputs(
            self.processor, images=[str(image)] if image is not None else None,
            prompts=formatted, image_token_index=self.model.config.image_token_index,
        ))
        ids = inputs.pop("input_ids")
        mask = inputs.pop("attention_mask", None)
        if ids.ndim != 2 or ids.shape[0] != 1 or not 0 < ids.shape[1] <= max_input_tokens:
            raise ValueError(f"배치는 1이어야 하며 입력 토큰은 1..{max_input_tokens}여야 합니다: {ids.shape}")
        if mask is not None and not bool(self.mx.all(mask == 1).item()):
            raise ValueError("패딩이 있는 입력은 지원하지 않습니다")
        visual_ids = {getattr(self.model.config, key, None) for key in ("image_token_id", "video_token_id")}
        if int(ids[0, -1].item()) in visual_ids:
            raise ValueError("마지막 입력 위치는 이미지가 아닌 텍스트여야 합니다")
        features = self.model.get_input_embeddings(ids, mask=mask, **inputs)
        state = QwenState(
            make_prompt_cache(self.model.language_model), features.to_dict(), features.rope_deltas,
        )
        return PreparedInput(ids, formatted, state)

    def forward(self, input_ids, state):
        outputs = self.model.language_model(input_ids, cache=state.cache, **state.kwargs)
        # 다음 단계는 이미지 재인코딩 없이 같은 캐시와 멀티모달 위치 정보를 사용한다.
        state.kwargs = {"rope_deltas": state.rope_deltas}
        return outputs.logits

    def project_logits(self, hidden):
        language = self.model.language_model
        decoder = language.model
        normalized = decoder.norm(hidden) if decoder.norm is not None else hidden
        if language.args.tie_word_embeddings:
            return decoder.embed_tokens.as_linear(normalized)
        return language.lm_head(normalized)
