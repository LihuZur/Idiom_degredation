"""Decoder LLM model runner implementation (STAGE3_PLAN §3.1)."""

import logging
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.base import FormattedInput, Model, ModelSpec, Precision, Prediction
from models.device import select_device

logger = logging.getLogger(__name__)


class DecoderRunner(Model):
    """Low-level transformers runner for instruction-tuned decoder models."""

    id: str
    device: torch.device
    _spec: ModelSpec
    _precision: Precision
    _fallback_template: str | None
    _tokenizer: Any
    _model: Any

    def __init__(
        self,
        spec: ModelSpec,
        *,
        precision: Precision | None = None,
        fallback_template: str | None = None,
    ) -> None:
        self._spec = spec
        self.id = spec.name
        self.device = select_device()
        self._fallback_template = fallback_template

        # Resolve precision: config > spec.default_precision, then cpu-force-fp32
        resolved_precision = precision if precision is not None else spec.default_precision
        if self.device.type == "cpu":
            if resolved_precision != "fp32":
                logger.warning(
                    f"Forcing precision to fp32 on CPU (requested: {resolved_precision})"
                )
                resolved_precision = "fp32"
        elif self.device.type == "mps" and resolved_precision == "bf16":
            try:
                # Test MPS bf16 compatibility
                torch.tensor([], dtype=torch.bfloat16, device="mps")
            except Exception:
                logger.warning("MPS does not support bf16 on this system. Falling back to fp32.")
                resolved_precision = "fp32"

        self._precision = resolved_precision

        dtype_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        torch_dtype = dtype_map[self._precision]

        # Explicit flushed prints (not tqdm) so there's always visible progress in
        # non-tty environments (e.g. Colab `!` cells), where HF's own download bars
        # can render invisibly for the whole download and this is the only feedback
        # until it's done - this step can take a long time on a first, cold run.
        print(f"Loading tokenizer for {spec.hf_repo}@{spec.revision}...", flush=True)
        self._tokenizer = AutoTokenizer.from_pretrained(
            spec.hf_repo,
            revision=spec.revision,
            padding_side="left",
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        print("Tokenizer loaded.", flush=True)

        print(
            f"Loading model {spec.hf_repo}@{spec.revision} "
            f"(dtype={torch_dtype}, device={self.device})... "
            "this downloads the weights on a first/cold run and can take a while",
            flush=True,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            spec.hf_repo,
            revision=spec.revision,
            dtype=torch_dtype,
        )
        self._model.to(self.device)
        self._model.eval()
        print(f"Model loaded and moved to {self.device}.", flush=True)

    def predict(self, batch: list[FormattedInput]) -> list[Prediction]:
        if not batch:
            return []

        prompts: list[str] = []
        for fi in batch:
            messages = fi.meta.get("messages", [])
            system_content = ""
            user_content = ""
            for msg in messages:
                if msg["role"] == "system":
                    system_content = msg["content"]
                elif msg["role"] == "user":
                    user_content = msg["content"]

            if self._tokenizer.chat_template:
                prompt = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                if not isinstance(prompt, str):
                    raise TypeError("Expected apply_chat_template to return a string")
            else:
                if not self._fallback_template:
                    raise ValueError(
                        f"Tokenizer for {self.id} has no chat template and no fallback template was provided."
                    )
                prompt = self._fallback_template.format(
                    system=system_content,
                    user=user_content,
                )
            prompts.append(prompt)

        # Retrieve generation kwargs from the first item
        gen_kwargs = batch[0].meta.get("generate_kwargs", {})
        temperature = float(gen_kwargs.get("temperature", 0.0))
        max_new_tokens = int(gen_kwargs.get("max_new_tokens", 32))
        do_sample = bool(gen_kwargs.get("do_sample", False))

        kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }
        if do_sample and temperature > 0.0:
            kwargs["temperature"] = temperature

        inputs = self._tokenizer(prompts, padding=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(**inputs, **kwargs)

        predictions: list[Prediction] = []
        for i, fi in enumerate(batch):
            input_len = inputs["input_ids"][i].shape[0]
            output_tokens = outputs[i][input_len:]
            raw_text = self._tokenizer.decode(output_tokens, skip_special_tokens=True).strip()
            predictions.append(
                Prediction(
                    id=fi.id,
                    raw=raw_text,
                    parsed=None,
                    meta={},
                )
            )

        return predictions
