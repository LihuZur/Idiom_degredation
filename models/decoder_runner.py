"""Decoder LLM model runner implementation (STAGE3_PLAN §3.1)."""

import logging
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.base import FormattedInput, Model, ModelSpec, Precision, Prediction
from models.device import select_device

logger = logging.getLogger(__name__)


def span_token_indices(offsets: list[tuple[int, int]], lo: int, hi: int) -> list[int]:
    """Indices of tokens whose character range overlaps [lo, hi) (special tokens have empty ranges and are skipped)."""
    return [n for n, (a, b) in enumerate(offsets) if b > a and a < hi and b > lo]


class DecoderRunner(Model):
    """Low-level transformers runner for instruction-tuned decoder models."""

    id: str
    device: torch.device
    _spec: ModelSpec
    _precision: Precision
    _fallback_template: str | None
    _system_role_supported: bool
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

        # Some chat templates (e.g. Gemma-2) reject a "system" message outright with
        # a jinja2 TemplateError - probe once at load time and fold system into the
        # user turn for those templates instead of failing every predict() call.
        self._system_role_supported = True
        if self._tokenizer.chat_template:
            try:
                self._tokenizer.apply_chat_template(
                    [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                self._system_role_supported = False

    def _build_prompt(self, fi: FormattedInput) -> str:
        messages = fi.meta.get("messages", [])
        system_content = ""
        user_content = ""
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            elif msg["role"] == "user":
                user_content = msg["content"]

        if not self._tokenizer.chat_template:
            if not self._fallback_template:
                raise ValueError(
                    f"Tokenizer for {self.id} has no chat template and no fallback template was provided."
                )
            return self._fallback_template.format(system=system_content, user=user_content)

        if system_content and not self._system_role_supported:
            chat_messages = [{"role": "user", "content": f"{system_content}\n\n{user_content}"}]
        else:
            chat_messages = messages
        prompt = self._tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if not isinstance(prompt, str):
            raise TypeError("Expected apply_chat_template to return a string")
        return prompt

    def score_options(
        self, batch: list[FormattedInput], options: list[list[str]], chunk: int = 8
    ) -> list[list[float]]:
        """Sum of log-probabilities of each option text as the continuation of the item's prompt (plan v3, outcome
        `p_gold`): the same prompt as `predict`, teacher-forced, no generation. One list of scores per item."""
        rows: list[
            tuple[int, list[int], int]
        ] = []  # (item index, full token ids, number of option tokens)
        for n, (fi, opts) in enumerate(zip(batch, options, strict=True)):
            prompt = self._build_prompt(fi)
            p_ids = self._tokenizer(prompt)["input_ids"]
            for o in opts:
                f_ids = self._tokenizer(prompt + o)["input_ids"]
                k = max(1, len(f_ids) - len(p_ids))
                rows.append((n, list(f_ids), k))
        scores: list[list[float]] = [[] for _ in batch]
        pad = int(self._tokenizer.pad_token_id)
        for start in range(0, len(rows), chunk):
            part = rows[start : start + chunk]
            width = max(len(r[1]) for r in part)
            ids = torch.tensor(
                [[pad] * (width - len(r[1])) + r[1] for r in part], device=self.device
            )
            mask = torch.tensor(
                [[0] * (width - len(r[1])) + [1] * len(r[1]) for r in part], device=self.device
            )
            with torch.no_grad():
                logp = torch.log_softmax(
                    self._model(input_ids=ids, attention_mask=mask).logits.float(), dim=-1
                )
            for j, (n, _, k) in enumerate(part):
                target = ids[j, width - k :]
                pred = logp[
                    j, width - k - 1 : width - 1
                ]  # logits at t predict token t + 1 (left padding)
                scores[n].append(float(pred.gather(1, target.unsqueeze(1)).sum()))
        return scores

    def hidden_states(
        self, fi: FormattedInput, text: str, span: tuple[int, int] | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Interpretability pilot (plan v3 F1): the residual stream of every layer for the item's prompt (the same
        prompt as `predict`), one item, no padding. Returns (last prompt token [layers + 1, hidden],
        mean over the tokens of `span` [layers + 1, hidden] or None). `span` is a character span inside `text`, and
        `text` must appear verbatim in the prompt; otherwise the span result is None."""
        prompt = self._build_prompt(fi)
        try:
            enc = self._tokenizer(prompt, return_tensors="pt", return_offsets_mapping=True)
            offsets = [tuple(o) for o in enc.pop("offset_mapping")[0].tolist()]
        except NotImplementedError:  # slow tokenizers have no offsets: last token only
            enc = self._tokenizer(prompt, return_tensors="pt")
            offsets, span = [], None
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self._model(**enc, output_hidden_states=True)
        states = (
            torch.stack([h[0] for h in out.hidden_states]).float().cpu()
        )  # [layers + 1, tokens, hidden]
        pooled = None
        start = prompt.find(text) if span is not None else -1
        if span is not None and start >= 0:
            idx = span_token_indices(offsets, start + span[0], start + span[1])
            if idx:
                pooled = states[:, idx, :].mean(dim=1)
        return states[:, -1, :], pooled

    def predict(self, batch: list[FormattedInput]) -> list[Prediction]:
        if not batch:
            return []

        prompts = [self._build_prompt(fi) for fi in batch]

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
