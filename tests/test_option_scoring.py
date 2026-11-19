"""Option scoring (p_gold): log-prob alignment under left padding, and p_gold in the evaluator (offline stubs)."""

from typing import Any, cast

import torch

from augmentation.base import AugmentedRow
from eval.config import EvalConfig
from eval.sst2 import Sst2Evaluator
from models.base import FormattedInput, Model, Prediction
from models.decoder_runner import DecoderRunner


class _Tok:
    """One token per character: 'a' -> 1, 'b' -> 2, ...; pad 0."""

    pad_token_id = 0

    def __call__(self, text: str) -> dict[str, list[int]]:
        return {"input_ids": [ord(c) - 96 for c in text]}


class _NextIsPlusOne(torch.nn.Module):
    """Logits that strongly predict token (current + 1) at every position."""

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Any:
        vocab = 30
        logits = torch.full((*input_ids.shape, vocab), -20.0)
        nxt = (input_ids + 1).clamp(max=vocab - 1)
        logits.scatter_(2, nxt.unsqueeze(-1), 20.0)
        return type("Out", (), {"logits": logits})()


def _runner() -> DecoderRunner:
    r = DecoderRunner.__new__(DecoderRunner)
    r._tokenizer = _Tok()  # type: ignore[assignment]
    r._model = _NextIsPlusOne()  # type: ignore[assignment]
    r.device = torch.device("cpu")
    r._build_prompt = lambda fi: fi.prompt  # type: ignore[method-assign]
    return r


def test_scores_follow_the_option_tokens_with_left_padding() -> None:
    batch = [
        FormattedInput(id="1", prompt="a", meta={}),
        FormattedInput(id="2", prompt="abcd", meta={}),
    ]
    # after "a" the model expects "b", then "c": "bc" is likely, "bd" is not; same after "abcd" for "ef" vs "eg"
    s = _runner().score_options(batch, [["bc", "bd"], ["ef", "eg", "e"]], chunk=3)
    assert s[0][0] > -1 and s[0][1] < -10
    assert s[1][0] > -1 and s[1][1] < -10 and s[1][2] > -1


def test_evaluator_attaches_p_gold() -> None:
    ev = Sst2Evaluator(EvalConfig(model="olmo-2-1b-instruct"))
    ev.score_options = True
    ex = AugmentedRow(
        id="1",
        variant="original",
        x="good film",
        y="1",
        augmenter_model="",
        prompt_hash="",
        meta={},
    )

    class _M:
        def score_options(
            self, batch: list[FormattedInput], options: list[list[str]]
        ) -> list[list[float]]:
            assert options == [["negative", "positive"]]
            return [[-3.0, -1.0]]

    out = ev.attach_option_scores(
        cast(Model, _M()), [ev.format(ex)], [ex], [Prediction(id="1", raw="positive", parsed=None)]
    )
    p = out[0].meta["p_gold"]
    assert 0.87 < p < 0.89  # exp(-1) / (exp(-1) + exp(-3))
    assert out[0].meta["option_codes"] == ["0", "1"]
