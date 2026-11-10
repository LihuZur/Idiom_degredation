import csv
from pathlib import Path
from typing import Any

import pytest

from augmentation.base import AugmentedRow
from eval.config import EvalConfig
from eval.reuse import (
    ReuseError,
    check_provenance,
    read_xy,
    reusable_predictions,
    verify_reuse,
)
from eval.sst2 import Sst2Evaluator
from models.base import FormattedInput, Prediction


def _row(id_: str, x: str = "great film", y: str = "1") -> AugmentedRow:
    return AugmentedRow(id=id_, variant="original", x=x, y=y, augmenter_model="", prompt_hash="")


def _prior(raws: dict[str, str]) -> dict[str, Any]:
    return {"per_task": [{"id": i, "original": {"raw": r}} for i, r in raws.items()]}


def _evaluator() -> Sst2Evaluator:
    return Sst2Evaluator(EvalConfig(model="m"))


def test_reuse_requires_identical_text_and_label() -> None:
    rows = [_row("a"), _row("b", x="changed"), _row("c", y="0"), _row("d"), _row("e")]
    prior_xy = {
        "a": ("great film", "1"),
        "b": ("great film", "1"),
        "c": ("great film", "1"),
        "e": ("great film", "1"),
    }
    prior = _prior({"a": "positive", "b": "positive", "c": "positive", "e": "negative"})
    reused, report = reusable_predictions(prior, "original", prior_xy, rows, _evaluator())
    assert set(reused) == {"a", "e"}
    assert report.reused == 2
    assert report.reasons == {"text_changed": 1, "label_changed": 1, "not_in_earlier_results": 1}


def test_reused_predictions_are_reparsed_and_rescored() -> None:
    prior = _prior({"a": "Positive.", "e": "no idea"})
    prior_xy = {"a": ("great film", "1"), "e": ("great film", "1")}
    reused, _ = reusable_predictions(
        prior, "original", prior_xy, [_row("a"), _row("e")], _evaluator()
    )
    assert reused["a"].meta["correct"] is True
    assert reused["e"].meta["parse_status"] == "unparseable"
    assert reused["e"].meta["correct"] is False
    assert reused["a"].meta["reused"] is True


def _check(prior: dict[str, Any], **override: Any) -> None:
    kw: dict[str, Any] = {
        "prompt_hash": "p",
        "model_id": "m",
        "model_revision": "r",
        "precision": "bf16",
        "resolved_config": {"seed": 0},
        **override,
    }
    check_provenance(
        prior,
        prompt_hash=kw["prompt_hash"],
        model_id=kw["model_id"],
        model_revision=kw["model_revision"],
        precision=kw["precision"],
        resolved_config=kw["resolved_config"],
    )


def test_check_provenance_rejects_any_difference() -> None:
    base = {
        "prompt_hash": "p",
        "model_id": "m",
        "model_revision": "r",
        "precision": "bf16",
        "resolved_config": {"seed": 0},
    }
    _check(base)
    for key, other in (
        ("prompt_hash", "q"),
        ("model_revision", "r2"),
        ("precision", "fp16"),
        ("resolved_config", {"seed": 1}),
    ):
        with pytest.raises(ReuseError, match=key):
            _check(base, **{key: other})


def test_read_xy(tmp_path: Path) -> None:
    path = tmp_path / "original.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "x", "y"])
        w.writeheader()
        w.writerow({"id": "a", "x": "hello", "y": "1"})
    assert read_xy(path) == {"a": ("hello", "1")}


class _Model:
    def __init__(self, outputs: dict[str, str]) -> None:
        self.id = "fake"
        self.outputs = outputs

    def predict(self, batch: list[FormattedInput]) -> list[Prediction]:
        return [Prediction(id=b.id, raw=self.outputs[b.id], parsed=None) for b in batch]


def test_verify_reuse_passes_and_fails() -> None:
    ev = _evaluator()
    rows = [_row("a"), _row("e")]
    prior_xy = {"a": ("great film", "1"), "e": ("great film", "1")}
    reused, _ = reusable_predictions(
        _prior({"a": "positive", "e": "negative"}), "original", prior_xy, rows, ev
    )
    assert verify_reuse(ev, _Model({"a": "positive", "e": "negative"}), rows, reused, 5, 0) == 2
    with pytest.raises(ReuseError, match="differ"):
        verify_reuse(ev, _Model({"a": "positive", "e": "positive"}), rows, reused, 5, 0)
