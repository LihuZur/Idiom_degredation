"""Unit tests for BaseEvaluator run_variant using FakeModel (STAGE3_PLAN §5.5)."""

import csv
from pathlib import Path
from typing import Literal

import torch

from augmentation.base import AugmentedRow
from eval.base import BaseEvaluator, RunResult
from eval.config import EvalConfig
from models.base import FormattedInput, Prediction


class FakeModel:
    """Mock model returning canned outputs."""

    def __init__(self) -> None:
        self.id = "fake-model"
        self.device = torch.device("cpu")
        self.outputs = ["positive", "negative", "invalid output", "positive", "negative"]
        self.idx = 0

    def predict(self, batch: list[FormattedInput]) -> list[Prediction]:
        preds = []
        for fi in batch:
            raw = self.outputs[self.idx % len(self.outputs)]
            self.idx += 1
            preds.append(Prediction(id=fi.id, raw=raw, parsed=None, meta={}))
        return preds


class DummyEvaluator(BaseEvaluator):
    """Concrete evaluator for testing BaseEvaluator pipeline."""

    dataset = "dummy"

    def format(self, ex: AugmentedRow) -> FormattedInput:
        return FormattedInput(
            id=ex.id,
            prompt=f"System: classify\nUser: {ex.x}",
            meta={"messages": [], "generate_kwargs": {}},
        )

    def parse(self, raw: str, ex: AugmentedRow) -> tuple[str | None, Literal["ok", "unparseable"]]:
        s = raw.strip().lower()
        if "positive" in s:
            return "1", "ok"
        if "negative" in s:
            return "0", "ok"
        return None, "unparseable"


def test_run_variant(tmp_path: Path) -> None:
    # Create a dummy CSV file with 5 rows
    csv_path = tmp_path / "original.csv"
    headers = [
        "id",
        "variant",
        "x",
        "y",
        "meta",
        "augmenter_model",
        "prompt_hash",
        "validators",
    ]
    rows = [
        ["1", "original", "I love this!", "1", "{}", "", "", ""],
        ["2", "original", "I hate this.", "0", "{}", "", "", ""],
        ["3", "original", "Unparseable sentence.", "1", "{}", "", "", ""],
        ["4", "original", "Awesome product.", "1", "{}", "", "", ""],
        ["5", "original", "Terrible support.", "0", "{}", "", "", ""],
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    cfg = EvalConfig(model="fake-model")
    evaluator = DummyEvaluator(cfg)
    model = FakeModel()

    # 1. Full run_variant
    res = evaluator.run_variant(model, csv_path)

    assert isinstance(res, RunResult)
    assert res.variant == "original"

    # We expect outputs:
    # row 1: "positive" -> parsed="1", correct=True
    # row 2: "negative" -> parsed="0", correct=True
    # row 3: "invalid output" -> parsed=None, correct=False, unparseable
    # row 4: "positive" -> parsed="1", correct=True
    # row 5: "negative" -> parsed="0", correct=True
    assert res.metrics["n"] == 5.0
    assert res.metrics["n_unparseable"] == 1.0
    assert res.metrics["unparseable_rate"] == 0.2
    assert res.metrics["accuracy"] == 0.8  # 4 out of 5 correct

    assert res.meta["unparseable_ids"] == ["3"]
    assert len(res.predictions) == 5

    # 2. Test limit parameter
    res_limit = evaluator.run_variant(model, csv_path, limit=3)
    assert res_limit.metrics["n"] == 3.0
    assert len(res_limit.predictions) == 3
