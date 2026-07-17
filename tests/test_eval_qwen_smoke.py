"""Integration/smoke test for Qwen model evaluation (STAGE3_PLAN §5.10)."""

import os
from pathlib import Path
from typing import cast

import pytest

from eval.config import EvalConfig
from eval.sst2 import Sst2Evaluator
from models.decoder_runner import DecoderRunner
from models.registry import get_model_class, get_model_spec


@pytest.mark.skipif(
    os.environ.get("RUN_BIG_MODEL_TESTS") != "1",
    reason="RUN_BIG_MODEL_TESTS=1 not set",
)
def test_eval_qwen_smoke() -> None:
    cfg = EvalConfig(model="qwen3.5-1.5b-instruct")
    evaluator = Sst2Evaluator(cfg)

    spec = get_model_spec("qwen3.5-1.5b-instruct")
    runner_cls = cast(type[DecoderRunner], get_model_class("qwen3.5-1.5b-instruct"))
    runner = runner_cls(spec)

    csv_path = Path("datasets_out/sst2/original.csv")
    assert csv_path.exists(), "sst2 original.csv must exist to run this integration test"

    res = evaluator.run_variant(runner, csv_path, limit=3)
    assert res.variant == "original"
    assert "accuracy" in res.metrics
    assert 0.0 <= res.metrics["accuracy"] <= 1.0
    assert len(res.predictions) == 3
