"""Tests for cleaning/pipeline.py (STAGE1_PLAN §5.3).

Uses a fake in-memory loader registered per-test under a unique name — does
NOT hit HF, per plan (real-HF tests live in test_sst2_loader.py / test_mmlu_loader.py).
"""

import itertools
import json
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import pytest

from cleaning.config import CleanConfig, LengthCfg
from cleaning.pipeline import Pipeline
from data.base import DatasetRow
from data.registry import DATASETS, register_dataset

_counter = itertools.count()


@pytest.fixture(autouse=True)
def _cleanup_fake_registrations() -> Iterable[None]:  # pyright: ignore[reportUnusedFunction]
    """Remove any fake loaders registered by this file so they don't leak
    into other tests (e.g. tests/test_smoke.py's exact registry check).
    """
    before = set(DATASETS)
    yield
    for name in set(DATASETS) - before:
        del DATASETS[name]


def _make_fake_loader(examples: list[DatasetRow]) -> str:
    """Register a fake loader yielding `examples` and return its dataset name."""
    dataset_name = f"fake_pipeline_{next(_counter)}"

    class _FakeLoader:
        name = dataset_name
        hf_dataset_id = "fake/fake"

        def __init__(self, *, hf_revision: str, normalize: list[str]) -> None:
            del hf_revision, normalize

        def load(self) -> Iterable[DatasetRow]:
            yield from examples

    register_dataset(dataset_name)(_FakeLoader)
    return dataset_name


def _cfg(
    dataset: str,
    *,
    seed: int = 0,
    min_tokens: int = 1,
    max_tokens: int = 10,
    max_rows: int | None = None,
    dedupe: bool = True,
) -> CleanConfig:
    return CleanConfig(
        dataset=dataset,
        seed=seed,
        hf_revision="fake-rev",
        length=LengthCfg(min_tokens=min_tokens, max_tokens=max_tokens),
        max_rows=max_rows,
        normalize=["nfc", "collapse_whitespace"],
        dedupe=dedupe,
    )


def test_length_filter_boundaries(tmp_path: Path) -> None:
    examples = [
        DatasetRow(id="1", x="one two", y=0, meta={}),  # 2 tokens: below min=3
        DatasetRow(id="2", x="one two three", y=0, meta={}),  # 3 tokens: at min
        DatasetRow(id="3", x="one two three four", y=0, meta={}),  # 4 tokens: at max
        DatasetRow(id="4", x="one two three four five", y=0, meta={}),  # 5: above max
    ]
    dataset = _make_fake_loader(examples)
    cfg = _cfg(dataset, min_tokens=3, max_tokens=4, dedupe=False)
    out_path = Pipeline(cfg, tmp_path, tmp_path / "config.yaml").run()
    df = pd.read_csv(out_path, keep_default_na=False, dtype={"id": str})
    assert set(df["id"]) == {"2", "3"}


def test_dedupe_collapses_identical_and_keeps_differing(tmp_path: Path) -> None:
    examples = [
        DatasetRow(id="1", x="hello world", y=0, meta={}),
        DatasetRow(id="2", x="hello world", y=0, meta={}),  # exact dup: dropped
        DatasetRow(id="3", x="hello world", y=1, meta={}),  # differs on y: kept
        DatasetRow(id="4", x="hello world", y=0, meta={"k": "v"}),  # differs on meta: kept
    ]
    dataset = _make_fake_loader(examples)
    cfg = _cfg(dataset, min_tokens=1, max_tokens=10, dedupe=True)
    out_path = Pipeline(cfg, tmp_path, tmp_path / "config.yaml").run()
    df = pd.read_csv(out_path, keep_default_na=False, dtype={"id": str})
    assert len(df) == 3
    assert set(df["id"]) == {"1", "3", "4"}


def test_shuffle_determinism_same_seed_same_order(tmp_path: Path) -> None:
    examples = [DatasetRow(id=str(i), x=f"word{i} filler", y=0, meta={}) for i in range(20)]
    dataset = _make_fake_loader(examples)
    cfg = _cfg(dataset, seed=42)
    out1 = Pipeline(cfg, tmp_path / "a", tmp_path / "config.yaml").run()
    out2 = Pipeline(cfg, tmp_path / "b", tmp_path / "config.yaml").run()
    order1 = pd.read_csv(out1, keep_default_na=False)["id"].tolist()
    order2 = pd.read_csv(out2, keep_default_na=False)["id"].tolist()
    assert order1 == order2


def test_shuffle_different_seeds_different_order(tmp_path: Path) -> None:
    examples = [DatasetRow(id=str(i), x=f"word{i} filler", y=0, meta={}) for i in range(20)]
    dataset = _make_fake_loader(examples)
    out1 = Pipeline(_cfg(dataset, seed=1), tmp_path / "a", tmp_path / "config.yaml").run()
    out2 = Pipeline(_cfg(dataset, seed=2), tmp_path / "b", tmp_path / "config.yaml").run()
    order1 = pd.read_csv(out1, keep_default_na=False)["id"].tolist()
    order2 = pd.read_csv(out2, keep_default_na=False)["id"].tolist()
    assert order1 != order2


def test_max_rows_cap_applied_after_shuffle(tmp_path: Path) -> None:
    examples = [DatasetRow(id=str(i), x=f"word{i} filler", y=0, meta={}) for i in range(20)]
    dataset = _make_fake_loader(examples)
    cfg = _cfg(dataset, seed=7, max_rows=5)
    out_path = Pipeline(cfg, tmp_path, tmp_path / "config.yaml").run()
    df = pd.read_csv(out_path, keep_default_na=False)
    assert len(df) == 5


def test_csv_round_trip_has_8_columns(tmp_path: Path) -> None:
    examples = [DatasetRow(id="1", x="hello world", y=1, meta={"a": 1})]
    dataset = _make_fake_loader(examples)
    out_path = Pipeline(_cfg(dataset), tmp_path, tmp_path / "config.yaml").run()
    df = pd.read_csv(out_path, keep_default_na=False)
    assert list(df.columns) == [
        "id",
        "variant",
        "x",
        "y",
        "meta",
        "augmenter_model",
        "prompt_hash",
        "validators",
    ]
    row = df.iloc[0]
    assert row["variant"] == "original"
    assert row["y"] == 1
    assert json.loads(row["meta"]) == {"a": 1}
    assert row["augmenter_model"] == ""
    assert row["prompt_hash"] == ""
    assert row["validators"] == ""


def test_sidecar_json_has_all_fields(tmp_path: Path) -> None:
    examples = [DatasetRow(id="1", x="hello world", y=1, meta={})]
    dataset = _make_fake_loader(examples)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("dummy: true")
    out_path = Pipeline(_cfg(dataset), tmp_path, config_path).run()
    sidecar_path = out_path.parent / "original.meta.json"
    sidecar = json.loads(sidecar_path.read_text())
    for field in (
        "stage",
        "dataset",
        "config_path",
        "config_hash",
        "resolved_config",
        "hf_dataset_id",
        "hf_revision",
        "tool_versions",
        "row_counts",
        "timestamp_utc",
    ):
        assert field in sidecar
    assert sidecar["row_counts"]["written"] == 1
