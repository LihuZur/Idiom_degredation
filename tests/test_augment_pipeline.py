"""Tests for augmentation/pipeline.py (STAGE2_CONTRACT Pipeline section)."""

import json
from pathlib import Path

import pandas as pd
import pytest

from augmentation.pipeline import AugmentPipeline
from cleaning.io import write_original_csv
from data.base import DatasetRow
from tests._augment_helpers import FakeClient, make_cfg

_REWRITTEN = "a rewritten sentence"


def _write_original(tmp_path: Path) -> Path:
    rows = [
        DatasetRow(id="1", x="hello world", y=1, meta={}),
        DatasetRow(id="2", x="goodbye world", y=0, meta={"k": "v"}),
        DatasetRow(id="3", x="another example", y=1, meta={}),
    ]
    out_path = tmp_path / "original.csv"
    write_original_csv(out_path, rows)
    return out_path


def _patch_build_client(monkeypatch: pytest.MonkeyPatch, client: FakeClient) -> None:
    """Make the pipeline's module-level `build_client` return `client` unconditionally."""

    def fake_build_client(provider: str, model: str) -> FakeClient:
        return client

    monkeypatch.setattr("augmentation.pipeline.build_client", fake_build_client)


def test_pipeline_produces_expected_output_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_csv = _write_original(tmp_path)
    fake = FakeClient(augment_result=_REWRITTEN, judge_result="PASS")
    _patch_build_client(monkeypatch, fake)
    cfg = make_cfg(cache_dir=tmp_path / "cache")
    out_dir = tmp_path / "datasets_out"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("dummy: true")

    paraphrase_csv, idiomatic_csv = AugmentPipeline(cfg, original_csv, out_dir, config_path).run()

    assert paraphrase_csv == out_dir / "sst2" / "paraphrase.csv"
    assert idiomatic_csv == out_dir / "sst2" / "idiomatic.csv"
    assert paraphrase_csv.exists()
    assert idiomatic_csv.exists()
    assert (out_dir / "sst2" / "paraphrase.meta.json").exists()
    assert (out_dir / "sst2" / "idiomatic.meta.json").exists()


def test_pipeline_rows_are_row_for_row_aligned_by_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_csv = _write_original(tmp_path)
    fake = FakeClient(augment_result=_REWRITTEN, judge_result="PASS")
    _patch_build_client(monkeypatch, fake)
    cfg = make_cfg(cache_dir=tmp_path / "cache")
    out_dir = tmp_path / "datasets_out"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("dummy: true")

    paraphrase_csv, idiomatic_csv = AugmentPipeline(cfg, original_csv, out_dir, config_path).run()

    original_df = pd.read_csv(original_csv, keep_default_na=False, dtype={"id": str})
    paraphrase_df = pd.read_csv(paraphrase_csv, keep_default_na=False, dtype={"id": str})
    idiomatic_df = pd.read_csv(idiomatic_csv, keep_default_na=False, dtype={"id": str})

    assert paraphrase_df["id"].tolist() == original_df["id"].tolist()
    assert idiomatic_df["id"].tolist() == original_df["id"].tolist()


def test_pipeline_rewritten_x_equals_client_output_not_verbatim_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_csv = _write_original(tmp_path)
    fake = FakeClient(augment_result=_REWRITTEN, judge_result="PASS")
    _patch_build_client(monkeypatch, fake)
    cfg = make_cfg(cache_dir=tmp_path / "cache")
    out_dir = tmp_path / "datasets_out"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("dummy: true")

    paraphrase_csv, idiomatic_csv = AugmentPipeline(cfg, original_csv, out_dir, config_path).run()

    original_df = pd.read_csv(original_csv, keep_default_na=False, dtype={"id": str})
    paraphrase_df = pd.read_csv(paraphrase_csv, keep_default_na=False, dtype={"id": str})
    idiomatic_df = pd.read_csv(idiomatic_csv, keep_default_na=False, dtype={"id": str})

    assert (paraphrase_df["x"] == _REWRITTEN).all()
    assert (idiomatic_df["x"] == _REWRITTEN).all()
    assert (paraphrase_df["x"] != original_df["x"]).all()


def test_pipeline_augmenter_model_column(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original_csv = _write_original(tmp_path)
    fake = FakeClient(augment_result=_REWRITTEN, judge_result="PASS")
    _patch_build_client(monkeypatch, fake)
    cfg = make_cfg(cache_dir=tmp_path / "cache")
    out_dir = tmp_path / "datasets_out"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("dummy: true")

    paraphrase_csv, idiomatic_csv = AugmentPipeline(cfg, original_csv, out_dir, config_path).run()

    for csv_path in (paraphrase_csv, idiomatic_csv):
        df = pd.read_csv(csv_path, keep_default_na=False, dtype={"id": str})
        assert (df["augmenter_model"] == "gemini/fake-model").all()


def test_pipeline_counts_are_correct(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original_csv = _write_original(tmp_path)
    fake = FakeClient(augment_result=_REWRITTEN, judge_result="PASS")
    _patch_build_client(monkeypatch, fake)
    cfg = make_cfg(cache_dir=tmp_path / "cache")
    out_dir = tmp_path / "datasets_out"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("dummy: true")

    pipeline = AugmentPipeline(cfg, original_csv, out_dir, config_path)
    paraphrase_csv, idiomatic_csv = pipeline.run()

    assert len(pd.read_csv(paraphrase_csv, keep_default_na=False, dtype={"id": str})) == 3
    assert len(pd.read_csv(idiomatic_csv, keep_default_na=False, dtype={"id": str})) == 3

    variant_extra_validator = {"paraphrase": "idiom_absence", "idiomatic": "idiom_presence"}
    for variant, extra in variant_extra_validator.items():
        sidecar_path = out_dir / "sst2" / f"{variant}.meta.json"
        sidecar = json.loads(sidecar_path.read_text())
        assert sidecar["row_counts"]["input_rows"] == 3
        assert sidecar["row_counts"]["augmented"] == 3
        assert sidecar["row_counts"]["written"] == 3
        # all rows pass on attempt 1 (fixed augment output, judge always PASS).
        assert sidecar["row_counts"]["validators_failed_by_name"] == {}
        assert sidecar["row_counts"]["validators_passed_by_name"]["semantic_similarity"] == 3
        assert sidecar["row_counts"]["validators_passed_by_name"]["label_preservation"] == 3
        assert sidecar["row_counts"]["validators_passed_by_name"][extra] == 3
        assert sidecar["cache_stats"]["misses"] == 3
        assert sidecar["cache_stats"]["hits"] == 0

    assert pipeline.last_counts["paraphrase"]["written"] == 3
    assert pipeline.last_cache_stats["paraphrase"] == {"hits": 0, "misses": 3}


def test_pipeline_second_run_hits_cache_with_zero_client_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_csv = _write_original(tmp_path)
    cache_dir = tmp_path / "cache"
    cfg = make_cfg(cache_dir=cache_dir)
    out_dir = tmp_path / "datasets_out"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("dummy: true")

    warm_client = FakeClient(augment_result=_REWRITTEN, judge_result="PASS")
    _patch_build_client(monkeypatch, warm_client)
    AugmentPipeline(cfg, original_csv, out_dir, config_path).run()

    # Fresh client (zeroed counters), same cache dir/model -> should be a pure
    # cache hit that never touches the client.
    fresh_client = FakeClient(augment_result=_REWRITTEN, judge_result="PASS")
    _patch_build_client(monkeypatch, fresh_client)
    second = AugmentPipeline(cfg, original_csv, out_dir, config_path)
    second.run()

    assert fresh_client.augment_calls == 0
    assert fresh_client.judge_calls == 0
    assert second.last_cache_stats["paraphrase"] == {"hits": 3, "misses": 0}
    assert second.last_cache_stats["idiomatic"] == {"hits": 3, "misses": 0}


def test_pipeline_changing_model_misses_warm_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_csv = _write_original(tmp_path)
    cache_dir = tmp_path / "cache"
    cfg = make_cfg(cache_dir=cache_dir)
    out_dir = tmp_path / "datasets_out"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("dummy: true")

    first_client = FakeClient(augment_result=_REWRITTEN, judge_result="PASS")
    _patch_build_client(monkeypatch, first_client)
    AugmentPipeline(cfg, original_csv, out_dir, config_path).run()

    # Cache key includes `f"{provider}/{model}"`; a different model is a fresh key.
    other_model_client = FakeClient(
        augment_result=_REWRITTEN, judge_result="PASS", model="other-fake-model"
    )
    _patch_build_client(monkeypatch, other_model_client)
    second = AugmentPipeline(cfg, original_csv, out_dir, config_path)
    second.run()

    assert other_model_client.augment_calls > 0
    assert second.last_cache_stats["paraphrase"] == {"hits": 0, "misses": 3}
    assert second.last_cache_stats["idiomatic"] == {"hits": 0, "misses": 3}
