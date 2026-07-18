"""Tests for augmentation/pipeline.py (STAGE2_CONTRACT Pipeline section)."""

import json
from pathlib import Path

import pandas as pd

from augmentation.config import AugmentConfig, CacheCfg, PromptsCfg
from augmentation.pipeline import AugmentPipeline
from cleaning.io import write_original_csv
from data.base import DatasetRow


def _cfg(cache_dir: Path, *, dataset: str = "sst2") -> AugmentConfig:
    return AugmentConfig(
        dataset=dataset,
        seed=0,
        augmenter="identity",
        prompts=PromptsCfg(paraphrase="paraphrase_v1.txt", idiomatic="idiomatic_v1.txt"),
        cache=CacheCfg(enabled=True, dir=str(cache_dir)),
    )


def _write_original(tmp_path: Path) -> Path:
    rows = [
        DatasetRow(id="1", x="hello world", y=1, meta={}),
        DatasetRow(id="2", x="goodbye world", y=0, meta={"k": "v"}),
        DatasetRow(id="3", x="another example", y=1, meta={}),
    ]
    out_path = tmp_path / "original.csv"
    write_original_csv(out_path, rows)
    return out_path


def test_pipeline_produces_expected_output_files(tmp_path: Path) -> None:
    original_csv = _write_original(tmp_path)
    cfg = _cfg(tmp_path / "cache")
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


def test_pipeline_rows_are_row_for_row_aligned_by_id(tmp_path: Path) -> None:
    original_csv = _write_original(tmp_path)
    cfg = _cfg(tmp_path / "cache")
    out_dir = tmp_path / "datasets_out"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("dummy: true")

    paraphrase_csv, idiomatic_csv = AugmentPipeline(cfg, original_csv, out_dir, config_path).run()

    original_df = pd.read_csv(original_csv, keep_default_na=False, dtype={"id": str})
    paraphrase_df = pd.read_csv(paraphrase_csv, keep_default_na=False, dtype={"id": str})
    idiomatic_df = pd.read_csv(idiomatic_csv, keep_default_na=False, dtype={"id": str})

    assert paraphrase_df["id"].tolist() == original_df["id"].tolist()
    assert idiomatic_df["id"].tolist() == original_df["id"].tolist()
    # identity augmenter copies x verbatim
    assert paraphrase_df["x"].tolist() == original_df["x"].tolist()
    assert idiomatic_df["x"].tolist() == original_df["x"].tolist()


def test_pipeline_counts_are_correct(tmp_path: Path) -> None:
    original_csv = _write_original(tmp_path)
    cfg = _cfg(tmp_path / "cache")
    out_dir = tmp_path / "datasets_out"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("dummy: true")

    pipeline = AugmentPipeline(cfg, original_csv, out_dir, config_path)
    paraphrase_csv, idiomatic_csv = pipeline.run()

    assert len(pd.read_csv(paraphrase_csv, keep_default_na=False, dtype={"id": str})) == 3
    assert len(pd.read_csv(idiomatic_csv, keep_default_na=False, dtype={"id": str})) == 3

    for variant in ("paraphrase", "idiomatic"):
        sidecar_path = out_dir / "sst2" / f"{variant}.meta.json"
        sidecar = json.loads(sidecar_path.read_text())
        assert sidecar["row_counts"]["input_rows"] == 3
        assert sidecar["row_counts"]["augmented"] == 3
        assert sidecar["row_counts"]["written"] == 3
        # identity is a no-op stub: everything always passes, nothing fails.
        assert sidecar["row_counts"]["validators_failed_by_name"] == {}
        assert sidecar["row_counts"]["validators_passed_by_name"]["semantic_similarity"] == 3
        assert sidecar["row_counts"]["validators_passed_by_name"]["label_preservation"] == 3
        assert sidecar["cache_stats"]["misses"] == 3
        assert sidecar["cache_stats"]["hits"] == 0

    assert pipeline.last_counts["paraphrase"]["written"] == 3
    assert pipeline.last_cache_stats["paraphrase"] == {"hits": 0, "misses": 3}


def test_pipeline_second_run_is_byte_identical(tmp_path: Path) -> None:
    original_csv = _write_original(tmp_path)
    cfg = _cfg(tmp_path / "cache")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("dummy: true")

    out_dir_1 = tmp_path / "out1"
    out_dir_2 = tmp_path / "out2"
    paraphrase_1, idiomatic_1 = AugmentPipeline(cfg, original_csv, out_dir_1, config_path).run()
    paraphrase_2, idiomatic_2 = AugmentPipeline(cfg, original_csv, out_dir_2, config_path).run()

    assert paraphrase_1.read_bytes() == paraphrase_2.read_bytes()
    assert idiomatic_1.read_bytes() == idiomatic_2.read_bytes()


def test_pipeline_second_run_hits_cache(tmp_path: Path) -> None:
    original_csv = _write_original(tmp_path)
    cache_dir = tmp_path / "cache"
    cfg = _cfg(cache_dir)
    out_dir = tmp_path / "datasets_out"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("dummy: true")

    AugmentPipeline(cfg, original_csv, out_dir, config_path).run()
    second = AugmentPipeline(cfg, original_csv, out_dir, config_path)
    second.run()

    assert second.last_cache_stats["paraphrase"] == {"hits": 3, "misses": 0}
    assert second.last_cache_stats["idiomatic"] == {"hits": 3, "misses": 0}
