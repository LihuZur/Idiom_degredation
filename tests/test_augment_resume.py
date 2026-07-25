"""Resume + graceful-stop tests for the Stage 2 pipeline (STAGE2_CONTRACT).

Rows are streamed to each variant CSV as they are accepted, so an interrupted
run leaves a durable partial CSV. A row that still fails after all retries stops
its variant gracefully (rows written so far are kept, no sidecar) instead of
discarding progress; re-running resumes from the saved rows and only augments
the remainder.
"""

from pathlib import Path

import pandas as pd
import pytest

from augmentation.io import VariantProgress, heal_variant_csv, read_variant_progress
from augmentation.pipeline import AugmentPipeline
from augmentation.providers.base import LLMError
from cleaning.io import write_original_csv
from data.base import DatasetRow
from tests._augment_helpers import FakeClient, make_cfg

_OK = "a rewritten sentence"


def _write_original(tmp_path: Path) -> Path:
    rows = [
        DatasetRow(id="1", x="first", y=1, meta={}),
        DatasetRow(id="2", x="second", y=0, meta={}),
        DatasetRow(id="3", x="third", y=1, meta={}),
    ]
    out_path = tmp_path / "original.csv"
    write_original_csv(out_path, rows)
    return out_path


def _patch_build_client(monkeypatch: pytest.MonkeyPatch, client: FakeClient) -> None:
    def fake_build_client(provider: str, model: str) -> FakeClient:
        return client

    monkeypatch.setattr("augmentation.pipeline.build_client", fake_build_client)


def _ids(csv_path: Path) -> list[str]:
    return pd.read_csv(csv_path, keep_default_na=False, dtype={"id": str})["id"].tolist()


def test_persistent_failure_stops_gracefully_and_keeps_written_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_csv = _write_original(tmp_path)
    out_dir = tmp_path / "out"
    cfg = make_cfg(cache_dir=tmp_path / "cache")

    # Row 1 augments fine; every later augment call fails, so row 2 exhausts its
    # retries and the paraphrase variant stops after row 1.
    def _augment(client: FakeClient) -> str:
        if client.augment_calls == 1:
            return _OK
        raise LLMError("boom")

    failing = FakeClient(augment_result=_augment, judge_result="PASS")
    _patch_build_client(monkeypatch, failing)

    pipeline = AugmentPipeline(cfg, original_csv, out_dir, tmp_path / "config.yaml")
    paraphrase_csv, idiomatic_csv = pipeline.run()  # does NOT raise

    assert pipeline.incomplete_variants == ["paraphrase"]
    assert paraphrase_csv.exists()
    assert _ids(paraphrase_csv) == ["1"]  # only the row that succeeded
    assert not (paraphrase_csv.parent / "paraphrase.meta.json").exists()  # incomplete
    assert pipeline.last_counts["paraphrase"]["written"] == 1
    # The run stopped before the idiomatic variant started.
    assert not idiomatic_csv.exists()
    assert "idiomatic" not in pipeline.last_counts


def test_rerun_resumes_and_only_augments_remaining_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_csv = _write_original(tmp_path)
    out_dir = tmp_path / "out"
    cfg = make_cfg(cache_dir=tmp_path / "cache")

    def _augment(client: FakeClient) -> str:
        if client.augment_calls == 1:
            return _OK
        raise LLMError("boom")

    _patch_build_client(monkeypatch, FakeClient(augment_result=_augment, judge_result="PASS"))
    AugmentPipeline(cfg, original_csv, out_dir, tmp_path / "config.yaml").run()

    # Re-run with a healthy client: paraphrase resumes (skips row 1, augments
    # rows 2-3), then idiomatic runs fresh (all 3).
    healthy = FakeClient(augment_result=_OK, judge_result="PASS")
    _patch_build_client(monkeypatch, healthy)
    resumed = AugmentPipeline(cfg, original_csv, out_dir, tmp_path / "config.yaml")
    paraphrase_csv, idiomatic_csv = resumed.run()

    assert resumed.incomplete_variants == []
    assert resumed.last_counts["paraphrase"]["skipped"] == 1
    assert resumed.last_counts["paraphrase"]["written"] == 3
    # Row 1 was skipped (from CSV), so only rows 2-3 of paraphrase were augmented
    # this run, plus all 3 of idiomatic.
    assert healthy.augment_calls == 2 + 3
    # Both variants complete, aligned to the original id order, with sidecars.
    assert _ids(paraphrase_csv) == ["1", "2", "3"]
    assert _ids(idiomatic_csv) == ["1", "2", "3"]
    assert (paraphrase_csv.parent / "paraphrase.meta.json").exists()
    assert (idiomatic_csv.parent / "idiomatic.meta.json").exists()


def test_changed_model_discards_stale_csv_and_reprocesses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_csv = _write_original(tmp_path)
    out_dir = tmp_path / "out"
    cfg = make_cfg(cache_dir=tmp_path / "cache")

    _patch_build_client(monkeypatch, FakeClient(augment_result=_OK, judge_result="PASS"))
    paraphrase_csv, _ = AugmentPipeline(cfg, original_csv, out_dir, tmp_path / "config.yaml").run()
    assert (paraphrase_csv.parent / "paraphrase.meta.json").exists()

    # A different model produces a different `augmenter_model`; the existing CSV
    # is stale and must be rewritten rather than resumed.
    other = FakeClient(augment_result=_OK, judge_result="PASS", model="other-model")
    _patch_build_client(monkeypatch, other)
    second = AugmentPipeline(cfg, original_csv, out_dir, tmp_path / "config.yaml")
    second.run()

    assert other.augment_calls > 0
    assert second.last_counts["paraphrase"]["skipped"] == 0
    df = pd.read_csv(paraphrase_csv, keep_default_na=False, dtype={"id": str})
    assert (df["augmenter_model"] == "gemini/other-model").all()


def test_heal_variant_csv_drops_torn_final_line(tmp_path: Path) -> None:
    path = tmp_path / "paraphrase.csv"
    # A clean header + one full row, then a torn partial row with no newline.
    path.write_text(
        "id,variant,x,y,meta,augmenter_model,prompt_hash,validators\n"
        '1,paraphrase,ok,1,{},gemini/fake-model,ph,{"v": {"passed": true}}\n'
        "2,paraphrase,half-writ",
        encoding="utf-8",
    )
    heal_variant_csv(path)
    progress = read_variant_progress(path)
    assert isinstance(progress, VariantProgress)
    assert progress.ids == ["1"]  # torn row 2 dropped, will be re-augmented
    assert progress.augmenter_model == "gemini/fake-model"
    assert progress.prompt_hash == "ph"
