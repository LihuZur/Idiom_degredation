"""Resume + graceful-skip tests for the Stage 2 pipeline (STAGE2_CONTRACT).

Rows are streamed to each variant CSV as they are accepted, so an interrupted
run leaves a durable partial CSV. A row that still fails after all retries is
skipped (recorded in the sidecar's `skipped_rows` + a durable manifest) and the
run continues; re-running resumes from the saved rows and does not retry the
skipped ones. After both variants are built their CSVs are reconciled to the
common id set so they stay aligned.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from augmentation.io import VariantProgress, heal_variant_csv, read_variant_progress
from augmentation.pipeline import AugmentPipeline
from augmentation.providers.base import EmptyResponseError, LLMError
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


def _skipped_rows(meta_path: Path) -> list[dict[str, object]]:
    return json.loads(meta_path.read_text())["skipped_rows"]


def _mark_row2_idiomatic() -> FakeClient:
    """A client whose rewrite of row 2 (`x="second"`) carries a "MARKED" idiom the
    `idiom_absence` judge rejects, so row 2 fails validation in paraphrase only
    (idiomatic's `idiom_presence` still passes)."""

    def _augment(client: FakeClient) -> str:
        return "MARKED rewrite" if "second" in client.calls[-1] else _OK

    def _judge(client: FakeClient) -> str:
        prompt = client.calls[-1]
        # Only the idiom_absence judge ("free of idiomatic") fails the marked row.
        return "FAIL" if ("free of idiomatic" in prompt and "MARKED" in prompt) else "PASS"

    return FakeClient(augment_result=_augment, judge_result=_judge)


def test_validation_failure_is_skipped_recorded_and_reconciled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_csv = _write_original(tmp_path)
    out_dir = tmp_path / "out"
    cfg = make_cfg(cache_dir=tmp_path / "cache")
    _patch_build_client(monkeypatch, _mark_row2_idiomatic())

    pipeline = AugmentPipeline(cfg, original_csv, out_dir, tmp_path / "config.yaml")
    paraphrase_csv, idiomatic_csv = pipeline.run()  # does NOT raise

    assert not pipeline.paused
    # Run completes; both variants reconcile to the common id set {1, 3}.
    assert _ids(paraphrase_csv) == ["1", "3"]
    assert _ids(idiomatic_csv) == ["1", "3"]
    assert (paraphrase_csv.parent / "paraphrase.meta.json").exists()
    assert (idiomatic_csv.parent / "idiomatic.meta.json").exists()
    assert pipeline.last_counts["paraphrase"]["dropped"] == 1
    assert pipeline.last_counts["idiomatic"]["dropped"] == 1

    # Row 2 recorded in BOTH sidecars: validation failure in paraphrase,
    # dropped-for-alignment ("unaligned") in idiomatic.
    assert _skipped_rows(paraphrase_csv.parent / "paraphrase.meta.json") == [
        {"id": "2", "reason": "validation_failed", "failing": ["idiom_absence"], "attempts": 3}
    ]
    assert _skipped_rows(idiomatic_csv.parent / "idiomatic.meta.json") == [
        {"id": "2", "reason": "unaligned", "failing": [], "attempts": 0}
    ]


def test_rerun_does_not_retry_validation_skips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_csv = _write_original(tmp_path)
    out_dir = tmp_path / "out"
    cfg = make_cfg(cache_dir=tmp_path / "cache")
    _patch_build_client(monkeypatch, _mark_row2_idiomatic())
    AugmentPipeline(cfg, original_csv, out_dir, tmp_path / "config.yaml").run()

    # Re-run with a healthy client: rows 1 & 3 resume from the CSVs and skipped
    # row 2 is in both manifests, so nothing is augmented again.
    healthy = FakeClient(augment_result=_OK, judge_result="PASS")
    _patch_build_client(monkeypatch, healthy)
    resumed = AugmentPipeline(cfg, original_csv, out_dir, tmp_path / "config.yaml")
    paraphrase_csv, idiomatic_csv = resumed.run()

    assert healthy.augment_calls == 0
    assert resumed.last_counts["paraphrase"]["resumed"] == 2
    assert _ids(paraphrase_csv) == ["1", "3"]
    assert _ids(idiomatic_csv) == ["1", "3"]


def test_empty_output_is_skipped_not_paused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_csv = _write_original(tmp_path)
    out_dir = tmp_path / "out"
    cfg = make_cfg(cache_dir=tmp_path / "cache")

    # Row 2 (x="second") always returns empty output — a row-specific failure,
    # so it must be SKIPPED (recorded), not pause the run in an infinite retry.
    def _augment(client: FakeClient) -> str:
        if "second" in client.calls[-1]:
            raise EmptyResponseError("empty")
        return _OK

    _patch_build_client(monkeypatch, FakeClient(augment_result=_augment, judge_result="PASS"))
    pipeline = AugmentPipeline(cfg, original_csv, out_dir, tmp_path / "config.yaml")
    paraphrase_csv, idiomatic_csv = pipeline.run()

    assert not pipeline.paused  # empty output does not pause the run
    assert _ids(paraphrase_csv) == ["1", "3"]
    assert _ids(idiomatic_csv) == ["1", "3"]
    para_skips = _skipped_rows(paraphrase_csv.parent / "paraphrase.meta.json")
    assert [r["id"] for r in para_skips] == ["2"]
    assert para_skips[0]["reason"] == "validation_failed"
    failing = para_skips[0]["failing"]
    assert isinstance(failing, list) and "empty_output" in failing[0]


def test_transient_error_pauses_without_recording_a_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_csv = _write_original(tmp_path)
    out_dir = tmp_path / "out"
    cfg = make_cfg(cache_dir=tmp_path / "cache")

    # Row 1 augments; row 2's three attempts (augment calls 2-4) all raise an
    # LLMError (e.g. a 429) — a transient failure, so the run pauses at row 2.
    def _augment(client: FakeClient) -> str:
        if client.augment_calls in (2, 3, 4):
            raise LLMError("429 rate limited")
        return _OK

    _patch_build_client(monkeypatch, FakeClient(augment_result=_augment, judge_result="PASS"))
    pipeline = AugmentPipeline(cfg, original_csv, out_dir, tmp_path / "config.yaml")
    paraphrase_csv, idiomatic_csv = pipeline.run()  # does NOT raise

    assert pipeline.paused
    assert _ids(paraphrase_csv) == ["1"]  # row 1 kept; run paused at row 2
    # No sidecar (incomplete) and — crucially — the transient failure is NOT
    # recorded as a skip, so a resume will retry row 2.
    assert not (paraphrase_csv.parent / "paraphrase.meta.json").exists()
    assert not (paraphrase_csv.parent / "paraphrase.skipped.json").exists()
    assert not idiomatic_csv.exists()

    # Resume with a healthy client: row 1 resumes, rows 2-3 are (re)tried and
    # succeed, idiomatic runs, and the run completes aligned.
    healthy = FakeClient(augment_result=_OK, judge_result="PASS")
    _patch_build_client(monkeypatch, healthy)
    resumed = AugmentPipeline(cfg, original_csv, out_dir, tmp_path / "config.yaml")
    resumed.run()

    assert not resumed.paused
    assert healthy.augment_calls == 2 + 3  # para rows 2-3 (row 1 resumed) + idiomatic 1-3
    assert _ids(paraphrase_csv) == ["1", "2", "3"]
    assert _ids(idiomatic_csv) == ["1", "2", "3"]
    assert (paraphrase_csv.parent / "paraphrase.meta.json").exists()


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
    assert second.last_counts["paraphrase"]["resumed"] == 0
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
