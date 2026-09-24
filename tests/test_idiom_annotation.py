"""Unit tests for `analysis.idiom_annotation` — extraction, resume, norm linking.

Every extraction test runs entirely offline via `_FakeExtractorClient` (no
network) and uses `tmp_path` so nothing touches the real `datasets_out/` or
`.hf_cache/`. Norm-linking tests use small in-memory tables, not the real
(gitignored) `data/norms/_files/` — see `tests/test_norms.py` for those.
"""

import csv
import json
from pathlib import Path

import pandas as pd
import pytest

from analysis.idiom_annotation import (
    AnnotateConfig,
    CacheCfg,
    DecodingCfg,
    ExtractorSpec,
    LimitsCfg,
    RetryCfg,
    RowInput,
    finalize_partial_csv,
    heal_partial_csv,
    link_to_norms,
    load_aligned_rows,
    parse_extraction,
    render_prompt,
    run_extraction,
)
from analysis.prompts.loader import load_prompt
from augmentation.providers.base import EmptyResponseError, LLMError
from data.norms.magpie import MagpieStats
from data.norms.opacity import OpacityEntry

# ---------- fake extractor client ----------


class _FakeExtractorClient:
    """Stand-in for `LLMClient` — returns a scripted response per call."""

    def __init__(
        self,
        *,
        provider: str = "openai",
        model: str = "fake-extractor",
        responses: list[str] | str = "kick the bucket",
        errors_before_response: int = 0,
    ) -> None:
        self.provider = provider
        self.model = model
        self.calls = 0
        self._responses = responses if isinstance(responses, list) else [responses]
        self._errors_before_response = errors_before_response

    def complete(
        self, *, system: str, user: str, temperature: float, max_output_tokens: int
    ) -> str:
        self.calls += 1
        if self.calls <= self._errors_before_response:
            raise LLMError("transient 500")
        idx = min(self.calls - self._errors_before_response, len(self._responses)) - 1
        return self._responses[idx]


def _make_variant_csvs(dataset_dir: Path, n_rows: int = 3) -> None:
    """Write minimal `original.csv`, `paraphrase.csv`, `idiomatic.csv` under `dataset_dir`."""
    dataset_dir.mkdir(parents=True, exist_ok=True)
    columns = ["id", "variant", "x", "y", "meta", "augmenter_model", "prompt_hash", "validators"]
    for variant, prefix in (("original", "orig"), ("paraphrase", "par"), ("idiomatic", "idi")):
        path = dataset_dir / f"{variant}.csv"
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for i in range(n_rows):
                writer.writerow(
                    {
                        "id": f"row/{i}",
                        "variant": variant,
                        "x": f"{prefix} text {i}",
                        "y": "0",
                        "meta": "{}",
                        "augmenter_model": "gemini/fake" if variant != "original" else "",
                        "prompt_hash": "deadbeef" if variant != "original" else "",
                        "validators": "{}",
                    }
                )


def _make_cfg(
    *, tmp_path: Path, cache_enabled: bool = True, max_rows: int | None = None
) -> AnnotateConfig:
    return AnnotateConfig(
        dataset="mmlu",
        prompt_file="idiom_extraction_v1.txt",
        extractor=ExtractorSpec(provider="openai", model="gpt-5.6-luna"),
        decoding=DecodingCfg(temperature=0.0, max_output_tokens=128),
        retry=RetryCfg(max_attempts=2, backoff_seconds=0.0),
        cache=CacheCfg(enabled=cache_enabled, dir=str(tmp_path / "cache")),
        limits=LimitsCfg(max_rows=max_rows),
        output_dir=str(tmp_path),
    )


def _patch_build_client(monkeypatch: pytest.MonkeyPatch, client: _FakeExtractorClient) -> None:
    """Replace `build_client` so `run_extraction` gets our fake instead of a real one."""

    def fake_build_client(_provider: str, _model: str) -> _FakeExtractorClient:
        return client

    def noop_check_env_key(_provider: str) -> None:
        return None

    monkeypatch.setattr("analysis.idiom_annotation.build_client", fake_build_client)
    monkeypatch.setattr("analysis.idiom_annotation.check_env_key", noop_check_env_key)


# ---------- parse_extraction ----------


def test_parse_extraction_single_idiom() -> None:
    assert parse_extraction("kick the bucket") == ["kick the bucket"]


def test_parse_extraction_multiple_idioms() -> None:
    assert parse_extraction("kick the bucket\nspill the beans") == [
        "kick the bucket",
        "spill the beans",
    ]


def test_parse_extraction_none() -> None:
    assert parse_extraction("NONE") == []


def test_parse_extraction_lowercases() -> None:
    assert parse_extraction("Kick The Bucket") == ["kick the bucket"]


def test_parse_extraction_rejects_empty() -> None:
    with pytest.raises(EmptyResponseError):
        parse_extraction("")


# ---------- prompt rendering ----------


def test_render_prompt_substitutes_all_slots() -> None:
    template = load_prompt("idiom_extraction_v1.txt")
    row = RowInput(
        row_id="row/0",
        original_text="He died.",
        paraphrase_text="He passed away.",
        idiomatic_text="He kicked the bucket.",
    )
    rendered = render_prompt(template, row)
    assert "He died." in rendered
    assert "He passed away." in rendered
    assert "He kicked the bucket." in rendered
    assert "{original}" not in rendered
    assert "{paraphrase}" not in rendered
    assert "{idiomatic}" not in rendered


# ---------- input loading ----------


def test_load_aligned_rows_emits_one_per_aligned_row(tmp_path: Path) -> None:
    _make_variant_csvs(tmp_path / "mmlu", n_rows=3)
    rows = load_aligned_rows(tmp_path / "mmlu")
    assert len(rows) == 3
    assert {r.row_id for r in rows} == {"row/0", "row/1", "row/2"}


def test_load_aligned_rows_drops_unaligned_rows(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "mmlu"
    _make_variant_csvs(dataset_dir, n_rows=3)
    par_path = dataset_dir / "paraphrase.csv"
    rows = list(csv.DictReader(par_path.open()))
    with par_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows[:-1])

    loaded = load_aligned_rows(dataset_dir)
    assert {r.row_id for r in loaded} == {"row/0", "row/1"}


# ---------- heal_partial_csv ----------


def test_heal_partial_csv_truncates_torn_line(tmp_path: Path) -> None:
    p = tmp_path / "partial.csv"
    p.write_bytes(b"header\nrow1\nrow2_partial_no_newline")
    heal_partial_csv(p)
    assert p.read_bytes() == b"header\nrow1\n"


# ---------- end-to-end via run_extraction ----------


def test_run_extraction_writes_partial_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _make_variant_csvs(tmp_path / "mmlu", n_rows=2)
    cfg = _make_cfg(tmp_path=tmp_path)
    client = _FakeExtractorClient(responses="kick the bucket")
    _patch_build_client(monkeypatch, client)

    stats = run_extraction(cfg)

    assert stats.api_calls == 2
    assert stats.cache_hits == 0
    partial = tmp_path / "mmlu" / "idiom_extractions.partial.csv"
    assert partial.exists()
    rows = list(csv.DictReader(partial.open()))
    assert len(rows) == 2
    assert json.loads(rows[0]["idioms_json"]) == ["kick the bucket"]


def test_run_extraction_second_call_is_all_cache_hits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_variant_csvs(tmp_path / "mmlu", n_rows=2)
    cfg = _make_cfg(tmp_path=tmp_path)
    client = _FakeExtractorClient(responses="kick the bucket")
    _patch_build_client(monkeypatch, client)

    _ = run_extraction(cfg)
    calls_after_first = client.calls
    finalize_partial_csv(tmp_path / "mmlu")  # move partial → final
    (tmp_path / "mmlu" / "idiom_extractions.partial.csv").unlink()

    stats2 = run_extraction(cfg)
    assert stats2.api_calls == 0
    assert stats2.cache_hits == 2
    assert client.calls == calls_after_first  # no new API calls made


def test_run_extraction_retries_transient_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_variant_csvs(tmp_path / "mmlu", n_rows=1)
    cfg = _make_cfg(tmp_path=tmp_path)
    client = _FakeExtractorClient(responses="kick the bucket", errors_before_response=1)
    _patch_build_client(monkeypatch, client)

    stats = run_extraction(cfg)
    assert stats.api_calls == 1
    assert stats.skipped_rows == 0


def test_run_extraction_skips_after_persistent_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_variant_csvs(tmp_path / "mmlu", n_rows=1)
    cfg = _make_cfg(tmp_path=tmp_path)
    client = _FakeExtractorClient(responses="kick the bucket", errors_before_response=99)
    _patch_build_client(monkeypatch, client)

    stats = run_extraction(cfg)
    assert stats.skipped_rows == 1
    assert stats.api_calls == 0


def test_run_extraction_max_rows_cap_stops_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_variant_csvs(tmp_path / "mmlu", n_rows=3)
    cfg = _make_cfg(tmp_path=tmp_path, max_rows=1)
    client = _FakeExtractorClient(responses="kick the bucket")
    _patch_build_client(monkeypatch, client)

    stats = run_extraction(cfg)
    assert stats.api_calls == 1

    partial = tmp_path / "mmlu" / "idiom_extractions.partial.csv"
    assert len(list(csv.DictReader(partial.open()))) == 1


def test_finalize_partial_csv_sorts_and_renames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_variant_csvs(tmp_path / "mmlu", n_rows=2)
    cfg = _make_cfg(tmp_path=tmp_path)
    client = _FakeExtractorClient(responses="kick the bucket")
    _patch_build_client(monkeypatch, client)

    run_extraction(cfg)
    final = finalize_partial_csv(tmp_path / "mmlu")
    assert final.name == "idiom_extractions.csv"
    assert final.exists()
    assert not (tmp_path / "mmlu" / "idiom_extractions.partial.csv").with_suffix(".tmp").exists()


# ---------- link_to_norms ----------


def test_link_to_norms_picks_highest_opacity_idiom() -> None:
    df = pd.DataFrame(
        [{"id": "row/0", "idioms_json": json.dumps(["kick the bucket", "spill the beans"])}]
    )
    opacity_table = {
        "kick the bucket": OpacityEntry(opacity=0.9, source="bulkes_tanner_2017"),
        "spill the beans": OpacityEntry(opacity=0.3, source="bulkes_tanner_2017"),
    }
    out = link_to_norms(df, opacity_table=opacity_table, magpie_stats={}, word_freq={})
    row = out.iloc[0]
    assert row["primary_idiom"] == "kick the bucket"
    assert row["opacity"] == 0.9
    assert row["opacity_source"] == "bulkes_tanner_2017"
    assert row["mean_opacity"] == pytest.approx(0.6)
    assert row["first_idiom_opacity"] == 0.9


def test_link_to_norms_no_coverage_yields_none_source() -> None:
    df = pd.DataFrame([{"id": "row/0", "idioms_json": json.dumps(["not a real idiom"])}])
    out = link_to_norms(df, opacity_table={}, magpie_stats={}, word_freq={})
    row = out.iloc[0]
    assert row["primary_idiom"] is None
    assert pd.isna(row["opacity"])
    assert row["opacity_source"] == "none"


def test_link_to_norms_computes_log_freq_and_magpie_fraction() -> None:
    df = pd.DataFrame([{"id": "row/0", "idioms_json": json.dumps(["kick the bucket"])}])
    opacity_table = {"kick the bucket": OpacityEntry(opacity=0.9, source="bulkes_tanner_2017")}
    magpie_stats = {
        "kick the bucket": MagpieStats(
            instance_count=5, idiomatic_fraction=0.8, mean_confidence=0.95
        )
    }
    word_freq = {"kick": 100, "the": 1_000_000, "bucket": 500}
    out = link_to_norms(
        df, opacity_table=opacity_table, magpie_stats=magpie_stats, word_freq=word_freq
    )
    row = out.iloc[0]
    assert row["log_freq"] == pytest.approx(__import__("math").log(100))
    assert row["magpie_idiomatic_fraction"] == 0.8


def test_link_to_norms_empty_idioms_list() -> None:
    df = pd.DataFrame([{"id": "row/0", "idioms_json": json.dumps([])}])
    out = link_to_norms(df, opacity_table={}, magpie_stats={}, word_freq={})
    row = out.iloc[0]
    assert row["num_idioms_extracted"] == 0
    assert row["primary_idiom"] is None
