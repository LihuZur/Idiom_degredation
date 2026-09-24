"""RQ2 idiom-extraction pass (paper_preparations/RQ2_idiom_properties.md §5 steps 2-3).

Runs a single LLM extractor over every aligned row in a dataset's Stage 2
output, asking it which idiom(s) the augmenter inserted into the `idiomatic`
variant (given `original` and `paraphrase` for contrast), then joins the
extracted idioms to the `data/norms/` opacity/frequency tables to produce
`datasets_out/{ds}/idiom_norms.csv`.

Design mirrors `analysis/meaning_preservation.py` (RQ1 judge pass): per-row
response cache, streaming partial CSV, retry-then-skip, pydantic config.
Simpler in one respect — one extractor model, not a two-judge cross-check.
"""

import csv
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from pydantic import BaseModel, Field, field_validator
from tqdm.auto import tqdm

from analysis.prompts.loader import load_prompt
from augmentation.cache import ResponseCache
from augmentation.providers.base import EmptyResponseError, LLMClient, LLMError, build_client
from cleaning.hashing import prompt_hash
from data.norms.frequency import idiom_frequency, load_word_frequencies
from data.norms.llm_opacity import load_llm_opacity_ratings
from data.norms.magpie import MagpieStats, load_magpie_stats
from data.norms.opacity import (
    OpacityEntry,
    build_fuzzy_opacity_index,
    content_words,
    load_opacity_norms,
)

_OUTPUT_COLUMNS = [
    "id",
    "idioms_json",
    "extractor_provider",
    "extractor_model",
    "prompt_version",
    "prompt_hash",
    "raw_output",
    "tokens_in_est",
    "tokens_out_est",
]

# Rough cost table in USD per 1M tokens (input, output); kept in sync
# manually with `analysis/meaning_preservation.py`'s `_PRICE_TABLE`. Used
# only for the progress sidecar.
_PRICE_TABLE: dict[str, tuple[float, float]] = {
    "openai/gpt-5.6-luna": (0.20, 1.20),
}


# ---------- config schema ----------


class ExtractorSpec(BaseModel, extra="forbid"):
    """The single LLM used to extract idioms from the `idiomatic` variant."""

    provider: str
    model: str
    # Overrides DecodingCfg.temperature for the extractor call. Needed for
    # vendor models (e.g. reasoning-tier) that reject anything but 1.0.
    temperature: float | None = None

    @field_validator("provider")
    @classmethod
    def _check_provider(cls, provider: str) -> str:
        if provider not in ("openai", "anthropic", "gemini"):
            raise ValueError(f"unknown provider: {provider!r}")
        return provider

    def label(self) -> str:
        """Canonical `provider/model` label used in cache keys and output rows."""
        return f"{self.provider}/{self.model}"


class DecodingCfg(BaseModel, extra="forbid"):
    """Deterministic-ish decoding params for the extractor call."""

    temperature: float = 0.0
    max_output_tokens: int = 256


class RetryCfg(BaseModel, extra="forbid"):
    """Retry-then-skip policy on transient / empty responses."""

    max_attempts: int = 3
    backoff_seconds: float = 2.0


class CacheCfg(BaseModel, extra="forbid"):
    """Response cache configuration for the extraction pass."""

    enabled: bool = True
    dir: str = ".hf_cache/annotate"


class LimitsCfg(BaseModel, extra="forbid"):
    """Safety cap for dry runs — prevents a runaway loop from spending the API budget."""

    max_rows: int | None = None
    """If set, at most this many *new* row-completions are performed before
    the run stops cleanly. Cached completions do not count. `None` disables
    the cap."""


class AnnotateConfig(BaseModel, extra="forbid"):
    """RQ2 idiom-extraction config, parsed from `configs/annotate/*.yaml`."""

    dataset: str
    prompt_file: str = "idiom_extraction_v1.txt"
    extractor: ExtractorSpec
    decoding: DecodingCfg = Field(default_factory=DecodingCfg)
    retry: RetryCfg = Field(default_factory=RetryCfg)
    cache: CacheCfg = Field(default_factory=CacheCfg)
    limits: LimitsCfg = Field(default_factory=LimitsCfg)
    output_dir: str = "datasets_out"


def load_config(path: Path) -> AnnotateConfig:
    """Parse and validate a `configs/annotate/*.yaml` file."""
    raw: Any = yaml.safe_load(path.read_text())
    return AnnotateConfig.model_validate(raw)


# ---------- data model ----------


@dataclass(slots=True, frozen=True)
class RowInput:
    """One aligned row: the original, its neutral paraphrase, and its idiomatic rewrite."""

    row_id: str
    original_text: str
    paraphrase_text: str
    idiomatic_text: str


@dataclass(slots=True, frozen=True)
class RowExtraction:
    """The extractor's result for one row: the idiom(s) it found in `idiomatic_text`."""

    row_id: str
    idioms: list[str]
    raw_output: str
    tokens_in_est: int
    tokens_out_est: int


@dataclass(slots=True)
class RunStats:
    """Tallies for one `run_extraction` invocation."""

    total_rows: int = 0
    cache_hits: int = 0
    api_calls: int = 0
    skipped_rows: int = 0
    tokens_in_est: int = 0
    tokens_out_est: int = 0
    dollars_est: float = 0.0


# ---------- input loading ----------


def _read_variant_csv(path: Path) -> dict[str, str]:
    """Read a Stage 2 variant CSV into `{row_id: x_text}`."""
    out: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_id = row.get("id") or ""
            if row_id:
                out[row_id] = row.get("x") or ""
    return out


def load_aligned_rows(dataset_dir: Path) -> list[RowInput]:
    """Return every row present in all three of `original`/`paraphrase`/`idiomatic` CSVs.

    Matches the alignment contract already enforced by
    `analysis.meaning_preservation.load_aligned_pairs`.
    """
    original = _read_variant_csv(dataset_dir / "original.csv")
    paraphrase = _read_variant_csv(dataset_dir / "paraphrase.csv")
    idiomatic = _read_variant_csv(dataset_dir / "idiomatic.csv")
    aligned_ids = sorted(set(original) & set(paraphrase) & set(idiomatic))
    return [
        RowInput(
            row_id=row_id,
            original_text=original[row_id],
            paraphrase_text=paraphrase[row_id],
            idiomatic_text=idiomatic[row_id],
        )
        for row_id in aligned_ids
    ]


# ---------- prompt rendering & response parsing ----------


def render_prompt(template: str, row: RowInput) -> str:
    """Substitute a `RowInput` into an `idiom_extraction_v1.txt`-style template."""
    return template.format(
        original=row.original_text,
        paraphrase=row.paraphrase_text,
        idiomatic=row.idiomatic_text,
    )


def parse_extraction(raw: str) -> list[str]:
    """Parse the extractor's response into a list of lowercase idiom surface strings.

    Expected format: one idiom per line, or the literal `NONE` on its own
    line if no idiom was inserted. Blank lines are ignored.

    Raises:
        EmptyResponseError: If the response has no non-blank content.
    """
    lines = [line.strip() for line in raw.strip().splitlines() if line.strip()]
    if not lines:
        raise EmptyResponseError("extractor returned empty output")
    if len(lines) == 1 and lines[0].upper() == "NONE":
        return []
    return [line.lower() for line in lines]


# ---------- durable output writer ----------


def heal_partial_csv(path: Path) -> None:
    """Truncate a torn final line left by a killed writer, mirroring `augmentation/io.py`."""
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("rb+") as f:
        data = f.read()
        if data.endswith(b"\n"):
            return
        last_newline = data.rfind(b"\n")
        f.seek(last_newline + 1 if last_newline >= 0 else 0)
        f.truncate()


class _PartialCsvAppender:
    """Append-only, per-row-flushed writer for `idiom_extractions.partial.csv`."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists() or path.stat().st_size == 0
        self._fh = path.open("a", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(
            self._fh, fieldnames=_OUTPUT_COLUMNS, quoting=csv.QUOTE_MINIMAL, lineterminator="\n"
        )
        if write_header:
            self._writer.writeheader()
            self._fh.flush()

    def append(
        self,
        extraction: RowExtraction,
        *,
        provider: str,
        model: str,
        prompt_version: str,
        prompt_hash_value: str,
    ) -> None:
        """Write one extraction record and flush it to disk."""
        self._writer.writerow(
            {
                "id": extraction.row_id,
                "idioms_json": json.dumps(extraction.idioms),
                "extractor_provider": provider,
                "extractor_model": model,
                "prompt_version": prompt_version,
                "prompt_hash": prompt_hash_value,
                "raw_output": extraction.raw_output,
                "tokens_in_est": extraction.tokens_in_est,
                "tokens_out_est": extraction.tokens_out_est,
            }
        )
        self._fh.flush()

    def close(self) -> None:
        """Close the underlying file handle."""
        self._fh.close()

    def __enter__(self) -> "_PartialCsvAppender":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def _read_done_ids(path: Path) -> set[str]:
    """Return the set of row ids already present in the partial CSV."""
    if not path.exists() or path.stat().st_size == 0:
        return set()
    done: set[str] = set()
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_id = row.get("id") or ""
            if row_id:
                done.add(row_id)
    return done


# ---------- token accounting ----------


def _estimate_tokens(text: str) -> int:
    """Crude token estimate (~4 chars per token) for the progress sidecar."""
    return max(1, len(text) // 4)


def _dollars_from_tokens(label: str, tokens_in: int, tokens_out: int) -> float:
    """Rough USD cost estimate using `_PRICE_TABLE`; returns 0 for unknown labels."""
    prices = _PRICE_TABLE.get(label)
    if prices is None:
        return 0.0
    price_in, price_out = prices
    return (tokens_in / 1_000_000) * price_in + (tokens_out / 1_000_000) * price_out


# ---------- orchestrator ----------


def check_env_key(provider: str) -> None:
    """Fail loud if the extractor's provider API key env var is missing."""
    provider_env = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GEMINI_API_KEY",
    }
    env = provider_env[provider]
    if not os.environ.get(env):
        raise LLMError(
            f"missing API key env var: {env} (needed by extractor provider {provider!r})\n"
            "Set it in your shell (or `source .env` if you use option A) and rerun."
        )


def _extract_once(
    *,
    client: LLMClient,
    row: RowInput,
    template: str,
    decoding: DecodingCfg,
    retry: RetryCfg,
) -> RowExtraction:
    """Call the extractor on one row with retry-on-transient.

    Returns an extraction on success. Raises `LLMError` on persistent
    transport failure or if no attempt produced a parseable response.
    """
    user_prompt = render_prompt(template, row)
    last_exc: BaseException | None = None
    for attempt in range(1, retry.max_attempts + 1):
        try:
            raw = client.complete(
                system="",
                user=user_prompt,
                temperature=decoding.temperature,
                max_output_tokens=decoding.max_output_tokens,
            )
            idioms = parse_extraction(raw)
            return RowExtraction(
                row_id=row.row_id,
                idioms=idioms,
                raw_output=raw,
                tokens_in_est=_estimate_tokens(user_prompt),
                tokens_out_est=_estimate_tokens(raw),
            )
        except EmptyResponseError as exc:
            last_exc = exc
            if attempt < retry.max_attempts and retry.backoff_seconds > 0:
                time.sleep(retry.backoff_seconds * (2 ** (attempt - 1)))
        except LLMError as exc:
            last_exc = exc
            if attempt < retry.max_attempts and retry.backoff_seconds > 0:
                time.sleep(retry.backoff_seconds * (2 ** (attempt - 1)))
    if last_exc is None:
        raise LLMError(f"extractor failed after {retry.max_attempts} attempts for {row.row_id}")
    raise LLMError(str(last_exc))


def _process_one_row(
    *,
    client: LLMClient,
    row: RowInput,
    template: str,
    template_hash: str,
    template_version: str,
    cache: ResponseCache,
    label: str,
    decoding: DecodingCfg,
    retry: RetryCfg,
) -> tuple[RowExtraction | None, str, str | None]:
    """Return `(extraction, status, detail)` for one row: status ∈ {hit, api, skip}.

    Cached extractions return `("hit", ...)`. New API calls return
    `("api", ...)` and are written to the cache inside this helper. Persistent
    failure returns `("skip", None, reason)`.
    """
    cached = cache.get(template_hash, label, row.row_id)
    if cached is not None:
        return (
            RowExtraction(
                row_id=row.row_id,
                idioms=list(cached.get("idioms", [])),
                raw_output=str(cached.get("raw_output", "")),
                tokens_in_est=int(cached.get("tokens_in_est", 0)),
                tokens_out_est=int(cached.get("tokens_out_est", 0)),
            ),
            "hit",
            None,
        )
    try:
        extraction = _extract_once(
            client=client, row=row, template=template, decoding=decoding, retry=retry
        )
    except LLMError as exc:
        return (None, "skip", str(exc))
    cache.put(
        template_hash,
        label,
        row.row_id,
        {
            "idioms": extraction.idioms,
            "raw_output": extraction.raw_output,
            "tokens_in_est": extraction.tokens_in_est,
            "tokens_out_est": extraction.tokens_out_est,
        },
    )
    return (extraction, "api", None)


def _record_result(
    *,
    extraction: RowExtraction,
    status: str,
    label: str,
    stats: RunStats,
    appender: "_PartialCsvAppender",
    provider: str,
    model: str,
    prompt_version: str,
    prompt_hash_value: str,
) -> None:
    """Fold one successful (hit or api) extraction into the run's tallies and CSV."""
    if status == "hit":
        stats.cache_hits += 1
    else:
        stats.api_calls += 1
        stats.tokens_in_est += extraction.tokens_in_est
        stats.tokens_out_est += extraction.tokens_out_est
        stats.dollars_est += _dollars_from_tokens(
            label, extraction.tokens_in_est, extraction.tokens_out_est
        )
    appender.append(
        extraction,
        provider=provider,
        model=model,
        prompt_version=prompt_version,
        prompt_hash_value=prompt_hash_value,
    )


def run_extraction(cfg: AnnotateConfig) -> RunStats:
    """Run the idiom-extraction pass end-to-end for one dataset.

    For each aligned row, in order:
    - Returns a cached extraction when present.
    - Otherwise calls the extractor with retry-on-transient, writes cache,
      appends the result to `idiom_extractions.partial.csv`.

    Returns a `RunStats` summary. On any `LimitsCfg.max_rows` cap or on
    Ctrl-C the caller is left with a durable partial CSV that a
    re-invocation will resume from (per-row cache + per-row streaming).
    """
    check_env_key(cfg.extractor.provider)

    dataset_dir = Path(cfg.output_dir) / cfg.dataset
    rows = load_aligned_rows(dataset_dir)

    template = load_prompt(cfg.prompt_file)
    template_hash = prompt_hash(template)
    template_version = cfg.prompt_file.removesuffix(".txt")

    partial_path = dataset_dir / "idiom_extractions.partial.csv"
    heal_partial_csv(partial_path)
    done_ids = _read_done_ids(partial_path)

    cache = ResponseCache(Path(cfg.cache.dir), enabled=cfg.cache.enabled)
    client = build_client(cfg.extractor.provider, cfg.extractor.model)
    label = cfg.extractor.label()

    decoding = cfg.decoding
    if cfg.extractor.temperature is not None:
        decoding = decoding.model_copy(update={"temperature": cfg.extractor.temperature})

    stats = RunStats(total_rows=len(rows))
    new_calls_allowed = cfg.limits.max_rows
    bar_total = (
        stats.total_rows
        if new_calls_allowed is None
        else min(stats.total_rows, len(done_ids) + new_calls_allowed)
    )

    with _PartialCsvAppender(partial_path) as appender:
        pbar = tqdm(total=bar_total, desc=f"annotate {cfg.dataset}", unit="row")
        pbar.update(len(done_ids))
        for row in rows:
            if row.row_id in done_ids:
                continue
            if new_calls_allowed is not None and new_calls_allowed <= 0:
                pbar.write("[limit] max_rows cap reached; stopping cleanly. Rerun to continue.")
                pbar.close()
                return stats

            extraction, status, detail = _process_one_row(
                client=client,
                row=row,
                template=template,
                template_hash=template_hash,
                template_version=template_version,
                cache=cache,
                label=label,
                decoding=decoding,
                retry=cfg.retry,
            )
            if new_calls_allowed is not None and status != "hit":
                new_calls_allowed -= 1
            if status == "skip":
                pbar.write(f"[skip] {row.row_id} — {detail}")
                stats.skipped_rows += 1
                pbar.update(1)
                continue

            assert extraction is not None
            _record_result(
                extraction=extraction,
                status=status,
                label=label,
                stats=stats,
                appender=appender,
                provider=cfg.extractor.provider,
                model=cfg.extractor.model,
                prompt_version=template_version,
                prompt_hash_value=template_hash,
            )
            pbar.update(1)
        pbar.close()
    return stats


def finalize_partial_csv(dataset_dir: Path) -> Path:
    """Rename the durable partial CSV to the final `idiom_extractions.csv`.

    Called after a run finishes cleanly (all rows done). Rewrites with
    `pandas` so the final file's row order is canonical (sorted by id).
    """
    partial = dataset_dir / "idiom_extractions.partial.csv"
    final = dataset_dir / "idiom_extractions.csv"
    if not partial.exists():
        raise FileNotFoundError(f"partial CSV missing: {partial}")
    df = pd.read_csv(partial, keep_default_na=False, dtype={"id": str})
    df = df.sort_values(["id"]).reset_index(drop=True)
    tmp = final.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(final)
    return final


# ---------- norm linking (RQ2 plan §5 step 2, §8 idiom-boundary convention) ----------


def link_to_norms(
    extractions_df: pd.DataFrame,
    *,
    opacity_table: dict[str, OpacityEntry],
    magpie_stats: dict[str, MagpieStats],
    word_freq: dict[str, int],
    fuzzy_index: dict[frozenset[str], OpacityEntry] | None = None,
    llm_opacity_table: dict[str, OpacityEntry] | None = None,
) -> pd.DataFrame:
    """Join each row's extracted idiom(s) to the opacity/frequency norm tables.

    Row-level `opacity` uses the highest-opacity idiom present among that
    row's extracted idioms (plan §8: "use the highest-opacity idiom present
    as the row-level opacity"). `first_idiom_opacity` and `mean_opacity` are
    also reported, for the "first idiom"/"mean opacity" robustness checks
    the plan calls for. Rows with no idiom covered by either opacity source
    get `opacity=NaN`, `opacity_source="none"`.

    Each extracted idiom is looked up in three tiers, in order: exact string
    match against `opacity_table`; failing that, a content-word-set match
    against `fuzzy_index` (`build_fuzzy_opacity_index`); failing that, an
    LLM-rated fallback from `llm_opacity_table` (`data.norms.llm_opacity`).

    Args:
        extractions_df: A frame shaped like `idiom_extractions.csv`
            (columns include `id`, `idioms_json`).
        opacity_table: From `data.norms.opacity.load_opacity_norms()`.
        magpie_stats: From `data.norms.magpie.load_magpie_stats()`.
        word_freq: From `data.norms.frequency.load_word_frequencies()`.
        fuzzy_index: From `data.norms.opacity.build_fuzzy_opacity_index()`.
            Omit to disable the fuzzy-match tier.
        llm_opacity_table: From `data.norms.llm_opacity.load_llm_opacity_ratings()`.
            Omit to disable the LLM-rated tier.

    Returns:
        One row per input row, with `id`, `num_idioms_extracted`,
        `primary_idiom`, `opacity`, `opacity_source`, `first_idiom_opacity`,
        `mean_opacity`, `log_freq`, `magpie_idiomatic_fraction`.
    """
    fuzzy_index = fuzzy_index or {}
    llm_opacity_table = llm_opacity_table or {}

    def _lookup(idiom: str) -> OpacityEntry | None:
        entry = opacity_table.get(idiom)
        if entry is not None:
            return entry
        entry = fuzzy_index.get(content_words(idiom))
        if entry is not None:
            return entry
        return llm_opacity_table.get(idiom)

    records: list[dict[str, Any]] = []
    for _, row in extractions_df.iterrows():
        row_id = str(row["id"])
        idioms_json = str(row["idioms_json"])
        idioms: list[str] = json.loads(idioms_json) if idioms_json else []
        covered = [
            (idiom, entry) for idiom in idioms for entry in (_lookup(idiom),) if entry is not None
        ]

        primary_idiom = None
        opacity = None
        opacity_source = "none"
        first_opacity = None
        mean_opacity = None
        log_freq = None
        idiomatic_fraction = None

        if covered:
            primary_idiom, primary_entry = max(covered, key=lambda pair: pair[1].opacity)
            opacity = primary_entry.opacity
            opacity_source = primary_entry.source
            first_opacity = covered[0][1].opacity
            mean_opacity = sum(entry.opacity for _, entry in covered) / len(covered)
            freq = idiom_frequency(primary_idiom, word_freq)
            if freq is not None and freq > 0:
                log_freq = math.log(freq)
            stats = magpie_stats.get(primary_idiom)
            if stats is not None:
                idiomatic_fraction = stats.idiomatic_fraction

        records.append(
            {
                "id": row_id,
                "num_idioms_extracted": len(idioms),
                "primary_idiom": primary_idiom,
                "opacity": opacity,
                "opacity_source": opacity_source,
                "first_idiom_opacity": first_opacity,
                "mean_opacity": mean_opacity,
                "log_freq": log_freq,
                "magpie_idiomatic_fraction": idiomatic_fraction,
            }
        )
    return pd.DataFrame.from_records(records)


def annotate_and_link(dataset_dir: Path) -> Path:
    """Load the finalized `idiom_extractions.csv`, join to norms, write `idiom_norms.csv`."""
    extractions_path = dataset_dir / "idiom_extractions.csv"
    df = pd.read_csv(extractions_path, keep_default_na=False, dtype={"id": str})
    opacity_table = load_opacity_norms()
    linked = link_to_norms(
        df,
        opacity_table=opacity_table,
        magpie_stats=load_magpie_stats(),
        word_freq=load_word_frequencies(),
        fuzzy_index=build_fuzzy_opacity_index(opacity_table),
        llm_opacity_table=load_llm_opacity_ratings(),
    )
    out_path = dataset_dir / "idiom_norms.csv"
    linked.to_csv(out_path, index=False)
    return out_path
