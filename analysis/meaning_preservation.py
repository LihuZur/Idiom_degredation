"""RQ1 meaning-preservation judge pass (paper_preparations/RQ1_meaning_preservation.md §5).

Runs one or more LLM judges over every (row, arm) pair in a dataset's Stage 2
output and produces `datasets_out/{ds}/meaning_validated.csv`. Reuses the
existing `ResponseCache` for per-pair-per-judge resume and streams merged rows
to a partial CSV so an interrupted run is durable and resumable.

Design mirrors `augmentation/pipeline.py`:
- Per-pair-per-judge cache file → no wasted API calls on rerun.
- Streaming append CSV writer → durable output mid-run.
- Retry-then-skip on transient errors → the whole run never dies on one row.
- Pydantic config with `extra="forbid"` → typos in YAML fail fast.
- Prompt-hash + config-hash sidecar → stale-cache detection at resume.
"""

import csv
import json
import os
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, Field, field_validator
from tqdm.auto import tqdm

from analysis.prompts.loader import load_prompt, render_context_block
from analysis.stats import cohen_kappa, gwet_ac1
from augmentation.cache import ResponseCache
from augmentation.providers.base import EmptyResponseError, LLMClient, LLMError, build_client
from cleaning.hashing import prompt_hash

Arm = Literal["par", "idi"]
_ARMS: tuple[Arm, ...] = ("par", "idi")

_OUTPUT_COLUMNS = [
    "id",
    "arm",
    "judge_provider",
    "judge_model",
    "prompt_version",
    "prompt_hash",
    "preserved",
    "verdict_raw",
    "reason",
    "tokens_in_est",
    "tokens_out_est",
]

# Rough cost table in USD per 1M tokens (input, output). Update when vendor
# prices change; used only for the progress sidecar so a runaway loop is
# noticed before the vendor's own limit fires.
_PRICE_TABLE: dict[str, tuple[float, float]] = {
    "openai/gpt-5.6-sol": (4.00, 20.00),
    "openai/gpt-5.6-terra": (2.00, 12.00),
    "openai/gpt-5.6-luna": (0.20, 1.20),
    "openai/gpt-6-astra": (10.00, 50.00),
    "anthropic/claude-sonnet-4-5": (3.00, 15.00),
    "anthropic/claude-opus-4-5": (15.00, 75.00),
    "anthropic/claude-haiku-4-5": (0.80, 4.00),
}


# ---------- config schema ----------


class JudgeSpec(BaseModel, extra="forbid"):
    """One judge (provider + pinned model) used in the two-judge cross-check."""

    provider: str
    model: str
    # Overrides DecodingCfg.temperature for this judge only. Needed because
    # some vendor models (e.g. reasoning-tier) reject any value but 1.0.
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
    """Deterministic-ish decoding params for a judge call."""

    temperature: float = 0.0
    max_output_tokens: int = 128


class RetryCfg(BaseModel, extra="forbid"):
    """Retry-then-skip policy on transient / empty responses."""

    max_attempts: int = 3
    backoff_seconds: float = 2.0


class CacheCfg(BaseModel, extra="forbid"):
    """Response cache configuration for the judge pass."""

    enabled: bool = True
    dir: str = ".hf_cache/validate"


class LimitsCfg(BaseModel, extra="forbid"):
    """Safety caps for dry runs — prevents a runaway loop from spending the API budget."""

    max_pairs: int | None = None
    """If set, at most this many *new* (id, arm) pair-completions are performed
    across all judges combined before the run stops cleanly. Cached completions
    do not count. `None` disables the cap."""


class ValidateConfig(BaseModel, extra="forbid"):
    """RQ1 meaning-preservation validation config, parsed from `configs/validate/*.yaml`."""

    dataset: str
    prompt_file: str = "meaning_preservation_v1.txt"
    judges: list[JudgeSpec] = Field(min_length=1)
    decoding: DecodingCfg = Field(default_factory=DecodingCfg)
    retry: RetryCfg = Field(default_factory=RetryCfg)
    cache: CacheCfg = Field(default_factory=CacheCfg)
    limits: LimitsCfg = Field(default_factory=LimitsCfg)
    output_dir: str = "datasets_out"


def load_config(path: Path) -> ValidateConfig:
    """Parse and validate a `configs/validate/*.yaml` file."""
    raw: Any = yaml.safe_load(path.read_text())
    return ValidateConfig.model_validate(raw)


# ---------- data model ----------


@dataclass(slots=True, frozen=True)
class PairInput:
    """One (row_id, arm) pair to be judged: original text vs. the arm's rewrite."""

    row_id: str
    arm: Arm
    original_text: str
    rewrite_text: str
    meta: dict[str, Any]


@dataclass(slots=True, frozen=True)
class PairVerdict:
    """One judge's verdict for one (row_id, arm) pair."""

    row_id: str
    arm: Arm
    judge_provider: str
    judge_model: str
    prompt_version: str
    prompt_hash: str
    preserved: bool
    verdict_raw: str
    reason: str
    tokens_in_est: int
    tokens_out_est: int


@dataclass(slots=True)
class RunStats:
    """Tallies for one `run_validation` invocation."""

    total_pairs: int = 0
    cache_hits: int = 0
    api_calls: int = 0
    skipped_rows: int = 0
    tokens_in_est: int = 0
    tokens_out_est: int = 0
    dollars_est: float = 0.0
    per_judge: dict[str, dict[str, int]] = field(default_factory=dict)


# ---------- input loading ----------


def _parse_meta(meta_str: str | None) -> dict[str, Any]:
    """Parse a `meta` CSV field, returning `{}` on empty/invalid JSON."""
    if not meta_str:
        return {}
    try:
        val = json.loads(meta_str)
        return val if isinstance(val, dict) else {}
    except json.JSONDecodeError:
        return {}


def _read_variant_csv(path: Path) -> dict[str, tuple[str, dict[str, Any]]]:
    """Read a Stage 2 variant CSV into `{row_id: (x_text, meta_dict)}`."""
    out: dict[str, tuple[str, dict[str, Any]]] = {}
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_id = row.get("id") or ""
            if not row_id:
                continue
            x = row.get("x") or ""
            meta = _parse_meta(row.get("meta"))
            out[row_id] = (x, meta)
    return out


def load_aligned_pairs(dataset_dir: Path) -> list[PairInput]:
    """Return every (row_id, arm) pair for which all three variant CSVs have an entry.

    An "aligned" row-id must appear in `original.csv`, `paraphrase.csv`, and
    `idiomatic.csv`; only aligned rows produce pairs. This matches the alignment
    contract already enforced by `scripts/variant_confusion.py`.
    """
    original = _read_variant_csv(dataset_dir / "original.csv")
    paraphrase = _read_variant_csv(dataset_dir / "paraphrase.csv")
    idiomatic = _read_variant_csv(dataset_dir / "idiomatic.csv")

    aligned_ids = sorted(set(original) & set(paraphrase) & set(idiomatic))
    pairs: list[PairInput] = []
    for row_id in aligned_ids:
        orig_text, meta = original[row_id]
        pairs.append(
            PairInput(
                row_id=row_id,
                arm="par",
                original_text=orig_text,
                rewrite_text=paraphrase[row_id][0],
                meta=meta,
            )
        )
        pairs.append(
            PairInput(
                row_id=row_id,
                arm="idi",
                original_text=orig_text,
                rewrite_text=idiomatic[row_id][0],
                meta=meta,
            )
        )
    return pairs


# ---------- prompt rendering & response parsing ----------


def render_prompt(template: str, pair: PairInput) -> str:
    """Substitute a `PairInput` into a `meaning_preservation_v1.txt`-style template."""
    variant_label = "paraphrase" if pair.arm == "par" else "idiomatic"
    context_block = render_context_block(pair.meta)
    return template.format(
        original=pair.original_text,
        variant=variant_label,
        rewrite=pair.rewrite_text,
        context_block=context_block,
    )


def parse_verdict(raw: str) -> tuple[bool, str]:
    """Parse a judge response into `(preserved, reason)`.

    The prompt requires exactly two lines (PASS/FAIL then a reason), but
    judges occasionally violate this and second-guess themselves mid-response
    (e.g. "Wait, let me reconsider... PASS ..."), emitting a second
    verdict+reason pair after the first. Found 2026-09-17 on ~10% of mnli's
    anthropic responses (vs. ~0.3-0.6% baseline noise on sst2/mmlu). We take
    the LAST line that starts with PASS/FAIL as the judge's final, most-
    considered verdict, not the first. Raises `EmptyResponseError` if no line
    starts with PASS/FAIL.
    """
    lines = [line.strip() for line in raw.strip().splitlines() if line.strip()]
    if not lines:
        raise EmptyResponseError("judge returned empty output")
    verdict_idxs = [i for i, line in enumerate(lines) if line.upper().startswith(("PASS", "FAIL"))]
    if not verdict_idxs:
        raise EmptyResponseError(f"judge did not start with PASS/FAIL: {lines[0]!r}")
    idx = verdict_idxs[-1]
    preserved = lines[idx].upper().startswith("PASS")
    reason = " ".join(lines[idx + 1 :])
    return preserved, reason


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
    """Append-only, per-row-flushed writer for `meaning_validated.partial.csv`."""

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

    def append(self, verdict: PairVerdict) -> None:
        """Write one verdict record and flush it to disk."""
        self._writer.writerow(
            {
                "id": verdict.row_id,
                "arm": verdict.arm,
                "judge_provider": verdict.judge_provider,
                "judge_model": verdict.judge_model,
                "prompt_version": verdict.prompt_version,
                "prompt_hash": verdict.prompt_hash,
                "preserved": int(verdict.preserved),
                "verdict_raw": verdict.verdict_raw,
                "reason": verdict.reason,
                "tokens_in_est": verdict.tokens_in_est,
                "tokens_out_est": verdict.tokens_out_est,
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


def _read_done_keys(path: Path) -> set[tuple[str, str, str]]:
    """Return `{(id, arm, judge_label)}` for rows already present in the partial CSV."""
    if not path.exists() or path.stat().st_size == 0:
        return set()
    done: set[tuple[str, str, str]] = set()
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_id = row.get("id") or ""
            arm = row.get("arm") or ""
            label = f"{row.get('judge_provider') or ''}/{row.get('judge_model') or ''}"
            if row_id and arm and label != "/":
                done.add((row_id, arm, label))
    return done


# ---------- token accounting ----------


def _estimate_tokens(text: str) -> int:
    """Crude token estimate (~4 chars per token) for the progress sidecar."""
    return max(1, len(text) // 4)


def _dollars_from_tokens(judge_label: str, tokens_in: int, tokens_out: int) -> float:
    """Rough USD cost estimate using `_PRICE_TABLE`; returns 0 for unknown labels."""
    prices = _PRICE_TABLE.get(judge_label)
    if prices is None:
        return 0.0
    price_in, price_out = prices
    return (tokens_in / 1_000_000) * price_in + (tokens_out / 1_000_000) * price_out


# ---------- orchestrator ----------


def _judge_once(
    *,
    client: LLMClient,
    pair: PairInput,
    template: str,
    template_hash: str,
    template_version: str,
    decoding: DecodingCfg,
    retry: RetryCfg,
) -> PairVerdict:
    """Call one judge on one pair with retry-on-transient. Skipped rows raise `LLMError`.

    Returns a verdict on success. Raises `EmptyResponseError` if no attempt
    produced a parseable PASS/FAIL response. Raises `LLMError` on persistent
    transport failure.
    """
    user_prompt = render_prompt(template, pair)
    last_exc: BaseException | None = None
    for attempt in range(1, retry.max_attempts + 1):
        try:
            raw = client.complete(
                system="",
                user=user_prompt,
                temperature=decoding.temperature,
                max_output_tokens=decoding.max_output_tokens,
            )
            preserved, reason = parse_verdict(raw)
            return PairVerdict(
                row_id=pair.row_id,
                arm=pair.arm,
                judge_provider=client.provider,
                judge_model=client.model,
                prompt_version=template_version,
                prompt_hash=template_hash,
                preserved=preserved,
                verdict_raw=raw,
                reason=reason,
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
        raise LLMError(
            f"judge failed after {retry.max_attempts} attempts for {pair.row_id}:{pair.arm}"
        )
    raise LLMError(str(last_exc))


def _verdict_from_cached(
    *,
    cached: dict[str, Any],
    pair: PairInput,
    judge: JudgeSpec,
    template_hash: str,
    template_version: str,
) -> PairVerdict:
    """Rebuild a `PairVerdict` from a cache entry."""
    return PairVerdict(
        row_id=pair.row_id,
        arm=pair.arm,
        judge_provider=judge.provider,
        judge_model=judge.model,
        prompt_version=template_version,
        prompt_hash=template_hash,
        preserved=bool(cached.get("preserved", False)),
        verdict_raw=str(cached.get("verdict_raw", "")),
        reason=str(cached.get("reason", "")),
        tokens_in_est=int(cached.get("tokens_in_est", 0)),
        tokens_out_est=int(cached.get("tokens_out_est", 0)),
    )


def _cache_payload(verdict: PairVerdict) -> dict[str, Any]:
    """Serialize a verdict for `ResponseCache.put`."""
    return {
        "preserved": verdict.preserved,
        "verdict_raw": verdict.verdict_raw,
        "reason": verdict.reason,
        "tokens_in_est": verdict.tokens_in_est,
        "tokens_out_est": verdict.tokens_out_est,
    }


def _cache_key(*, template_hash: str, judge: JudgeSpec, pair: PairInput) -> tuple[str, str, str]:
    """Cache key tuple for one (judge, pair): prompt-invalidated + judge-scoped + arm-scoped."""
    return (template_hash, judge.label(), f"{pair.row_id}:{pair.arm}")


def _check_env_keys(judges: Iterable[JudgeSpec]) -> None:
    """Fail loud if any configured judge's API key env var is missing."""
    provider_env = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GEMINI_API_KEY",
    }
    missing: list[str] = []
    for judge in judges:
        env = provider_env[judge.provider]
        if not os.environ.get(env):
            missing.append(f"{env} (needed by judge {judge.label()})")
    if missing:
        raise LLMError(
            "missing API key env vars:\n  - "
            + "\n  - ".join(missing)
            + "\nSet them in your shell (or `source .env` if you use option A) and rerun."
        )


def _build_clients(judges: Iterable[JudgeSpec]) -> dict[str, LLMClient]:
    """Instantiate one `LLMClient` per judge, keyed by `provider/model`."""
    clients: dict[str, LLMClient] = {}
    for judge in judges:
        clients[judge.label()] = build_client(judge.provider, judge.model)
    return clients


def _write_progress_sidecar(path: Path, stats: RunStats, cfg: ValidateConfig) -> None:
    """Write `judge_progress.json` — informational; the cache is the source of truth."""
    payload = {
        "dataset": cfg.dataset,
        "judges": [j.label() for j in cfg.judges],
        "total_pairs": stats.total_pairs,
        "cache_hits": stats.cache_hits,
        "api_calls": stats.api_calls,
        "skipped_rows": stats.skipped_rows,
        "tokens_in_est": stats.tokens_in_est,
        "tokens_out_est": stats.tokens_out_est,
        "dollars_est": round(stats.dollars_est, 4),
        "per_judge": stats.per_judge,
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def finalize_partial_csv(dataset_dir: Path) -> Path:
    """Rename the durable partial CSV to the final `meaning_validated.csv`.

    Called after a run finishes cleanly (all judges, all pairs done). Rewrites
    with `pandas` so the final file's row order is canonical (sorted by id, arm,
    judge_provider, judge_model) instead of insertion order.
    """
    partial = dataset_dir / "meaning_validated.partial.csv"
    final = dataset_dir / "meaning_validated.csv"
    if not partial.exists():
        raise FileNotFoundError(f"partial CSV missing: {partial}")
    df = pd.read_csv(partial, keep_default_na=False, dtype={"id": str, "arm": str})
    df = df.sort_values(["id", "arm", "judge_provider", "judge_model"]).reset_index(drop=True)
    tmp = final.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(final)
    return final


# ---------- accept rule & agreement (RQ1 plan §5 steps 4-6) ----------


def compute_accept_rule(verdicts_path: Path) -> pd.DataFrame:
    """Pivot long-format judge verdicts into one row per id with `accepted_pair`.

    Reads a `meaning_validated(.partial).csv`-shaped file (one row per
    id x arm x judge). A row is `accepted_pair=True` iff EVERY judge present
    in the file marked `preserved=True` on BOTH arms. A judge/arm combo that
    hasn't been judged yet counts as not-accepted (rather than raising), so
    this is safe to call on an in-progress partial CSV.

    Also writes the result to `meaning_accept.csv` next to `verdicts_path`.

    Args:
        verdicts_path: Path to a long-format verdicts CSV (partial or final).

    Returns:
        One row per `id`, with a `{judge_provider}/{judge_model}__{arm}`
        boolean column per judge x arm plus `accepted_pair`.

    Raises:
        FileNotFoundError: If `verdicts_path` does not exist.
    """
    if not verdicts_path.exists():
        raise FileNotFoundError(f"verdicts CSV missing: {verdicts_path}")
    df = pd.read_csv(verdicts_path, keep_default_na=False, dtype={"id": str, "arm": str})
    df["judge_label"] = df["judge_provider"] + "/" + df["judge_model"]
    df["preserved"] = df["preserved"].astype(bool)
    wide = df.pivot(index="id", columns=["judge_label", "arm"], values="preserved")
    # Reindex to the full judge x arm grid (both `_ARMS`) so a judge/arm combo
    # that's entirely absent so far (e.g. a judge hasn't reached this arm at
    # all yet) still yields a column of NaN instead of silently vanishing —
    # otherwise `.all(axis=1)` below would ignore it and over-accept rows.
    judges = sorted(df["judge_label"].unique())
    full_columns = pd.MultiIndex.from_product([judges, _ARMS])
    wide = wide.reindex(columns=full_columns)
    wide.columns = [f"{judge}__{arm}" for judge, arm in wide.columns]
    wide = wide.sort_index(axis=1).reset_index()
    verdict_cols = [c for c in wide.columns if c != "id"]
    wide["accepted_pair"] = wide[verdict_cols].fillna(False).all(axis=1)
    out_path = verdicts_path.parent / "meaning_accept.csv"
    wide.to_csv(out_path, index=False)
    return wide


_EXPECTED_JUDGE_COUNT = 2


def judge_agreement_kappa(verdicts_path: Path) -> float:
    """Cohen's kappa between the two judges on the pair-judgment task (H1d/H1e).

    Compares `preserved` verdicts matched by `(id, arm)`; only pairs judged
    by BOTH judges are included.

    Args:
        verdicts_path: Path to a long-format verdicts CSV (partial or final).

    Returns:
        Kappa in `[-1, 1]`, or `NaN` if no `(id, arm)` pair has been judged
        by both judges yet.

    Raises:
        FileNotFoundError: If `verdicts_path` does not exist.
        ValueError: If the file does not contain exactly two distinct judges.
    """
    if not verdicts_path.exists():
        raise FileNotFoundError(f"verdicts CSV missing: {verdicts_path}")
    df = pd.read_csv(verdicts_path, keep_default_na=False, dtype={"id": str, "arm": str})
    df["judge_label"] = df["judge_provider"] + "/" + df["judge_model"]
    df["preserved"] = df["preserved"].astype(bool)
    labels = sorted(df["judge_label"].unique())
    if len(labels) != _EXPECTED_JUDGE_COUNT:
        raise ValueError(
            f"expected exactly 2 judges in {verdicts_path}, found {len(labels)}: {labels}"
        )
    wide = df.pivot(index=["id", "arm"], columns="judge_label", values="preserved").dropna()
    if wide.empty:
        return float("nan")
    return cohen_kappa(
        wide[labels[0]].to_numpy(dtype=bool).tolist(),
        wide[labels[1]].to_numpy(dtype=bool).tolist(),
    )


def judge_agreement_ac1(verdicts_path: Path) -> float:
    """Gwet's AC1 between the two judges on the pair-judgment task (H1d).

    Prefer this over `judge_agreement_kappa` when the judges' accept rates
    are highly skewed (near 0% or 100%), where Cohen's kappa can understate
    agreement (the "kappa paradox"). See `analysis.stats.gwet_ac1`.

    Args:
        verdicts_path: Path to a long-format verdicts CSV (partial or final).

    Returns:
        AC1 in `[-1, 1]`, or `NaN` if no `(id, arm)` pair has been judged
        by both judges yet.

    Raises:
        FileNotFoundError: If `verdicts_path` does not exist.
        ValueError: If the file does not contain exactly two distinct judges.
    """
    if not verdicts_path.exists():
        raise FileNotFoundError(f"verdicts CSV missing: {verdicts_path}")
    df = pd.read_csv(verdicts_path, keep_default_na=False, dtype={"id": str, "arm": str})
    df["judge_label"] = df["judge_provider"] + "/" + df["judge_model"]
    df["preserved"] = df["preserved"].astype(bool)
    labels = sorted(df["judge_label"].unique())
    if len(labels) != _EXPECTED_JUDGE_COUNT:
        raise ValueError(
            f"expected exactly 2 judges in {verdicts_path}, found {len(labels)}: {labels}"
        )
    wide = df.pivot(index=["id", "arm"], columns="judge_label", values="preserved").dropna()
    if wide.empty:
        return float("nan")
    return gwet_ac1(
        wide[labels[0]].to_numpy(dtype=bool).tolist(),
        wide[labels[1]].to_numpy(dtype=bool).tolist(),
    )


def stratify_audit_sample(
    accept_df: pd.DataFrame,
    *,
    judge_labels: list[str],
    n_both_accept: int = 200,
    n_disagree: int = 200,
    n_both_reject: int = 100,
    seed: int = 0,
) -> pd.DataFrame:
    """Draw the 500-row stratified human-audit sample (RQ1 plan §5 step 6).

    Buckets each `id` by per-judge, both-arms acceptance (a judge "accepts" a
    row iff it marked `preserved=True` on both the `par` and `idi` arms):
    - both_accept: every judge accepts the row (equivalent to `accepted_pair`).
    - both_reject: every judge rejects the row, and all judges agree with
      each other.
    - disagree: judges' per-row accept flags are not unanimous.

    Args:
        accept_df: Output of `compute_accept_rule` (one row per id, with
            `{judge_label}__par` / `{judge_label}__idi` boolean columns).
        judge_labels: The `provider/model` labels to stratify over.
        n_both_accept: Rows to draw from the both-accept stratum.
        n_disagree: Rows to draw from the disagree stratum.
        n_both_reject: Rows to draw from the both-reject stratum.
        seed: Seed for the `numpy.random.default_rng` used for sampling.

    Returns:
        The concatenated sample (`n_both_accept + n_disagree + n_both_reject`
        rows) with an added `audit_stratum` column.

    Raises:
        ValueError: If any stratum has fewer rows available than requested.
    """
    per_judge_accept = pd.DataFrame(
        {
            label: accept_df[f"{label}__par"].fillna(False)
            & accept_df[f"{label}__idi"].fillna(False)
            for label in judge_labels
        }
    )
    both_accept_mask = per_judge_accept.all(axis=1)
    both_reject_mask = (~per_judge_accept).all(axis=1)
    disagree_mask = ~both_accept_mask & ~both_reject_mask

    rng = np.random.default_rng(seed)

    def _sample(mask: pd.Series, n: int, stratum: str) -> pd.DataFrame:
        pool = accept_df.loc[mask]
        if len(pool) < n:
            raise ValueError(f"only {len(pool)} rows in '{stratum}' stratum, need {n}")
        idx = rng.choice(pool.index.to_numpy(), size=n, replace=False)
        return pool.loc[idx].assign(audit_stratum=stratum)

    parts = [
        _sample(both_accept_mask, n_both_accept, "both_accept"),
        _sample(disagree_mask, n_disagree, "disagree"),
        _sample(both_reject_mask, n_both_reject, "both_reject"),
    ]
    return pd.concat(parts, ignore_index=True)


_HUMAN_LABEL_COLUMNS = ["human_preserved_par", "human_preserved_idi"]


def attach_audit_context(
    sample: pd.DataFrame, dataset_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a stratified sample into a blind human-facing sheet + a scoring key.

    `stratify_audit_sample`'s output has no text to read and full judge
    verdicts attached — useless for a blind human audit as-is. This splits it:
    the human-facing sheet gets the actual `original`/`paraphrase`/`idiomatic`
    text plus empty `human_preserved_par`/`human_preserved_idi` columns to
    fill in by hand, and drops every judge-verdict column so the audit stays
    blind (RQ1 plan §5 step 6). The scoring key keeps `id` and the judge
    columns for later comparison against the filled-in human labels (H1e).

    Args:
        sample: Output of `stratify_audit_sample` (one row per id, with
            `{judge_label}__par` / `{judge_label}__idi` boolean columns).
        dataset_dir: Directory holding `original.csv`, `paraphrase.csv`,
            `idiomatic.csv` for this dataset.

    Returns:
        `(blind_sheet, scoring_key)` — do not show `scoring_key` to the
        author while they are labeling `blind_sheet`.
    """
    original = _read_variant_csv(dataset_dir / "original.csv")
    paraphrase = _read_variant_csv(dataset_dir / "paraphrase.csv")
    idiomatic = _read_variant_csv(dataset_dir / "idiomatic.csv")

    judge_cols = [c for c in sample.columns if "__" in c]
    scoring_key = cast("pd.DataFrame", sample[["id", *judge_cols]]).copy()

    blind_sheet = cast("pd.DataFrame", sample[["id", "audit_stratum"]]).copy()
    blind_sheet["original_text"] = [original[i][0] for i in blind_sheet["id"]]
    blind_sheet["paraphrase_text"] = [paraphrase[i][0] for i in blind_sheet["id"]]
    blind_sheet["idiomatic_text"] = [idiomatic[i][0] for i in blind_sheet["id"]]
    for col in _HUMAN_LABEL_COLUMNS:
        blind_sheet[col] = ""
    return blind_sheet, scoring_key


def human_judge_agreement(audit_sheet_path: Path, audit_key_path: Path) -> pd.DataFrame:
    """Cohen's kappa between filled-in human audit labels and each judge (H1e).

    Reads the labeled `audit_sample.csv` (blind sheet, `human_preserved_par`/
    `human_preserved_idi` filled in by hand) and its `audit_sample_key.csv`
    sidecar (judge verdicts for the same rows), merges them on `id`, and
    computes one Cohen's kappa per judge, pooling both arms (matching
    `judge_agreement_kappa`'s `(id, arm)` granularity). Rows with a blank or
    unrecognized human label are excluded rather than erroring, so a
    partially-labeled sheet can be scored.

    Args:
        audit_sheet_path: Path to the labeled blind sheet (`audit_sample.csv`).
        audit_key_path: Path to the matching scoring key
            (`audit_sample_key.csv`, as written by `attach_audit_context`).

    Returns:
        One row per judge label with `n_rated`, `kappa`, `ac1`,
        `raw_agreement`, `human_accept_rate`, and `judge_accept_rate`.

    Raises:
        FileNotFoundError: If either path does not exist.
    """
    if not audit_sheet_path.exists():
        raise FileNotFoundError(f"audit sheet missing: {audit_sheet_path}")
    if not audit_key_path.exists():
        raise FileNotFoundError(f"audit scoring key missing: {audit_key_path}")

    sheet = pd.read_csv(audit_sheet_path, dtype={"id": str})
    key = pd.read_csv(audit_key_path, dtype={"id": str})
    merged = sheet.merge(key, on="id", validate="one_to_one")

    judge_labels = sorted({c.rsplit("__", 1)[0] for c in key.columns if "__" in c})
    rows: list[dict[str, Any]] = []
    for judge_label in judge_labels:
        human_vals: list[bool] = []
        judge_vals: list[bool] = []
        for arm in _ARMS:
            human_col = f"human_preserved_{arm}"
            judge_col = f"{judge_label}__{arm}"
            human_raw = merged[human_col].astype(str).str.strip().str.upper()
            is_rated = human_raw.isin(["TRUE", "FALSE"])
            human_vals.extend(human_raw[is_rated].eq("TRUE").tolist())
            judge_vals.extend(merged.loc[is_rated, judge_col].astype(bool).tolist())
        n_rated = len(human_vals)
        kappa = cohen_kappa(human_vals, judge_vals) if n_rated else float("nan")
        ac1 = gwet_ac1(human_vals, judge_vals) if n_rated else float("nan")
        raw_agreement = (
            sum(h == j for h, j in zip(human_vals, judge_vals, strict=True)) / n_rated
            if n_rated
            else float("nan")
        )
        rows.append(
            {
                "judge_label": judge_label,
                "n_rated": n_rated,
                "kappa": kappa,
                "ac1": ac1,
                "raw_agreement": raw_agreement,
                "human_accept_rate": (sum(human_vals) / n_rated) if n_rated else float("nan"),
                "judge_accept_rate": (sum(judge_vals) / n_rated) if n_rated else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _process_one_pair(
    *,
    judge: JudgeSpec,
    client: LLMClient,
    pair: PairInput,
    template: str,
    template_hash: str,
    template_version: str,
    cache: ResponseCache,
    decoding: DecodingCfg,
    retry: RetryCfg,
) -> tuple[PairVerdict | None, str, str | None]:
    """Return `(verdict, status, detail)` for one (judge, pair): status ∈ {hit, api, skip}.

    Cached verdicts return `("hit", verdict, None)`. New API calls return
    `("api", verdict, None)`. Persistent failure returns `("skip", None, reason)`
    where `reason` is the underlying error message. Any successful API call is
    written to the cache inside this helper so the caller never has to remember.
    """
    ph, model_label, input_id = _cache_key(template_hash=template_hash, judge=judge, pair=pair)
    cached = cache.get(ph, model_label, input_id)
    if cached is not None:
        return (
            _verdict_from_cached(
                cached=cached,
                pair=pair,
                judge=judge,
                template_hash=template_hash,
                template_version=template_version,
            ),
            "hit",
            None,
        )
    try:
        verdict = _judge_once(
            client=client,
            pair=pair,
            template=template,
            template_hash=template_hash,
            template_version=template_version,
            decoding=decoding
            if judge.temperature is None
            else decoding.model_copy(update={"temperature": judge.temperature}),
            retry=retry,
        )
    except LLMError as exc:
        return (None, "skip", str(exc))
    cache.put(ph, model_label, input_id, _cache_payload(verdict))
    return (verdict, "api", None)


def _record_result(
    *,
    verdict: PairVerdict,
    status: str,
    judge_label: str,
    stats: RunStats,
    per: dict[str, int],
    appender: "_PartialCsvAppender",
) -> None:
    """Fold one successful (hit or api) verdict into the run's tallies and CSV."""
    if status == "hit":
        stats.cache_hits += 1
        per["cache_hits"] += 1
    else:
        stats.api_calls += 1
        per["api_calls"] += 1
    appender.append(verdict)
    stats.tokens_in_est += verdict.tokens_in_est
    stats.tokens_out_est += verdict.tokens_out_est
    stats.dollars_est += _dollars_from_tokens(
        judge_label, verdict.tokens_in_est, verdict.tokens_out_est
    )
    if verdict.preserved:
        per["preserved_true"] += 1
    else:
        per["preserved_false"] += 1


def run_validation(cfg: ValidateConfig) -> RunStats:
    """Run the meaning-preservation judge pass end-to-end for one dataset.

    Loads aligned (row_id, arm) pairs and, for each pair in turn, runs every
    configured judge on it before moving to the next pair (so `--max-pairs N`
    gives you N fully cross-judged pairs, not N calls to just the first judge).
    For each (pair, judge):
    - Returns a cached verdict when present.
    - Otherwise calls the judge with retry-on-transient, writes cache, appends
      the verdict to `meaning_validated.partial.csv`.

    Returns a `RunStats` summary. On any `LimitsCfg.max_pairs` cap or on Ctrl-C
    the caller is left with a durable partial CSV that a re-invocation will
    resume from (per-pair cache + per-row streaming).
    """
    _check_env_keys(cfg.judges)

    dataset_dir = Path(cfg.output_dir) / cfg.dataset
    pairs = load_aligned_pairs(dataset_dir)

    template = load_prompt(cfg.prompt_file)
    template_hash = prompt_hash(template)
    template_version = cfg.prompt_file.removesuffix(".txt")

    partial_path = dataset_dir / "meaning_validated.partial.csv"
    heal_partial_csv(partial_path)
    done_keys = _read_done_keys(partial_path)

    cache = ResponseCache(Path(cfg.cache.dir), enabled=cfg.cache.enabled)
    clients = _build_clients(cfg.judges)

    stats = RunStats(total_pairs=len(pairs) * len(cfg.judges))
    for judge in cfg.judges:
        stats.per_judge[judge.label()] = {
            "cache_hits": 0,
            "api_calls": 0,
            "skipped": 0,
            "preserved_true": 0,
            "preserved_false": 0,
        }

    progress_path = dataset_dir / "judge_progress.json"
    new_api_calls_allowed = cfg.limits.max_pairs
    # Bar total reflects what this invocation can actually reach: with a cap,
    # tqdm's ETA would otherwise extrapolate against the full dataset and show
    # a wildly misleading multi-hour estimate for a run that stops in seconds.
    bar_total = (
        stats.total_pairs
        if new_api_calls_allowed is None
        else min(stats.total_pairs, len(done_keys) + new_api_calls_allowed)
    )

    with _PartialCsvAppender(partial_path) as appender:
        pbar = tqdm(total=bar_total, desc=f"validate {cfg.dataset}", unit="pair")
        pbar.update(len(done_keys))
        for pair in pairs:
            for judge in cfg.judges:
                client = clients[judge.label()]
                per = stats.per_judge[judge.label()]
                if (pair.row_id, pair.arm, judge.label()) in done_keys:
                    continue
                if new_api_calls_allowed is not None and new_api_calls_allowed <= 0:
                    pbar.write(
                        "[limit] max_pairs cap reached; stopping cleanly. Rerun to continue."
                    )
                    _write_progress_sidecar(progress_path, stats, cfg)
                    pbar.close()
                    return stats
                verdict, status, detail = _process_one_pair(
                    judge=judge,
                    client=client,
                    pair=pair,
                    template=template,
                    template_hash=template_hash,
                    template_version=template_version,
                    cache=cache,
                    decoding=cfg.decoding,
                    retry=cfg.retry,
                )
                if status != "hit" and new_api_calls_allowed is not None:
                    new_api_calls_allowed -= 1
                if status == "skip":
                    pbar.write(f"[skip] {judge.label()} {pair.row_id}:{pair.arm} — {detail}")
                    stats.skipped_rows += 1
                    per["skipped"] += 1
                    pbar.update(1)
                    continue
                assert verdict is not None
                _record_result(
                    verdict=verdict,
                    status=status,
                    judge_label=judge.label(),
                    stats=stats,
                    per=per,
                    appender=appender,
                )
                pbar.update(1)
        pbar.close()

    _write_progress_sidecar(progress_path, stats, cfg)
    return stats
