"""Stage 2 pipeline: original.csv -> paraphrase.csv + idiomatic.csv (STAGE2_CONTRACT)."""

import csv
import json
import platform
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

from augmentation.base import AugmentedRow, Augmenter, ValidationResult, Validator, Variant
from augmentation.cache import ResponseCache
from augmentation.config import AugmentConfig
from augmentation.io import (
    AugmentRowCounts,
    AugmentSidecar,
    AugmentToolVersions,
    CacheStats,
    SkippedRow,
    VariantCsvAppender,
    heal_variant_csv,
    read_skips_manifest,
    read_variant_progress,
    reconcile_variant_csvs,
    skips_manifest_path,
    write_sidecar,
    write_skips_manifest,
)
from augmentation.llm_validators import build_judge
from augmentation.prompts.loader import load_prompt
from augmentation.providers.base import LLMClient, LLMError, build_client
from augmentation.registry import get_augmenter
from augmentation.validators import SemanticSimilarityValidator
from cleaning.hashing import config_hash, prompt_hash
from data.base import DatasetRow

_VARIANT_VALIDATORS: dict[Variant, list[str]] = {
    "paraphrase": ["semantic_similarity", "label_preservation", "idiom_absence"],
    "idiomatic": ["semantic_similarity", "label_preservation", "idiom_presence"],
}


class AugmentationError(RuntimeError):
    """Raised when a row still fails after all retry attempts.

    `transient` distinguishes the two causes, which the pipeline handles very
    differently: a genuine validation failure (`transient=False` — the judges
    rejected every attempt, e.g. an original that already contains an idiom) is
    a permanent skip, whereas an API/transport failure such as a 429 rate-limit
    or 5xx (`transient=True`) is not the row's fault and must NOT be recorded as
    a skip — the run pauses so it can resume once the condition clears."""

    def __init__(
        self,
        *,
        row_id: str,
        variant: Variant,
        failing: list[str],
        attempts: int,
        transient: bool = False,
    ) -> None:
        self.row_id = row_id
        self.variant = variant
        self.failing = failing
        self.attempts = attempts
        self.transient = transient
        super().__init__(
            f"augmentation failed for id={row_id!r} variant={variant!r} after "
            f"{attempts} attempt(s){' (transient)' if transient else ''}; failing: {failing}"
        )


def _parse_meta(meta_str: str | None) -> dict[str, Any]:
    """Parse a `meta` CSV field, returning `{}` on empty/invalid JSON."""
    if not meta_str:
        return {}
    try:
        val = json.loads(meta_str)
        return val if isinstance(val, dict) else {}
    except json.JSONDecodeError:
        return {}


@dataclass(slots=True)
class _VariantBuild:
    """Running tallies for one variant. Rows are streamed straight to the output
    CSV as they are accepted (not held here), so memory stays bounded regardless
    of dataset size and an interrupted run leaves a resumable partial CSV."""

    written: int = 0
    hits: int = 0
    misses: int = 0
    skipped: int = 0
    paused: bool = False
    """Set when a transient/API error (e.g. a 429 rate-limit) stopped the variant
    mid-run. The partial CSV is kept for resume; no row is recorded as skipped."""
    passed_by_name: dict[str, int] = field(default_factory=dict[str, int])
    failed_by_name: dict[str, int] = field(default_factory=dict[str, int])
    skipped_rows: list[SkippedRow] = field(default_factory=list[SkippedRow])
    """Rows excluded from this variant's CSV (validation failures + rows dropped
    to keep variants aligned). Persisted so a resumed run does not retry them."""


class AugmentPipeline:
    """Dataset-agnostic Stage 2 augmentation orchestrator (STAGE2_CONTRACT).

    Resolves the augmenter/validators/prompts purely through `cfg` and the
    registries — no dataset- or augmenter-specific branching lives here.

    Accepted rows are streamed to the variant CSV as they are produced, so an
    interrupted run leaves a durable partial CSV that a re-run resumes from (only
    the remaining rows are augmented). A row that still fails after
    `cfg.retry.max_attempts` is **skipped** — recorded in the sidecar's
    `skipped_rows` and a durable manifest (so a resume does not retry it) — and
    the run continues. After both variants finish, their CSVs are reconciled to
    the common id set so they stay row-for-row aligned.
    """

    def __init__(
        self, cfg: AugmentConfig, input_csv: Path, out_dir: Path, config_path: Path
    ) -> None:
        self._cfg = cfg
        self._input_csv = input_csv
        self._out_dir = out_dir
        self._config_path = config_path
        self.last_counts: dict[str, dict[str, int]] = {}
        self.last_cache_stats: dict[str, dict[str, int]] = {}
        self.paused = False
        """True if the run stopped on a transient/API error (e.g. quota) before
        finishing. The partial CSVs are resumable; sidecars are not written."""

    def _read_original(self, csv_path: Path) -> list[DatasetRow]:
        """Read `id, x, y, meta` columns from a Stage 1 CSV into `DatasetRow`s."""
        with csv_path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return [
                DatasetRow(
                    id=row["id"],
                    x=row["x"],
                    y=row["y"],
                    meta=_parse_meta(row.get("meta")),
                )
                for row in reader
            ]

    def _backoff(self, attempt: int) -> None:
        """Sleep an exponentially increasing delay before retry `attempt + 1`."""
        base = self._cfg.retry.backoff_seconds
        if base > 0:
            time.sleep(base * (2 ** (attempt - 1)))

    def _build_validators(self, variant: Variant, client: LLMClient) -> list[Validator]:
        """Construct the validator chain for `variant`, sharing `client` with judges (D6)."""
        cfg = self._cfg
        validators: list[Validator] = []
        for name in _VARIANT_VALIDATORS[variant]:
            if name == "semantic_similarity":
                validators.append(SemanticSimilarityValidator())
            else:
                validators.append(
                    build_judge(
                        name,
                        client=client,
                        temperature=cfg.judge.temperature,
                        max_output_tokens=cfg.judge.max_output_tokens,
                    )
                )
        return validators

    def _augment_with_retry(
        self,
        augmenter: Augmenter,
        validators: list[Validator],
        row: DatasetRow,
        variant: Variant,
    ) -> tuple[AugmentedRow, list[ValidationResult]]:
        """Augment `row` and validate it, retrying on failure per `cfg.retry`.

        Raises `AugmentationError` when no attempt succeeds. `transient=True` iff
        every attempt failed with an `LLMError` (a transport/API failure such as
        a 429) and none actually reached — and was rejected by — the validators;
        that case is a pause-and-resume, not a skip."""
        cfg = self._cfg
        failing: list[str] = []
        saw_validation_fail = False
        for attempt in range(1, cfg.retry.max_attempts + 1):
            try:
                aug = augmenter.augment(row)
                results = [v.validate(aug, row) for v in validators]
            except LLMError as exc:
                failing = [f"llm_error: {exc}"]
                if attempt < cfg.retry.max_attempts:
                    self._backoff(attempt)
                continue
            failing = [r.name for r in results if not r.passed]
            if not failing:
                return aug, results
            saw_validation_fail = True
            if attempt < cfg.retry.max_attempts:
                self._backoff(attempt)
        raise AugmentationError(
            row_id=row.id,
            variant=variant,
            failing=failing,
            attempts=cfg.retry.max_attempts,
            transient=not saw_validation_fail,
        )

    def _resume(
        self, out_path: Path, augmenter_model: str, prompt_hash_val: str
    ) -> tuple[set[str], set[str], _VariantBuild]:
        """Load prior progress for `out_path`, returning `(done_ids, skip_ids, build)`.

        The output CSV is the resume checkpoint. A torn tail line is healed
        first; rows whose `augmenter_model`/`prompt_hash` match the current run
        are kept (`done_ids`) so only the remaining rows are re-augmented. Rows
        recorded in the skip manifest (`skip_ids`) previously failed and are not
        retried. A CSV from a different model/prompt is stale and discarded, and
        its skip manifest with it (started fresh)."""
        manifest = skips_manifest_path(out_path)
        heal_variant_csv(out_path)
        prior = read_variant_progress(out_path)
        compatible = (
            bool(prior.ids)
            and prior.augmenter_model == augmenter_model
            and prior.prompt_hash == prompt_hash_val
        )
        if not compatible:
            out_path.unlink(missing_ok=True)
            manifest.unlink(missing_ok=True)
            return set(), set(), _VariantBuild()
        done_ids = set(prior.ids)
        skipped_rows = read_skips_manifest(manifest)
        skip_ids = {r.id for r in skipped_rows}
        build = _VariantBuild(
            written=len(done_ids),
            skipped=len(done_ids),
            passed_by_name=dict(prior.passed_by_name),
            failed_by_name=dict(prior.failed_by_name),
            skipped_rows=skipped_rows,
        )
        return done_ids, skip_ids, build

    def _build_variant(
        self,
        variant: Variant,
        prompt_hash_val: str,
        template: str,
        rows: list[DatasetRow],
        cache: ResponseCache,
        client: LLMClient,
        out_path: Path,
    ) -> _VariantBuild:
        """Augment every row for one variant (cache-aware) and run its validators.

        Accepted rows are appended to `out_path` and flushed as they are
        produced. On (re)start, rows already present in `out_path` are skipped
        so only the remaining rows are augmented."""
        cfg = self._cfg
        augmenter = get_augmenter(cfg.augmenter)(
            variant=variant,
            prompt_hash=prompt_hash_val,
            client=client,
            prompt_template=template,
            temperature=cfg.decoding.temperature,
            max_output_tokens=cfg.decoding.max_output_tokens,
        )
        validators = self._build_validators(variant, client)
        augmenter_model = f"{client.provider}/{client.model}"
        manifest = skips_manifest_path(out_path)

        done_ids, skip_ids, build = self._resume(out_path, augmenter_model, prompt_hash_val)
        progress = tqdm(rows, desc=f"[augment {variant}]", unit="row")
        with VariantCsvAppender(out_path) as appender:
            for row in progress:
                if row.id in done_ids or row.id in skip_ids:
                    continue
                cached = cache.get(prompt_hash_val, augmenter_model, row.id)
                if cached is not None and isinstance(cached.get("validators"), list):
                    build.hits += 1
                    aug = AugmentedRow(
                        id=row.id,
                        variant=variant,
                        x=str(cached["x"]),
                        y=row.y,
                        augmenter_model=augmenter_model,
                        prompt_hash=prompt_hash_val,
                        meta=dict(row.meta),
                    )
                    results = [
                        ValidationResult(
                            name=str(v["name"]),
                            passed=bool(v["passed"]),
                            score=v["score"],
                            details=v["details"],
                        )
                        for v in cached["validators"]
                    ]
                else:
                    build.misses += 1
                    try:
                        aug, results = self._augment_with_retry(augmenter, validators, row, variant)
                    except AugmentationError as err:
                        if err.transient:
                            # A transport/API failure (e.g. a 429 rate-limit) —
                            # not this row's fault. Pause: keep the rows already
                            # written (flushed), record NOTHING as skipped, and
                            # stop so a re-run resumes once the condition clears.
                            build.paused = True
                            progress.write(
                                f"[augment {variant}] paused at id={err.row_id!r}: "
                                f"{err.failing} after {err.attempts} attempt(s). "
                                f"{build.written} row(s) saved — re-run to resume."
                            )
                            break
                        # Genuine validation failure (e.g. an original that
                        # already contains an idiom has no idiom-free paraphrase).
                        # Skip it, record it durably so a resume won't retry it,
                        # and keep going. Alignment is restored by reconciliation.
                        build.skipped_rows.append(
                            SkippedRow(
                                id=err.row_id,
                                reason="validation_failed",
                                failing=err.failing,
                                attempts=err.attempts,
                            )
                        )
                        skip_ids.add(err.row_id)
                        write_skips_manifest(manifest, build.skipped_rows)
                        progress.write(
                            f"[augment {variant}] skipped id={err.row_id!r}: still failing "
                            f"{err.failing} after {err.attempts} attempt(s)."
                        )
                        continue
                    cache.put(
                        prompt_hash_val,
                        augmenter_model,
                        row.id,
                        {
                            "x": aug.x,
                            "validators": [
                                {
                                    "name": r.name,
                                    "passed": r.passed,
                                    "score": r.score,
                                    "details": r.details,
                                }
                                for r in results
                            ],
                        },
                    )

                for vr in results:
                    tally = build.passed_by_name if vr.passed else build.failed_by_name
                    tally[vr.name] = tally.get(vr.name, 0) + 1

                appender.append(aug, results)
                build.written += 1
                progress.set_postfix(  # pyright: ignore[reportUnknownMemberType]
                    hits=build.hits,
                    misses=build.misses,
                    resumed=build.skipped,
                    dropped=len(build.skipped_rows),
                )
        return build

    def _record_counts(self, variant: Variant, input_rows: int, build: _VariantBuild) -> None:
        """Record a variant's per-run counts/cache stats for the CLI to report."""
        self.last_counts[variant] = {
            "input_rows": input_rows,
            "written": build.written,
            "resumed": build.skipped,
            "dropped": len(build.skipped_rows),
            **{f"passed_{name}": n for name, n in build.passed_by_name.items()},
        }
        self.last_cache_stats[variant] = {"hits": build.hits, "misses": build.misses}

    def _write_sidecar(
        self,
        variant: Variant,
        prompt_file: str,
        prompt_hash_val: str,
        input_rows: int,
        build: _VariantBuild,
        client: LLMClient,
        out_path: Path,
    ) -> None:
        """Write a variant's sidecar (its CSV is already on disk), recording every
        skipped/dropped row in `skipped_rows`."""
        cfg = self._cfg
        resolved_config: dict[str, Any] = cfg.model_dump()
        sidecar = AugmentSidecar(
            dataset=cfg.dataset,
            variant=variant,
            config_path=str(self._config_path.resolve()),
            config_hash=config_hash(resolved_config),
            resolved_config=resolved_config,
            augmenter_model=f"{client.provider}/{client.model}",
            prompt_file=prompt_file,
            prompt_hash=prompt_hash_val,
            tool_versions=AugmentToolVersions(
                python=platform.python_version(),
                pandas=pd.__version__,
            ),
            row_counts=AugmentRowCounts(
                input_rows=input_rows,
                augmented=build.written,
                validators_passed_by_name=build.passed_by_name,
                validators_failed_by_name=build.failed_by_name,
                written=build.written,
                skipped=build.skipped,
                dropped=len(build.skipped_rows),
            ),
            cache_stats=CacheStats(hits=build.hits, misses=build.misses),
            skipped_rows=sorted(build.skipped_rows, key=lambda r: r.id),
            timestamp_utc=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        )
        write_sidecar(out_path.parent / f"{variant}.meta.json", sidecar)

    def _reconcile(
        self, out_paths: dict[Variant, Path], builds: dict[Variant, _VariantBuild]
    ) -> None:
        """Drop rows to the id-set common to all variants so they stay aligned.

        A row skipped in one variant (e.g. a paraphrase that can't shed its
        source idiom) must not survive in the others. Rows removed here are
        recorded as `"unaligned"` skips (and persisted) so a resume won't
        re-add them, and each variant's `written` count is corrected."""
        _surviving, removed_by_path = reconcile_variant_csvs(list(out_paths.values()))
        for variant, build in builds.items():
            removed = removed_by_path.get(out_paths[variant], set())
            known = {r.id for r in build.skipped_rows}
            new_unaligned = sorted(removed - known)
            if not new_unaligned:
                continue
            build.skipped_rows.extend(
                SkippedRow(id=rid, reason="unaligned") for rid in new_unaligned
            )
            build.written -= len(new_unaligned)
            write_skips_manifest(skips_manifest_path(out_paths[variant]), build.skipped_rows)

    def run(self) -> tuple[Path, Path]:
        """Run both variants, returning `(paraphrase_csv, idiomatic_csv)`.

        Rows are streamed to each CSV as accepted. A row that still fails
        validation after all retries is skipped (recorded in the sidecar's
        `skipped_rows` + a durable manifest) and the run continues; a transient
        API error (e.g. a 429 rate-limit) instead **pauses** the run — partial
        CSVs are kept for resume, no row is recorded as skipped, and no sidecar
        is written (`self.paused` is set). Once both variants complete their CSVs
        are reconciled to the common id set so they stay row-for-row aligned,
        then sidecars are written."""
        cfg = self._cfg
        rows = self._read_original(self._input_csv)
        cache = ResponseCache(Path(cfg.cache.dir), enabled=cfg.cache.enabled)
        client = build_client(cfg.augmenter, cfg.augmenter_model)

        prompt_files: dict[Variant, str] = {
            "paraphrase": cfg.prompts.paraphrase,
            "idiomatic": cfg.prompts.idiomatic,
        }
        out_paths: dict[Variant, Path] = {
            variant: self._out_dir / cfg.dataset / f"{variant}.csv"
            for variant in ("paraphrase", "idiomatic")
        }
        prompt_hashes: dict[Variant, tuple[str, str]] = {}

        builds: dict[Variant, _VariantBuild] = {}
        for variant in ("paraphrase", "idiomatic"):
            prompt_file = prompt_files[variant]
            template = load_prompt(prompt_file)
            ph = prompt_hash(template)
            prompt_hashes[variant] = (prompt_file, ph)
            build = self._build_variant(
                variant, ph, template, rows, cache, client, out_paths[variant]
            )
            builds[variant] = build
            self._record_counts(variant, len(rows), build)
            if build.paused:
                # Transient/API failure: stop before reconciling or writing
                # sidecars, leaving resumable partial CSVs.
                self.paused = True
                return out_paths["paraphrase"], out_paths["idiomatic"]

        self._reconcile(out_paths, builds)

        for variant, build in builds.items():
            prompt_file, ph = prompt_hashes[variant]
            self._record_counts(variant, len(rows), build)
            self._write_sidecar(
                variant, prompt_file, ph, len(rows), build, client, out_paths[variant]
            )

        return out_paths["paraphrase"], out_paths["idiomatic"]
