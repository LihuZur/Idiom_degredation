"""Stage 2 pipeline: original.csv -> paraphrase.csv + idiomatic.csv (STAGE2_CONTRACT)."""

import csv
import json
import platform
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from augmentation.base import AugmentedRow, ValidationResult, Variant
from augmentation.cache import ResponseCache
from augmentation.config import AugmentConfig
from augmentation.io import (
    AugmentRowCounts,
    AugmentSidecar,
    AugmentToolVersions,
    CacheStats,
    write_sidecar,
    write_variant_csv,
)
from augmentation.prompts.loader import load_prompt
from augmentation.registry import get_augmenter, get_validator
from cleaning.hashing import config_hash, prompt_hash
from data.base import DatasetRow

_VARIANT_VALIDATORS: dict[Variant, list[str]] = {
    "paraphrase": ["semantic_similarity", "label_preservation", "idiom_absence"],
    "idiomatic": ["semantic_similarity", "label_preservation", "idiom_presence"],
}


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
    """Accumulated output for one variant: rows plus cache/validator tallies."""

    output_rows: list[tuple[AugmentedRow, list[ValidationResult]]] = field(default_factory=list)
    hits: int = 0
    misses: int = 0
    passed_by_name: dict[str, int] = field(default_factory=dict[str, int])
    failed_by_name: dict[str, int] = field(default_factory=dict[str, int])


class AugmentPipeline:
    """Dataset-agnostic Stage 2 augmentation orchestrator (STAGE2_CONTRACT).

    Resolves the augmenter/validators/prompts purely through `cfg` and the
    registries — no dataset- or augmenter-specific branching lives here.
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

    def _build_variant(
        self,
        variant: Variant,
        prompt_hash_val: str,
        rows: list[DatasetRow],
        cache: ResponseCache,
    ) -> _VariantBuild:
        """Augment every row for one variant (cache-aware) and run its validators."""
        cfg = self._cfg
        augmenter = get_augmenter(cfg.augmenter)(variant=variant, prompt_hash=prompt_hash_val)
        validators = [get_validator(name)() for name in _VARIANT_VALIDATORS[variant]]

        build = _VariantBuild()
        for row in rows:
            cached = cache.get(prompt_hash_val, cfg.augmenter, row.id)
            if cached is not None:
                build.hits += 1
                aug = AugmentedRow(
                    id=row.id,
                    variant=variant,
                    x=str(cached["x"]),
                    y=row.y,
                    augmenter_model=cfg.augmenter,
                    prompt_hash=prompt_hash_val,
                    meta=dict(row.meta),
                )
            else:
                build.misses += 1
                aug = augmenter.augment(row)
                cache.put(prompt_hash_val, cfg.augmenter, row.id, {"x": aug.x})

            results = [validator.validate(aug) for validator in validators]
            for vr in results:
                tally = build.passed_by_name if vr.passed else build.failed_by_name
                tally[vr.name] = tally.get(vr.name, 0) + 1

            build.output_rows.append((aug, results))
        return build

    def _write_variant(
        self,
        variant: Variant,
        prompt_file: str,
        prompt_hash_val: str,
        input_rows: int,
        build: _VariantBuild,
    ) -> Path:
        """Write one variant's CSV + sidecar and record its counts/cache stats."""
        cfg = self._cfg
        out_path = self._out_dir / cfg.dataset / f"{variant}.csv"
        write_variant_csv(out_path, build.output_rows)

        resolved_config: dict[str, Any] = cfg.model_dump()
        written = len(build.output_rows)
        sidecar = AugmentSidecar(
            dataset=cfg.dataset,
            variant=variant,
            config_path=str(self._config_path.resolve()),
            config_hash=config_hash(resolved_config),
            resolved_config=resolved_config,
            augmenter_model=cfg.augmenter,
            prompt_file=prompt_file,
            prompt_hash=prompt_hash_val,
            tool_versions=AugmentToolVersions(
                python=platform.python_version(),
                pandas=pd.__version__,
            ),
            row_counts=AugmentRowCounts(
                input_rows=input_rows,
                augmented=input_rows,
                validators_passed_by_name=build.passed_by_name,
                validators_failed_by_name=build.failed_by_name,
                written=written,
            ),
            cache_stats=CacheStats(hits=build.hits, misses=build.misses),
            timestamp_utc=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        )
        write_sidecar(out_path.parent / f"{variant}.meta.json", sidecar)

        self.last_counts[variant] = {
            "input_rows": input_rows,
            "augmented": input_rows,
            "written": written,
            **{f"passed_{name}": n for name, n in build.passed_by_name.items()},
            **{f"failed_{name}": n for name, n in build.failed_by_name.items()},
        }
        self.last_cache_stats[variant] = {"hits": build.hits, "misses": build.misses}
        return out_path

    def run(self) -> tuple[Path, Path]:
        """Run both variants end-to-end, returning `(paraphrase_csv, idiomatic_csv)`."""
        cfg = self._cfg
        rows = self._read_original(self._input_csv)
        cache = ResponseCache(Path(cfg.cache.dir), enabled=cfg.cache.enabled)

        prompt_files: dict[Variant, str] = {
            "paraphrase": cfg.prompts.paraphrase,
            "idiomatic": cfg.prompts.idiomatic,
        }

        out_paths: dict[Variant, Path] = {}
        for variant in ("paraphrase", "idiomatic"):
            prompt_file = prompt_files[variant]
            ph = prompt_hash(load_prompt(prompt_file))
            build = self._build_variant(variant, ph, rows, cache)
            out_paths[variant] = self._write_variant(variant, prompt_file, ph, len(rows), build)

        return out_paths["paraphrase"], out_paths["idiomatic"]
