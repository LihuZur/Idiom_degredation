"""Stage 1 pipeline: raw Examples -> datasets_out/{ds}/original.csv (STAGE1_PLAN §3.2)."""

import platform
import random
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import datasets as hf_datasets
import pandas as pd

from cleaning import io, normalize
from cleaning.config import CleanConfig
from cleaning.hashing import canonical_json, config_hash
from cleaning.io import CleanSidecar, RowCounts, ToolVersions
from data.base import DatasetRow
from data.registry import get_dataset


class Pipeline:
    """Dataset-agnostic Stage 1 cleaner orchestrator (STAGE1_PLAN §3.2).

    Orchestrates load -> normalize -> length-filter -> dedupe -> shuffle ->
    cap -> write. No `if dataset == ...` branching lives here; dataset-
    specific behavior (id scheme, `meta` composition) lives in the loader.
    """

    def __init__(self, cfg: CleanConfig, out_dir: Path, config_path: Path) -> None:
        self.dataset = cfg.dataset
        self.last_counts: dict[str, int] = {}
        self._cfg = cfg
        self._out_dir = out_dir
        self._config_path = config_path

    def run(self) -> Path:
        cfg = self._cfg
        loader_cls = get_dataset(cfg.dataset)
        loader = loader_cls(hf_revision=cfg.hf_revision, normalize=cfg.normalize)

        raw = list(loader.load())
        counts: dict[str, int] = {"raw_loaded": len(raw)}

        normalized = [replace(ex, x=normalize.apply(cfg.normalize, ex.x)) for ex in raw]
        counts["after_normalize"] = len(normalized)

        length_filtered = [
            ex
            for ex in normalized
            if cfg.length.min_tokens <= len(ex.x.split()) <= cfg.length.max_tokens
        ]
        counts["after_length_filter"] = len(length_filtered)

        if cfg.dedupe:
            deduped: list[DatasetRow] = []
            seen: set[tuple[str, str, str]] = set()
            for ex in length_filtered:
                key = (ex.x, canonical_json(ex.meta), str(ex.y))
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(ex)
        else:
            deduped = list(length_filtered)
        counts["after_dedupe"] = len(deduped)

        survivors = list(deduped)
        random.Random(cfg.seed).shuffle(survivors)
        if cfg.max_rows is not None:
            survivors = survivors[: cfg.max_rows]
        counts["after_shuffle_and_cap"] = len(survivors)
        counts["written"] = len(survivors)

        out_path = self._out_dir / cfg.dataset / "original.csv"
        io.write_original_csv(out_path, survivors)

        resolved_config: dict[str, Any] = cfg.model_dump()
        sidecar = CleanSidecar(
            stage="clean",
            dataset=cfg.dataset,
            config_path=str(self._config_path.resolve()),
            config_hash=config_hash(resolved_config),
            resolved_config=resolved_config,
            hf_dataset_id=loader.hf_dataset_id,
            hf_revision=cfg.hf_revision,
            tool_versions=ToolVersions(
                python=platform.python_version(),
                datasets=hf_datasets.__version__,
                pandas=pd.__version__,
            ),
            row_counts=RowCounts(**counts),
            timestamp_utc=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        )
        io.write_sidecar(out_path.parent / "original.meta.json", sidecar)
        self.last_counts = counts
        return out_path
