"""Explicit id-alignment check between original/paraphrase/idiomatic (STAGE2_CONTRACT)."""

from pathlib import Path

import pandas as pd

from augmentation.config import AugmentConfig, CacheCfg, PromptsCfg
from augmentation.pipeline import AugmentPipeline
from cleaning.io import write_original_csv
from data.base import DatasetRow


def test_original_paraphrase_idiomatic_share_same_ordered_id_list(tmp_path: Path) -> None:
    rows = [
        DatasetRow(id="a1", x="first example", y=0, meta={}),
        DatasetRow(id="a2", x="second example", y=1, meta={}),
        DatasetRow(id="a3", x="third example", y=0, meta={}),
    ]
    original_csv = tmp_path / "original.csv"
    write_original_csv(original_csv, rows)

    cfg = AugmentConfig(
        dataset="sst2",
        seed=0,
        augmenter="identity",
        prompts=PromptsCfg(paraphrase="paraphrase_v1.txt", idiomatic="idiomatic_v1.txt"),
        cache=CacheCfg(enabled=True, dir=str(tmp_path / "cache")),
    )
    out_dir = tmp_path / "datasets_out"
    paraphrase_csv, idiomatic_csv = AugmentPipeline(
        cfg, original_csv, out_dir, tmp_path / "config.yaml"
    ).run()

    original_ids = pd.read_csv(original_csv, keep_default_na=False, dtype={"id": str})[
        "id"
    ].tolist()
    paraphrase_ids = pd.read_csv(paraphrase_csv, keep_default_na=False, dtype={"id": str})[
        "id"
    ].tolist()
    idiomatic_ids = pd.read_csv(idiomatic_csv, keep_default_na=False, dtype={"id": str})[
        "id"
    ].tolist()

    assert original_ids == ["a1", "a2", "a3"]
    assert paraphrase_ids == original_ids
    assert idiomatic_ids == original_ids
