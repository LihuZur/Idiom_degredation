"""Explicit id-alignment check between original/paraphrase/idiomatic (STAGE2_CONTRACT)."""

from pathlib import Path

import pandas as pd
import pytest

from augmentation.pipeline import AugmentPipeline
from cleaning.io import write_original_csv
from data.base import DatasetRow
from tests._augment_helpers import FakeClient, make_cfg


def test_original_paraphrase_idiomatic_share_same_ordered_id_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [
        DatasetRow(id="a1", x="first example", y=0, meta={}),
        DatasetRow(id="a2", x="second example", y=1, meta={}),
        DatasetRow(id="a3", x="third example", y=0, meta={}),
    ]
    original_csv = tmp_path / "original.csv"
    write_original_csv(original_csv, rows)

    fake = FakeClient(augment_result="a rewritten sentence", judge_result="PASS")

    def fake_build_client(provider: str, model: str) -> FakeClient:
        return fake

    monkeypatch.setattr("augmentation.pipeline.build_client", fake_build_client)

    cfg = make_cfg(cache_dir=tmp_path / "cache")
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
