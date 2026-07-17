from pathlib import Path

import pytest
from click import ClickException

from scripts.eval import resolve_variants


def test_resolve_variants_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = "sst2"
    base_dir = tmp_path / "datasets_out" / dataset
    base_dir.mkdir(parents=True, exist_ok=True)

    # 1. Only original.csv exists
    (base_dir / "original.csv").touch()

    resolved = resolve_variants(dataset, None, base_dir=base_dir.parent)
    assert resolved == ["original"]

    # 2. Both original.csv and paraphrase.csv exist
    (base_dir / "paraphrase.csv").touch()
    resolved = resolve_variants(dataset, None, base_dir=base_dir.parent)
    assert resolved == ["original", "paraphrase"]


def test_resolve_variants_explicit(tmp_path: Path) -> None:
    dataset = "sst2"
    base_dir = tmp_path / "datasets_out" / dataset
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "original.csv").touch()

    # Requesting existing variant should succeed
    resolved = resolve_variants(dataset, "original", base_dir=base_dir.parent)
    assert resolved == ["original"]

    # Requesting missing variant should raise click exception
    with pytest.raises(ClickException):
        resolve_variants(dataset, "original,paraphrase", base_dir=base_dir.parent)
