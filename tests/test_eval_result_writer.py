import json
from pathlib import Path

import pytest

from eval.io import load_result, write_result


def test_result_roundtrip(tmp_path: Path) -> None:
    file_path = tmp_path / "result.json"
    dummy_data = {
        "stage": "eval",
        "dataset": "sst2",
        "model_id": "qwen3.5-1.5b-instruct",
        "metrics": {"original": {"accuracy": 0.85}},
    }

    write_result(file_path, dummy_data)
    assert file_path.exists()

    loaded = load_result(file_path)
    assert loaded == dummy_data


def test_invalid_json_loaded(tmp_path: Path) -> None:
    file_path = tmp_path / "invalid.json"
    file_path.write_text("invalid json")

    with pytest.raises(json.JSONDecodeError):
        load_result(file_path)
