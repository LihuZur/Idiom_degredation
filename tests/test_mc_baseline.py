"""MMLU-replacement baseline: field detection, label base, balanced sampling, prompts and parser (offline)."""

import csv
import json
from pathlib import Path

import pytest

from data.mc_baseline import normalize, sample_balanced, write_dataset
from eval.mc_context import build_user, parse_letter


def _siqa(n: int) -> list[dict[str, str]]:
    return [
        {"context": f"ctx {i}", "question": "How would X feel?", "answerA": "a", "answerB": "b", "answerC": "c",
         "label": str(1 + i % 3)}
        for i in range(n)
    ]  # fmt: skip


def test_normalize_detects_one_based_labels_and_answer_columns() -> None:
    rows = normalize(_siqa(6), "siqa")
    assert rows[0]["choices"] == ["a", "b", "c"]
    assert {r["label"] for r in rows} == {"0", "1", "2"}


def test_normalize_zero_based_and_failures() -> None:
    cos = [{"id": "q", "context": "c", "question": "q", **{f"answer{k}": str(k) for k in range(4)}, "label": k}
           for k in range(4)]  # fmt: skip
    assert [r["label"] for r in normalize(cos, "cosmosqa")] == ["0", "1", "2", "3"]
    with pytest.raises(ValueError, match="missing columns"):
        normalize([{"text": "x"}], "bad")
    with pytest.raises(ValueError, match="labels"):
        normalize([{**_siqa(1)[0], "label": "7"}], "bad")


def test_balanced_sample_and_written_csv(tmp_path: Path) -> None:
    rows = normalize(_siqa(90), "siqa")
    s = sample_balanced(rows, 30, 1)
    assert len(s) == 30 and sorted({r["label"] for r in s}) == ["0", "1", "2"]
    assert sample_balanced(rows, 30, 1) == s
    p = write_dataset(s, tmp_path / "noctx", with_context=False)
    row = next(csv.DictReader(p.open()))
    assert row["x"] == "" and json.loads(row["meta"])["choices"] == ["a", "b", "c"]


def test_prompt_and_parser() -> None:
    assert build_user("", "Q?", ["x", "y", "z"], False).startswith("Question: Q?")
    assert "Text: story" in build_user("story", "Q?", ["x", "y", "z"], True)
    assert parse_letter("B", 3) == "1" and parse_letter("(c)", 3) == "2"
    assert parse_letter("D", 3) is None  # only A-C allowed with 3 options
    assert parse_letter("The answer is A because a person", 3) == "0"
