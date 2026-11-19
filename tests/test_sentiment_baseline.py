"""SST-2-replacement baseline: column discovery, binary labels, balanced sampling (offline)."""

import pytest

from data.sentiment_baseline import binary_label, normalize, sample_balanced


def test_binary_labels_drop_neutral_and_mixed() -> None:
    assert (
        binary_label("positive") == "1"
        and binary_label(0) == "0"
        and binary_label("Negative") == "0"
    )
    assert binary_label("neutral") is None and binary_label("mixed") is None


def test_normalize_finds_columns_and_keeps_source() -> None:
    rows = [{"sentence": "great", "gold_label": "positive"}, {"sentence": "bad", "gold_label": "negative"},
            {"sentence": "ok", "gold_label": "neutral"}]  # fmt: skip
    out = normalize(rows, "dyn")
    assert [r["label"] for r in out] == ["1", "0"]
    b = normalize([{"text": "x", "sentiment": 1, "source": "Google"}], "bes")
    assert b[0]["source"] == "Google"
    with pytest.raises(ValueError, match="no text column"):
        normalize([{"foo": "x", "label": 1}], "bad")


def test_balanced_sample() -> None:
    rows = normalize(
        [{"text": f"t{i}", "label": "positive" if i % 4 == 0 else "negative"} for i in range(100)],
        "x",
    )
    s = sample_balanced(rows, 20, 1)
    assert sum(r["label"] == "1" for r in s) == sum(r["label"] == "0" for r in s) == 10
