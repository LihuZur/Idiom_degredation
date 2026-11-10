import hashlib
from typing import Any

import pytest

import data.agnews as agnews_module
from augmentation.base import AugmentedRow
from cleaning.normalize import html_artifacts
from data.agnews import LABEL_NAMES, AgNewsLoader
from data.registry import get_dataset
from eval.agnews import AgNewsEvaluator
from eval.config import EvalConfig
from eval.registry import get_evaluator


def _ex(x: str = "Some news.") -> AugmentedRow:
    return AugmentedRow(id="a", variant="original", x=x, y="0", augmenter_model="", prompt_hash="")


def test_html_artifacts_repairs_ag_news_damage() -> None:
    assert html_artifacts("in the country #39;s offices") == "in the country's offices"
    assert html_artifacts("dusty\\Saharan town") == "dusty Saharan town"
    assert html_artifacts("fall to #36;5 million") == "fall to $5 million"
    assert html_artifacts("a &lt;B&gt;deal&lt;/B&gt; it #146;s fine") == "a deal it's fine"


@pytest.mark.parametrize(
    ("raw", "code"),
    [
        ("World", "0"),
        ("Sports.", "1"),
        ("Business", "2"),
        ("Sci/Tech", "3"),
        ("Technology", "3"),
        ("Science/Technology", "3"),
        ("business world", "2"),
        ("The answer is Sports", "1"),
    ],
)
def test_parse_maps_labels(raw: str, code: str) -> None:
    ev = AgNewsEvaluator(EvalConfig(model="m"))
    assert ev.parse(raw, _ex()) == (code, "ok")


def test_parse_unparseable() -> None:
    ev = AgNewsEvaluator(EvalConfig(model="m"))
    assert ev.parse("I am not sure", _ex()) == (None, "unparseable")


def test_format_puts_snippet_in_user_message() -> None:
    fi = AgNewsEvaluator(EvalConfig(model="m")).format(_ex("Stocks rally."))
    assert "Stocks rally." in fi.meta["messages"][1]["content"]
    assert "Sci/Tech" in fi.meta["messages"][0]["content"]


def test_registered_under_agnews() -> None:
    assert get_dataset("agnews") is AgNewsLoader
    assert get_evaluator("agnews") is AgNewsEvaluator


def test_loader_ids_labels_and_meta(monkeypatch: pytest.MonkeyPatch) -> None:
    rows: list[dict[str, Any]] = [
        {"text": "It #39;s news", "label": 2},
        {"text": "Goal scored", "label": 1},
    ]
    seen: dict[str, Any] = {}

    def fake_load(id_: str, **kwargs: Any) -> list[dict[str, Any]]:
        seen.update(id=id_, **kwargs)
        return rows

    monkeypatch.setattr(agnews_module, "load_dataset", fake_load)
    out = list(AgNewsLoader(hf_revision="rev", normalize=["html_artifacts"]).load())
    assert seen == {"id": "fancyzhx/ag_news", "revision": "rev", "split": "test"}
    assert [r.y for r in out] == [2, 1]
    assert [r.meta["label_name"] for r in out] == [LABEL_NAMES[2], LABEL_NAMES[1]]
    assert out[0].id == hashlib.sha256(b"It's news").hexdigest()[:16]
    assert out[0].x == "It #39;s news"  # text is cleaned by Stage 1, ids use the cleaned form


def test_wide_loader_keeps_test_rows_and_drops_conflicting_train_duplicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    splits: dict[str, list[dict[str, Any]]] = {
        "test": [{"text": "Goal scored", "label": 1}],
        "train": [
            {"text": "Goal scored", "label": 1},  # repeats a test row
            {"text": "Chip maker grows", "label": 2},  # also labelled 3 below: ambiguous
            {"text": "Peace talks resume", "label": 0},
            {"text": "Chip maker grows", "label": 3},
            {"text": "Peace talks resume", "label": 0},  # exact repeat
            {"text": "Shares fall", "label": 2},
        ],
    }

    def fake_load(id_: str, **kwargs: Any) -> list[dict[str, Any]]:
        return splits[kwargs["split"]]

    monkeypatch.setattr(agnews_module, "load_dataset", fake_load)
    out = list(agnews_module.AgNewsWideLoader(hf_revision="rev", normalize=[]).load())
    assert [r.x for r in out] == ["Goal scored", "Peace talks resume", "Shares fall"]
    assert out[0].meta == {"label_name": LABEL_NAMES[1]}  # test row exactly as in `agnews`
    assert out[1].meta == {"label_name": LABEL_NAMES[0], "split": "train"}
