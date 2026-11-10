"""Window mode (prompt v7): a short window may be rewritten; the literal is built by code."""

import json
from pathlib import Path

from augmentation.idiom_generation import build_variants, load_inventory, validate
from data.norms.idiom_text import IdiomIndex
from tests._generation_helpers import write_inventory

X = "After the long war the two sides stopped fighting at last."


def _index(tmp_path: Path) -> IdiomIndex:
    final, _ = write_inventory(tmp_path)
    _, rows = load_inventory(final)
    return IdiomIndex(rows, min_tokens=2, min_content=1)


def _reply(**over: str) -> str:
    base = {
        "window": "the two sides stopped fighting",
        "choice": "1",
        "idiomatic_window": "the two sides buried the hatchet",
        "idiom_phrase": "buried the hatchet",
        "literal_phrase": "made peace",
        "verdict": "KEEP",
    }
    return json.dumps({**base, **over})


def test_window_reply_builds_literal_by_swapping_only_the_idiom(tmp_path: Path) -> None:
    p = validate(_reply(), X, {"1", "2", "3"}, _index(tmp_path))
    assert p.ok, p.reason
    assert p.literal_text == "the two sides made peace"
    idiomatic, literal, _ = build_variants(X, p)
    assert idiomatic == "After the long war the two sides buried the hatchet at last."
    assert literal == "After the long war the two sides made peace at last."


def test_window_failures(tmp_path: Path) -> None:
    index = _index(tmp_path)
    ids = {"1", "2", "3"}

    def reason(**over: str) -> str:
        return validate(_reply(**over), X, ids, index).reason

    assert reason(idiom_phrase="spill the beans") == "idiom_phrase_not_found"
    assert reason(literal_phrase="") == "idiom_phrase_not_found"
    assert reason(idiom_phrase="the hatchet") == "idiom_not_present"  # swap must cover the idiom
    assert reason(literal_phrase="buried the hatchet") == "idiom_phrase_not_found"
    rewritten = "rivals buried the hatchet"
    assert reason(idiomatic_window=rewritten) == "window_rewritten_too_much"
    assert reason(window="sides stopped fighting at last and more words here") == "span_not_unique"


def test_idiom_replacing_most_of_a_short_window_is_not_a_rewrite(tmp_path: Path) -> None:
    x = "Meanwhile, he is being looked into by the board."
    reply = _reply(
        window="is being looked into",
        idiomatic_window="is under the microscope",
        idiom_phrase="under the microscope",
        literal_phrase="under investigation",
        choice="9",
    )
    final, _ = write_inventory(tmp_path)
    _, rows = load_inventory(final)
    rows.append(
        {
            "norm_idiom": "under the microscope",
            "midas_ids": "9",
            "definitions": json.dumps(["Under close scrutiny."]),
            "midas_forms": json.dumps(["under the microscope"]),
        }
    )
    p = validate(reply, x, {"9"}, IdiomIndex(rows, min_tokens=2, min_content=1))
    assert p.ok, p.reason


def test_article_mismatch_is_rejected(tmp_path: Path) -> None:
    from augmentation.idiom_generation import article_mismatch  # noqa: PLC0415

    x = "the city is a living, breathing creature."
    assert article_mismatch(x, "living, breathing", "alive and kicking", "thriving")
    assert not article_mismatch(x, "living, breathing", "thriving and lively", "busy")
    y = "it was an honest mistake"
    assert not article_mismatch(y, "honest mistake", "honest slip", "honest error")
    assert not article_mismatch("they ran a race", "ran", "made tracks", "hurried")
