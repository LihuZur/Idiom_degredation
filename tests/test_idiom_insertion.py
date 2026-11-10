"""v3 insertion generator: prompt rendering, the code checks, the mixed shortlist and the retry."""

import json
import random
import re
from collections.abc import Sequence
from pathlib import Path

from augmentation.idiom_generation import Item
from augmentation.idiom_insertion import (
    DATASET_TEXT,
    DATASET_TEXT_V2,
    EXAMPLE_IDS,
    TEXTS_BY_TEMPLATE,
    article_errors,
    generate_insert,
    insert_context,
    insert_label,
    mixed_shortlist,
    new_article_error,
    render_insert_prompt,
    validate_insert,
    words_kept,
)
from tests._generation_helpers import X, setup_inventory

IDIOMATIC = "They buried the hatchet after the long war."
LITERAL = "They made peace after the long war."


def _reply(**over: str) -> str:
    base = {
        "choice": "1",
        "idiomatic": IDIOMATIC,
        "idiom_text": "buried the hatchet",
        "literal_text": "made peace",
        "literal": LITERAL,
        "check": "same label; same meaning; natural",
        "verdict": "KEEP",
    }
    return json.dumps(base | over)


def test_valid_reply_passes(tmp_path: Path) -> None:
    _, index = setup_inventory(tmp_path)
    p = validate_insert(_reply(), X, {"1", "2"}, index)
    assert p.ok
    assert p.literal == LITERAL


def test_literal_must_be_idiomatic_with_only_the_idiom_replaced(tmp_path: Path) -> None:
    _, index = setup_inventory(tmp_path)
    reply = _reply(literal="They made peace after the war.")
    assert validate_insert(reply, X, {"1"}, index).reason == "literal_mismatch"


def test_rejections(tmp_path: Path) -> None:
    _, index = setup_inventory(tmp_path)
    cases = {
        "none": _reply(choice="NONE"),
        "self_rejected": _reply(verdict="NONE"),
        "unknown_choice": _reply(choice="3"),
        "no_change": _reply(idiomatic=X, idiom_text="stopped fighting", literal=X),
        "idiom_not_present": _reply(
            idiomatic="They patched things up after the long war.",
            idiom_text="patched things up",
            literal=LITERAL,
        ),
        "idiom_in_literal": _reply(
            literal_text="spilled the beans",
            literal="They spilled the beans after the long war.",
        ),
        "bad_json": "not json",
    }
    for reason, raw in cases.items():
        assert validate_insert(raw, X, {"1", "2"}, index).reason == reason, reason


def test_words_kept() -> None:
    assert words_kept(X, X) == 1.0
    assert 0.5 < words_kept(X, IDIOMATIC) < 1.0


def test_mixed_shortlist_without_retriever_is_random_and_complete(tmp_path: Path) -> None:
    inv, _ = setup_inventory(tmp_path)
    item = Item("sst2", "a", X, "1", {})
    short = mixed_shortlist(item, inv, None, random.Random(0), n_fit=2, n_random=1)
    assert sorted(i.idiom_id for i in short) == ["1", "2", "3"]


def test_prompt_has_no_unfilled_slot_and_shows_fixed_context(tmp_path: Path) -> None:
    inv, _ = setup_inventory(tmp_path)
    mnli = Item("mnli", "m", X, "1", {"hypothesis": "There was a war.", "label_name": "neutral"})
    prompt = render_insert_prompt(mnli, inv)
    assert not re.search(r"\{[a-z_]+\}", prompt)
    assert 'Hypothesis (fixed, never changed): "There was a war."' in prompt
    assert "Text (label: neutral):" in prompt
    assert DATASET_TEXT["mnli"].example.splitlines()[0] in prompt
    sst2 = Item("sst2", "s", X, "0", {})
    assert insert_context(sst2) == ""
    assert "Hypothesis" not in render_insert_prompt(sst2, inv).split("Idioms (id")[1]


def test_mmlu_label_shows_letter_and_choice() -> None:
    item = Item("mmlu", "q", "Q?", "2", {"choices": ["a", "b", "c", "d"], "answer_index": 2})
    assert insert_label(item) == "C) c"
    assert insert_context(item).startswith(
        "Answer choices (fixed, never changed): A) a B) b C) c D) d"
    )


def test_example_sentences_are_excluded_ids() -> None:
    assert {t.example_id for t in DATASET_TEXT.values()} == EXAMPLE_IDS


def test_second_try_gets_a_fresh_shortlist_and_skips_idiomatic_originals(tmp_path: Path) -> None:
    inv, index = setup_inventory(tmp_path)
    items = [
        Item("sst2", "a", X, "1", {}),
        Item("sst2", "b", "They spilled the beans at dinner.", "1", {}),
    ]
    seen: list[list[str]] = []

    def complete(prompts: Sequence[str]) -> list[str | None]:
        seen.append(list(prompts))
        return [_reply(choice="NONE") if len(seen) == 1 else _reply() for _ in prompts]

    out = generate_insert(items, inv, index, complete, None, model="m", p_hash="h", cap=5, seed=0)
    assert [o.reason for o in out] == ["ok", "original_has_idiom"]
    assert out[0].attempts == 2
    assert out[0].history == ("none",)
    assert len(seen) == 2
    assert out[0].record is not None
    assert out[0].record["original_words_kept"] < 1.0


def test_new_article_error_is_caught_anywhere_in_the_rewrite(tmp_path: Path) -> None:
    assert article_errors("making a entertaining film") == 1
    assert article_errors("an hour and a unicorn") == 0
    assert new_article_error("a fine film", "making a entertaining film")
    assert not new_article_error("a entertaining film", "a entertaining film, warts and all")
    _, index = setup_inventory(tmp_path)
    reply = _reply(
        idiomatic="They buried the hatchet after a long war, a end at last.",
        literal="They made peace after a long war, a end at last.",
    )
    assert validate_insert(reply, X, {"1"}, index).reason == "article_mismatch"


def test_v2_rules_differ_from_v1_only_for_sst2_and_mmlu() -> None:
    changed = {d for d in DATASET_TEXT if DATASET_TEXT[d] != DATASET_TEXT_V2[d]}
    assert changed == {"sst2", "mmlu"}
    assert TEXTS_BY_TEMPLATE["idiom_insert_v1.txt"] is DATASET_TEXT
    assert "numbers, names and technical terms" in DATASET_TEXT_V2["mmlu"].rule


def test_per_item_seed_gives_the_same_shortlist_in_any_sample(tmp_path: Path) -> None:
    from augmentation.idiom_insertion import item_rng  # noqa: PLC0415

    inv, index = setup_inventory(tmp_path)
    a = Item("sst2", "a", X, "1", {})
    b = Item("sst2", "b", "The film was long.", "1", {})
    prompts: dict[str, list[str]] = {"alone": [], "pair": []}

    def recorder(key: str):
        def complete(ps: Sequence[str]) -> list[str | None]:
            prompts[key].extend(ps)
            return [_reply() for _ in ps]

        return complete

    generate_insert([a], inv, index, recorder("alone"), None, model="m", p_hash="h", cap=5, seed=3)
    generate_insert(
        [b, a], inv, index, recorder("pair"), None, model="m", p_hash="h", cap=5, seed=3
    )
    assert prompts["alone"][0] == prompts["pair"][1]
    assert item_rng(3, "a", 1).random() == item_rng(3, "a", 1).random()
    assert item_rng(3, "a", 1).random() != item_rng(3, "a", 2).random()
