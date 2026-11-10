import json

from data.norms.idiom_text import IdiomIndex, compile_form, gloss_for, tokenize


def _row(idiom: str, *forms: str, definition: str = "meaning") -> dict[str, str]:
    return {
        "norm_idiom": idiom,
        "midas_ids": "1",
        "definitions": json.dumps([definition]),
        "midas_forms": json.dumps(list(forms)),
    }


def test_tokenize_reattaches_detached_possessive() -> None:
    assert tokenize("Britain 's water") == ["britain's", "water"]


def test_compile_form_skips_short_or_function_only_forms() -> None:
    assert compile_form("all ears") is None
    assert compile_form("out of the") is None
    assert compile_form("bury the hatchet") is not None


def test_find_handles_inflection_and_possessive_slots() -> None:
    index = IdiomIndex(
        [
            _row("Bury the hatchet", "bury the hatchet"),
            _row("Take one's breath away", "take someone's breath away"),
        ]
    )
    assert [h.ref.idiom for h in index.find("They finally buried the hatchet.")] == [
        "Bury the hatchet"
    ]
    assert [h.ref.idiom for h in index.find("The view took her breath away")] == [
        "Take one's breath away"
    ]
    assert index.find("She buried the treasure") == []


def test_find_matches_slot_words_across_up_to_three_tokens() -> None:
    index = IdiomIndex([_row("Cook someone's goose", "cook someone's goose")])
    assert len(index.find("He cooked the old man's goose")) == 1


def test_gloss_for_returns_replacement_tokens() -> None:
    index = IdiomIndex([_row("Bury the hatchet", "bury the hatchet")])
    sentence = "They decided to bury the hatchet after years."
    (hit,) = index.find(sentence)
    assert gloss_for(sentence, "They decided to make peace after years.", hit) == "make peace"
