"""POC-2 pieces (plan v3 §1b, step 8d): the blind generator prompt (O37) and the label-balance report (O39)."""

from augmentation.idiom_generation import Idiom, Item
from augmentation.idiom_insertion import (
    insert_prompt_hash,
    render_insert_prompt,
)
from scripts.checker import label_balance

_IDIOM = Idiom(
    "7", "7", "spill the beans", ("spill the beans",), ("To reveal a secret.",), "", None, None, None,
    "", None, None,
)  # fmt: skip


def _mnli() -> Item:
    meta = {"hypothesis": "HYPOTHESIS-TEXT stays fixed."}
    return Item("mnli", "1n", "XTEXT1 said nothing.", "0", meta)


def _mmlu() -> Item:
    meta = {"choices": ["CHOICE-A", "CHOICE-B", "CHOICE-C", "CHOICE-D"], "answer_index": 2}
    return Item("mmlu", "q1", "XTEXT2 asks something?", "2", meta)


def test_blind_prompt_hides_label_task_and_context() -> None:
    cases = [(i, v) for i in (_mnli(), _mmlu(), Item("sst2", "s1", "a SENTENCE-TEXT .", "1", {}))
             for v in ("idiom_insert_v3.txt", "idiom_insert_v4.txt")]  # fmt: skip
    for item, version in cases:
        blind = render_insert_prompt(item, [_IDIOM], version)
        assert item.x in blind and "spill the beans" in blind
        for hidden in ("label", "HYPOTHESIS-TEXT", "CHOICE-", "movie review", "premise", "multiple-choice",
                       "sentiment", "positive", "negative", "entailment"):  # fmt: skip
            assert hidden not in blind.lower() if hidden.islower() else hidden not in blind, hidden
        assert "{" not in blind.split("Idioms (id")[1]  # every placeholder filled
    seen = render_insert_prompt(_mmlu(), [_IDIOM], "idiom_insert_v2.txt")
    assert "CHOICE-C" in seen and "label" in seen  # the frozen v2 prompt still shows them


def test_blind_prompt_has_its_own_cache_namespace() -> None:
    assert insert_prompt_hash("mnli", "idiom_insert_v3.txt") != insert_prompt_hash("mnli")
    assert len({insert_prompt_hash(d, "idiom_insert_v3.txt") for d in ("sst2", "mnli")}) == 1


def test_label_balance_counts_and_test() -> None:
    drawn = {f"k{i}": ("a" if i < 50 else "b") for i in range(100)}
    kept = {k for k in drawn if k != "k0"}
    accepted = {k for k in kept if drawn[k] == "a"}  # every "b" rejected: badly unbalanced
    rows = label_balance(drawn, kept, accepted, pool=["a"] * 500 + ["b"] * 500)
    by = {(r["stage"], r["label"]): r for r in rows}
    assert by[("drawn", "a")]["n"] == 50 and by[("kept", "a")]["n"] == 49
    assert by[("kept", "a")]["p_vs_drawn"] > 0.5
    assert by[("accepted", "b")]["n"] == 0 and by[("accepted", "a")]["p_vs_drawn"] < 1e-6
    assert by[("drawn_vs_pool", "")]["p_vs_drawn"] > 0.9


def test_v4_restores_small_additions_and_keeps_every_advisor_rule() -> None:
    v4 = render_insert_prompt(_mmlu(), [_IDIOM], "idiom_insert_v4.txt")
    assert "add a small detail" in v4 and "nothing more" not in v4
    assert "same overall conclusions" in v4  # do not change the label, without naming it
    assert "technical or specialised terms exactly" in v4
    assert insert_prompt_hash("mmlu", "idiom_insert_v4.txt") != insert_prompt_hash(
        "mmlu", "idiom_insert_v3.txt"
    )


def test_v5_adds_the_question_and_judgement_rules_and_keeps_v4() -> None:
    v5 = render_insert_prompt(_mmlu(), [_IDIOM], "idiom_insert_v5.txt")
    assert "must stay exactly that: do not complete it, do not answer it" in v5
    assert "true or false" in v5 and "right or wrong" in v5
    assert "its order is random and means nothing" in v5
    assert "same overall conclusions" in v5 and "technical or specialised terms exactly" in v5
    assert "CHOICE-" not in v5 and "label" not in v5.lower()


def test_question_tail_check_catches_answered_questions_only() -> None:
    from augmentation.idiom_insertion import question_tail_changed  # noqa: PLC0415

    q = "attitude change is greatest when the level of discrepancy is:"
    assert question_tail_changed(
        q, "attitude change is greatest when the level of discrepancy is a happy medium."
    )
    assert not question_tail_changed(
        q, "in a nutshell, attitude change is greatest when the level of discrepancy is:"
    )
    blank = "Descartes reasons that thinking shows that _____."
    assert question_tail_changed(
        blank, "Descartes reasons that thinking shows that he is alive and kicking."
    )
    assert not question_tail_changed(
        blank, "Out of the blue, Descartes reasons that thinking shows that _____."
    )


def test_bank_orders_are_fixed_spread_and_differ() -> None:
    from augmentation.idiom_insertion import bank_in_order, bank_order_index  # noqa: PLC0415

    pool = [
        Idiom(
            str(n),
            str(n),
            f"idiom {n}",
            (f"idiom {n}",),
            ("x",),
            "",
            None,
            None,
            None,
            "",
            None,
            None,
        )
        for n in range(40)
    ]
    assert bank_in_order(pool, 3) == bank_in_order(
        pool, 3
    )  # same order every time (cacheable prefix)
    assert bank_in_order(pool, 3) != bank_in_order(pool, 4)
    ks = {bank_order_index(f"item{n}", 1) for n in range(200)}
    assert ks == set(range(8))  # items spread over all orders
    assert (
        bank_order_index("a", 2) == (bank_order_index("a", 1) + 1) % 8
    )  # a retry takes the next order


def test_majority_vote() -> None:
    from augmentation.checker import Verdict  # noqa: PLC0415
    from scripts.checker import majority  # noqa: PLC0415

    ok, bad = Verdict(True, action="valid"), Verdict(True, action="invalid")
    runs = [{"k": ("1", bad)}, {"k": ("2", ok)}, {"k": ("3", bad)}]
    assert majority(runs)["k"][1].rejects and majority(runs)["k"][0] == "1"
    runs = [{"k": ("1", bad)}, {"k": ("2", ok)}, {"k": ("3", ok)}]
    assert not majority(runs)["k"][1].rejects


def test_siqa_blind_prompt_hides_question_and_checker_shows_it() -> None:
    from augmentation.checker import render_checker_prompt  # noqa: PLC0415
    from augmentation.idiom_insertion import insert_label  # noqa: PLC0415

    meta = {
        "question": "QTEXT How would Sydney feel?",
        "choices": ["CH-sad", "CH-happy", "CH-bored"],
    }
    item = Item("siqa", "s1", "Sydney felt bad afterwards.", "0", meta)
    blind = render_insert_prompt(item, [_IDIOM], "idiom_insert_v5.txt")
    assert item.x in blind and "QTEXT" not in blind and "CH-" not in blind
    assert insert_label(item) == "A) CH-sad"
    rec = {
        "dataset": "siqa", "label": "A) CH-sad", "question": meta["question"], "choices": meta["choices"],
        "original": item.x, "idiomatic": item.x, "literal": item.x, "idiom": "spill the beans",
        "definition": "reveal", "idiom_text": "x", "literal_text": "y",
    }  # fmt: skip
    p = render_checker_prompt(rec, "checker_v1_structured.txt")
    assert "<question_never_changed>QTEXT" in p and "C) CH-bored" in p and "A) CH-sad" in p


def test_v6_shows_options_unlabelled_and_overlap_check() -> None:
    from augmentation.idiom_insertion import option_overlap_gained  # noqa: PLC0415

    v6 = render_insert_prompt(_mmlu(), [_IDIOM], "idiom_insert_v6.txt")
    tail = v6.split("Idioms (id")[1]
    assert "Related lines" in tail and "- CHOICE-C" in tail
    assert "C)" not in tail and "label" not in v6.lower() and "answer" not in tail.lower()
    assert "stock opening phrase" in v6 and "Match the register" in v6
    q = "attitude change is greatest when the level of discrepancy is:"
    opts = ["small.", "moderate.", "large.", "unpredictable."]
    assert option_overlap_gained(
        q, "attitude change is greatest when the level of discrepancy is moderate:", opts
    )
    assert not option_overlap_gained(
        q, "in a nutshell, attitude change is greatest when the level of discrepancy is:", opts
    )
