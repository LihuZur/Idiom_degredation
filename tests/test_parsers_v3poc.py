"""Answer parsers: cases found in the v3 POC evaluation (2026-11-11)."""

from eval.agnews import AgNewsEvaluator
from eval.mmlu import MmluEvaluator

AG = AgNewsEvaluator.__new__(AgNewsEvaluator)
MM = MmluEvaluator.__new__(MmluEvaluator)


def test_agnews_label_glued_to_the_explanation() -> None:
    assert AG.parse("WorldThe news snippet provided is about", None) == ("0", "ok")  # type: ignore[arg-type]
    # the leading label wins over a label word echoed later
    assert AG.parse("SportsYou are a news topic classifier. Reply: World", None) == ("1", "ok")  # type: ignore[arg-type]
    assert AG.parse("The topic is Business", None) == ("2", "ok")  # type: ignore[arg-type]
    assert AG.parse("Crime", None) == (None, "unparseable")  # type: ignore[arg-type]


def test_mmlu_leading_letter_and_article_a() -> None:
    assert MM.parse("AA is the most comprehensive definition", None) == ("0", "ok")  # type: ignore[arg-type]
    assert MM.parse("ASocrates believed", None) == ("0", "ok")  # type: ignore[arg-type]
    assert MM.parse("(C) because", None) == ("2", "ok")  # type: ignore[arg-type]
    assert MM.parse("b", None) == ("1", "ok")  # type: ignore[arg-type]
    # the article "a" is not option A; "All" is not a letter answer
    assert MM.parse("Socrates believed that a virtue is B", None) == ("1", "ok")  # type: ignore[arg-type]
    assert MM.parse("All of the above is a trap, answer: C", None) == ("2", "ok")  # type: ignore[arg-type]
