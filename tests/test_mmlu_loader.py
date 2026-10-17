"""MMLU loader tests — hits the real HF hub (STAGE1_PLAN §5.6, Q17)."""

import pytest

from data.mmlu import MmluChoicesLoader, MmluLoader, decode_choices, encode_choices

_HF_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"


def test_mmlu_loader_yields_expected_rows() -> None:
    loader = MmluLoader(hf_revision=_HF_REVISION, normalize=["nfc", "collapse_whitespace"])
    rows = list(loader.load())

    subjects = {row.meta["subject"] for row in rows}
    assert len(subjects) == 57

    for row in rows:
        choices = row.meta["choices"]
        assert isinstance(choices, list)
        assert len(choices) == 4
        assert all(isinstance(c, str) for c in choices)
        assert row.meta["answer_index"] in {0, 1, 2, 3}

        subject, _, index = row.id.partition("/")
        assert subject == row.meta["subject"]
        assert index.isdigit()


def test_encode_decode_choices_round_trip() -> None:
    choices = ["Mercury", "Venus", "Earth", "Mars"]
    encoded = encode_choices(choices)
    assert encoded == "A) Mercury\nB) Venus\nC) Earth\nD) Mars"
    assert decode_choices(encoded) == choices


def test_decode_choices_rejects_wrong_line_count() -> None:
    with pytest.raises(ValueError, match="expected 4 choice lines"):
        decode_choices("A) only one line")


def test_decode_choices_rejects_wrong_prefix() -> None:
    with pytest.raises(ValueError, match="expected line to start with"):
        decode_choices("A) x\nB) y\nZ) z\nD) w")


def test_mmlu_choices_loader_selects_the_same_ids_as_mmlu_loader() -> None:
    """RQ3 §4/H3c: same row ids as `mmlu` so the two are directly comparable."""
    mmlu_ids = [row.id for row in MmluLoader(hf_revision=_HF_REVISION, normalize=[]).load()]
    choices_ids = [
        row.id for row in MmluChoicesLoader(hf_revision=_HF_REVISION, normalize=[]).load()
    ]
    assert choices_ids == mmlu_ids


def test_mmlu_choices_loader_swaps_x_and_meta_roles() -> None:
    loader = MmluChoicesLoader(hf_revision=_HF_REVISION, normalize=["nfc", "collapse_whitespace"])
    first = next(iter(loader.load()))

    # x = encoded choices (the field Stage 2 rewrites); question is now fixed meta.
    assert first.x == encode_choices(decode_choices(first.x))
    assert isinstance(first.meta["question"], str) and first.meta["question"]
    assert first.meta["answer_index"] == first.y
    assert "choices" not in first.meta


def test_mmlu_choices_loader_meta_carries_no_unexpected_keys() -> None:
    loader = MmluChoicesLoader(hf_revision=_HF_REVISION, normalize=["nfc", "collapse_whitespace"])
    for row in loader.load():
        assert set(row.meta) == {"question", "subject", "answer_index"}
