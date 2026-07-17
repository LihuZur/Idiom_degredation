"""MMLU loader tests — hits the real HF hub (STAGE1_PLAN §5.6, Q17)."""

from data.mmlu import MmluLoader

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
