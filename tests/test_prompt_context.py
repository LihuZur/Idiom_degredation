"""Tests for augmentation prompt context injection (D9: meta as read-only context)."""

from augmentation.prompts.loader import load_prompt, render_context


def test_render_context_empty_when_no_choices() -> None:
    assert render_context({}) == ""
    assert render_context({"subject": "math"}) == ""
    assert render_context({"choices": []}) == ""


def test_render_context_renders_choices_as_labeled_list() -> None:
    ctx = render_context({"choices": ["Paris", "Rome", "Berlin", "Madrid"], "answer_index": 0})
    assert "A) Paris" in ctx
    assert "B) Rome" in ctx
    assert "C) Berlin" in ctx
    assert "D) Madrid" in ctx


def test_render_context_omits_answer_index_and_subject() -> None:
    # The correct-answer index must never leak into the prompt.
    ctx = render_context({"choices": ["3", "4", "5", "6"], "answer_index": 1, "subject": "math"})
    assert "answer_index" not in ctx
    assert "1" not in ctx.replace("3", "").replace("4", "").replace("5", "").replace("6", "")
    assert "math" not in ctx


def test_templates_have_context_and_text_slots() -> None:
    for name in ("paraphrase_v1.txt", "idiomatic_v1.txt"):
        template = load_prompt(name)
        assert "{context}" in template
        assert "{x}" in template
        assert "{y}" in template


def test_template_fills_cleanly_for_mmlu_and_sst2() -> None:
    for name in ("paraphrase_v1.txt", "idiomatic_v1.txt"):
        template = load_prompt(name)
        # MMLU-style: choices appear as context, stem is the text to rewrite.
        mmlu = template.format(
            x="What is 2+2?",
            y="1",
            context=render_context({"choices": ["3", "4", "5", "6"], "answer_index": 1}),
        )
        assert "Answer choices:" in mmlu
        assert "B) 4" in mmlu
        assert "What is 2+2?" in mmlu
        # SST-2-style: no context section at all.
        sst2 = template.format(x="a fine film", y="1", context=render_context({}))
        assert "Answer choices:" not in sst2
        assert "a fine film" in sst2
