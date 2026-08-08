"""Tests for augmentation prompt context injection (D9: meta as read-only context)."""

from augmentation.prompts.loader import load_prompt, render_context


def test_render_context_empty_when_no_choices() -> None:
    assert render_context({}) == ""
    assert render_context({"subject": "math"}) == ""
    assert render_context({"choices": []}) == ""
    # MNLI bookkeeping alone renders nothing — only allowlisted keys do.
    assert render_context({"prompt_id": 63735, "genre": "slate"}) == ""


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


def test_render_context_choices_output_is_unchanged_byte_for_byte() -> None:
    """Regression guard: MMLU's rendered context must not drift.

    `prompt_hash` covers only the raw template text, so a change here would leave
    every MMLU cache entry silently valid under an identical hash.
    """
    assert render_context({"choices": ["3", "4", "5", "6"], "answer_index": 1}) == (
        "Answer choices:\nA) 3\nB) 4\nC) 5\nD) 6\n\n"
    )


def test_render_context_renders_hypothesis_and_label_name_for_mnli() -> None:
    ctx = render_context(
        {
            "hypothesis": "Everyone really likes the newest benefits",
            "label_name": "neutral",
            "prompt_id": 63735,
            "genre": "slate",
        }
    )
    # The judge template is hardcoded, so the hypothesis block carries its own
    # constraint (R9) — an entailment label cannot be judged without both sides.
    assert "Everyone really likes the newest benefits" in ctx
    assert "Hypothesis (fixed reference" in ctx
    assert "must not restate" in ctx
    assert "Gold label in words: neutral" in ctx
    assert ctx.endswith("\n\n")


def test_render_context_omits_prompt_id_and_genre() -> None:
    ctx = render_context(
        {
            "hypothesis": "He smiled at Vrenna",
            "label_name": "entailment",
            "prompt_id": 91383,
            "genre": "fiction",
        }
    )
    assert "91383" not in ctx
    assert "fiction" not in ctx
    assert "prompt_id" not in ctx
    assert "genre" not in ctx


def test_render_context_ignores_empty_or_non_string_hypothesis() -> None:
    assert render_context({"hypothesis": ""}) == ""
    assert render_context({"hypothesis": None}) == ""
    assert render_context({"label_name": ""}) == ""


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


def test_nli_templates_have_context_and_text_slots() -> None:
    for name in ("paraphrase_nli_v1.txt", "idiomatic_nli_v1.txt"):
        template = load_prompt(name)
        assert "{context}" in template
        assert "{x}" in template
        assert "{y}" in template


def test_nli_templates_forbid_encoding_the_answer() -> None:
    """R7: the rewrite must not restate the hypothesis or reveal the label."""
    for name in ("paraphrase_nli_v1.txt", "idiomatic_nli_v1.txt"):
        template = load_prompt(name)
        assert "restate" in template
        assert "hint at the label" in template


def test_nli_template_fills_cleanly_for_mnli() -> None:
    for name in ("paraphrase_nli_v1.txt", "idiomatic_nli_v1.txt"):
        template = load_prompt(name)
        rendered = template.format(
            x="The new rights are nice enough",
            y="1",
            context=render_context(
                {
                    "hypothesis": "Everyone really likes the newest benefits",
                    "label_name": "neutral",
                    "prompt_id": 63735,
                    "genre": "slate",
                }
            ),
        )
        assert "Everyone really likes the newest benefits" in rendered
        assert "Gold label in words: neutral" in rendered
        assert "The new rights are nice enough" in rendered
        assert "Answer choices:" not in rendered
        assert "63735" not in rendered
        assert "slate" not in rendered
