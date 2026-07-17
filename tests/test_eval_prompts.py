"""Unit tests for evaluation prompts and hashes (STAGE3_PLAN §5.2)."""

import hashlib

import eval.prompts.mmlu as mmlu_prompts
import eval.prompts.sst2 as sst2_prompts


def get_prompt_hash(system: str, user_template: str) -> str:
    """Calculate the stable prompt hash (STAGE3_PLAN §1)."""
    content = system + "\x1e" + user_template
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def test_sst2_prompt() -> None:
    expected_user = "Sentence: hello\nAnswer:"
    assert sst2_prompts.USER_TEMPLATE.format(x="hello") == expected_user
    assert "sentiment classifier" in sst2_prompts.SYSTEM


def test_mmlu_prompt() -> None:
    expected_user = "Question: What is 1+1?\nA. 1\nB. 2\nC. 3\nD. 4\nAnswer:"
    rendered = mmlu_prompts.USER_TEMPLATE.format(
        question="What is 1+1?", A="1", B="2", C="3", D="4"
    )
    assert rendered == expected_user
    assert "multiple-choice question" in mmlu_prompts.SYSTEM


def test_prompt_hash_stability() -> None:
    hash_val1 = get_prompt_hash(sst2_prompts.SYSTEM, sst2_prompts.USER_TEMPLATE)
    hash_val2 = get_prompt_hash(sst2_prompts.SYSTEM, sst2_prompts.USER_TEMPLATE)
    assert hash_val1 == hash_val2
    assert len(hash_val1) == 16
    assert isinstance(hash_val1, str)
