"""Prompts for MNLI evaluation (MNLI_DATASET_PLAN §5.2)."""

SYSTEM = (
    "You are a natural language inference classifier. Read the premise and the "
    "hypothesis, then respond with exactly one word: "
    '"entailment", "neutral", or "contradiction".'
)

USER_TEMPLATE = """Premise: {premise}
Hypothesis: {hypothesis}
Answer:"""
