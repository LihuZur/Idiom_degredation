"""Prompts for SST-2 evaluation (STAGE3_PLAN §1)."""

SYSTEM = (
    "You are a sentiment classifier. Read the sentence and respond with "
    'exactly one word: either "positive" or "negative".'
)

USER_TEMPLATE = """Sentence: {x}
Answer:"""
