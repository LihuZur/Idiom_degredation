"""Prompts for MMLU evaluation (STAGE3_PLAN §1)."""

SYSTEM = (
    "You are answering a multiple-choice question. Read the question "
    "and the four choices, then respond with exactly one letter: A, B, C, or D."
)

USER_TEMPLATE = """Question: {question}
A. {A}
B. {B}
C. {C}
D. {D}
Answer:"""
