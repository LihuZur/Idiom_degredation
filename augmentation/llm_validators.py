"""LLM-judge validators (D2/D6): real label-preservation + idiom presence/absence.

Each validator reuses the shared augmenter `LLMClient` (D6), renders a frozen,
hashed judge template, and parses a strict PASS/FAIL verdict. Malformed or empty
verdicts fail closed (`passed=False`) so the pipeline's retry-then-abort loop
(D3) re-augments rather than emitting an unvalidated row.
"""

from augmentation.base import AugmentedRow, ValidationResult, Validator
from augmentation.prompts.loader import load_prompt, render_context
from augmentation.providers.base import LLMClient
from augmentation.registry import register_validator
from cleaning.hashing import prompt_hash
from data.base import DatasetRow

# Frozen judge templates, one per validator (loaded + hashed at construction).
JUDGE_PROMPTS: dict[str, str] = {
    "label_preservation": "judge_label_preservation_v1.txt",
    "idiom_presence": "judge_idiom_presence_v1.txt",
    "idiom_absence": "judge_idiom_absence_v1.txt",
}


def parse_verdict(text: str) -> tuple[bool, str]:
    """Map a judge response to `(passed, verdict)`.

    Returns `(True, "PASS")` / `(False, "FAIL")` for an unambiguous verdict and
    `(False, "MALFORMED")` when neither or both tokens appear (fail closed).
    """
    upper = text.upper()
    has_pass = "PASS" in upper
    has_fail = "FAIL" in upper
    if has_pass and not has_fail:
        return True, "PASS"
    if has_fail and not has_pass:
        return False, "FAIL"
    return False, "MALFORMED"


class _LLMJudgeValidator:
    """Base for the three LLM-judge validators.

    Subclasses set `name` (a key of `JUDGE_PROMPTS`) and `_render` to build the
    judge's user prompt from the rewritten row (+ original, for label checks).
    """

    name: str

    def __init__(
        self,
        *,
        client: LLMClient,
        prompt_template: str,
        temperature: float,
        max_output_tokens: int,
    ) -> None:
        self._client = client
        self._template = prompt_template
        self._prompt_hash = prompt_hash(prompt_template)
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens

    def _render(self, ex: AugmentedRow, original: DatasetRow) -> str:
        raise NotImplementedError

    def validate(self, ex: AugmentedRow, original: DatasetRow) -> ValidationResult:
        """Ask the judge and return a strict PASS/FAIL `ValidationResult`."""
        user = self._render(ex, original)
        raw = self._client.complete(
            system="",
            user=user,
            temperature=self._temperature,
            max_output_tokens=self._max_output_tokens,
        )
        passed, verdict = parse_verdict(raw)
        return ValidationResult(
            name=self.name,
            passed=passed,
            score=None,
            details={
                "verdict": verdict,
                "judge_model": f"{self._client.provider}/{self._client.model}",
                "judge_prompt_hash": self._prompt_hash,
            },
        )


@register_validator("label_preservation")
class LabelPreservationValidator(_LLMJudgeValidator):
    """LLM-judge: is the gold label still correct for the rewritten text (M8)?"""

    name: str = "label_preservation"

    def _render(self, ex: AugmentedRow, original: DatasetRow) -> str:
        return self._template.format(
            y=original.y,
            context=render_context(ex.meta),
            original=original.x,
            rewritten=ex.x,
        )


@register_validator("idiom_presence")
class IdiomPresenceValidator(_LLMJudgeValidator):
    """LLM-judge: does the (idiomatic-variant) rewrite contain idioms?"""

    name: str = "idiom_presence"

    def _render(self, ex: AugmentedRow, original: DatasetRow) -> str:
        return self._template.format(context=render_context(ex.meta), rewritten=ex.x)


@register_validator("idiom_absence")
class IdiomAbsenceValidator(_LLMJudgeValidator):
    """LLM-judge: is the (paraphrase-variant control) rewrite free of idioms?"""

    name: str = "idiom_absence"

    def _render(self, ex: AugmentedRow, original: DatasetRow) -> str:
        return self._template.format(context=render_context(ex.meta), rewritten=ex.x)


_JUDGE_CLASSES: dict[str, type[_LLMJudgeValidator]] = {
    "label_preservation": LabelPreservationValidator,
    "idiom_presence": IdiomPresenceValidator,
    "idiom_absence": IdiomAbsenceValidator,
}


def build_judge(
    name: str, *, client: LLMClient, temperature: float, max_output_tokens: int
) -> Validator:
    """Construct the LLM-judge validator `name` with its frozen template + `client`.

    `name` must be a key of `JUDGE_PROMPTS`; the shared augmenter `client` is
    reused for judging (D6).
    """
    return _JUDGE_CLASSES[name](
        client=client,
        prompt_template=load_prompt(JUDGE_PROMPTS[name]),
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
