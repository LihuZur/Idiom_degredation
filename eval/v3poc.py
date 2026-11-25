"""Evaluators for the v3 POC sets (THESIS_PLAN_v3 §8 step 8c): E3 + POC pairs, not yet validated. Same prompts
and parsers as the pilot datasets, registered under new names so POC results never mix with other results."""

from eval.agnews import AgNewsEvaluator
from eval.mmlu import MmluEvaluator
from eval.mnli import MnliEvaluator
from eval.registry import register_evaluator
from eval.sst2 import Sst2Evaluator


@register_evaluator("sst2_v3poc")
class Sst2V3PocEvaluator(Sst2Evaluator):
    dataset = "sst2_v3poc"


@register_evaluator("mnli_v3poc")
class MnliV3PocEvaluator(MnliEvaluator):
    dataset = "mnli_v3poc"


@register_evaluator("mmlu_v3poc")
class MmluV3PocEvaluator(MmluEvaluator):
    dataset = "mmlu_v3poc"


@register_evaluator("agnews_v3poc")
class AgNewsV3PocEvaluator(AgNewsEvaluator):
    dataset = "agnews_v3poc"


# POC-2 (plan v3 step 8d, advisors 2026-11-15): pairs from the blind prompt v3 that passed the cheap filter.
@register_evaluator("sst2_v3poc2")
class Sst2V3Poc2Evaluator(Sst2Evaluator):
    dataset = "sst2_v3poc2"


@register_evaluator("mnli_v3poc2")
class MnliV3Poc2Evaluator(MnliEvaluator):
    dataset = "mnli_v3poc2"


@register_evaluator("mmlu_v3poc2")
class MmluV3Poc2Evaluator(MmluEvaluator):
    dataset = "mmlu_v3poc2"


@register_evaluator("agnews_v3poc2")
class AgNewsV3Poc2Evaluator(AgNewsEvaluator):
    dataset = "agnews_v3poc2"


# POC-3 (plan v3 step 8e, 2026-11-17): prompt v5, 20 fit + 10 random idioms; pairs accepted by both luna (3 votes)
# and the Claude review. No MMLU (decision pending, O47).
@register_evaluator("sst2_v3poc3")
class Sst2V3Poc3Evaluator(Sst2Evaluator):
    dataset = "sst2_v3poc3"


@register_evaluator("mnli_v3poc3")
class MnliV3Poc3Evaluator(MnliEvaluator):
    dataset = "mnli_v3poc3"


@register_evaluator("agnews_v3poc3")
class AgNewsV3Poc3Evaluator(AgNewsEvaluator):
    dataset = "agnews_v3poc3"


# Full study, MNLI (plan v3 O53, 2026-11-25): 795 pairs (265 per label) that passed luna (3 votes) and the blind AI
# review, words kept >= 80%, at most 5 per idiom. Same prompt and parser as POC-3, so the two are comparable.
@register_evaluator("mnli_full")
class MnliFullEvaluator(MnliEvaluator):
    dataset = "mnli_full"
