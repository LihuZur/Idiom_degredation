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
