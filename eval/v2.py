"""Evaluators for the v2 datasets (THESIS_PLAN_v2 §3): same prompts and parsers as the pilot
datasets, registered under new names so v2 results never overwrite pilot results."""

from eval.agnews import AgNewsEvaluator
from eval.mmlu import MmluEvaluator
from eval.mnli import MnliEvaluator
from eval.registry import register_evaluator
from eval.sst2 import Sst2Evaluator


@register_evaluator("sst2_v2")
class Sst2V2Evaluator(Sst2Evaluator):
    dataset = "sst2_v2"


@register_evaluator("mnli_v2")
class MnliV2Evaluator(MnliEvaluator):
    dataset = "mnli_v2"


@register_evaluator("mmlu_v2")
class MmluV2Evaluator(MmluEvaluator):
    dataset = "mmlu_v2"


@register_evaluator("agnews_v2")
class AgNewsV2Evaluator(AgNewsEvaluator):
    dataset = "agnews_v2"
