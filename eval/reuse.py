"""Strict reuse of earlier Stage 3 predictions (THESIS_PLAN_v2 §3.5, §10 step 10).

A prediction may be reused only if it was produced for byte-identical input by the same model
revision, precision, decoding config and prompt; anything else must be run on the GPU again.
`verify_reuse` re-runs a small random sample and fails loudly if any output differs.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from augmentation.base import AugmentedRow
from models.base import Prediction


class ReuseError(ValueError):
    """The earlier results are not a valid source for reuse."""


@dataclass(slots=True)
class ReuseReport:
    reused: int = 0
    reasons: dict[str, int] = field(default_factory=dict[str, int])

    def skip(self, reason: str) -> None:
        self.reasons[reason] = self.reasons.get(reason, 0) + 1


def read_xy(csv_path: Path) -> dict[str, tuple[str, str]]:
    """id -> (x, y) of a variant CSV, as strings, for the text-identity check."""
    with csv_path.open(encoding="utf-8", newline="") as f:
        return {r["id"]: (r["x"], r["y"]) for r in csv.DictReader(f)}


def check_provenance(
    prior: dict[str, Any],
    *,
    prompt_hash: str,
    model_id: str,
    model_revision: Any,
    precision: str,
    resolved_config: dict[str, Any],
) -> None:
    """Raise `ReuseError` unless the earlier run used the same prompt, model and decoding."""
    for key, want in (
        ("prompt_hash", prompt_hash),
        ("model_id", model_id),
        ("model_revision", model_revision),
        ("precision", precision),
        ("resolved_config", resolved_config),
    ):
        if prior.get(key) != want:
            raise ReuseError(
                f"cannot reuse: {key} differs (earlier {prior.get(key)!r}, now {want!r})"
            )


def reusable_predictions(
    prior: dict[str, Any],
    prior_variant: str,
    prior_xy: dict[str, tuple[str, str]],
    rows: list[AugmentedRow],
    evaluator: Any,
) -> tuple[dict[str, Prediction], ReuseReport]:
    """Earlier predictions that are valid for `rows`: same id, same input text, same label.

    Outputs are re-parsed and re-scored with the current evaluator, so scoring never depends on
    the earlier run's parser. Rows that fail any check are left out and must be run.
    """
    raw_by_id = {
        t["id"]: t[prior_variant]["raw"] for t in prior.get("per_task", []) if prior_variant in t
    }
    out: dict[str, Prediction] = {}
    report = ReuseReport()
    for ex in rows:
        if ex.id not in raw_by_id:
            report.skip("not_in_earlier_results")
            continue
        old = prior_xy.get(ex.id)
        if old is None:
            report.skip("not_in_earlier_csv")
        elif old[0] != ex.x:
            report.skip("text_changed")
        elif old[1] != str(ex.y):
            report.skip("label_changed")
        else:
            raw = raw_by_id[ex.id]
            parsed, status = evaluator.parse(raw, ex)
            correct = evaluator.score(parsed, ex.y)
            out[ex.id] = Prediction(
                id=ex.id,
                raw=raw,
                parsed=parsed,
                meta={"parse_status": status, "correct": correct, "reused": True},
            )
            report.reused += 1
    return out, report


def verify_reuse(
    evaluator: Any,
    model: Any,
    rows: list[AugmentedRow],
    reused: dict[str, Prediction],
    n: int,
    seed: int,
) -> int:
    """Re-run `n` reused rows on the model; raise `ReuseError` if any raw output differs."""
    pool = [ex for ex in rows if ex.id in reused]
    sample = random.Random(seed).sample(pool, min(n, len(pool)))
    if not sample:
        return 0
    fresh, _ = evaluator._run_inference(model, sample)
    bad = [p.id for p in fresh if p.raw != reused[p.id].raw]
    if bad:
        raise ReuseError(
            f"{len(bad)} of {len(sample)} re-run outputs differ from the earlier ones "
            f"(e.g. ids {bad[:5]}); do not reuse these predictions"
        )
    return len(sample)
