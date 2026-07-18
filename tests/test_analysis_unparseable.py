"""Regression: unparseable model outputs have ``parsed: null`` and must load.

A real Stage 3 run records ``parsed: None`` with ``parse_status != "ok"`` when
the model output cannot be parsed into a label. The complete-triple invariant
still holds (the variant key is present), so the loader must accept a null
``parsed`` rather than reject the whole run. Analysis derives numbers from
``correct``, so an unparseable output is simply an incorrect one.
"""

import json
from pathlib import Path

from analysis.aggregate import aggregate
from analysis.results import correct_by_variant, load_result


def _write_with_unparseable(results_dir: Path) -> Path:
    """Write a complete-triple result where one variant/task is unparseable."""

    def entry(correct: bool, *, parseable: bool = True) -> dict[str, object]:
        return {
            "raw": "mixed",
            "parsed": "0" if parseable else None,
            "parse_status": "ok" if parseable else "unparseable",
            "correct": correct,
        }

    per_task = [
        {
            "id": "task-0",
            "y": "0",
            "original": entry(True),
            "paraphrase": entry(True),
            "idiomatic": entry(True),
        },
        {
            "id": "task-1",
            "y": "0",
            # idiomatic output was unparseable -> parsed is null, correct is False
            "original": entry(True),
            "paraphrase": entry(True),
            "idiomatic": entry(False, parseable=False),
        },
    ]
    data = {
        "dataset": "sst2",
        "model_id": "modelA",
        "model_revision": "rev0",
        "config_hash": "cfg0",
        "prompt_hash": "ph0",
        "variants_run": ["original", "paraphrase", "idiomatic"],
        "per_task": per_task,
    }
    path = results_dir / "sst2" / "modelA.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def test_load_result_accepts_null_parsed(tmp_path: Path) -> None:
    path = _write_with_unparseable(tmp_path / "results")
    result = load_result(path)

    # The unparseable variant kept its null `parsed` and its correctness flag.
    idiomatic = result.per_task[1].variants["idiomatic"]
    assert idiomatic.parsed is None
    assert idiomatic.parse_status == "unparseable"
    assert idiomatic.correct is False

    # correctness flags are unaffected by the null parse
    assert correct_by_variant(result)["idiomatic"] == [True, False]


def test_aggregate_handles_unparseable_run(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    _write_with_unparseable(results_dir)
    df = aggregate(results_dir, n_resamples=200, ci=0.95, seed=0)

    assert df.shape == (1, 17)
    # idiomatic accuracy = 1/2 (the unparseable task counts as incorrect).
    assert df.loc[0, "acc_idiomatic"] == 0.5
    assert df.loc[0, "acc_original"] == 1.0
