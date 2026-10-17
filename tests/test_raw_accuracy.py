from analysis.aggregate import RunStats
from analysis.raw_accuracy import raw_accuracy_table, sole_cause_table, to_wide
from analysis.results import PerTask, PerTaskVariant, ResultFile, correct_by_variant
from analysis.stats import paired_bootstrap


def _run(model_id: str, flags: dict[str, list[bool]]) -> RunStats:
    n = len(flags["original"])
    per_task = [
        PerTask(
            id=str(i),
            y="a",
            variants={
                v: PerTaskVariant(raw="", parsed="a", parse_status="ok", correct=flags[v][i])
                for v in flags
            },
        )
        for i in range(n)
    ]
    result = ResultFile(
        dataset="sst2",
        model_id=model_id,
        model_revision="r",
        config_hash="c",
        prompt_hash="p",
        variants_run=list(flags),
        per_task=per_task,
    )
    boot = paired_bootstrap(correct_by_variant(result), n_resamples=200, ci=0.95, seed=0)
    return RunStats(result=result, boot=boot)


def test_raw_accuracy_counts_and_mean() -> None:
    a = _run(
        "a",
        {
            "original": [True, True, False, False],
            "paraphrase": [True, False, False, False],
            "idiomatic": [True, True, True, False],
        },
    )
    b = _run(
        "b",
        {
            "original": [True, True, True, True],
            "paraphrase": [True, True, True, False],
            "idiomatic": [True, True, False, False],
        },
    )
    long = raw_accuracy_table([a, b], ("sst2",))

    row_a = long[(long.model_id == "a") & (long.variant == "idiomatic")].iloc[0]
    assert (row_a["correct"], row_a["n"], row_a["acc"]) == (3, 4, 0.75)
    mean_par = long[(long.model_id == "MEAN") & (long.variant == "paraphrase")].iloc[0]
    assert mean_par["acc"] == (0.25 + 0.75) / 2

    wide = to_wide(long)
    assert list(wide.model_id) == ["a", "b", "MEAN"]
    assert wide[wide.model_id == "MEAN"].iloc[0]["original"] == "75.00"


def test_sole_cause_counts_and_pooled() -> None:
    # Row-wise (orig, par, idi): a: (T,T,F) -> b; (T,F,T) -> c; (T,T,F) -> b; (F,T,F) ignored.
    a = _run(
        "a",
        {
            "original": [True, True, True, False],
            "paraphrase": [True, False, True, True],
            "idiomatic": [False, True, False, False],
        },
    )
    b = _run(
        "b",
        {
            "original": [True, True],
            "paraphrase": [True, True],
            "idiomatic": [False, True],
        },
    )
    sole = sole_cause_table([a, b], ("sst2",))

    row_a = sole[sole.model_id == "a"].iloc[0]
    assert (row_a["n_orig_correct"], row_a["b"], row_a["c"], row_a["diff"]) == (3, 2, 1, 1)
    pooled = sole[sole.model_id == "POOLED"].iloc[0]
    assert (pooled["b"], pooled["c"], pooled["n_orig_correct"]) == (3, 1, 5)
    # continuity-corrected McNemar: (|3-1|-1)^2 / 4 = 0.25
    assert abs(pooled["chi2"] - 0.25) < 1e-9

    wide = to_wide(raw_accuracy_table([a, b], ("sst2",)), sole)
    assert wide[wide.model_id == "POOLED"].iloc[0]["b / c"] == "3 / 1"
