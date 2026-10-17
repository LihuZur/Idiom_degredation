from analysis.aggregate import RunStats
from analysis.raw_accuracy import raw_accuracy_table, to_wide
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
