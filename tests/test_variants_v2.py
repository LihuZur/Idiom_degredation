from pathlib import Path

import pytest

from analysis.aggregate import bootstrap_by_run
from analysis.raw_accuracy import raw_accuracy_table, sole_cause_table
from analysis.results import VARIANTS_V2, correct_by_variant, load_result, variants_for
from analysis.stats import paired_bootstrap
from tests._stage4_helpers import write_result_file

_N_RES = 50


def _flags(*rows: tuple[bool, bool, bool]) -> tuple[list[bool], list[bool], list[bool]]:
    a, b, c = zip(*rows, strict=True)
    return list(a), list(b), list(c)


def _v2_file(tmp_path: Path, model: str = "m1") -> Path:
    orig, ctrl, idi = _flags(
        (True, True, True), (True, True, False), (True, False, True), (False, True, False)
    )
    return write_result_file(
        tmp_path,
        dataset="sst2_v2",
        model_id=model,
        correct_by_variant={"original": orig, "span_literal": ctrl, "idiomatic": idi},
        variants_run=list(VARIANTS_V2),
    )


def test_variants_for_selects_family() -> None:
    assert variants_for("sst2") == ("original", "paraphrase", "idiomatic")
    assert variants_for("agnews_v2") == VARIANTS_V2


def test_load_v2_result_and_correctness(tmp_path: Path) -> None:
    result = load_result(_v2_file(tmp_path))
    flags = correct_by_variant(result)
    assert list(flags) == list(VARIANTS_V2)
    assert flags["span_literal"] == [True, True, False, True]


def test_v2_result_missing_span_literal_raises(tmp_path: Path) -> None:
    path = write_result_file(
        tmp_path,
        dataset="mnli_v2",
        model_id="m1",
        correct_by_variant={"original": [True], "paraphrase": [True], "idiomatic": [True]},
        variants_run=["original", "paraphrase", "idiomatic"],
    )
    with pytest.raises(ValueError, match="span_literal"):
        load_result(path)


def test_legacy_result_still_requires_paraphrase(tmp_path: Path) -> None:
    path = write_result_file(
        tmp_path,
        dataset="sst2",
        model_id="m1",
        correct_by_variant={"original": [True], "span_literal": [True], "idiomatic": [True]},
        variants_run=["original", "span_literal", "idiomatic"],
    )
    with pytest.raises(ValueError, match="paraphrase"):
        load_result(path)


def test_extra_declared_variant_is_loaded(tmp_path: Path) -> None:
    path = write_result_file(
        tmp_path,
        dataset="sst2_v2",
        model_id="m1",
        correct_by_variant={
            "original": [True, False],
            "span_literal": [True, True],
            "idiomatic": [False, True],
            "paraphrase": [True, False],
        },
        variants_run=["original", "span_literal", "idiomatic", "paraphrase"],
    )
    result = load_result(path)
    assert "paraphrase" in result.per_task[0].variants
    assert list(correct_by_variant(result)) == list(VARIANTS_V2)


def test_paired_bootstrap_uses_custom_variant_names() -> None:
    flags = {
        "original": [True, True, False, True],
        "span_literal": [True, False, False, True],
        "idiomatic": [False, False, False, True],
    }
    res = paired_bootstrap(flags, n_resamples=_N_RES, ci=0.9, seed=0, variants=VARIANTS_V2)
    assert set(res.acc) == set(VARIANTS_V2)
    assert res.delta_paraphrase.point == pytest.approx(-0.25)  # control - original
    assert res.delta_idiom.point == pytest.approx(-0.25)  # idiomatic - control


def test_bootstrap_by_run_filters_by_family(tmp_path: Path) -> None:
    _v2_file(tmp_path)
    write_result_file(
        tmp_path,
        dataset="sst2",
        model_id="m1",
        correct_by_variant={
            "original": [True, False, True],
            "paraphrase": [True, True, False],
            "idiomatic": [False, True, True],
        },
    )
    legacy = bootstrap_by_run(tmp_path, n_resamples=_N_RES, ci=0.9, seed=0, family="legacy")
    v2 = bootstrap_by_run(tmp_path, n_resamples=_N_RES, ci=0.9, seed=0, family="v2")
    assert [r.result.dataset for r in legacy] == ["sst2"]
    assert [r.result.dataset for r in v2] == ["sst2_v2"]
    assert "span_literal" in v2[0].boot.acc


def test_raw_and_sole_cause_tables_for_v2(tmp_path: Path) -> None:
    _v2_file(tmp_path)
    runs = bootstrap_by_run(tmp_path, n_resamples=_N_RES, ci=0.9, seed=0, family="v2")
    long = raw_accuracy_table(runs, ("sst2_v2",))
    assert set(long["variant"]) == set(VARIANTS_V2)
    sole = sole_cause_table(runs, ("sst2_v2",))
    row = sole[sole["model_id"] == "m1"].iloc[0]
    # original-correct rows: 0 (all), 1 (ctrl ok, idiom wrong -> b), 2 (ctrl wrong, idiom ok -> c)
    assert (int(row["b"]), int(row["c"]), int(row["n_orig_correct"])) == (1, 1, 3)
