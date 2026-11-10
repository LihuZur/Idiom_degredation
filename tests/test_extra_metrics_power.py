from pathlib import Path

import numpy as np
import pytest

from analysis.extra_metrics import parse_failure_table, rank_stability_table, unrobustness_table
from analysis.power import item_differences, minimum_detectable_effect, simulate_power
from analysis.results import ResultFile, load_result
from tests._stage4_helpers import write_result_file


def _load(
    tmp_path: Path, model: str, orig: list[bool], para: list[bool], idi: list[bool]
) -> ResultFile:
    path = write_result_file(
        tmp_path,
        dataset="sst2",
        model_id=model,
        correct_by_variant={"original": orig, "paraphrase": para, "idiomatic": idi},
    )
    return load_result(path)


def test_unrobustness_is_symmetric_and_net_change_is_directional(tmp_path: Path) -> None:
    r = _load(
        tmp_path,
        "m",
        [True, True, False, False],
        [True, False, True, False],
        [True, True, True, True],
    )
    table = unrobustness_table([r])
    para = table[table["variant"] == "paraphrase"].iloc[0]
    assert (para["n_broken"], para["n_fixed"]) == (1, 1)
    assert para["U"] == pytest.approx(0.5)
    assert para["net_change"] == pytest.approx(0.0)
    idi = table[table["variant"] == "idiomatic"].iloc[0]
    assert (idi["U"], idi["net_change"]) == (pytest.approx(0.5), pytest.approx(0.5))


def test_parse_failure_rate_counts_non_ok(tmp_path: Path) -> None:
    r = _load(tmp_path, "m", [True, True], [True, True], [True, True])
    r.per_task[0].variants["idiomatic"].parse_status = "unparseable"
    table = parse_failure_table([r])
    row = table[table["variant"] == "idiomatic"].iloc[0]
    assert (row["n_unparseable"], row["rate"]) == (1, pytest.approx(0.5))


def test_rank_stability_perfect_when_order_is_unchanged(tmp_path: Path) -> None:
    hi = [True] * 8 + [False] * 2
    lo = [True] * 4 + [False] * 6
    mid = [True] * 6 + [False] * 4
    rs = [_load(tmp_path, m, v, v, v) for m, v in (("a", hi), ("b", lo), ("c", mid))]
    table = rank_stability_table(rs)
    assert set(table["spearman_rho"].round(6)) == {1.0}


def test_power_grows_with_n_and_effect_and_matches_mde(tmp_path: Path) -> None:
    r1 = _load(
        tmp_path,
        "m1",
        [True] * 6,
        [True, True, False, True, False, True],
        [True, False, False, True, True, True],
    )
    r2 = _load(
        tmp_path,
        "m2",
        [True] * 6,
        [True, True, True, True, False, False],
        [False, True, False, True, False, True],
    )
    d = item_differences([r1, r2], "sst2", "paraphrase", "idiomatic")
    assert len(d) == 6
    rng = np.random.default_rng(0)
    low = simulate_power(d, 50, 0.0, alpha=0.05, reps=400, rng=rng)
    high = simulate_power(d, 400, 0.3, alpha=0.05, reps=400, rng=rng)
    assert low < 0.15
    assert high > 0.9
    sd = float(d.std(ddof=1))
    assert minimum_detectable_effect(sd, 400, alpha=0.05, power=0.8) < minimum_detectable_effect(
        sd, 100, alpha=0.05, power=0.8
    )
