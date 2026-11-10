"""v3 analysis on synthetic data with a known idiom effect (offline)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.v3_analysis import fit_main, holm, load_long, per_model


def _synthetic(effect: float, seed: int = 0, n_items: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    models = {"qwen3.5-0.5b-instruct": -0.5, "qwen3.5-7b-instruct": 1.0, "olmo-2-1b-instruct": 0.0}
    rows = []
    item_diff = rng.normal(0, 1, n_items)
    for m, strength in models.items():
        for i in range(n_items):
            for v in ("idiomatic", "literal"):
                logit = 1.0 + strength + item_diff[i] + (effect if v == "idiomatic" else 0.0)
                rows.append(
                    {
                        "dataset": "toy",
                        "model": m,
                        "log_params": np.log(
                            {
                                "qwen3.5-0.5b-instruct": 0.5,
                                "qwen3.5-7b-instruct": 7.0,
                                "olmo-2-1b-instruct": 1.0,
                            }[m]
                        ),
                        "item": f"i{i}",
                        "idiom": f"id{i // 3}",
                        "version": v,
                        "correct": int(rng.random() < 1 / (1 + np.exp(-logit))),
                        "parse_ok": True,
                    }
                )
    return pd.DataFrame(rows)


def test_main_test_recovers_a_negative_effect_and_no_effect() -> None:
    hurt = fit_main(_synthetic(-0.8))
    assert hurt["log_odds_idiomatic"] < -0.4 and hurt["ci_high"] < 0 and hurt["p_approx"] < 0.01
    null = fit_main(_synthetic(0.0, seed=1))
    assert null["ci_low"] < 0 < null["ci_high"]


def test_per_model_counts_and_bootstrap() -> None:
    pm = per_model(_synthetic(-0.8), n_boot=300)
    assert len(pm) == 3
    assert (pm["b_wrong_only_idiomatic"] > pm["c_wrong_only_literal"]).all()
    assert (pm["gap_pp"] < 0).all() and (pm["boot_ci_high"] < 0).all()


def test_holm() -> None:
    assert holm([0.01, 0.04, 0.03, 0.5]) == [0.04, 0.09, 0.09, 0.5]


def test_load_long_reads_result_files(tmp_path: Path) -> None:
    data = tmp_path / "data" / "toy_v3poc"
    data.mkdir(parents=True)
    meta = json.dumps({"idiom": {"idiom": "spill the beans"}})
    (data / "idiomatic.csv").write_text(
        f'id,meta\na,"{meta.replace(chr(34), chr(34) * 2)}"\n', encoding="utf-8"
    )
    res = tmp_path / "results" / "toy_v3poc"
    res.mkdir(parents=True)
    task = {
        "id": "a",
        "y": "1",
        "idiomatic": {"correct": False, "parse_status": "ok"},
        "span_literal": {"correct": True, "parse_status": "ok"},
        "original": {"correct": True, "parse_status": "fail"},
    }
    (res / "m.json").write_text(
        json.dumps({"model_id": "olmo-2-1b-instruct", "per_task": [task]}), encoding="utf-8"
    )
    df = load_long(tmp_path / "results", "toy_v3poc", data_root=tmp_path / "data")
    assert set(df["version"]) == {"idiomatic", "literal", "original"}
    assert df.loc[df["version"] == "idiomatic", "idiom"].iloc[0] == "spill the beans"
    assert not df.loc[df["version"] == "original", "parse_ok"].iloc[0]


def test_holm_skips_missing_p_values() -> None:
    out = holm([0.01, float("nan"), 0.04])
    assert out[0] == 0.02 and out[2] == 0.04 and np.isnan(out[1])
