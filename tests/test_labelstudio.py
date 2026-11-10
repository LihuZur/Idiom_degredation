import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pytest

from augmentation.idiom_generation import Item, generate, write_outputs
from augmentation.labelstudio import (
    agreement,
    blind_accuracy,
    blind_config_xml,
    blind_tasks,
    flatten_export,
    gold_config_xml,
    gold_tasks,
    load_generated,
    stratified_gold_sample,
)
from tests._generation_helpers import GOOD, X, setup_inventory


def _generated(tmp_path: Path) -> list[dict[str, Any]]:
    inv, index = setup_inventory(tmp_path)
    items = [Item("sst2", f"id{n}", X, "1", {}) for n in range(4)]
    outcomes = generate(items, inv, index, lambda ps: [GOOD] * len(ps), model="m", p_hash="ph", k=3)
    pool = {
        i.id: {
            "id": i.id,
            "variant": "original",
            "x": i.x,
            "y": i.y,
            "meta": "{}",
            "augmenter_model": "",
            "prompt_hash": "",
            "validators": "",
        }
        for i in items
    }
    write_outputs(tmp_path / "out", outcomes, pool, model="m", p_hash="ph", meta={})
    return load_generated(tmp_path / "out")


def test_gold_tasks_show_all_three_texts_and_idiom_info(tmp_path: Path) -> None:
    sample = stratified_gold_sample(_generated(tmp_path), 2, 0)
    task = gold_tasks(sample)[0]["data"]
    assert task["original"] == X
    assert task["idiomatic"] == "They buried the hatchet after the long war."
    assert task["span_literal"] == "They made peace after the long war."
    assert (task["idiom"], task["span"]) == ("Bury the hatchet", "stopped fighting")
    xml = gold_config_xml()
    for crit in (
        "idiomatic_figurative",
        "span_literal_slot_fit",
        "idiomatic_correction",
        "span_literal_action",
    ):
        assert f'name="{crit}"' in xml


def test_blind_tasks_hide_original_and_variant_name(tmp_path: Path) -> None:
    tasks = blind_tasks("sst2", _generated(tmp_path)[:2], 0)
    assert len(tasks) == 4
    xml = blind_config_xml("sst2")
    assert "$text" in xml
    assert "$original" not in xml
    assert "$variant" not in xml
    assert "$gold" not in xml
    assert "not sure" in xml


def _export() -> list[dict[str, Any]]:
    def ann(who: int, answers: dict[str, str]) -> dict[str, Any]:
        return {
            "completed_by": who,
            "result": [
                {"from_name": k, "type": "choices", "value": {"choices": [v]}}
                for k, v in answers.items()
            ],
        }

    return [
        {
            "data": {"id": "a"},
            "annotations": [
                ann(1, {"idiomatic_meaning": "yes"}),
                ann(2, {"idiomatic_meaning": "yes"}),
            ],
        },
        {
            "data": {"id": "b"},
            "annotations": [
                ann(1, {"idiomatic_meaning": "yes"}),
                ann(2, {"idiomatic_meaning": "no"}),
            ],
        },
        {
            "data": {"id": "c"},
            "annotations": [
                ann(1, {"idiomatic_meaning": "no"}),
                ann(2, {"idiomatic_meaning": "no"}),
            ],
        },
        {
            "data": {"id": "d"},
            "annotations": [
                ann(1, {"idiomatic_meaning": "yes"}),
                ann(2, {"idiomatic_meaning": "yes"}),
            ],
        },
    ]


def test_agreement_reports_kappa_per_criterion() -> None:
    (row,) = agreement(flatten_export(_export()))
    assert row["criterion"] == "idiomatic_meaning"
    assert row["n_items"] == 4
    assert row["agreement"] == pytest.approx(0.75)
    assert row["cohen_kappa"] == pytest.approx(0.5)


def test_blind_accuracy_counts_not_sure_as_wrong() -> None:
    def ann(who: int, value: str) -> dict[str, Any]:
        return {
            "completed_by": who,
            "result": [{"from_name": "label", "value": {"choices": [value]}}],
        }

    export = [
        {
            "data": {"id": "a", "variant": "idiomatic", "gold": "1"},
            "annotations": [ann(1, "positive")],
        },
        {
            "data": {"id": "b", "variant": "idiomatic", "gold": "0"},
            "annotations": [ann(1, "not sure")],
        },
    ]
    (rec,) = blind_accuracy(flatten_export(export), "sst2")
    assert (rec["n"], rec["accuracy"]) == (2, 0.5)
    json.dumps(export)


def test_label_studio_configs_are_well_formed_xml() -> None:
    ET.fromstring(gold_config_xml())
    for dataset in ("sst2", "mnli", "agnews"):
        ET.fromstring(blind_config_xml(dataset))
