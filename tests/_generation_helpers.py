"""Shared fixtures for the generation tests: a three-idiom inventory and one good model reply."""

import csv
import json
from pathlib import Path
from typing import Any

from augmentation.idiom_generation import load_inventory
from data.norms.idiom_text import IdiomIndex

FIELDS = [
    "norm_idiom",
    "norm_source",
    "midas_ids",
    "midas_forms",
    "definitions",
    "example_sentences",
    "opacity",
    "familiarity",
    "min_word_count_enwiki",
]
IDIOMS = [
    ("Bury the hatchet", "1", "bury the hatchet", "To make peace.", 0.2),
    ("Spill the beans", "2", "spill the beans", "To reveal a secret.", 0.5),
    ("Break the ice", "3", "break the ice", "To ease social tension.", 0.8),
]
X = "They stopped fighting after the long war."


def write_inventory(tmp: Path) -> tuple[Path, Path]:
    final = tmp / "final.csv"
    with final.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for name, mid, form, defin, op in IDIOMS:
            w.writerow(
                {
                    "norm_idiom": name,
                    "norm_source": "bulkes_tanner_2017",
                    "midas_ids": mid,
                    "midas_forms": json.dumps([form]),
                    "definitions": json.dumps([defin]),
                    "example_sentences": json.dumps([f"Example of {form}."]),
                    "opacity": op,
                    "familiarity": "4.0",
                    "min_word_count_enwiki": "1000",
                }
            )
    domains = tmp / "domains.csv"
    domains.write_text(
        "idiom_id,idiom,domain\n1,bury,war-politics\n2,spill,everyday\n3,break,sports\n"
    )
    return final, domains


def setup_inventory(tmp_path: Path) -> tuple[list[Any], IdiomIndex]:
    final, domains = write_inventory(tmp_path)
    inv, rows = load_inventory(final, domains_csv=domains)
    return inv, IdiomIndex(rows, min_tokens=2, min_content=1)


GOOD = json.dumps(
    {
        "choice": "1",
        "span": "stopped fighting",
        "idiom_text": "buried the hatchet",
        "literal_text": "made peace",
    }
)
