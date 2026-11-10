"""Early stop at the kept-item target."""

from pathlib import Path
from typing import Any

from augmentation.idiom_generation import Item, generate, prompt_hash, span_prompt_hash
from tests._generation_helpers import GOOD, X, setup_inventory


def test_generate_stops_after_the_wave_that_reaches_the_target(tmp_path: Path) -> None:
    inv, index = setup_inventory(tmp_path)
    items = [Item("sst2", f"id{n}", X, "1", {}) for n in range(10)]
    seen: list[int] = []

    def complete(prompts: Any) -> list[str | None]:
        seen.append(len(prompts))
        return [GOOD for _ in prompts]

    out = generate(items, inv, index, complete, model="m", p_hash="ph", k=3, wave=4, target_kept=5)
    assert len(out) == 8
    assert seen == [4, 4]
    assert sum(o.ok for o in out) == 8


def test_items_that_already_contain_an_inventory_idiom_are_skipped_without_api_calls(
    tmp_path: Path,
) -> None:
    inv, index = setup_inventory(tmp_path)
    items = [
        Item("sst2", "idiomatic", "They buried the hatchet after the long war.", "1", {}),
        Item("sst2", "plain", X, "1", {}),
    ]
    sent: list[str] = []

    def complete(prompts: Any) -> list[str | None]:
        sent.extend(prompts)
        return [GOOD for _ in prompts]

    out = generate(items, inv, index, complete, model="m", p_hash="ph", k=3)
    assert [(o.item.id, o.reason) for o in out] == [
        ("idiomatic", "original_has_idiom"),
        ("plain", "ok"),
    ]
    assert len(sent) == 1 and X in sent[0]


def test_an_idiom_is_never_kept_more_often_than_the_cap_even_inside_one_wave(
    tmp_path: Path,
) -> None:
    inv, index = setup_inventory(tmp_path)
    items = [Item("sst2", f"id{n}", X, "1", {}) for n in range(5)]
    out = generate(
        items,
        inv,
        index,
        lambda ps: [GOOD for _ in ps],
        model="m",
        p_hash="ph",
        k=3,
        cap=2,
        wave=10,
    )
    assert [o.reason for o in out] == ["ok", "ok", "idiom_cap", "idiom_cap", "idiom_cap"]


def test_prompt_hashes_track_what_shapes_each_stage() -> None:
    assert prompt_hash("sst2", True) != prompt_hash("sst2", False)
    assert span_prompt_hash("sst2") == span_prompt_hash("sst2")
    assert span_prompt_hash("sst2") != span_prompt_hash("mnli")
