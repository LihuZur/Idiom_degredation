import json

from data.norms.domains import IdiomLine, batches, parse_domains, render, tag_all


def _lines(n: int) -> list[IdiomLine]:
    return [IdiomLine(str(i), f"idiom {i}", "meaning") for i in range(n)]


def test_parse_keeps_only_valid_ids_and_domains() -> None:
    raw = 'Sure: {"1": "Sports", "2": "cooking", "9": "everyday", "3": "war-politics"}'
    assert parse_domains(raw, ["1", "2", "3"]) == {"1": "sports", "3": "war-politics"}
    assert parse_domains("no json", ["1"]) == {}


def test_batches_and_prompt() -> None:
    assert [len(b) for b in batches(_lines(95))] == [40, 40, 15]
    assert "0: idiom 0 - meaning" in render(_lines(2))


def test_tag_all_reasks_for_missing_ids() -> None:
    calls: list[str] = []

    def complete(prompt: str) -> str:
        calls.append(prompt)
        ids = [line.split(":")[0] for line in prompt.splitlines() if ": idiom " in line]
        # the first round forgets id "1"
        answer = {i: "everyday" for i in ids if not (len(calls) == 1 and i == "1")}
        return json.dumps(answer)

    tags = tag_all(_lines(3), complete)
    assert set(tags) == {"0", "1", "2"}
    assert len(calls) == 2
