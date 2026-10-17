"""Unit tests for `analysis.meaning_preservation` — resume, cache, crash-safety.

Every test runs entirely offline via `_FakeJudgeClient` (no network) and uses
`tmp_path` so nothing touches the real `datasets_out/` or `.hf_cache/`.
"""

import csv
import json
from pathlib import Path

import pytest

from analysis.meaning_preservation import (
    CacheCfg,
    DecodingCfg,
    JudgeSpec,
    LimitsCfg,
    PairInput,
    RetryCfg,
    ValidateConfig,
    attach_audit_context,
    compute_accept_rule,
    finalize_partial_csv,
    heal_partial_csv,
    human_judge_agreement,
    judge_agreement_ac1,
    judge_agreement_kappa,
    load_aligned_pairs,
    parse_verdict,
    render_prompt,
    run_validation,
    stratify_audit_sample,
)
from analysis.prompts.loader import load_prompt
from analysis.stats import gwet_ac1
from augmentation.providers.base import EmptyResponseError, LLMError

# ---------- fake judge client ----------


class _FakeJudgeClient:
    """Stand-in for `LLMClient` — returns a scripted PASS/FAIL/error per call."""

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        responses: list[str] | str = "PASS\nlooks fine",
        errors_before_response: int = 0,
    ) -> None:
        self.provider = provider
        self.model = model
        self.calls = 0
        self._responses = responses if isinstance(responses, list) else [responses]
        self._errors_before_response = errors_before_response

    def complete(
        self, *, system: str, user: str, temperature: float, max_output_tokens: int
    ) -> str:
        self.calls += 1
        if self.calls <= self._errors_before_response:
            raise LLMError("transient 500")
        idx = min(self.calls - self._errors_before_response, len(self._responses)) - 1
        return self._responses[idx]


def _make_variant_csvs(dataset_dir: Path, n_rows: int = 3) -> None:
    """Write minimal `original.csv`, `paraphrase.csv`, `idiomatic.csv` under `dataset_dir`."""
    dataset_dir.mkdir(parents=True, exist_ok=True)
    columns = ["id", "variant", "x", "y", "meta", "augmenter_model", "prompt_hash", "validators"]
    for variant, prefix in (("original", "orig"), ("paraphrase", "par"), ("idiomatic", "idi")):
        path = dataset_dir / f"{variant}.csv"
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for i in range(n_rows):
                writer.writerow(
                    {
                        "id": f"row/{i}",
                        "variant": variant,
                        "x": f"{prefix} text {i}",
                        "y": "0",
                        "meta": "{}",
                        "augmenter_model": "gemini/fake" if variant != "original" else "",
                        "prompt_hash": "deadbeef" if variant != "original" else "",
                        "validators": "{}",
                    }
                )


def _make_cfg(
    *, tmp_path: Path, cache_enabled: bool = True, max_pairs: int | None = None
) -> ValidateConfig:
    return ValidateConfig(
        dataset="mmlu",
        prompt_file="meaning_preservation_v1.txt",
        judges=[
            JudgeSpec(provider="openai", model="gpt-5.6-sol"),
            JudgeSpec(provider="anthropic", model="claude-sonnet-4-5"),
        ],
        decoding=DecodingCfg(temperature=0.0, max_output_tokens=128),
        retry=RetryCfg(max_attempts=2, backoff_seconds=0.0),
        cache=CacheCfg(enabled=cache_enabled, dir=str(tmp_path / "cache")),
        limits=LimitsCfg(max_pairs=max_pairs),
        output_dir=str(tmp_path),
    )


def _patch_build_clients(
    monkeypatch: pytest.MonkeyPatch, clients: dict[str, _FakeJudgeClient]
) -> None:
    """Replace `_build_clients` so `run_validation` gets our fakes instead of real ones."""

    def fake_build_clients(judges: list[JudgeSpec]) -> dict[str, _FakeJudgeClient]:
        return {j.label(): clients[j.label()] for j in judges}

    def noop_check_env_keys(_judges: list[JudgeSpec]) -> None:
        return None

    monkeypatch.setattr("analysis.meaning_preservation._build_clients", fake_build_clients)
    monkeypatch.setattr("analysis.meaning_preservation._check_env_keys", noop_check_env_keys)


# ---------- parse_verdict ----------


def test_parse_verdict_pass() -> None:
    preserved, reason = parse_verdict("PASS\nMeaning intact.")
    assert preserved is True
    assert reason == "Meaning intact."


def test_parse_verdict_fail_with_multi_line_reason() -> None:
    preserved, reason = parse_verdict("FAIL\nAdds a hedge.\nAlso reorders.")
    assert preserved is False
    assert reason == "Adds a hedge. Also reorders."


def test_parse_verdict_case_insensitive() -> None:
    assert parse_verdict("pass\nok")[0] is True
    assert parse_verdict("fail\nnope")[0] is False


def test_parse_verdict_rejects_neither() -> None:
    with pytest.raises(EmptyResponseError):
        parse_verdict("MAYBE\nunsure")


def test_parse_verdict_rejects_empty() -> None:
    with pytest.raises(EmptyResponseError):
        parse_verdict("")


# ---------- prompt rendering ----------


def test_render_prompt_substitutes_all_slots() -> None:
    template = load_prompt("meaning_preservation_v1.txt")
    pair = PairInput(
        row_id="row/0",
        arm="idi",
        original_text="The train arrived on time.",
        rewrite_text="The train arrived on the dot.",
        meta={},
    )
    rendered = render_prompt(template, pair)
    assert "The train arrived on time." in rendered
    assert "The train arrived on the dot." in rendered
    assert "REWRITE_TYPE: idiomatic" in rendered
    # No unrendered placeholders left over.
    assert "{original}" not in rendered
    assert "{rewrite}" not in rendered
    assert "{context_block}" not in rendered


def test_render_prompt_par_arm_says_paraphrase() -> None:
    template = load_prompt("meaning_preservation_v1.txt")
    pair = PairInput(
        row_id="row/0",
        arm="par",
        original_text="a",
        rewrite_text="b",
        meta={},
    )
    assert "REWRITE_TYPE: paraphrase" in render_prompt(template, pair)


def test_render_prompt_renders_mnli_hypothesis() -> None:
    template = load_prompt("meaning_preservation_v1.txt")
    pair = PairInput(
        row_id="mnli/0",
        arm="idi",
        original_text="premise",
        rewrite_text="rewritten premise",
        meta={"hypothesis": "This is the fixed hypothesis."},
    )
    rendered = render_prompt(template, pair)
    assert "This is the fixed hypothesis." in rendered
    assert "CONTEXT" in rendered


def test_render_prompt_renders_mmlu_choices_with_answer_index() -> None:
    template = load_prompt("meaning_preservation_v1.txt")
    pair = PairInput(
        row_id="mmlu/0",
        arm="idi",
        original_text="Which planet is closest to the sun?",
        rewrite_text="Which planet sits nearest to the sun?",
        meta={"choices": ["Mercury", "Venus", "Earth", "Mars"], "answer_index": 0},
    )
    rendered = render_prompt(template, pair)
    assert "Mercury" in rendered
    assert "correct choice remains index 0" in rendered


def test_render_prompt_renders_mnli_hypothesis_premise() -> None:
    """RQ3 §4: mnli_hypothesis fixes the premise instead of the hypothesis."""
    template = load_prompt("meaning_preservation_v1.txt")
    pair = PairInput(
        row_id="mnli_hypothesis/0",
        arm="idi",
        original_text="hypothesis",
        rewrite_text="rewritten hypothesis",
        meta={"premise": "This is the fixed premise."},
    )
    rendered = render_prompt(template, pair)
    assert "This is the fixed premise." in rendered
    assert "CONTEXT" in rendered


def test_render_prompt_renders_mmlu_choices_question() -> None:
    """RQ3 §4: mmlu_choices fixes the question stem instead of the choices."""
    template = load_prompt("meaning_preservation_v1.txt")
    pair = PairInput(
        row_id="mmlu_choices/0",
        arm="idi",
        original_text="A) Mercury\nB) Venus\nC) Earth\nD) Mars",
        rewrite_text="A) The first rock from the sun\nB) Venus\nC) Earth\nD) Mars",
        meta={"question": "Which planet is closest to the sun?"},
    )
    rendered = render_prompt(template, pair)
    assert "Which planet is closest to the sun?" in rendered
    assert "correct choice still uniquely correct" in rendered


# ---------- input loading ----------


def test_load_aligned_pairs_emits_par_and_idi_per_aligned_row(tmp_path: Path) -> None:
    _make_variant_csvs(tmp_path / "mmlu", n_rows=3)
    pairs = load_aligned_pairs(tmp_path / "mmlu")
    # 3 aligned rows * 2 arms = 6 pairs
    assert len(pairs) == 6
    assert {p.arm for p in pairs} == {"par", "idi"}
    assert {p.row_id for p in pairs} == {"row/0", "row/1", "row/2"}


def test_load_aligned_pairs_drops_unaligned_rows(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "mmlu"
    _make_variant_csvs(dataset_dir, n_rows=3)
    # remove row/2 from paraphrase only → row/2 is no longer aligned
    par_path = dataset_dir / "paraphrase.csv"
    rows = list(csv.DictReader(par_path.open()))
    with par_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows[:-1])

    pairs = load_aligned_pairs(dataset_dir)
    assert {p.row_id for p in pairs} == {"row/0", "row/1"}


# ---------- heal_partial_csv ----------


def test_heal_partial_csv_truncates_torn_line(tmp_path: Path) -> None:
    p = tmp_path / "partial.csv"
    p.write_bytes(b"header\nrow1\nrow2_partial_no_newline")
    heal_partial_csv(p)
    assert p.read_bytes() == b"header\nrow1\n"


def test_heal_partial_csv_on_clean_file_is_noop(tmp_path: Path) -> None:
    p = tmp_path / "partial.csv"
    p.write_bytes(b"header\nrow1\n")
    heal_partial_csv(p)
    assert p.read_bytes() == b"header\nrow1\n"


def test_heal_partial_csv_on_empty_file_is_noop(tmp_path: Path) -> None:
    p = tmp_path / "partial.csv"
    p.write_bytes(b"")
    heal_partial_csv(p)
    assert p.read_bytes() == b""


# ---------- end-to-end via run_validation ----------


def test_run_validation_writes_partial_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _make_variant_csvs(tmp_path / "mmlu", n_rows=2)
    cfg = _make_cfg(tmp_path=tmp_path)
    clients = {
        "openai/gpt-5.6-sol": _FakeJudgeClient(
            provider="openai", model="gpt-5.6-sol", responses="PASS\nfine"
        ),
        "anthropic/claude-sonnet-4-5": _FakeJudgeClient(
            provider="anthropic", model="claude-sonnet-4-5", responses="PASS\nfine"
        ),
    }
    _patch_build_clients(monkeypatch, clients)

    stats = run_validation(cfg)
    # 2 rows x 2 arms x 2 judges = 8 pair-judgments.
    assert stats.total_pairs == 8
    assert stats.api_calls == 8
    assert stats.cache_hits == 0

    partial = tmp_path / "mmlu" / "meaning_validated.partial.csv"
    assert partial.exists()
    df_rows = list(csv.DictReader(partial.open()))
    assert len(df_rows) == 8
    assert {r["arm"] for r in df_rows} == {"par", "idi"}
    assert all(r["preserved"] == "1" for r in df_rows)


def test_run_validation_second_call_is_all_cache_hits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_variant_csvs(tmp_path / "mmlu", n_rows=2)
    cfg = _make_cfg(tmp_path=tmp_path)
    clients = {
        "openai/gpt-5.6-sol": _FakeJudgeClient(
            provider="openai", model="gpt-5.6-sol", responses="PASS\nfine"
        ),
        "anthropic/claude-sonnet-4-5": _FakeJudgeClient(
            provider="anthropic", model="claude-sonnet-4-5", responses="PASS\nfine"
        ),
    }
    _patch_build_clients(monkeypatch, clients)

    _ = run_validation(cfg)
    # Second call — no new API calls; every pair is either a cache hit or already
    # present in the partial CSV (which we skip).
    finalize_partial_csv(tmp_path / "mmlu")  # move partial → final
    # Clear the partial so the re-run walks judges again (simulates the "resume
    # after the previous run finalized" case): rebuild the partial from cache.
    # We keep the cache; expect ONLY cache hits.
    partial = tmp_path / "mmlu" / "meaning_validated.partial.csv"
    partial.unlink()
    stats2 = run_validation(cfg)
    assert stats2.api_calls == 0
    assert stats2.cache_hits == 8


def test_run_validation_resumes_after_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ctrl-C mid-run leaves a durable partial CSV; rerun completes without re-judging."""
    _make_variant_csvs(tmp_path / "mmlu", n_rows=3)
    cfg = _make_cfg(tmp_path=tmp_path, max_pairs=4)  # stop after 4 new API calls
    clients = {
        "openai/gpt-5.6-sol": _FakeJudgeClient(
            provider="openai", model="gpt-5.6-sol", responses="PASS\nfine"
        ),
        "anthropic/claude-sonnet-4-5": _FakeJudgeClient(
            provider="anthropic", model="claude-sonnet-4-5", responses="PASS\nfine"
        ),
    }
    _patch_build_clients(monkeypatch, clients)

    stats1 = run_validation(cfg)
    assert stats1.api_calls == 4
    partial = tmp_path / "mmlu" / "meaning_validated.partial.csv"
    assert partial.exists()
    rows_after_crash = list(csv.DictReader(partial.open()))
    assert len(rows_after_crash) == 4

    # Rerun without the cap — should judge only the remaining pairs.
    cfg_full = _make_cfg(tmp_path=tmp_path, max_pairs=None)
    stats2 = run_validation(cfg_full)
    assert stats2.api_calls == 8  # 3 rows * 2 arms * 2 judges = 12 total; 4 done → 8 new
    assert stats2.cache_hits == 0

    final = finalize_partial_csv(tmp_path / "mmlu")
    df_final = list(csv.DictReader(final.open()))
    assert len(df_final) == 12  # every (id, arm, judge) present exactly once


def test_run_validation_retries_transient_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_variant_csvs(tmp_path / "mmlu", n_rows=1)
    cfg = _make_cfg(tmp_path=tmp_path)
    clients = {
        "openai/gpt-5.6-sol": _FakeJudgeClient(
            provider="openai",
            model="gpt-5.6-sol",
            responses="PASS\nfine",
            errors_before_response=1,  # one 500, then success
        ),
        "anthropic/claude-sonnet-4-5": _FakeJudgeClient(
            provider="anthropic", model="claude-sonnet-4-5", responses="PASS\nfine"
        ),
    }
    _patch_build_clients(monkeypatch, clients)

    stats = run_validation(cfg)
    # 1 row * 2 arms * 2 judges = 4 pairs; all succeed after retry.
    assert stats.skipped_rows == 0
    assert stats.api_calls == 4


def test_run_validation_skips_after_persistent_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_variant_csvs(tmp_path / "mmlu", n_rows=1)
    cfg = _make_cfg(tmp_path=tmp_path)
    # OpenAI judge fails every call; Anthropic succeeds.
    openai_client = _FakeJudgeClient(
        provider="openai", model="gpt-5.6-sol", responses="PASS\nfine", errors_before_response=999
    )
    clients = {
        "openai/gpt-5.6-sol": openai_client,
        "anthropic/claude-sonnet-4-5": _FakeJudgeClient(
            provider="anthropic", model="claude-sonnet-4-5", responses="PASS\nfine"
        ),
    }
    _patch_build_clients(monkeypatch, clients)

    stats = run_validation(cfg)
    # OpenAI: 2 pairs x 2 retries = 4 failing calls; both pairs skipped.
    # Anthropic: 2 successful pairs.
    assert stats.skipped_rows == 2
    assert stats.api_calls == 2  # only the anthropic successes count as api_calls


def test_run_validation_max_pairs_cap_stops_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_variant_csvs(tmp_path / "mmlu", n_rows=3)
    cfg = _make_cfg(tmp_path=tmp_path, max_pairs=2)
    clients = {
        "openai/gpt-5.6-sol": _FakeJudgeClient(
            provider="openai", model="gpt-5.6-sol", responses="PASS\nfine"
        ),
        "anthropic/claude-sonnet-4-5": _FakeJudgeClient(
            provider="anthropic", model="claude-sonnet-4-5", responses="PASS\nfine"
        ),
    }
    _patch_build_clients(monkeypatch, clients)

    stats = run_validation(cfg)
    assert stats.api_calls == 2
    # Partial CSV kept for the resume case.
    partial = tmp_path / "mmlu" / "meaning_validated.partial.csv"
    assert partial.exists()
    assert len(list(csv.DictReader(partial.open()))) == 2


def test_missing_api_key_env_raises_helpful_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`_check_env_keys` runs by default and errors on missing OPENAI_API_KEY."""
    _make_variant_csvs(tmp_path / "mmlu", n_rows=1)
    cfg = _make_cfg(tmp_path=tmp_path)
    # Do NOT patch `_check_env_keys` this time; ensure envs are cleared.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LLMError) as excinfo:
        run_validation(cfg)
    assert "OPENAI_API_KEY" in str(excinfo.value)
    assert "ANTHROPIC_API_KEY" in str(excinfo.value)


def test_progress_sidecar_written(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _make_variant_csvs(tmp_path / "mmlu", n_rows=1)
    cfg = _make_cfg(tmp_path=tmp_path)
    clients = {
        "openai/gpt-5.6-sol": _FakeJudgeClient(
            provider="openai", model="gpt-5.6-sol", responses="PASS\nfine"
        ),
        "anthropic/claude-sonnet-4-5": _FakeJudgeClient(
            provider="anthropic", model="claude-sonnet-4-5", responses="PASS\nfine"
        ),
    }
    _patch_build_clients(monkeypatch, clients)

    _ = run_validation(cfg)

    sidecar_path = tmp_path / "mmlu" / "judge_progress.json"
    assert sidecar_path.exists()
    payload = json.loads(sidecar_path.read_text())
    assert payload["dataset"] == "mmlu"
    assert set(payload["judges"]) == {"openai/gpt-5.6-sol", "anthropic/claude-sonnet-4-5"}
    assert payload["api_calls"] == 4


# ---------- accept rule, kappa, audit sample ----------

_JUDGE_A = ("openai", "gpt-5.6-sol")
_JUDGE_B = ("anthropic", "claude-sonnet-4-5")


def _write_verdicts_csv(path: Path, rows: list[tuple[str, str, tuple[str, str], bool]]) -> None:
    """Write a minimal long-format verdicts CSV: `(id, arm, (provider, model), preserved)`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "id",
        "arm",
        "judge_provider",
        "judge_model",
        "prompt_version",
        "prompt_hash",
        "preserved",
        "verdict_raw",
        "reason",
        "tokens_in_est",
        "tokens_out_est",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row_id, arm, (provider, model), preserved in rows:
            writer.writerow(
                {
                    "id": row_id,
                    "arm": arm,
                    "judge_provider": provider,
                    "judge_model": model,
                    "prompt_version": "v1",
                    "prompt_hash": "deadbeef",
                    "preserved": int(preserved),
                    "verdict_raw": "PASS" if preserved else "FAIL",
                    "reason": "ok" if preserved else "drift",
                    "tokens_in_est": 10,
                    "tokens_out_est": 2,
                }
            )


def test_compute_accept_rule_both_judges_pass_both_arms(tmp_path: Path) -> None:
    path = tmp_path / "sst2" / "meaning_validated.csv"
    _write_verdicts_csv(
        path,
        [
            ("row/0", "par", _JUDGE_A, True),
            ("row/0", "idi", _JUDGE_A, True),
            ("row/0", "par", _JUDGE_B, True),
            ("row/0", "idi", _JUDGE_B, True),
        ],
    )
    df = compute_accept_rule(path)
    assert len(df) == 1
    assert bool(df.loc[0, "accepted_pair"]) is True
    assert (path.parent / "meaning_accept.csv").exists()


def test_compute_accept_rule_one_fail_rejects_row(tmp_path: Path) -> None:
    path = tmp_path / "sst2" / "meaning_validated.csv"
    _write_verdicts_csv(
        path,
        [
            ("row/0", "par", _JUDGE_A, True),
            ("row/0", "idi", _JUDGE_A, False),
            ("row/0", "par", _JUDGE_B, True),
            ("row/0", "idi", _JUDGE_B, True),
        ],
    )
    df = compute_accept_rule(path)
    assert bool(df.loc[0, "accepted_pair"]) is False


def test_compute_accept_rule_missing_judge_verdict_rejects_row(tmp_path: Path) -> None:
    path = tmp_path / "sst2" / "meaning_validated.csv"
    _write_verdicts_csv(
        path,
        [
            ("row/0", "par", _JUDGE_A, True),
            ("row/0", "idi", _JUDGE_A, True),
            ("row/0", "par", _JUDGE_B, True),
            # openai skipped row/0:idi for judge B -> not yet accepted.
        ],
    )
    df = compute_accept_rule(path)
    assert bool(df.loc[0, "accepted_pair"]) is False


def test_compute_accept_rule_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        compute_accept_rule(tmp_path / "nope.csv")


def test_judge_agreement_kappa_perfect_agreement(tmp_path: Path) -> None:
    path = tmp_path / "sst2" / "meaning_validated.csv"
    _write_verdicts_csv(
        path,
        [
            ("row/0", "par", _JUDGE_A, True),
            ("row/0", "par", _JUDGE_B, True),
            ("row/1", "par", _JUDGE_A, False),
            ("row/1", "par", _JUDGE_B, False),
        ],
    )
    assert judge_agreement_kappa(path) == pytest.approx(1.0)


def test_judge_agreement_kappa_ignores_unmatched_pairs(tmp_path: Path) -> None:
    path = tmp_path / "sst2" / "meaning_validated.csv"
    _write_verdicts_csv(
        path,
        [
            ("row/0", "par", _JUDGE_A, True),
            ("row/0", "par", _JUDGE_B, True),
            ("row/1", "par", _JUDGE_A, False),
            # row/1 not yet judged by B -> excluded from the kappa computation.
        ],
    )
    assert judge_agreement_kappa(path) == pytest.approx(1.0)


def test_judge_agreement_kappa_requires_exactly_two_judges(tmp_path: Path) -> None:
    path = tmp_path / "sst2" / "meaning_validated.csv"
    _write_verdicts_csv(path, [("row/0", "par", _JUDGE_A, True)])
    with pytest.raises(ValueError, match="expected exactly 2 judges"):
        judge_agreement_kappa(path)


def test_judge_agreement_ac1_perfect_agreement(tmp_path: Path) -> None:
    path = tmp_path / "sst2" / "meaning_validated.csv"
    _write_verdicts_csv(
        path,
        [
            ("row/0", "par", _JUDGE_A, True),
            ("row/0", "par", _JUDGE_B, True),
            ("row/1", "par", _JUDGE_A, False),
            ("row/1", "par", _JUDGE_B, False),
        ],
    )
    assert judge_agreement_ac1(path) == pytest.approx(1.0)


def test_judge_agreement_ac1_requires_exactly_two_judges(tmp_path: Path) -> None:
    path = tmp_path / "sst2" / "meaning_validated.csv"
    _write_verdicts_csv(path, [("row/0", "par", _JUDGE_A, True)])
    with pytest.raises(ValueError, match="expected exactly 2 judges"):
        judge_agreement_ac1(path)


def test_stratify_audit_sample_buckets_and_sizes(tmp_path: Path) -> None:
    rows: list[tuple[str, str, tuple[str, str], bool]] = []
    # 2 both-accept rows, 2 disagree rows, 2 both-reject rows.
    for i in range(2):
        rid = f"accept/{i}"
        rows += [
            (rid, arm, judge, True) for arm in ("par", "idi") for judge in (_JUDGE_A, _JUDGE_B)
        ]
    for i in range(2):
        rid = f"disagree/{i}"
        rows += [(rid, arm, _JUDGE_A, True) for arm in ("par", "idi")]
        rows += [(rid, arm, _JUDGE_B, False) for arm in ("par", "idi")]
    for i in range(2):
        rid = f"reject/{i}"
        rows += [
            (rid, arm, judge, False) for arm in ("par", "idi") for judge in (_JUDGE_A, _JUDGE_B)
        ]
    path = tmp_path / "sst2" / "meaning_validated.csv"
    _write_verdicts_csv(path, rows)

    accept_df = compute_accept_rule(path)
    judge_labels = [f"{p}/{m}" for p, m in (_JUDGE_A, _JUDGE_B)]
    sample = stratify_audit_sample(
        accept_df,
        judge_labels=judge_labels,
        n_both_accept=2,
        n_disagree=2,
        n_both_reject=2,
        seed=0,
    )
    assert len(sample) == 6
    counts = sample["audit_stratum"].value_counts().to_dict()
    assert counts == {"both_accept": 2, "disagree": 2, "both_reject": 2}
    assert set(sample.loc[sample["audit_stratum"] == "both_accept", "id"]) == {
        "accept/0",
        "accept/1",
    }
    assert set(sample.loc[sample["audit_stratum"] == "disagree", "id"]) == {
        "disagree/0",
        "disagree/1",
    }
    assert set(sample.loc[sample["audit_stratum"] == "both_reject", "id"]) == {
        "reject/0",
        "reject/1",
    }


def test_stratify_audit_sample_raises_when_stratum_too_small(tmp_path: Path) -> None:
    path = tmp_path / "sst2" / "meaning_validated.csv"
    _write_verdicts_csv(
        path,
        [
            ("row/0", "par", _JUDGE_A, True),
            ("row/0", "idi", _JUDGE_A, True),
            ("row/0", "par", _JUDGE_B, True),
            ("row/0", "idi", _JUDGE_B, True),
        ],
    )
    accept_df = compute_accept_rule(path)
    judge_labels = [f"{p}/{m}" for p, m in (_JUDGE_A, _JUDGE_B)]
    with pytest.raises(ValueError, match="both_accept"):
        stratify_audit_sample(
            accept_df, judge_labels=judge_labels, n_both_accept=5, n_disagree=0, n_both_reject=0
        )


def test_attach_audit_context_splits_blind_sheet_and_scoring_key(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "sst2"
    _make_variant_csvs(dataset_dir, n_rows=1)
    path = dataset_dir / "meaning_validated.csv"
    _write_verdicts_csv(
        path,
        [
            ("row/0", "par", _JUDGE_A, True),
            ("row/0", "idi", _JUDGE_A, True),
            ("row/0", "par", _JUDGE_B, True),
            ("row/0", "idi", _JUDGE_B, True),
        ],
    )
    accept_df = compute_accept_rule(path)
    judge_labels = [f"{p}/{m}" for p, m in (_JUDGE_A, _JUDGE_B)]
    sample = stratify_audit_sample(
        accept_df, judge_labels=judge_labels, n_both_accept=1, n_disagree=0, n_both_reject=0
    )

    blind_sheet, scoring_key = attach_audit_context(sample, dataset_dir)

    assert list(blind_sheet["id"]) == ["row/0"]
    assert blind_sheet.loc[0, "original_text"] == "orig text 0"
    assert blind_sheet.loc[0, "paraphrase_text"] == "par text 0"
    assert blind_sheet.loc[0, "idiomatic_text"] == "idi text 0"
    assert blind_sheet.loc[0, "human_preserved_par"] == ""
    assert blind_sheet.loc[0, "human_preserved_idi"] == ""
    # Blind sheet must not leak any judge verdict — that's the whole point.
    assert not any("__" in c for c in blind_sheet.columns)

    # Scoring key keeps the judge verdicts, for later human-vs-judge kappa.
    assert list(scoring_key["id"]) == ["row/0"]
    judge_cols = [c for c in scoring_key.columns if "__" in c]
    assert len(judge_cols) == 4  # 2 judges x 2 arms
    assert all(scoring_key.loc[0, c] for c in judge_cols)


def test_human_judge_agreement_scores_each_judge(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "sst2"
    _make_variant_csvs(dataset_dir, n_rows=2)
    path = dataset_dir / "meaning_validated.csv"
    _write_verdicts_csv(
        path,
        [
            ("row/0", "par", _JUDGE_A, True),
            ("row/0", "idi", _JUDGE_A, True),
            ("row/0", "par", _JUDGE_B, True),
            ("row/0", "idi", _JUDGE_B, False),
            ("row/1", "par", _JUDGE_A, False),
            ("row/1", "idi", _JUDGE_A, False),
            ("row/1", "par", _JUDGE_B, False),
            ("row/1", "idi", _JUDGE_B, False),
        ],
    )
    accept_df = compute_accept_rule(path)
    judge_labels = [f"{p}/{m}" for p, m in (_JUDGE_A, _JUDGE_B)]
    sample = stratify_audit_sample(
        accept_df, judge_labels=judge_labels, n_both_accept=0, n_disagree=1, n_both_reject=1
    )
    blind_sheet, scoring_key = attach_audit_context(sample, dataset_dir)
    blind_sheet["human_preserved_par"] = "TRUE"
    blind_sheet["human_preserved_idi"] = "FALSE"

    sheet_path = dataset_dir / "audit_sample.csv"
    key_path = dataset_dir / "audit_sample_key.csv"
    blind_sheet.to_csv(sheet_path, index=False)
    scoring_key.to_csv(key_path, index=False)

    result = human_judge_agreement(sheet_path, key_path)

    assert set(result["judge_label"]) == set(judge_labels)
    assert (result["n_rated"] == 4).all()  # 2 rows x 2 arms, all labeled
    judge_a_row = result.set_index("judge_label").loc["openai/gpt-5.6-sol"]
    # Human said par=True/idi=False for both rows; judge A said
    # row/0=(True,True) and row/1=(False,False) -> agree on 2/4 (idi/row0, par/row1 mismatch).
    assert judge_a_row["raw_agreement"] == pytest.approx(0.5)
    assert "ac1" in result.columns
    human_vals = [True, True, False, False]  # par: row0,row1 ; idi: row0,row1
    judge_a_vals = [True, False, True, False]  # par: row0,row1 ; idi: row0,row1
    assert judge_a_row["ac1"] == pytest.approx(gwet_ac1(human_vals, judge_a_vals))
