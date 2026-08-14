"""Smoke tests: package imports, registries start empty, device helper works."""

from pathlib import Path

import pytest
import torch

from analysis.aggregate import aggregate
from analysis.plots import plot_cross_dataset_summary, plot_dataset_cross_model
from augmentation import registry as aug_reg
from data import registry as data_reg
from models import registry as model_reg
from models.device import select_device

_N_RESAMPLES = 200
_CI = 0.95
_SEED = 0


# The 17 models the report evaluates. New registrations may be added freely, but
# none of these may disappear: the published results depend on them. Asserting a
# subset rather than the exact list is deliberate — an exact list went stale every
# time a model was registered, which is what this test used to fail on.
_REPORTED_MODELS = frozenset(
    {
        "falcon3-3b-instruct",
        "falcon3-7b-instruct",
        "gemma-4-2b-instruct",
        "gemma-4-9b-instruct",
        "granite-3.1-2b-instruct",
        "h2o-danube3-4b-chat",
        "mistral-7b-instruct-v0.3",
        "olmo-2-1b-instruct",
        "olmo-2-7b-instruct",
        "phi-4-mini-instruct",
        "qwen3.5-0.5b-instruct",
        "qwen3.5-1.5b-instruct",
        "qwen3.5-3b-instruct",
        "qwen3.5-7b-instruct",
        "smollm2-1.7b-instruct",
        "stablelm-2-1.6b-chat",
        "yi-1.5-6b-chat",
    }
)


def test_dataset_registry_has_sst2_mmlu_and_mnli() -> None:
    assert set(data_reg.list_datasets()) == {"sst2", "mmlu", "mnli"}
    models = model_reg.list_models()
    assert models == sorted(models), "list_models() must return sorted ids"
    assert len(models) == len(set(models)), "duplicate model ids registered"
    missing = sorted(_REPORTED_MODELS - set(models))
    assert not missing, f"models used in the report are no longer registered: {missing}"
    assert aug_reg.list_augmenters() == ["anthropic", "gemini", "openai"]
    assert set(aug_reg.list_validators()) == {
        "semantic_similarity",
        "label_preservation",
        "idiom_presence",
        "idiom_absence",
    }


def test_select_device_returns_valid_torch_device() -> None:
    dev = select_device()
    assert isinstance(dev, torch.device)
    assert dev.type in {"cuda", "mps", "cpu"}


@pytest.mark.xfail(
    reason=(
        "Stale since real augmentation landed (R12). Two independent problems: the "
        "identity-phase premise is no longer true — paraphrase/idiomatic now differ "
        "from original, so deltas are not zero — and `results/` has no tracked files, "
        "so this reads whatever happens to be on the local disk. Kept as xfail rather "
        "than skip so it flags if it ever starts passing. Real fix is separate scope."
    ),
    strict=False,
)
def test_stage4_over_committed_results_is_identity_phase(tmp_path: Path) -> None:
    """End-to-end Stage 4 over the committed `results/` dir (real identity-phase data).

    In the identity augmentation phase, paraphrase/idiomatic variants are
    identical to the original text, so every model's outputs — and hence
    correctness — match across variants exactly. Every delta should
    therefore be exactly zero with a p-value of 1.0.
    """
    results_dir = Path("results")

    df = aggregate(results_dir, n_resamples=_N_RESAMPLES, ci=_CI, seed=_SEED)

    assert not df.empty
    assert (df["delta_paraphrase"] == 0.0).all()
    assert (df["delta_idiom"] == 0.0).all()
    assert (df["delta_paraphrase_p"] == 1.0).all()
    assert (df["delta_idiom_p"] == 1.0).all()

    out_dir = tmp_path / "analysis_out"
    dataset = sorted(df["dataset"].unique())[0]

    dataset_html = plot_dataset_cross_model(
        dataset, results_dir, out_dir=out_dir, n_resamples=_N_RESAMPLES, ci=_CI, seed=_SEED
    )
    summary_html = plot_cross_dataset_summary(
        results_dir, out_dir=out_dir, n_resamples=_N_RESAMPLES, ci=_CI, seed=_SEED
    )

    assert dataset_html.exists()
    assert summary_html.exists()
    assert list(out_dir.rglob("*.png")) == []
