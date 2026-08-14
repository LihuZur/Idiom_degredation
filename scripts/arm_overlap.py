"""Per-arm rewrite depth: is the paraphrase arm matched to the idiomatic arm on how
much text it actually changes? This is Yang et al.'s objection to paraphrase
baselines, and the answer is no -- the idiomatic arm is the shallower edit on all
three datasets (report §4.1, "Cross-dataset magnitudes carry an augmenter confound").
"""

import re
from pathlib import Path

import click
import numpy as np
import pandas as pd

VARIANTS = ("original", "paraphrase", "idiomatic")

# `\w+` is the tokenisation that reproduces the paraphrase-arm figures already in the
# report (0.33 MNLI / 0.39 SST-2 / 0.54 MMLU). Do not change it without rechecking
# those three numbers in main.tex: other tokenisations move SST-2 to 0.38 and MMLU
# from 0.54 to 0.50.
_WORD = re.compile(r"\w+")


def _tokens(text: object) -> list[str]:
    return _WORD.findall(str(text).lower())


def _jaccard(a: object, b: object) -> float:
    """Token-set overlap; 1.0 for two empty strings, which have nothing to differ on."""
    left, right = set(_tokens(a)), set(_tokens(b))
    union = left | right
    return len(left & right) / len(union) if union else 1.0


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--datasets-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("datasets_out"),
    show_default=True,
    help="Stage 1/2 output root containing {dataset}/{variant}.csv.",
)
def main(datasets_dir: Path) -> None:
    """Print median rewrite depth per arm, over the rows aligned across all variants."""
    for dataset in ("sst2", "mmlu", "mnli"):
        # `x` is exactly the rewritten field for all three datasets (SST-2 sentence,
        # MMLU stem, MNLI premise); everything else lives in `meta`.
        frames = {
            variant: pd.read_csv(datasets_dir / dataset / f"{variant}.csv").set_index("id")
            for variant in VARIANTS
        }
        ids = frames["original"].index
        for variant in ("paraphrase", "idiomatic"):
            ids = ids.intersection(frames[variant].index)

        original = frames["original"].loc[ids].x
        orig_len = np.median([len(_tokens(text)) for text in original])
        click.echo(f"{dataset.upper():5s} n={len(ids):5d}  orig_med_tokens={orig_len:.0f}")

        for arm in ("paraphrase", "idiomatic"):
            rewritten = frames[arm].loc[ids].x
            overlap = np.median([_jaccard(a, b) for a, b in zip(original, rewritten, strict=True)])
            delta = np.median(
                [
                    len(_tokens(b)) - len(_tokens(a))
                    for a, b in zip(original, rewritten, strict=True)
                ]
            )
            length = np.median([len(_tokens(text)) for text in rewritten])
            click.echo(
                f"   {arm:11s} median Jaccard w/ orig = {overlap:.3f}"
                f"   median token delta = {delta:+.0f}   median len = {length:.0f}"
            )

        # Direct paraphrase-vs-idiomatic overlap, for reference: the two arms are
        # rewrites of the same source, not of each other.
        cross = np.median(
            [
                _jaccard(a, b)
                for a, b in zip(
                    frames["paraphrase"].loc[ids].x,
                    frames["idiomatic"].loc[ids].x,
                    strict=True,
                )
            ]
        )
        click.echo(f"   para vs idi median Jaccard = {cross:.3f}")


if __name__ == "__main__":
    main()
