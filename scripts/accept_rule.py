"""RQ1 accept-rule CLI — merges judge verdicts into one `accepted_pair` flag per row.

Usage:
    uv run idiom-accept-rule datasets_out/sst2/meaning_validated.csv
    uv run idiom-accept-rule datasets_out/sst2/meaning_validated.partial.csv
"""

import math
from pathlib import Path

import click

from analysis.meaning_preservation import (
    compute_accept_rule,
    judge_agreement_ac1,
    judge_agreement_kappa,
)


@click.command()
@click.argument("verdicts_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def main(verdicts_path: Path) -> None:
    """Compute the accept rule and inter-judge kappa for one dataset's verdicts."""
    df = compute_accept_rule(verdicts_path)
    n = len(df)
    n_accept = int(df["accepted_pair"].sum())  # pyright: ignore[reportArgumentType]
    out_path = verdicts_path.parent / "meaning_accept.csv"
    click.echo(f"wrote {out_path}")
    if n:
        click.echo(f"accepted_pair: {n_accept}/{n} rows ({n_accept / n:.1%})")
    else:
        click.echo("no rows found")

    try:
        kappa = judge_agreement_kappa(verdicts_path)
        ac1 = judge_agreement_ac1(verdicts_path)
    except ValueError as exc:
        click.echo(f"kappa: skipped ({exc})")
    else:
        if math.isnan(kappa):
            click.echo("agreement: no overlap yet")
        else:
            click.echo(
                f"judge x judge Gwet's AC1: {ac1:.3f} "
                f"[Cohen's kappa={kappa:.3f}, reference only -- unreliable under this skewed prevalence]"
            )


if __name__ == "__main__":
    main()
