"""RQ1 human-vs-judge kappa scorer (RQ1 plan §5 step 6, H1e).

Reads a labeled `audit_sample.csv` (blind sheet, hand-filled) and its
`audit_sample_key.csv` sidecar (judge verdicts for the same rows), and prints
Cohen's kappa and Gwet's AC1 between the human labels and each judge.

Usage:
    uv run idiom-audit-score datasets_out/sst2/audit_sample.csv
"""

from pathlib import Path

import click

from analysis.meaning_preservation import human_judge_agreement


@click.command()
@click.argument("audit_sheet_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def main(audit_sheet_path: Path) -> None:
    """Score a labeled human-audit sheet against the judges that produced it."""
    audit_key_path = audit_sheet_path.parent / "audit_sample_key.csv"
    result = human_judge_agreement(audit_sheet_path, audit_key_path)
    for _, row in result.iterrows():
        n_rated = int(row["n_rated"])  # pyright: ignore[reportArgumentType]
        click.echo(
            f"{row['judge_label']}: ac1={row['ac1']:.3f} "
            f"raw_agreement={row['raw_agreement']:.1%} n={n_rated} "
            f"(human_accept={row['human_accept_rate']:.1%}, "
            f"judge_accept={row['judge_accept_rate']:.1%}) "
            f"[kappa={row['kappa']:.3f}, reference only -- unreliable under this skewed prevalence]"
        )


if __name__ == "__main__":
    main()
