"""RQ1 human-audit sampler — stratified 500-row sample (RQ1 plan §5 step 6).

Draws 200 both-accept + 200 disagree + 100 both-reject rows and writes:
- `audit_sample.csv` — blind sheet (text + empty human-label columns) for the
  author to fill in by hand. Judge verdicts are NOT included.
- `audit_sample_key.csv` — the judge verdicts for the same rows, for scoring
  human-vs-judge kappa afterwards. Do not open this while labeling.

Usage:
    uv run idiom-audit-sample datasets_out/sst2/meaning_validated.csv
"""

from pathlib import Path

import click

from analysis.meaning_preservation import (
    attach_audit_context,
    compute_accept_rule,
    stratify_audit_sample,
)


@click.command()
@click.argument("verdicts_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--seed", type=int, default=0, help="RNG seed for the stratified sample.")
@click.option("--n-both-accept", type=int, default=200, show_default=True)
@click.option("--n-disagree", type=int, default=200, show_default=True)
@click.option("--n-both-reject", type=int, default=100, show_default=True)
def main(
    verdicts_path: Path,
    seed: int,
    n_both_accept: int,
    n_disagree: int,
    n_both_reject: int,
) -> None:
    """Draw and write the stratified human-audit sample for one dataset."""
    accept_df = compute_accept_rule(verdicts_path)
    judge_labels = sorted({c.rsplit("__", 1)[0] for c in accept_df.columns if "__" in c})
    sample = stratify_audit_sample(
        accept_df,
        judge_labels=judge_labels,
        n_both_accept=n_both_accept,
        n_disagree=n_disagree,
        n_both_reject=n_both_reject,
        seed=seed,
    )
    blind_sheet, scoring_key = attach_audit_context(sample, verdicts_path.parent)

    sheet_path = verdicts_path.parent / "audit_sample.csv"
    key_path = verdicts_path.parent / "audit_sample_key.csv"
    blind_sheet.to_csv(sheet_path, index=False)
    scoring_key.to_csv(key_path, index=False)
    click.echo(f"wrote {len(blind_sheet)} blind audit rows to {sheet_path}")
    click.echo(f"wrote scoring key (do not open while labeling) to {key_path}")


if __name__ == "__main__":
    main()
