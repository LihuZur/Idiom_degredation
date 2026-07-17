"""Per-dataset cross-model Plotly figure over all three variants.

Scaffold stub. See ARCHITECTURE §2.7 ("per-dataset cross-model view").
"""

import click


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--dataset", type=str, required=True, help="Registered dataset name.")
def main(dataset: str) -> None:
    """Build the per-dataset cross-model Plotly figure."""
    raise NotImplementedError(f"scripts/plot_dataset.py is a scaffold stub (dataset: {dataset})")


if __name__ == "__main__":
    main()
