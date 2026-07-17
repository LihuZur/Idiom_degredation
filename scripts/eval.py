"""Stage 3 CLI — run one model on a dataset's variant triple.

Scaffold stub. See README §10 and ARCHITECTURE §2.6.
"""

import click


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--dataset", type=str, required=True, help="Registered dataset name.")
@click.option(
    "--model",
    type=str,
    required=True,
    help="Registered model id (see models/registry.py).",
)
def main(dataset: str, model: str) -> None:
    """Stage 3: evaluate one model on original + paraphrase + idiomatic."""
    raise NotImplementedError(
        f"scripts/eval.py is a scaffold stub (dataset: {dataset}, model: {model})"
    )


if __name__ == "__main__":
    main()
