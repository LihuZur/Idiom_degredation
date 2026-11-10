"""Plan item counts: simulated power of the primary comparison from item-level pilot data."""

from pathlib import Path

import click

from analysis.io import write_markdown, write_table
from analysis.power import power_table
from analysis.results import discover_results, load_result


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--results",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("results"),
    show_default=True,
)
@click.option("--out-dir", type=click.Path(path_type=Path), default=Path("analysis/tables"))
@click.option("--datasets", default="sst2,mnli,mmlu", show_default=True)
@click.option("--control", default="paraphrase", show_default=True, help="Control variant name.")
@click.option("--treat", default="idiomatic", show_default=True)
@click.option("--n-grid", default="500,1000,1500,1800", show_default=True)
@click.option("--effects-pp", default="0.5,1.0,2.0", show_default=True)
@click.option("--alpha", type=float, default=0.05, show_default=True)
@click.option(
    "--holm-k", type=int, default=4, show_default=True, help="Datasets in the Holm family."
)
@click.option("--target-power", type=float, default=0.8, show_default=True)
@click.option("--reps", type=int, default=2000, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
def main(
    results: Path,
    out_dir: Path,
    datasets: str,
    control: str,
    treat: str,
    n_grid: str,
    effects_pp: str,
    alpha: float,
    holm_k: int,
    target_power: float,
    reps: int,
    seed: int,
) -> None:
    """Power vs item count for planted effects, at the Holm-adjusted worst-case alpha."""
    loaded = [load_result(p) for p in discover_results(results)]
    table = power_table(
        loaded,
        [d.strip() for d in datasets.split(",") if d.strip()],
        control=control,
        treat=treat,
        n_grid=[int(x) for x in n_grid.split(",")],
        effects_pp=[float(x) for x in effects_pp.split(",")],
        alpha=alpha / holm_k,
        target_power=target_power,
        reps=reps,
        seed=seed,
    )
    write_table(table, out_dir / "power_simulation.csv")
    write_markdown(table.round(4), out_dir / "power_simulation.md")
    click.echo(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
