"""Command-line interface: ``fedsechealth <command> --config configs/xxx.yaml``."""

from __future__ import annotations

from pathlib import Path

import typer
import yaml
from rich.console import Console

from . import experiments
from .config import load_config

app = typer.Typer(add_completion=False, help="FedSecHealth: privacy & security testbed for FL.")
console = Console()

ConfigOpt = typer.Option(None, "--config", "-c", help="YAML experiment config.")
SetOpt = typer.Option(None, "--set", "-s", help="Override, e.g. -s fl.rounds=5 (repeatable).")


def _cfg(config: Path | None, overrides: list[str] | None):
    parsed = {}
    for item in overrides or []:
        key, _, value = item.partition("=")
        parsed[key] = yaml.safe_load(value)
    return load_config(config, parsed)


@app.command()
def train(config: Path | None = ConfigOpt, set_: list[str] | None = SetOpt) -> None:
    """Federated training vs. centralized and local-only baselines."""
    experiments.experiment_train(_cfg(config, set_), log=console.print)


@app.command()
def attack(config: Path | None = ConfigOpt, set_: list[str] | None = SetOpt) -> None:
    """Gradient inversion attacks (analytic, iDLG, DLG) with and without DP."""
    experiments.experiment_attack(_cfg(config, set_), log=console.print)


@app.command()
def tradeoff(config: Path | None = ConfigOpt, set_: list[str] | None = SetOpt) -> None:
    """Sweep the privacy budget epsilon: accuracy vs. attack success."""
    experiments.experiment_tradeoff(_cfg(config, set_), log=console.print)


@app.command()
def demo(
    config: Path | None = ConfigOpt,
    set_: list[str] | None = SetOpt,
    epsilon: float = typer.Option(5.0, help="Privacy budget for the DP column."),
) -> None:
    """Show a real patient record next to what a curious server reconstructs."""
    console.print(experiments.demo_reconstruction(_cfg(config, set_), epsilon=epsilon))


if __name__ == "__main__":
    app()
