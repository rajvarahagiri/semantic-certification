from __future__ import annotations

import json
from pathlib import Path
import typer

from .budget import optimistic_budget_check
from .collect import collect_probe
from .config import load_config
from .pin_models import pin_models
from .probe import run_probe
from .test0 import run_test0

app = typer.Typer(no_args_is_help=True, help="Semantic Certification Controller Phase 0")


@app.command("pin-models")
def pin_models_cmd(
    config: str = typer.Option("config/phase0.yaml", "--config"),
    out: str = typer.Option("artifacts/model_lock.json", "--out"),
):
    cfg = load_config(config)
    lock = pin_models(cfg.models["full_model"]["repo_id"], cfg.models["gate_model"]["repo_id"], out)
    typer.echo(json.dumps(lock, indent=2))


@app.command("collect-probe")
def collect_probe_cmd(
    config: str = typer.Option("config/phase0.yaml", "--config"),
    project_dir: str = typer.Option(".", "--project-dir"),
    limit: int | None = typer.Option(None, "--limit"),
):
    cfg = load_config(config)
    summary = collect_probe(cfg, Path(project_dir) / "data", limit=limit)
    typer.echo(json.dumps(summary, indent=2))


@app.command("run-probe")
def run_probe_cmd(
    config: str = typer.Option("config/phase0.yaml", "--config"),
    project_dir: str = typer.Option(".", "--project-dir"),
    mock: bool = typer.Option(False, "--mock", help="Test-only; never use mock results for Phase 0."),
):
    cfg = load_config(config)
    summary = run_probe(cfg, project_dir, mock=mock)
    typer.echo(json.dumps(summary, indent=2))


@app.command("test0")
def test0_cmd(
    project_dir: str = typer.Option(".", "--project-dir"),
):
    result = run_test0(project_dir)
    typer.echo(json.dumps(result, indent=2))


@app.command("budget-check")
def budget_check_cmd(
    lead_changes_per_day: float = typer.Option(...),
    populations: int = typer.Option(...),
    overlap_factor: float = typer.Option(1.0),
    epsilon: float = typer.Option(0.005),
    budget_fraction: float = typer.Option(0.05),
):
    result = optimistic_budget_check(
        lead_changes_per_day, budget_fraction, epsilon, populations, overlap_factor
    )
    typer.echo(json.dumps(result, indent=2))


if __name__ == "__main__":
    app()
