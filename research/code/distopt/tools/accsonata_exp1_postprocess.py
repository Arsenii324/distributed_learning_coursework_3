"""Parse and plot outputs from `run_accsonata_exp1_sweeps.py`.

The sweep script prints a human-readable log. This tool turns it into:
- rows.csv: one row per (sweep, param, seed, algorithm)
- summary.csv: median across seeds per (sweep, param, algorithm)
- a couple PNG plots for quick comparison

Usage:
  python -m research.code.distopt.tools.accsonata_exp1_postprocess \
    --log research/artifacts/2026-04-01/accsonata_exp1/accsonata_exp1_sweep.txt \
    --outdir research/artifacts/2026-04-01/accsonata_exp1
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


_CASE_A_RE = re.compile(
    r"^\[n_local=(?P<n_local>\d+) seed=(?P<seed>\d+)\] beta/mu=(?P<beta_over_mu>[0-9.eE+-]+)\s*$"
)
_CASE_B_RE = re.compile(
    r"^\[lambda_reg=(?P<lambda_reg>[0-9.eE+-]+) seed=(?P<seed>\d+)\] kappa_g=(?P<kappa_g>[0-9.eE+-]+)\s*$"
)
_ALGO_RE = re.compile(
    r"^(?P<algo>[A-Za-z0-9\-]+):\s+mix=(?P<mix>\d+)\s+grad/node=(?P<grad_per_node>\d+)\s+reached=(?P<reached>True|False)\s+avg_sq=(?P<avg_sq>[0-9.eE+-]+)\s+gap=(?P<gap>[0-9.eE+-]+)\s*$"
)


@dataclass
class CaseContext:
    sweep: str
    seed: int
    n_local: int | None = None
    beta_over_mu: float | None = None
    lambda_reg: float | None = None
    kappa_g: float | None = None


def parse_log(text: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    ctx: CaseContext | None = None

    current_sweep: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("=== Sweep A"):
            current_sweep = "beta_over_mu"
            continue
        if line.startswith("=== Sweep B"):
            current_sweep = "kappa"
            continue

        m_a = _CASE_A_RE.match(line)
        if m_a:
            if current_sweep is None:
                current_sweep = "beta_over_mu"
            ctx = CaseContext(
                sweep=current_sweep,
                seed=int(m_a.group("seed")),
                n_local=int(m_a.group("n_local")),
                beta_over_mu=float(m_a.group("beta_over_mu")),
            )
            continue

        m_b = _CASE_B_RE.match(line)
        if m_b:
            if current_sweep is None:
                current_sweep = "kappa"
            ctx = CaseContext(
                sweep=current_sweep,
                seed=int(m_b.group("seed")),
                lambda_reg=float(m_b.group("lambda_reg")),
                kappa_g=float(m_b.group("kappa_g")),
            )
            continue

        m_algo = _ALGO_RE.match(line)
        if m_algo and ctx is not None:
            rows.append(
                {
                    "sweep": ctx.sweep,
                    "seed": ctx.seed,
                    "n_local": ctx.n_local,
                    "beta_over_mu": ctx.beta_over_mu,
                    "lambda_reg": ctx.lambda_reg,
                    "kappa_g": ctx.kappa_g,
                    "algorithm": m_algo.group("algo"),
                    "mix_rounds": int(m_algo.group("mix")),
                    "grad_evals_per_node": int(m_algo.group("grad_per_node")),
                    "reached": (m_algo.group("reached") == "True"),
                    "avg_sq": float(m_algo.group("avg_sq")),
                    "gap": float(m_algo.group("gap")),
                }
            )
            continue

    return pd.DataFrame(rows)


def _save_plots(df: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    # Sweep A: x = beta_over_mu (median per n_local), y = median mix_rounds.
    df_a = df[df["sweep"] == "beta_over_mu"].copy()
    if not df_a.empty:
        summary_a = (
            df_a.groupby(["n_local", "algorithm"], as_index=False)
            .agg(
                beta_over_mu=("beta_over_mu", "median"),
                mix_rounds=("mix_rounds", "median"),
                grad_evals_per_node=("grad_evals_per_node", "median"),
                reached_rate=("reached", "mean"),
            )
            .sort_values(["beta_over_mu", "algorithm"])
        )

        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        for algo, g in summary_a.groupby("algorithm"):
            ax.plot(g["beta_over_mu"], g["mix_rounds"], marker="o", label=algo)
        ax.set_xscale("log")
        ax.set_xlabel("median beta/mu_g (across seeds)")
        ax.set_ylabel("median mix_rounds to stop")
        ax.set_title("ACC-SONATA Exp1 Sweep A (vary n_local)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir / "sweepA_mix_vs_beta_over_mu.png", dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        for algo, g in summary_a.groupby("algorithm"):
            ax.plot(
                g["beta_over_mu"],
                g["grad_evals_per_node"],
                marker="o",
                label=algo,
            )
        ax.set_xscale("log")
        ax.set_xlabel("median beta/mu_g (across seeds)")
        ax.set_ylabel("median grad_evals_per_node")
        ax.set_title("ACC-SONATA Exp1 Sweep A (vary n_local)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir / "sweepA_grad_vs_beta_over_mu.png", dpi=200)
        plt.close(fig)

    # Sweep B: x = lambda_reg (given), y = median mix_rounds.
    df_b = df[df["sweep"] == "kappa"].copy()
    if not df_b.empty:
        summary_b = (
            df_b.groupby(["lambda_reg", "algorithm"], as_index=False)
            .agg(
                kappa_g=("kappa_g", "median"),
                mix_rounds=("mix_rounds", "median"),
                grad_evals_per_node=("grad_evals_per_node", "median"),
                reached_rate=("reached", "mean"),
            )
            .sort_values(["lambda_reg", "algorithm"])
        )

        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        for algo, g in summary_b.groupby("algorithm"):
            ax.plot(g["lambda_reg"], g["mix_rounds"], marker="o", label=algo)
        ax.set_xscale("log")
        ax.set_xlabel("lambda_reg")
        ax.set_ylabel("median mix_rounds to stop")
        ax.set_title("ACC-SONATA Exp1 Sweep B (vary lambda_reg)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir / "sweepB_mix_vs_lambda_reg.png", dpi=200)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    args = parser.parse_args()

    log_path = Path(args.log)
    outdir = Path(args.outdir)
    text = log_path.read_text(encoding="utf-8")

    df = parse_log(text)
    if df.empty:
        raise SystemExit("No rows parsed from log (format changed?)")

    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "rows.csv", index=False)

    summary_cols = ["sweep", "algorithm", "seed"]
    # Summary by sweep parameter.
    group_cols: list[str]
    summaries: list[pd.DataFrame] = []

    df_a = df[df["sweep"] == "beta_over_mu"].copy()
    if not df_a.empty:
        summaries.append(
            df_a.groupby(["n_local", "algorithm"], as_index=False)
            .agg(
                beta_over_mu=("beta_over_mu", "median"),
                mix_rounds=("mix_rounds", "median"),
                grad_evals_per_node=("grad_evals_per_node", "median"),
                reached_rate=("reached", "mean"),
            )
            .assign(sweep="beta_over_mu")
        )

    df_b = df[df["sweep"] == "kappa"].copy()
    if not df_b.empty:
        summaries.append(
            df_b.groupby(["lambda_reg", "algorithm"], as_index=False)
            .agg(
                kappa_g=("kappa_g", "median"),
                mix_rounds=("mix_rounds", "median"),
                grad_evals_per_node=("grad_evals_per_node", "median"),
                reached_rate=("reached", "mean"),
            )
            .assign(sweep="kappa")
        )

    if summaries:
        summary_df = pd.concat(summaries, ignore_index=True)
        summary_df.to_csv(outdir / "summary.csv", index=False)

    _save_plots(df, outdir)


if __name__ == "__main__":
    main()
