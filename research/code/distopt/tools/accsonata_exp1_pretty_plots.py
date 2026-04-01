"""Make nicer ACC-SONATA Exp.1 sweep plots (Russian + mathtext).

Reads existing CSV artifacts (does not overwrite original plots/data).

Usage:
  python -m research.code.distopt.tools.accsonata_exp1_pretty_plots \
    --summary research/artifacts/2026-04-01/accsonata_exp1/summary.csv \
    --outdir  research/artifacts/2026-04-01/accsonata_exp1/pretty
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


def _apply_style() -> None:
    # Matplotlib's default DejaVu Sans typically supports Cyrillic.
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 220,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linestyle": "-",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 2.0,
            "lines.markersize": 6.0,
        }
    )


def _algo_order(algo: str) -> tuple[int, str]:
    order = {
        "ACC-SONATA-F": 0,
        "ACC-SONATA-L": 1,
        "EXTRA": 2,
        "GradientTracking": 3,
        "MUDAG": 4,
    }
    return (order.get(algo, 99), algo)


def plot_sweep_a(summary: pd.DataFrame, outdir: Path, *, tag: str) -> None:
    a = summary[summary["sweep"] == "beta_over_mu"].copy()
    if a.empty:
        return

    a = a.sort_values(["beta_over_mu", "algorithm"], key=lambda s: s)

    def _plot(metric: str, ylabel: str, filename: str) -> None:
        fig, ax = plt.subplots(figsize=(7.6, 4.4))
        for algo in sorted(a["algorithm"].unique(), key=_algo_order):
            g = a[a["algorithm"] == algo]
            ax.plot(g["beta_over_mu"], g[metric], marker="o", label=algo)

        ax.set_xscale("log")
        ax.set_xlabel(r"медиана $\beta/\mu_g$ (по сидам)")
        ax.set_ylabel(ylabel)
        ax.set_title(r"ACC-SONATA Exp.1 — Sweep A (от $n_{local}$)")
        ax.legend(loc="best", frameon=True)
        fig.tight_layout()
        fig.savefig(outdir / filename)
        plt.close(fig)

    _plot(
        metric="mix_rounds",
        ylabel="медиана раундов связи до остановки",
        filename=f"sweepA_mix_vs_beta_over_mu_{tag}.png",
    )
    _plot(
        metric="grad_evals_per_node",
        ylabel="медиана градиентных вызовов на узел",
        filename=f"sweepA_grad_vs_beta_over_mu_{tag}.png",
    )


def plot_sweep_b(summary: pd.DataFrame, outdir: Path, *, tag: str) -> None:
    b = summary[summary["sweep"] == "kappa"].copy()
    if b.empty:
        return

    b = b.sort_values(["lambda_reg", "algorithm"], key=lambda s: s)

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for algo in sorted(b["algorithm"].unique(), key=_algo_order):
        g = b[b["algorithm"] == algo]
        ax.plot(g["lambda_reg"], g["mix_rounds"], marker="o", label=algo)

    ax.set_xscale("log")
    ax.set_xlabel(r"$\lambda_{reg}$")
    ax.set_ylabel("медиана раундов связи до остановки")
    ax.set_title(r"ACC-SONATA Exp.1 — Sweep B (от $\lambda_{reg}$)")
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(outdir / f"sweepB_mix_vs_lambda_reg_{tag}.png")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=str, required=True)
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument(
        "--tag",
        type=str,
        default="ru",
        help="Filename tag to avoid overwriting (e.g. ru2)",
    )
    args = ap.parse_args()

    summary_path = Path(args.summary)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    _apply_style()

    summary = pd.read_csv(summary_path)
    tag = str(args.tag)
    plot_sweep_a(summary, outdir, tag=tag)
    plot_sweep_b(summary, outdir, tag=tag)


if __name__ == "__main__":
    main()
