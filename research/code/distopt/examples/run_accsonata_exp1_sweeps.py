from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..accsonata_generators import RidgeSpec, make_accsonata_er_graph, make_accsonata_ridge_problem
from ..algorithms import AccSonataF, AccSonataL, EXTRA, GradientTracking, MUDAG
from ..problems import DistributedQuadraticProblem
from ..runner import MaxMixRounds, TargetAvgSqDistToXStarAllNodes, run_experiment


@dataclass(frozen=True)
class ExpConfig:
    m_agents: int
    p_edge: float
    lazy_for_psd: float
    d: int
    noise_std: float
    mu0: float
    L0: float
    eps_sq: float
    max_mix_rounds: int
    log_every: int
    acc_eps_stop: float
    mudag_c_K: float


def _parse_int_list(s: str) -> list[int]:
    if not s.strip():
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _pick_step_sizes(problem) -> dict[str, float]:
    stats = problem.ensure_stats()
    # Use a conservative bound for stability in heterogeneous regimes.
    L = float(stats.L_l)
    return {
        "extra": 0.2 / L,
        "gt": 0.1 / L,
    }


def _cheb_steps(problem) -> int:
    chi = float(problem.graph.ensure_stats().chi)
    return max(2, int(np.floor(np.sqrt(chi))))


def _avg_sq_reached(res, *, eps_sq: float) -> bool:
    return float(res.final["avg_sq_dist_to_x_star_all_nodes"]) <= float(eps_sq)


def _run_suite(problem, problem_psd, *, cfg: ExpConfig, seed: int) -> list[dict[str, float | int | str]]:
    steps = _pick_step_sizes(problem)

    cheb_steps = _cheb_steps(problem)

    algs: list[tuple[str, Any, DistributedQuadraticProblem]] = []

    # Baselines (minimal tuning; consistent with the LIBSVM demo).
    algs.append(("EXTRA", EXTRA(alpha=float(steps["extra"])), problem))
    algs.append(("GradientTracking", GradientTracking(alpha=float(steps["gt"])), problem))

    # MUDAG requires PSD mixing, so run it on the lazified problem.
    algs.append(
        (
            "MUDAG",
            MUDAG(c_K=float(cfg.mudag_c_K), fastmix_stop_eps_sq=float(cfg.eps_sq)),
            problem_psd,
        )
    )

    # ACC-SONATA-L: primary curve.
    algs.append(
        (
            "ACC-SONATA-L",
            AccSonataL(cheb_steps=int(cheb_steps), eps_stop=float(cfg.acc_eps_stop)),
            problem,
        )
    )

    # ACC-SONATA-F: include only when beta > mu_g holds (required by check()).
    pstats = problem.ensure_stats()
    if float(pstats.beta) > float(pstats.mu_g):
        algs.append(
            (
                "ACC-SONATA-F",
                AccSonataF(cheb_steps=int(cheb_steps), eps_stop=float(cfg.acc_eps_stop)),
                problem,
            )
        )

    stop = [TargetAvgSqDistToXStarAllNodes(cfg.eps_sq), MaxMixRounds(cfg.max_mix_rounds)]

    rows: list[dict[str, float | int | str]] = []
    for name, alg, prob in algs:
        X0 = np.zeros((prob.n, prob.d), dtype=np.float64)
        res = run_experiment(prob, alg, stop=stop, seed=seed, X0=X0, log_every=int(cfg.log_every))
        rows.append(
            {
                "alg": str(name),
                "mix_rounds": int(res.final["mix_rounds"]),
                "grad_evals_per_node": int(res.final["grad_evals_per_node"]),
                "reached_tol": int(_avg_sq_reached(res, eps_sq=cfg.eps_sq)),
                "avg_sq": float(res.final["avg_sq_dist_to_x_star_all_nodes"]),
                "objective_gap": float(res.final["objective_gap"]),
            }
        )

    return rows


def _summarize_rows(rows: list[dict[str, float | int | str]]) -> str:
    parts = []
    for r in rows:
        parts.append(
            f"{r['alg']}: mix={r['mix_rounds']} grad/node={r['grad_evals_per_node']} "
            f"reached={bool(r['reached_tol'])} avg_sq={r['avg_sq']:.2e} gap={r['objective_gap']:.2e}"
        )
    return "\n".join(parts)


def sweep_beta_over_mu(
    *,
    cfg: ExpConfig,
    seeds: list[int],
    n_local_grid: list[int],
    lambda_reg: float,
    paper_protocol: bool,
    x_true_mean: float,
) -> None:
    print("\n=== Sweep A: vary n_local (beta/mu_g axis) ===")

    for n_local in n_local_grid:
        beta_over_mu_vals: list[float] = []
        for seed in seeds:
            graph = make_accsonata_er_graph(cfg.m_agents, cfg.p_edge, seed=seed, lazy=0.0)
            graph_psd = make_accsonata_er_graph(cfg.m_agents, cfg.p_edge, seed=seed, lazy=cfg.lazy_for_psd)

            spec = RidgeSpec(
                d=cfg.d,
                n_local=int(n_local),
                mu0=cfg.mu0,
                L0=cfg.L0,
                lambda_reg=float(lambda_reg),
                noise_std=cfg.noise_std,
            )

            problem = make_accsonata_ridge_problem(
                graph,
                spec,
                seed=seed,
                no_spectral_scaling=True,
                paper_protocol=bool(paper_protocol),
                x_true_mean=float(x_true_mean),
            )
            problem_psd = make_accsonata_ridge_problem(
                graph_psd,
                spec,
                seed=seed,
                no_spectral_scaling=True,
                paper_protocol=bool(paper_protocol),
                x_true_mean=float(x_true_mean),
            )

            pstats = problem.ensure_stats()
            beta_over_mu_vals.append(float(pstats.beta) / float(pstats.mu_g))

            rows = _run_suite(problem, problem_psd, cfg=cfg, seed=seed)
            print(f"\n[n_local={n_local} seed={seed}] beta/mu={beta_over_mu_vals[-1]:.3e}")
            print(_summarize_rows(rows))

        med = float(np.median(beta_over_mu_vals))
        print(f"\n[n_local={n_local}] median beta/mu_g across seeds: {med:.3e}")


def sweep_kappa(
    *,
    cfg: ExpConfig,
    seeds: list[int],
    lambda_grid: list[float],
    n_local: int,
    paper_protocol: bool,
    x_true_mean: float,
) -> None:
    print("\n=== Sweep B: vary lambda_reg (kappa_g axis) ===")

    for lam in lambda_grid:
        kappa_vals: list[float] = []
        for seed in seeds:
            graph = make_accsonata_er_graph(cfg.m_agents, cfg.p_edge, seed=seed, lazy=0.0)
            graph_psd = make_accsonata_er_graph(cfg.m_agents, cfg.p_edge, seed=seed, lazy=cfg.lazy_for_psd)

            spec = RidgeSpec(
                d=cfg.d,
                n_local=int(n_local),
                mu0=cfg.mu0,
                L0=cfg.L0,
                lambda_reg=float(lam),
                noise_std=cfg.noise_std,
            )

            problem = make_accsonata_ridge_problem(
                graph,
                spec,
                seed=seed,
                no_spectral_scaling=True,
                paper_protocol=bool(paper_protocol),
                x_true_mean=float(x_true_mean),
            )
            problem_psd = make_accsonata_ridge_problem(
                graph_psd,
                spec,
                seed=seed,
                no_spectral_scaling=True,
                paper_protocol=bool(paper_protocol),
                x_true_mean=float(x_true_mean),
            )

            pstats = problem.ensure_stats()
            kappa_vals.append(float(pstats.kappa_g))

            rows = _run_suite(problem, problem_psd, cfg=cfg, seed=seed)
            print(f"\n[lambda_reg={lam:.3e} seed={seed}] kappa_g={kappa_vals[-1]:.3e}")
            print(_summarize_rows(rows))

        med = float(np.median(kappa_vals))
        print(f"\n[lambda_reg={lam:.3e}] median kappa_g across seeds: {med:.3e}")


def main() -> None:
    ap = argparse.ArgumentParser(description="ACC-SONATA Exp.1 synthetic ridge sweeps (quadratic-only).")

    ap.add_argument("--seeds", default="0,1,2", help="Comma-separated list of integer seeds")

    ap.add_argument("--m_agents", type=int, default=30)
    ap.add_argument("--p_edge", type=float, default=0.5)
    ap.add_argument("--lazy_for_psd", type=float, default=0.5)

    ap.add_argument("--d", type=int, default=50)
    ap.add_argument("--noise_std", type=float, default=0.1)
    ap.add_argument("--mu0", type=float, default=1.0)
    ap.add_argument("--L0", type=float, default=1000.0)
    ap.add_argument(
        "--paper_protocol",
        action="store_true",
        help="Use ACC-SONATA Exp.1 synthetic protocol: features ~ N(0,Sigma) with eigs in [mu0,L0], labels from shared x_true.",
    )
    ap.add_argument(
        "--x_true_mean",
        type=float,
        default=5.0,
        help="Mean shift for x_true when --paper_protocol is set (paper uses 5.0).",
    )

    ap.add_argument("--eps_sq", type=float, default=1e-4, help="Squared residual tolerance")
    ap.add_argument("--max_mix_rounds", type=int, default=20000)
    ap.add_argument("--log_every", type=int, default=20)

    ap.add_argument(
        "--acc_eps_stop",
        type=float,
        default=1e-30,
        help="Disable embedded NEXT early-stop by setting very small (recommended for sweeps)",
    )

    ap.add_argument("--mudag_c_K", type=float, default=0.2)

    # Paper Exp.1 sweep (i) sets lambda=0 and varies n_local over [100, 40000].
    ap.add_argument("--beta_lambda_reg", type=float, default=0.0)
    ap.add_argument("--beta_n_local_grid", default="100,200,400,800,1600,3200")

    ap.add_argument("--kappa_n_local", type=int, default=800)
    ap.add_argument("--kappa_lambda_grid", default="0.0,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1")

    args = ap.parse_args()

    seeds = _parse_int_list(str(args.seeds))
    if not seeds:
        raise ValueError("--seeds must be a non-empty comma-separated list")

    cfg = ExpConfig(
        m_agents=int(args.m_agents),
        p_edge=float(args.p_edge),
        lazy_for_psd=float(args.lazy_for_psd),
        d=int(args.d),
        noise_std=float(args.noise_std),
        mu0=float(args.mu0),
        L0=float(args.L0),
        eps_sq=float(args.eps_sq),
        max_mix_rounds=int(args.max_mix_rounds),
        log_every=int(args.log_every),
        acc_eps_stop=float(args.acc_eps_stop),
        mudag_c_K=float(args.mudag_c_K),
    )

    n_local_grid = [int(x.strip()) for x in str(args.beta_n_local_grid).split(",") if x.strip()]
    lambda_grid = [float(x.strip()) for x in str(args.kappa_lambda_grid).split(",") if x.strip()]

    sweep_beta_over_mu(
        cfg=cfg,
        seeds=seeds,
        n_local_grid=n_local_grid,
        lambda_reg=float(args.beta_lambda_reg),
        paper_protocol=bool(args.paper_protocol),
        x_true_mean=float(args.x_true_mean),
    )
    sweep_kappa(
        cfg=cfg,
        seeds=seeds,
        lambda_grid=lambda_grid,
        n_local=int(args.kappa_n_local),
        paper_protocol=bool(args.paper_protocol),
        x_true_mean=float(args.x_true_mean),
    )


if __name__ == "__main__":
    main()
