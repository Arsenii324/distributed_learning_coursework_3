from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from ..datasets import download_libsvm_dataset, get_data_dir, load_svmlight, make_distributed_ridge_problem_from_dataset
from ..generators import erdos_renyi_adjacency, make_graph_from_adjacency
from ..runner import MaxMixRounds, run_experiment
from ..algorithms import AccSonataL, DGD, EXTRA, GradientTracking, MUDAG


@dataclass(frozen=True)
class RunConfig:
    dataset: str
    m_agents: int
    p_edge: float
    lambda_reg: float
    max_mix_rounds: int
    lazy_for_psd: float
    seed: int


def _pick_step_sizes(problem) -> dict[str, float]:
    stats = problem.ensure_stats()
    L = float(stats.L_g)
    # Conservative defaults; tuning is paper-specific and out of scope for this demo.
    return {
        "dgd": 0.2 / L,
        "extra": 0.2 / L,
        "gt": 0.1 / L,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Download a LIBSVM dataset, build a ridge quadratic, run distopt algorithms.")
    ap.add_argument("--dataset", default="a9a", choices=["a9a", "w8a", "SUSY"], help="Dataset name")
    ap.add_argument("--m_agents", type=int, default=30, help="Number of agents/nodes")
    ap.add_argument("--p_edge", type=float, default=0.5, help="ER edge probability")
    ap.add_argument("--lambda_reg", type=float, default=1.0, help="Ridge regularization")
    ap.add_argument("--max_mix_rounds", type=int, default=2000, help="Communication budget")
    ap.add_argument("--lazy_for_psd", type=float, default=0.5, help="Lazification for PSD-required algorithms (e.g., MUDAG)")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    ap.add_argument(
        "--acc_cheb_steps",
        type=int,
        default=None,
        help="Chebyshev mixing steps for ACC-SONATA-L. Default: max(2, floor(sqrt(chi))) for the chosen graph.",
    )
    ap.add_argument(
        "--insecure_download",
        action="store_true",
        help="Disable TLS certificate verification for dataset download (sets DISTOPT_INSECURE_DOWNLOAD=1).",
    )

    args = ap.parse_args()

    cfg = RunConfig(
        dataset=args.dataset,
        m_agents=int(args.m_agents),
        p_edge=float(args.p_edge),
        lambda_reg=float(args.lambda_reg),
        max_mix_rounds=int(args.max_mix_rounds),
        lazy_for_psd=float(args.lazy_for_psd),
        seed=int(args.seed),
    )

    if args.insecure_download:
        import os

        os.environ["DISTOPT_INSECURE_DOWNLOAD"] = "1"

    print(f"Data dir: {get_data_dir()}")
    dl = download_libsvm_dataset(cfg.dataset)
    print(f"Downloaded: {dl.path}")
    print(f"Using file: {dl.decompressed_path}")

    X, y = load_svmlight(dl.decompressed_path)
    print(f"Loaded X shape={X.shape}, nnz={X.nnz}, y shape={y.shape}")

    # Build a connected-ish ER graph (no rejection sampling here; increase p_edge if needed).
    adj = erdos_renyi_adjacency(cfg.m_agents, cfg.p_edge, seed=cfg.seed)

    graph = make_graph_from_adjacency(adj, lazy=0.0)
    graph_psd = make_graph_from_adjacency(adj, lazy=cfg.lazy_for_psd)

    problem = make_distributed_ridge_problem_from_dataset(
        graph,
        X,
        y,
        m_agents=cfg.m_agents,
        lambda_reg=cfg.lambda_reg,
        partition="shuffle",
        seed=cfg.seed,
        dataset_name=cfg.dataset,
    )

    # For MUDAG, use PSD-lazified graph.
    problem_psd = make_distributed_ridge_problem_from_dataset(
        graph_psd,
        X,
        y,
        m_agents=cfg.m_agents,
        lambda_reg=cfg.lambda_reg,
        partition="shuffle",
        seed=cfg.seed,
        dataset_name=cfg.dataset,
    )

    steps = _pick_step_sizes(problem)

    chi = float(problem.graph.ensure_stats().chi)
    acc_cheb_steps = (
        int(args.acc_cheb_steps)
        if args.acc_cheb_steps is not None
        else max(2, int(np.floor(np.sqrt(chi))))
    )
    if acc_cheb_steps <= 0:
        raise ValueError("--acc_cheb_steps must be positive")

    algorithms = [
        ("DGD", DGD(alpha=steps["dgd"]), problem),
        ("EXTRA", EXTRA(alpha=steps["extra"]), problem),
        ("GradientTracking", GradientTracking(alpha=steps["gt"]), problem),
        ("MUDAG", MUDAG(c_K=2.0), problem_psd),
        ("ACC-SONATA-L", AccSonataL(cheb_steps=acc_cheb_steps), problem),
    ]

    stop = MaxMixRounds(cfg.max_mix_rounds)

    print("\nRunning...")
    for name, alg, prob in algorithms:
        res = run_experiment(prob, alg, stop=stop, seed=cfg.seed, log_every=10)
        last = res.history[-1]
        x_bar = np.asarray(res.final["x_bar"], dtype=np.float64)
        gap_final = prob.global_value(x_bar) - prob.global_value(prob.x_star)
        print(
            f"{name:16s} | mix={int(res.final['mix_rounds']):6d} "
            f"grad/node={int(res.final['grad_evals_per_node']):6d} "
            f"gap={gap_final:.3e} "
            f"cons={last['consensus_error']:.3e}"
        )


if __name__ == "__main__":
    main()
