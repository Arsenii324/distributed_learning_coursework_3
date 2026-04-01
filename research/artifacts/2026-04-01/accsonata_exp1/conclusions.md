# ACC-SONATA Exp.1 sweep — conclusions (2026-04-01)

This note summarizes the outputs produced by running:

`python -m research.code.distopt.examples.run_accsonata_exp1_sweeps --paper_protocol --seeds 0,1,2 --max_mix_rounds 50000 --log_every 500 --beta_n_local_grid 100,200,400,800,1600 --beta_lambda_reg 0.0 --kappa_lambda_grid 0.0,1e-4,1e-3,1e-2,1e-1 --kappa_n_local 800`

## Where the artifacts are

- Raw stdout log: `accsonata_exp1_sweep.txt`
- Parsed data: `rows.csv` (per seed), `summary.csv` (median across seeds)
- Plots:
  - `sweepA_mix_vs_beta_over_mu.png`
  - `sweepA_grad_vs_beta_over_mu.png`
  - `sweepB_mix_vs_lambda_reg.png`

All of the above are in this folder.

## Setup reminder (what “stop” means here)

All algorithms are stopped by:
- `TargetAvgSqDistToXStarAllNodes(eps_sq)` with `eps_sq = 1e-4`, OR
- `MaxMixRounds(max_mix_rounds)` with `max_mix_rounds = 50000`.

So the primary runtime metric is *communication* (`mix_rounds`), and `grad_evals_per_node` gives the compute footprint.

## Main empirical findings

### 1) Sweep A (vary `n_local` ⇒ vary `beta/mu_g`)

Across the observed range (median `beta/mu_g` from ~240 to ~1103):

- **ACC-SONATA variants dominate by a wide margin.**
  - Relative to EXTRA, the median communication speedup `EXTRA / mix_rounds` is:
    - **ACC-SONATA-F:** ~5.5× to ~8.8× faster
    - **ACC-SONATA-L:** ~3.6× to ~5.9× faster
  - In *compute* (`grad_evals_per_node`), the separation is even larger:
    - **ACC-SONATA-F:** ~20× to ~30× fewer gradients than EXTRA
    - **ACC-SONATA-L:** ~13× to ~21× fewer gradients than EXTRA

- **ACC-SONATA-F vs ACC-SONATA-L tradeoff:**
  - At *lower* `beta/mu_g` (easier heterogeneity), **ACC-SONATA-F is best** (e.g. ~3120 mix rounds at `beta/mu_g≈240`).
  - At the *largest* `beta/mu_g` point in the sweep, **ACC-SONATA-L slightly wins** (median ~7700 vs ~8224 mix rounds).
  - Interpretation: **F is better when the regime is “more favorable”; L is more stable as `beta/mu_g` increases**.

- **MUDAG is compute-efficient but communication-heavy in this regime.**
  - It uses **~3×–5× fewer gradients than EXTRA**, but
  - its `mix_rounds` are **higher than EXTRA** for most of the sweep (i.e., a worse comm tradeoff here).

- **Robustness / reaching tolerance:**
  - ACC-SONATA-{F,L} had **100% reached rate** in Sweep A.
  - EXTRA and MUDAG had **2/3 reached rate** at the two hardest `n_local` settings (100 and 200).
  - GradientTracking had **0% reached rate** throughout Sweep A (always hit `max_mix_rounds`).

- **Hardest observed case (largest `beta/mu_g` in the log):**
  - For `beta/mu_g≈1237` (seed 1), **EXTRA and MUDAG did not reach** the `1e-4` target within 50k mix rounds, while both ACC-SONATA variants did (≈9–10k mix rounds).

### 2) Sweep B (vary `lambda_reg` ⇒ intended to vary `kappa_g`)

In this run, `kappa_g` only shifts modestly (median ~1008 down to ~915 as `lambda_reg` increases to 1e-1), and correspondingly:

- All successful methods improve only modestly in `mix_rounds` (roughly **single-digit to ~10%** reductions from `lambda_reg=0` to `1e-1`).
- GradientTracking still fails to reach the tolerance (0% reached).

Interpretation: **this particular `lambda_reg` grid doesn’t substantially change global conditioning in the generated instances**, so performance is dominated by the other aspects of the protocol (graph + heterogeneity).

## Interpreting the GradientTracking outcome

GradientTracking is implemented in a standard DIGing-style form (2 mixes + 1 gradient per iteration), but in this sweep it never reaches the strict `avg_sq ≤ 1e-4` target within 50k mix rounds.

Two grounded caveats before treating this as “GT is bad”:
- The sweep script explicitly labels baselines as “minimal tuning”. GT step size is set by `_pick_step_sizes()` as `alpha = 0.1 / L_l` (local smoothness). In many GT analyses, stability can depend on both smoothness and graph contraction constants; a more conservative `alpha` (or tuned `alpha`) may be required here.
- The sweep uses a **communication budget** (mix rounds). GT spending 2 mixes/iter means its gradient budget is half the comm budget. That said, even by objective gap / avg_sq it appears far from converged in some seeds.

So: **the current sweep output is strong evidence that “untuned GT under this protocol is not competitive,” not that the method is inherently incapable.**

## Takeaways relevant to “average-case / structural metrics” framing

- This synthetic ridge protocol looks decisively in favor of **ACC-SONATA (Chebyshev-accelerated communication)**: it delivers robust convergence with an order-of-magnitude comm reduction vs EXTRA.
- The Sweep A curves show a clear dependence on the heterogeneity axis (`beta/mu_g`): ACC-SONATA-F degrades as `beta/mu_g` increases, while ACC-SONATA-L is relatively flat.
- Because the baseline GT configuration does not reach the target at all, this sweep **does not currently support** testing subtle predictiveness claims (e.g., whether a χ-like metric explains variance between successful methods). It *does* cleanly answer the algorithm-selection question for this protocol: **use ACC-SONATA**.
