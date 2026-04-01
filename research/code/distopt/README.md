# distopt

`distopt` is a **dense NumPy** experiment harness for decentralized optimization on static graphs.

It is intentionally scoped to the setting used throughout the repo’s research notes:
- static **undirected** simple topology (stored as adjacency)
- a **symmetric doubly-stochastic** mixing matrix `W` (stored explicitly; can be overridden)
- local objectives are **strongly convex quadratics**
- the global objective is always the average:  
  $f(x)=\frac1n\sum_{i=1}^n f_i(x)$

The package is designed for *paper-scale* experiments (small to “mediocre” `(n,d)`), so it prioritizes correctness + transparency over wall-clock performance.

## Architectural overview

The flow is:

1. **Topology** → `Graph(adj, W)`
2. **Task** → `DistributedQuadraticProblem(graph, A, b)`
3. **Algorithm** → `Algorithm.init_state(...)` + repeated `Algorithm.step(...)`
4. **Runner** → `run_experiment(...)` which logs metrics into an `ExperimentResult`

Key design decisions:
- **Adjacency and `W` are both stored.** Adjacency is the physical topology; `W` is the one-round communication operator.
- **Costs are explicit counters, not time.** We track `mix_rounds` and `grad_evals_per_node`.
- **Algorithms are state machines.** They expose their internal state clearly (e.g. `X`, trackers, previous iterates).
- **Metrics are split:** static stats are cached on `Graph`/`Problem`, dynamic metrics come from the runner, and algorithm-specific diagnostics come from `algorithm.diagnostics()`.

## Package layout (modules and responsibilities)

### `graphs.py`
- `Graph`: stores
  - `adj` (bool, shape `(n,n)`) undirected with no self-loops
  - `W` (float, shape `(n,n)`) symmetric + doubly-stochastic by default
  - `L = I - W`
- `Graph.mix(X)`: applies one communication round (`X ← W X`)
- `Graph.ensure_stats()`: cached spectral stats:
  - eigenvalues of `W` and `L`
  - one-sided spectral gap `γ = 1 - λ₂(W)`
  - condition number `χ = λ_max(L) / λ_min⁺(L)`
- `metropolis_mixing_matrix(adj, lazy=...)`: default constructor for `W`

### `problems.py`
- `DistributedQuadraticProblem(graph, A, b)` with
  - `A` shape `(n,d,d)`, symmetric PD per node
  - `b` shape `(n,d)`
- Local objective:
  - $f_i(x) = \tfrac12 x^\top A_i x - b_i^\top x$
  - $\nabla f_i(x) = A_i x - b_i$
- Cached aggregates:
  - `A_bar = mean(A_i)`, `b_bar = mean(b_i)`
  - exact optimum `x_star = solve(A_bar, b_bar)`
- `ensure_stats()` computes exact (dense) constants:
  - local: `L_i = λ_max(A_i)`, `μ_i = λ_min(A_i)`
  - global: `L_g = λ_max(Ā)`, `μ_g = λ_min(Ā)`
  - `β = max_i ||A_i - Ā||₂` (spectral norm)
  - optional `δ = max_{i<j} ||A_i - A_j||₂` (on-demand; can be expensive)

### `oracles.py`
- `Counters(mix_rounds, grad_evals_per_node)` lives in algorithm state.
- `CostedOracles(problem, counters)` enforces accounting:
  - `mix(X)` increments `mix_rounds` and applies `W X`
  - `local_grad(X)` increments `grad_evals_per_node` and computes stacked local gradients

**Important:** algorithms should call mixing/gradients through `CostedOracles` (not `W @ X` directly), otherwise counters won’t reflect the algorithm’s intended cost model.

### `algorithms/`
Baseline algorithms implemented as transparent state machines (see `algorithms/README.md` for details).

- `DGD`: one mix + one local gradient per iteration
- `EXTRA`: one mix + one local gradient per iteration (after the first step)
- `GradientTracking`: two mixes (for `X` and tracker) + one local gradient per iteration
- `MUDAG`: an outer momentum method with an inner Chebyshev/FastMix loop (many mixes per outer iteration; requires PSD `W`, typically enforced by lazification `lazy=0.5`)

### `metrics.py`
- `default_metrics(problem, algorithm, state)` returns the default dynamic metrics:
  - `t`, `mix_rounds`, `grad_evals_per_node`
  - `dist_to_x_star = ||x̄ - x*||₂`
  - `avg_sq_dist_to_x_star_all_nodes = (1/n) ||X - 1 x*^T||_F^2` (MATLAB-style residual used in the archived MUDAG code)
  - `objective_gap = f(x̄) - f(x*)`
  - `consensus_error = ||X - 1 x̄^T||_F`

### `runner.py`
- Stop conditions:
  - `MaxIters`, `MaxMixRounds`
  - `TargetXStarDist` (default style)
  - `TargetAvgSqDistToXStarAllNodes` (MATLAB-style residual threshold; note this is a *squared* tolerance)
  - `TargetObjectiveGap`
- `run_experiment(problem, algorithm, stop=..., log_every=..., metric_fns=...)`:
  - stops when **any** stop condition triggers
  - logs at `t=0` and then every `log_every`
  - merges metrics from:
    1) `default_metrics`
    2) user metric functions
    3) `algorithm.diagnostics()`
- Returns `ExperimentResult` with:
  - `history`: list of scalar dicts
  - `final`: includes `x_bar` (as a NumPy array) + final counters

### `generators.py`
Helpers to create instances for experiments:
- adjacency: `path_adjacency`, `cycle_adjacency`, `complete_adjacency`, `erdos_renyi_adjacency`
- graph builder: `make_graph_from_adjacency(adj, ..., W_override=...)`
- quadratic families:
  - `make_random_spd_problem`
  - `make_shared_eigenbasis_problem`
  - `make_wishart_ridge_problem`

## Minimal quickstart

```python
import numpy as np

from research.code.distopt.generators import path_adjacency, make_graph_from_adjacency, make_wishart_ridge_problem
from research.code.distopt.algorithms import DGD
from research.code.distopt.runner import run_experiment, MaxIters

adj = path_adjacency(8)
graph = make_graph_from_adjacency(adj, lazy=0.1)
problem = make_wishart_ridge_problem(graph, d=5, m_per_node=50, lambda_reg=1.0, seed=0)

rng = np.random.default_rng(0)
X0 = rng.normal(size=(graph.n, problem.d))

res = run_experiment(problem, DGD(alpha=0.05), stop=MaxIters(200), X0=X0, log_every=20)
print(res.history[-1]["objective_gap"], res.history[-1]["consensus_error"])
```

## Canonical notebook workflow (graphs × families × algorithms × step sizes)

Most paper-style experiments follow the same structure:

1. Choose an adjacency (topology) → build a `Graph` (and thus a default `W`).
2. Choose a quadratic family → build a `DistributedQuadraticProblem`.
3. Choose algorithm(s) and stopping criteria.
4. Sweep `alpha` (or other hyperparameters).
5. Compare runs by **mixing rounds** / **gradient evaluations** and by the logged metrics.

### A note on what to compare

The `t` counter is an “algorithm iteration” and is **not** always comparable across methods.

- DGD and EXTRA do ~1 mix per iteration.
- Gradient tracking does **2 mixes per iteration** in this implementation (one for `X`, one for the tracker).

So when comparing different methods, prefer:
- `mix_rounds`
- `grad_evals_per_node`

### 1) Pick a topology (adjacency) and build a graph

```python
from research.code.distopt.generators import (
  path_adjacency,
  cycle_adjacency,
  complete_adjacency,
  erdos_renyi_adjacency,
  make_graph_from_adjacency,
)

n = 20
adj = path_adjacency(n)
# adj = cycle_adjacency(n)
# adj = complete_adjacency(n)
# adj = erdos_renyi_adjacency(n, p=0.2, seed=0)

graph = make_graph_from_adjacency(adj, lazy=0.1)
gstats = graph.ensure_stats()
print("gamma=", gstats.gamma, "chi=", gstats.chi)
```

If you want to override the default mixing matrix `W` (still storing adjacency):

```python
W_custom = graph.W.copy()  # replace with your own symmetric doubly-stochastic W
graph2 = make_graph_from_adjacency(adj, W_override=W_custom)
```

### 2) Pick a quadratic family

Recommended starting point (“ERM-like” surrogate): Wishart / ridge.

```python
from research.code.distopt.generators import make_wishart_ridge_problem

problem = make_wishart_ridge_problem(
  graph,
  d=10,
  m_per_node=50,
  lambda_reg=1.0,
  noise_std=0.0,
  seed=0,
)

pstats = problem.ensure_stats()
print("kappa_g=", pstats.kappa_g, "beta=", pstats.beta)
```

Other useful families:

```python
from research.code.distopt.generators import make_random_spd_problem, make_shared_eigenbasis_problem

problem_random = make_random_spd_problem(graph, d=10, mu=1.0, L=10.0, seed=0)
problem_shared = make_shared_eigenbasis_problem(graph, d=10, mu=1.0, L=10.0, seed=0)
```

### 3) Choose algorithms and stop conditions

```python
from research.code.distopt.algorithms import DGD, EXTRA, GradientTracking
from research.code.distopt.runner import MaxIters, TargetObjectiveGap

stop = [MaxIters(500), TargetObjectiveGap(1e-10)]

alg_factories = {
  "DGD": lambda alpha: DGD(alpha=alpha),
  "EXTRA": lambda alpha: EXTRA(alpha=alpha),
  "GT": lambda alpha: GradientTracking(alpha=alpha),
}
```

### 4) Sweep `alpha` and compare

```python
import numpy as np

from research.code.distopt.runner import run_experiment

rng = np.random.default_rng(0)
X0 = rng.normal(size=(graph.n, problem.d))

alphas = [0.2, 0.1, 0.05, 0.02]

rows = []
for alpha in alphas:
  for alg_name, make_alg in alg_factories.items():
    res = run_experiment(
      problem,
      make_alg(alpha),
      stop=stop,
      X0=X0,
      log_every=25,
    )
    last = res.history[-1]
    rows.append(
      {
        "alg": alg_name,
        "alpha": alpha,
        "t": int(res.final["t"]),
        "mix": int(res.final["mix_rounds"]),
        "grad": int(res.final["grad_evals_per_node"]),
        "dist": last["dist_to_x_star"],
        "gap": last["objective_gap"],
        "cons": last["consensus_error"],
      }
    )

for r in rows:
  print(r)
```

Notes:
- If an algorithm diverges, you’ll typically see `dist_to_x_star` and/or `consensus_error` explode.
- Gradient tracking can be more sensitive to `alpha` than DGD/EXTRA; shrink `alpha` when needed.

### 5) Add custom metrics

Metric functions are callables `(problem, algorithm, state) -> dict[str, float]`.

Example: log the global gradient norm at the average iterate.

```python
import numpy as np

from research.code.distopt.runner import MaxIters

def global_grad_norm(problem, algorithm, state):
  _ = algorithm
  g = problem.global_grad(state.x_bar)
  return {"global_grad_norm": float(np.linalg.norm(g))}

res = run_experiment(
  problem,
  DGD(alpha=0.05),
  stop=MaxIters(200),
  X0=X0,
  log_every=10,
  metric_fns=[global_grad_norm],
)
print(res.history[-1]["global_grad_norm"])
```

### 6) Running “graph × family” grids

```python
graphs = {
  "path": make_graph_from_adjacency(path_adjacency(20), lazy=0.1),
  "cycle": make_graph_from_adjacency(cycle_adjacency(20), lazy=0.1),
}

families = {
  "wishart": lambda g: make_wishart_ridge_problem(g, d=10, m_per_node=50, seed=0),
  "shared": lambda g: make_shared_eigenbasis_problem(g, d=10, seed=0),
}

all_results = []
for gname, g in graphs.items():
  for fname, make_prob in families.items():
    prob = make_prob(g)
    res = run_experiment(prob, EXTRA(alpha=0.05), stop=MaxIters(300), log_every=50)
    all_results.append({"graph": gname, "family": fname, "res": res})
```

## Extension points

- New graphs / mixing rules: add a new mixing-matrix constructor in `graphs.py` and keep the same invariants (symmetric + doubly-stochastic by default).
- New problem families: add generators in `generators.py` that return `(problem, metadata)`.
- New algorithms:
  - implement the `Algorithm` protocol in `algorithms/`
  - store counters in state
  - use `state.oracles(problem)` to perform `mix`/`local_grad` and keep accounting correct.
- New metrics: pass additional metric functions to `run_experiment(..., metric_fns=[...])`.

## Notes on step sizes

This package intentionally does **not** auto-tune step sizes. Some methods (notably gradient tracking) can diverge for large `alpha` depending on `W` and the quadratic family.

The expected workflow is to:
- start with conservative `alpha`
- sweep `alpha` for stability if needed
- compare algorithms using the cost counters rather than runtime.
