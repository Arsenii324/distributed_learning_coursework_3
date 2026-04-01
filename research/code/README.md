# research/code

This folder contains **notebook-first experimental code** used by the materials under `research/`.

It is intentionally lightweight:
- **Clarity over performance** (dense NumPy, paper-scale sizes).
- **Reproducibility over convenience** (explicit seeds in generators; explicit cost counters).
- **Small, composable pieces** (graph ↔ problem ↔ algorithm ↔ runner).

## Package(s)

### `research.code.distopt`
A minimal harness for **decentralized / distributed optimization experiments** on **static undirected graphs**, focused on **strongly convex quadratics**.

The harness is designed to:
- represent graph topology via adjacency + a one-round mixing matrix `W`
- represent distributed quadratic tasks via per-node `(A_i, b_i)`
- run baseline decentralized algorithms as transparent state machines
- track costs as **mixing rounds** and **gradient evaluations** (no wall-clock timing)
- log dynamic metrics (e.g. `||x̄-x*||`, objective gap, consensus error)

See `research/code/distopt/README.md` for the architecture and the intended workflow.

## Relationship to archived external code

The folder `research/Acceleration-in-Distributed-Optimization-Under-Similarity-main/` is an **archived codebase from a paper** downloaded online.

It can serve as a reference, but this repo’s experimental harness is implemented independently under `research/code/` to avoid architectural coupling or hidden assumptions.

## Imports

A small `research/__init__.py` is present so that notebooks can import code as a normal package:

```python
from research.code.distopt.generators import path_adjacency, make_graph_from_adjacency
from research.code.distopt.generators import make_wishart_ridge_problem
from research.code.distopt.algorithms import DGD
from research.code.distopt.runner import run_experiment, MaxIters

adj = path_adjacency(10)
graph = make_graph_from_adjacency(adj)
problem = make_wishart_ridge_problem(graph, d=5, m_per_node=50, seed=0)

res = run_experiment(problem, DGD(alpha=0.05), stop=MaxIters(200), log_every=10)
print(res.history[-1])
```
