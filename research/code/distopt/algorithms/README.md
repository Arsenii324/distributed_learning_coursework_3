# distopt.algorithms

This folder contains baseline decentralized optimization algorithms implemented as **transparent state machines**.

## Design goals

- **Readability and correctness** over squeezing constants.
- **Explicit cost accounting** via `Counters` and `CostedOracles`.
- **Simple interop** with the runner: `run_experiment(problem, algorithm, ...)`.

## The `Algorithm` interface

Algorithms follow the `Algorithm` protocol in `base.py`:

- `check(problem)`: validate assumptions / parameters
- `init_state(problem, X0=None, seed=None)`: produce an initial state
- `step(problem, state)`: compute the next state (one logical iteration)
- `diagnostics(problem, state) -> dict[str, float]`: algorithm-owned scalar diagnostics (e.g. `alpha`)

### State conventions

All algorithms are expected to expose at least:
- `state.t`: iteration counter (int)
- `state.X`: stacked local iterates (shape `(n,d)`)
- `state.counters`: `Counters(mix_rounds, grad_evals_per_node)`
- `state.x_bar`: average iterate (shape `(d,)`), usually provided as a property

Algorithms may additionally store auxiliary variables (previous iterates, trackers, etc.).

### Cost accounting

**Rule:** Mixing and gradient computations should be performed through

```python
orc = state.oracles(problem)
WX = orc.mix(state.X)           # counts 1 mix round
G  = orc.local_grad(state.X)    # counts 1 grad eval per node
```

This keeps `mix_rounds` and `grad_evals_per_node` consistent across algorithms.

If an algorithm bypasses this (e.g. uses `problem.graph.W @ X` directly), the counters will be wrong.

## Implemented algorithms

### DGD (`dgd.py`)
Decentralized gradient descent:

- Update:
  - `X^{k+1} = W X^k - α ∇F(X^k)`
- Cost per iteration:
  - 1 mixing round
  - 1 local gradient evaluation per node

### EXTRA (`extra.py`)
EXTRA (Shi et al.) for symmetric doubly-stochastic `W`.

- First step (implemented as DGD-style):
  - `X^1 = W X^0 - α ∇F(X^0)`
- Subsequent steps use the update (matching the repo notes):
  - `X^{k+1} = X^k + W X^k - W~ X^{k-1} - α(∇F(X^k) - ∇F(X^{k-1}))`
  - where `W~ = (I + W)/2`.
- Implementation detail:
  - caches `W X^{k-1}` so that each new iteration performs **one** fresh mixing.
- Cost per iteration (after the first step):
  - 1 mixing round
  - 1 local gradient evaluation per node

### Gradient Tracking (`gradient_tracking.py`)
DIGing-style gradient tracking:

- Update:
  - `X^{k+1} = W X^k - α Y^k`
  - `Y^{k+1} = W Y^k + ∇F(X^{k+1}) - ∇F(X^k)`
- Initialization:
  - computes one gradient to set `Y^0 = ∇F(X^0)`
- Cost per iteration:
  - 2 mixing rounds (one for `X`, one for `Y`)
  - 1 local gradient evaluation per node (at `X^{k+1}`)

### MUDAG (`mudag.py`)
MUDAG with a Chebyshev/FastMix inner mixing loop (mirroring the archived MATLAB implementation).

- Assumptions:
  - `W` should be **PSD** (eigenvalues in $[0,1]$). In the MATLAB workflow this is enforced via lazification: $W \leftarrow (W+I)/2$.
  - In this repo, construct such a matrix via `Graph.from_adjacency(..., lazy=0.5)` (or `make_graph_from_adjacency(..., lazy=0.5)`).

- Inner-loop length:
  - `K = ceil(c_K / sqrt(1-λ₂(W)) * log((M/L) * κ_g))`
  - with `M = L_l` (worst local smoothness) and `L = L_g` (global smoothness).

- Cost per **outer** iteration (this implementation):
  - `K+1` mixing rounds (FastMix runs `k=0..K`)
  - 1 local gradient eval per node (the previous gradient is cached to avoid recomputation; update is unchanged)

- MATLAB-style inner-loop early stop (optional):
  - The archived MATLAB `Mudag.m` breaks out of FastMix early once the target residual is met.
  - In this repo, this is exposed as `fastmix_stop_eps_sq` (set it to the same squared tolerance used by
    `TargetAvgSqDistToXStarAllNodes` for paper-faithful behavior).

Note: `t` is an outer-iteration counter; compare methods via `mix_rounds` / `grad_evals_per_node`.

## How to add a new algorithm

1. Create a new file under this folder.
2. Define a `@dataclass` for your algorithm hyperparameters (step sizes, momentum, etc.).
3. Define a state dataclass storing `t`, `X`, `counters`, plus any auxiliary variables.
4. In `step`, use `state.oracles(problem)` for mixing/grad.
5. Export it in `__init__.py`.

Tip: keep `diagnostics()` scalar-only so it can be merged into the `history` table without bloating logs.
