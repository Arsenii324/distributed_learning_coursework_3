from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..oracles import Counters, CostedOracles
from ..problems import DistributedQuadraticProblem
from .base import init_X0


def _avg_sq_dist_to_x_star_all_nodes(problem: DistributedQuadraticProblem, X: np.ndarray) -> float:
    X = np.asarray(X, dtype=np.float64)
    diff = X - problem.x_star[None, :]
    return float(np.linalg.norm(diff, ord="fro") ** 2 / float(problem.n))


def _default_chebyshev_steps(problem: DistributedQuadraticProblem) -> int:
    chi = float(problem.graph.ensure_stats().chi)
    if not np.isfinite(chi) or chi <= 0:
        return 1
    return max(1, int(np.floor(np.sqrt(chi))))


def _default_inner_comm_budget_from_ratio(ratio: float) -> int:
    # MATLAB uses ceil(log(ratio)). Ensure at least 1.
    if not (ratio > 0) or not np.isfinite(ratio):
        return 1
    return max(1, int(np.ceil(np.log(float(ratio)))))


def _chebyshev3_mix(
    orc: CostedOracles,
    X: np.ndarray,
    steps: int,
    *,
    chi: float,
    lambda_max_L: float,
) -> np.ndarray:
    """Chebyshev-accelerated mixing (MATLAB `chebyshev3.m`) with correct comm accounting.

    This applies a polynomial in W that approximates averaging.

    Cost model: each internal application of W is done via `orc.mix`, so
    `steps` increments `mix_rounds` by exactly `steps`.

    Notes:
    - When `chi` is too close to 1, the MATLAB formula becomes ill-conditioned.
      In that case, we fall back to `steps` plain mixing steps.
    """

    if steps <= 0:
        return np.asarray(X, dtype=np.float64)

    X = np.asarray(X, dtype=np.float64)

    if not (np.isfinite(chi) and np.isfinite(lambda_max_L) and lambda_max_L > 0):
        # Conservative fallback.
        out = X
        for _ in range(int(steps)):
            out = orc.mix(out)
        return out

    if chi <= 1.0 + 1e-12:
        out = X
        for _ in range(int(steps)):
            out = orc.mix(out)
        return out

    root_chi = float(np.sqrt(chi))

    c1 = float((root_chi - 1.0) / (root_chi + 1.0))
    c2 = float((chi + 1.0) / (chi - 1.0))
    c3 = float(2.0 * chi / ((1.0 + chi) * lambda_max_L))

    T = int(steps)
    theta = float(lambda_max_L * (1.0 + c1 ** (2 * T)) / (1.0 + c1**T) ** 2)

    def apply_M(U: np.ndarray) -> np.ndarray:
        # (I - c3 L)U = (1-c3)U + c3 WU, with one counted mixing per application.
        WU = orc.mix(U)
        return (1.0 - c3) * U + c3 * WU

    x00 = X
    a0 = 1.0
    a1 = c2
    x0 = X
    x1 = c2 * apply_M(X)

    for _ in range(1, T):
        a2 = 2.0 * c2 * a1 - a0
        x2 = 2.0 * c2 * apply_M(x1) - x0
        x0, x1 = x1, x2
        a0, a1 = a1, a2

    tmp_out = x00 - x1 / a1
    out = x00 - theta * tmp_out
    return out


def _grad_with_embedding(
    orc: CostedOracles,
    X: np.ndarray,
    *,
    tau: float,
    aux_X: np.ndarray,
) -> np.ndarray:
    G = orc.local_grad(X)
    if tau:
        G = G + float(tau) * (np.asarray(X) - np.asarray(aux_X))
    return G


def _next_F_quadratic(
    problem: DistributedQuadraticProblem,
    orc: CostedOracles,
    *,
    X_init: np.ndarray,
    tau: float,
    aux_X: np.ndarray,
    Y_init: np.ndarray,
    beta: float,
    cheb_steps: int,
    gamma: float,
    comm_budget: int,
    eps_stop: float,
    chi: float,
    lambda_max_L: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Embedded NEXT-F (quadratic-only) matching MATLAB `NEXT_F.m` (embedded=true).

    Important: `comm_budget` corresponds to the paper/MATLAB inner iteration
    count `T` (number of NEXT inner updates). Communication cost is accounted
    separately by counting `orc.mix` calls inside `_chebyshev3_mix`.
    """

    X = np.asarray(X_init, dtype=np.float64)
    aux_X = np.asarray(aux_X, dtype=np.float64)
    Y = np.asarray(Y_init, dtype=np.float64)

    # Initial gradient at X (counts 1 grad eval per node).
    Grad = _grad_with_embedding(orc, X, tau=tau, aux_X=aux_X)

    n, d = problem.n, problem.d
    I_d = np.eye(d, dtype=np.float64)
    M = problem.A + (float(beta) + float(tau)) * I_d[None, :, :]

    for _ in range(int(comm_budget)):
        # Local "full" surrogate solve.
        rhs = problem.b + float(beta) * X + float(tau) * aux_X + Grad - Y

        v = np.empty_like(X)
        for i in range(n):
            v[i] = np.linalg.solve(M[i], rhs[i])

        tmp_X = float(gamma) * (v - X) + X
        X1 = _chebyshev3_mix(orc, tmp_X, cheb_steps, chi=chi, lambda_max_L=lambda_max_L)

        new_Grad = _grad_with_embedding(orc, X1, tau=tau, aux_X=aux_X)
        tmp_Y = Y + (new_Grad - Grad)
        Y1 = _chebyshev3_mix(orc, tmp_Y, cheb_steps, chi=chi, lambda_max_L=lambda_max_L)

        # Early-stop shortcut (used in MATLAB validation scripts) should not
        # return an inconsistent (X, Y) pair; otherwise the outer acceleration
        # can drift away even after getting close to x*.
        if _avg_sq_dist_to_x_star_all_nodes(problem, X1) <= float(eps_stop):
            return X1, Y1

        Y = Y1

        Grad = new_Grad
        X = X1

    return X, Y


def _next_L_quadratic(
    problem: DistributedQuadraticProblem,
    orc: CostedOracles,
    *,
    X_init: np.ndarray,
    tau: float,
    aux_X: np.ndarray,
    Y_init: np.ndarray,
    L_reg: float,
    cheb_steps: int,
    gamma: float,
    comm_budget: int,
    eps_stop: float,
    chi: float,
    lambda_max_L: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Embedded NEXT-L (quadratic-only) matching MATLAB `NEXT_L.m` (embedded=true).

    Important: `comm_budget` corresponds to the paper/MATLAB inner iteration
    count `T` (number of NEXT inner updates). Communication cost is accounted
    separately by counting `orc.mix` calls inside `_chebyshev3_mix`.
    """

    X = np.asarray(X_init, dtype=np.float64)
    aux_X = np.asarray(aux_X, dtype=np.float64)
    Y = np.asarray(Y_init, dtype=np.float64)

    step_size = float(1.0 / (float(L_reg) + float(tau)))

    Grad = _grad_with_embedding(orc, X, tau=tau, aux_X=aux_X)

    for _ in range(int(comm_budget)):
        v = X - step_size * Y
        tmp_X = float(gamma) * (v - X) + X
        X1 = _chebyshev3_mix(orc, tmp_X, cheb_steps, chi=chi, lambda_max_L=lambda_max_L)

        new_Grad = _grad_with_embedding(orc, X1, tau=tau, aux_X=aux_X)
        tmp_Y = Y + (new_Grad - Grad)
        Y1 = _chebyshev3_mix(orc, tmp_Y, cheb_steps, chi=chi, lambda_max_L=lambda_max_L)

        # Early-stop shortcut (used in MATLAB validation scripts) should not
        # return an inconsistent (X, Y) pair; otherwise the outer acceleration
        # can drift away even after getting close to x*.
        if _avg_sq_dist_to_x_star_all_nodes(problem, X1) <= float(eps_stop):
            return X1, Y1

        Y = Y1

        Grad = new_Grad
        X = X1

    return X, Y


@dataclass
class AccSonataState:
    t: int
    X: np.ndarray
    counters: Counters

    # Internal variables:
    Z: np.ndarray
    X_acc: np.ndarray
    X_acc_prev: np.ndarray
    Y: np.ndarray

    # Fixed hyperparams/constants:
    alpha: float
    tau: float
    inner_comm_budget: int
    cheb_steps: int
    gamma: float
    eps_stop: float

    chi: float
    lambda_max_L: float
    mu_reg: float
    L_reg: float
    beta: float

    @property
    def x_bar(self) -> np.ndarray:
        return self.X.mean(axis=0)

    def oracles(self, problem: DistributedQuadraticProblem) -> CostedOracles:
        return CostedOracles(problem, self.counters)


@dataclass
class AccSonataF:
    """ACC-SONATA-F (quadratic-only) — MATLAB-faithful port.

    Reference: `Acc_NEXT_F.m` calling embedded `NEXT_F.m`.

    Defaults match the MATLAB validation script:
    - `gamma = 1`
    - `cheb_steps = floor(sqrt(chi))`
    - `inner_comm_budget = ceil(log(beta/mu_reg))`
    - `eps_stop = 1e-4` (squared error)

    Note: `eps_stop` is used only for the embedded early-stop shortcut.
    The runner’s stop conditions still determine overall termination.
    """

    inner_comm_budget: int | None = None
    cheb_steps: int | None = None
    gamma: float = 1.0
    eps_stop: float = 1e-4
    name: str = "ACC-SONATA-F"

    def check(self, problem: DistributedQuadraticProblem) -> None:
        gstats = problem.graph.ensure_stats()
        if not bool(gstats.connected):
            raise ValueError("ACC-SONATA requires a connected graph")

        pstats = problem.ensure_stats()
        mu_reg = float(pstats.mu_g)
        beta = float(pstats.beta)
        if not (mu_reg > 0):
            raise ValueError("Require mu_g > 0")
        if not (beta > mu_reg):
            raise ValueError(
                "ACC-SONATA-F requires beta > mu_g for the default theory parameters. "
                f"Got beta={beta:.3e}, mu_g={mu_reg:.3e}."
            )

        if self.inner_comm_budget is not None and self.inner_comm_budget <= 0:
            raise ValueError("inner_comm_budget must be positive")
        if self.cheb_steps is not None and self.cheb_steps <= 0:
            raise ValueError("cheb_steps must be positive")
        if not (self.gamma > 0):
            raise ValueError("gamma must be positive")
        if not (self.eps_stop > 0):
            raise ValueError("eps_stop must be positive")

    def init_state(
        self,
        problem: DistributedQuadraticProblem,
        *,
        X0: np.ndarray | None = None,
        seed: int | None = None,
    ) -> AccSonataState:
        _ = seed
        X_init = init_X0(problem, X0=X0, seed=seed)

        pstats = problem.ensure_stats()
        gstats = problem.graph.ensure_stats()

        mu_reg = float(pstats.mu_g)
        L_reg = float(pstats.L_g)
        beta = float(pstats.beta)

        alpha = float(np.sqrt(mu_reg / beta))
        tau = float(beta - mu_reg)

        inner_comm_budget = (
            int(self.inner_comm_budget)
            if self.inner_comm_budget is not None
            else _default_inner_comm_budget_from_ratio(beta / mu_reg)
        )

        cheb_steps = int(self.cheb_steps) if self.cheb_steps is not None else _default_chebyshev_steps(problem)

        counters = Counters()
        orc = CostedOracles(problem, counters)
        # MATLAB initializes y as grad(x, tau=0, aux=0).
        Y0 = orc.local_grad(X_init)

        return AccSonataState(
            t=0,
            X=X_init,
            counters=counters,
            Z=X_init.copy(),
            X_acc=X_init.copy(),
            X_acc_prev=X_init.copy(),
            Y=Y0,
            alpha=alpha,
            tau=tau,
            inner_comm_budget=int(inner_comm_budget),
            cheb_steps=int(cheb_steps),
            gamma=float(self.gamma),
            eps_stop=float(self.eps_stop),
            chi=float(gstats.chi),
            lambda_max_L=float(gstats.lambda_max_L),
            mu_reg=mu_reg,
            L_reg=L_reg,
            beta=beta,
        )

    def step(self, problem: DistributedQuadraticProblem, state: AccSonataState) -> AccSonataState:
        orc = state.oracles(problem)

        # Embedded NEXT_F call with y-shift.
        Y_in = state.Y + float(state.tau) * (state.X_acc_prev - state.X_acc)
        new_Z, new_Y = _next_F_quadratic(
            problem,
            orc,
            X_init=state.Z,
            tau=state.tau,
            aux_X=state.X_acc,
            Y_init=Y_in,
            beta=state.beta,
            cheb_steps=state.cheb_steps,
            gamma=state.gamma,
            comm_budget=state.inner_comm_budget,
            eps_stop=state.eps_stop,
            chi=state.chi,
            lambda_max_L=state.lambda_max_L,
        )

        coef = float((1.0 - state.alpha) / (1.0 + state.alpha))
        new_X_acc = new_Z + coef * (new_Z - state.Z)

        return AccSonataState(
            t=int(state.t) + 1,
            X=new_Z,
            counters=state.counters,
            Z=new_Z,
            X_acc=new_X_acc,
            X_acc_prev=state.X_acc,
            Y=new_Y,
            alpha=state.alpha,
            tau=state.tau,
            inner_comm_budget=state.inner_comm_budget,
            cheb_steps=state.cheb_steps,
            gamma=state.gamma,
            eps_stop=state.eps_stop,
            chi=state.chi,
            lambda_max_L=state.lambda_max_L,
            mu_reg=state.mu_reg,
            L_reg=state.L_reg,
            beta=state.beta,
        )

    def diagnostics(self, problem: DistributedQuadraticProblem, state: AccSonataState) -> dict[str, float]:
        _ = problem
        return {
            "accsonata_alpha": float(state.alpha),
            "accsonata_tau": float(state.tau),
            "accsonata_inner_comm_budget": float(state.inner_comm_budget),
            "accsonata_cheb_steps": float(state.cheb_steps),
            "accsonata_gamma": float(state.gamma),
            "accsonata_eps_stop": float(state.eps_stop),
            "accsonata_mu_reg": float(state.mu_reg),
            "accsonata_L_reg": float(state.L_reg),
            "accsonata_beta": float(state.beta),
        }


@dataclass
class AccSonataL:
    """ACC-SONATA-L (quadratic-only) — MATLAB-faithful port.

    Reference: `Cata_NEXT_L.m` calling embedded `NEXT_L.m`.

    Defaults match the MATLAB validation script:
    - `gamma = 1`
    - `cheb_steps = floor(sqrt(chi))`
    - `inner_comm_budget = ceil(log(L_reg/mu_reg))`
    - `eps_stop = 1e-4` (squared error)
    """

    inner_comm_budget: int | None = None
    cheb_steps: int | None = None
    gamma: float = 1.0
    eps_stop: float = 1e-4
    name: str = "ACC-SONATA-L"

    def check(self, problem: DistributedQuadraticProblem) -> None:
        gstats = problem.graph.ensure_stats()
        if not bool(gstats.connected):
            raise ValueError("ACC-SONATA requires a connected graph")

        pstats = problem.ensure_stats()
        mu_reg = float(pstats.mu_g)
        L_reg = float(pstats.L_g)
        if not (mu_reg > 0 and L_reg > 0):
            raise ValueError("Require positive mu_g and L_g")
        if self.inner_comm_budget is not None and self.inner_comm_budget <= 0:
            raise ValueError("inner_comm_budget must be positive")
        if self.cheb_steps is not None and self.cheb_steps <= 0:
            raise ValueError("cheb_steps must be positive")
        if not (self.gamma > 0):
            raise ValueError("gamma must be positive")
        if not (self.eps_stop > 0):
            raise ValueError("eps_stop must be positive")

    def init_state(
        self,
        problem: DistributedQuadraticProblem,
        *,
        X0: np.ndarray | None = None,
        seed: int | None = None,
    ) -> AccSonataState:
        _ = seed
        X_init = init_X0(problem, X0=X0, seed=seed)

        pstats = problem.ensure_stats()
        gstats = problem.graph.ensure_stats()

        mu_reg = float(pstats.mu_g)
        L_reg = float(pstats.L_g)
        beta = float(pstats.beta)

        alpha = float(np.sqrt(mu_reg / L_reg))
        tau = float(L_reg - mu_reg)

        inner_comm_budget = (
            int(self.inner_comm_budget)
            if self.inner_comm_budget is not None
            else _default_inner_comm_budget_from_ratio(L_reg / mu_reg)
        )

        cheb_steps = int(self.cheb_steps) if self.cheb_steps is not None else _default_chebyshev_steps(problem)

        counters = Counters()
        orc = CostedOracles(problem, counters)
        Y0 = orc.local_grad(X_init)

        return AccSonataState(
            t=0,
            X=X_init,
            counters=counters,
            Z=X_init.copy(),
            X_acc=X_init.copy(),
            X_acc_prev=X_init.copy(),
            Y=Y0,
            alpha=alpha,
            tau=tau,
            inner_comm_budget=int(inner_comm_budget),
            cheb_steps=int(cheb_steps),
            gamma=float(self.gamma),
            eps_stop=float(self.eps_stop),
            chi=float(gstats.chi),
            lambda_max_L=float(gstats.lambda_max_L),
            mu_reg=mu_reg,
            L_reg=L_reg,
            beta=beta,
        )

    def step(self, problem: DistributedQuadraticProblem, state: AccSonataState) -> AccSonataState:
        orc = state.oracles(problem)

        Y_in = state.Y + float(state.tau) * (state.X_acc_prev - state.X_acc)
        new_Z, new_Y = _next_L_quadratic(
            problem,
            orc,
            X_init=state.Z,
            tau=state.tau,
            aux_X=state.X_acc,
            Y_init=Y_in,
            L_reg=state.L_reg,
            cheb_steps=state.cheb_steps,
            gamma=state.gamma,
            comm_budget=state.inner_comm_budget,
            eps_stop=state.eps_stop,
            chi=state.chi,
            lambda_max_L=state.lambda_max_L,
        )

        coef = float((1.0 - state.alpha) / (1.0 + state.alpha))
        new_X_acc = new_Z + coef * (new_Z - state.Z)

        return AccSonataState(
            t=int(state.t) + 1,
            X=new_Z,
            counters=state.counters,
            Z=new_Z,
            X_acc=new_X_acc,
            X_acc_prev=state.X_acc,
            Y=new_Y,
            alpha=state.alpha,
            tau=state.tau,
            inner_comm_budget=state.inner_comm_budget,
            cheb_steps=state.cheb_steps,
            gamma=state.gamma,
            eps_stop=state.eps_stop,
            chi=state.chi,
            lambda_max_L=state.lambda_max_L,
            mu_reg=state.mu_reg,
            L_reg=state.L_reg,
            beta=state.beta,
        )

    def diagnostics(self, problem: DistributedQuadraticProblem, state: AccSonataState) -> dict[str, float]:
        _ = problem
        return {
            "accsonata_alpha": float(state.alpha),
            "accsonata_tau": float(state.tau),
            "accsonata_inner_comm_budget": float(state.inner_comm_budget),
            "accsonata_cheb_steps": float(state.cheb_steps),
            "accsonata_gamma": float(state.gamma),
            "accsonata_eps_stop": float(state.eps_stop),
            "accsonata_mu_reg": float(state.mu_reg),
            "accsonata_L_reg": float(state.L_reg),
            "accsonata_beta": float(state.beta),
        }
