"""One rebalance as a convex problem.

    max_w   -f.w  -  lambda * risk(w)  -  gamma * |w - w_old|_1  [ - kappa/2 * ||w - w_old||^2 ]
    s.t.    |B' w| <= beta_tol          (one row per benchmark)
            |w|_1  <= max_gross
            -cap_short <= w <= cap_long

w is signed notional over equity (short < 0); f is the per-day funding a long pays, so a short
on a positive-funding coin earns -f.w > 0. risk(w) is sum(resid_var * w^2), or w'Sigma w when a
covariance is given. gamma is the execution penalty spread over the expected holding period;
the optional quadratic term (kappa, off by default) damps noise-driven weight changes.

Then the no-trade band: coins whose weight would move less than min_trade keep their old weight
and the rest is re-solved, so skipping small trades never breaks neutrality.
"""

from dataclasses import dataclass
from typing import Optional

import cvxpy as cp
import numpy as np


@dataclass
class Problem:
    coins: list
    f: np.ndarray              # per-day funding forecast (long pays if > 0)
    B: np.ndarray              # coins x benchmarks betas
    resid_var: np.ndarray      # daily residual variance
    cap_long: np.ndarray       # >= 0
    cap_short: np.ndarray      # >= 0
    w_old: np.ndarray
    Sigma: Optional[np.ndarray] = None
    cond: float = float("nan")         # condition number of Sigma before the PSD projection
    carry: Optional[np.ndarray] = None # raw funding forecast, when f is drift-adjusted


def _solve(p: Problem, lam, gamma, max_gross, beta_tol, pinned=None, kappa=0.0):
    w = cp.Variable(len(p.coins))
    risk = (cp.quad_form(w, cp.psd_wrap(p.Sigma)) if p.Sigma is not None
            else cp.sum(cp.multiply(p.resid_var, cp.square(w))))
    cons = [cp.abs(p.B.T @ w) <= beta_tol, cp.norm1(w) <= max_gross,
            w <= p.cap_long, w >= -p.cap_short]
    if pinned is not None and pinned.any():
        cons.append(w[np.flatnonzero(pinned)] == p.w_old[pinned])
    obj = -p.f @ w - lam * risk - gamma * cp.norm1(w - p.w_old)
    if kappa:
        obj = obj - kappa / 2 * cp.sum_squares(w - p.w_old)
    prob = cp.Problem(cp.Maximize(obj), cons)
    try:
        prob.solve(solver=cp.CLARABEL)
    except cp.SolverError:
        return None
    if prob.status not in ("optimal", "optimal_inaccurate") or w.value is None:
        return None
    out = np.asarray(w.value, dtype=float)
    out[np.abs(out) < 1e-6] = 0.0
    # an "optimal_inaccurate" answer can sit far outside the constraints; keep only feasible ones
    slack = 1e-4
    if (np.abs(out).sum() > max_gross * (1 + slack) + slack
            or np.any(np.abs(p.B.T @ out) > beta_tol + slack)
            or np.any(out > p.cap_long + slack) or np.any(out < -p.cap_short - slack)):
        return None
    return out


def solve(p: Problem, lam, gamma, max_gross, beta_tol, min_trade, kappa=0.0):
    """Returns (weights, status). On failure the old book is kept (status 'infeasible')."""
    w = _solve(p, lam, gamma, max_gross, beta_tol, kappa=kappa)
    if w is None:
        return p.w_old.copy(), "infeasible"
    allowed = (p.w_old <= p.cap_long + 1e-12) & (p.w_old >= -p.cap_short - 1e-12)
    small = (np.abs(w - p.w_old) < min_trade) & (np.abs(w - p.w_old) > 0) & allowed
    if not small.any():
        return w, "optimal"
    w2 = _solve(p, lam, gamma, max_gross, beta_tol, pinned=small, kappa=kappa)
    return (w2, "no_trade_band") if w2 is not None else (w, "optimal")
