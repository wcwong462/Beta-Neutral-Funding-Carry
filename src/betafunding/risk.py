"""Covariance for the optimiser's risk term.

With 100-400 tradable coins and a 30-90 day window there are fewer days than coins, so the
sample covariance is singular and the optimiser would lever whatever pair the noise makes look
uncorrelated. The model's default risk term is each coin's residual variance (no correlations);
when correlations are wanted:

    lw     : Ledoit-Wolf shrinkage of the sample toward a scaled identity - analytic intensity,
             invertible at any size (the model's option, risk.model: lw)
    sample : rolling sample covariance, blended short / long window (research comparison)
    ewma   : RiskMetrics, weight lambda^k on the k-th most recent bar (research comparison)
"""

import numpy as np
import pandas as pd

METHODS = ("sample", "lw", "ewma")


def covariance(ret: pd.DataFrame, day, coins, fallback_var, method="lw", window_days=60,
               short_days=7, short_weight=0.5, ewma_lambda=0.94, bars_per_day=1):
    """Daily-scale covariance of `coins` over the window ending at the close of `day`.

    `ret` holds the bars to estimate from (daily, or intraday with `bars_per_day` > 1). A coin
    with too little history gets `fallback_var` (its residual variance) on the diagonal and no
    covariance. Projected onto the PSD cone for the QP. Returns (Sigma, condition number before
    the projection; inf when singular).
    """
    end = day + pd.Timedelta(days=1)
    R = ret.loc[(ret.index >= end - pd.Timedelta(days=window_days)) & (ret.index < end), coins]
    minp = max(int(bars_per_day * 5), 5)
    enough = (R.notna().sum() >= minp).to_numpy()

    if method == "sample":
        short = R[R.index >= end - pd.Timedelta(days=short_days)]
        c_long = R.cov(min_periods=minp)
        c_short = short.cov(min_periods=max(int(bars_per_day * 3), 3))
        S = (short_weight * c_short.fillna(c_long) + (1 - short_weight) * c_long).to_numpy()
    elif method in ("lw", "ewma"):
        S = np.full((len(coins), len(coins)), np.nan)
        if enough.any():
            X = R.loc[:, enough]
            X = (X - X.mean()).fillna(0.0).to_numpy()
            if method == "lw":
                from sklearn.covariance import ledoit_wolf
                sub = ledoit_wolf(X, assume_centered=True)[0]
            else:
                w = ewma_lambda ** np.arange(len(X))[::-1]
                sub = (X * (w / w.sum())[:, None]).T @ X
            S[np.ix_(enough, enough)] = sub
    else:
        raise ValueError(f"unknown covariance method {method!r}; one of {METHODS}")
    S = S * bars_per_day

    d = np.diag(S).copy()
    bad = ~np.isfinite(d) | (d <= 0)
    d[bad] = np.asarray(fallback_var, dtype=float)[bad]
    S = np.nan_to_num(S, nan=0.0)
    S[bad, :] = 0.0
    S[:, bad] = 0.0
    np.fill_diagonal(S, d)
    vals, vecs = np.linalg.eigh((S + S.T) / 2)
    cond = float(vals.max() / vals.min()) if vals.min() > 0 else np.inf
    return (vecs * np.clip(vals, 1e-10, None)) @ vecs.T, cond
