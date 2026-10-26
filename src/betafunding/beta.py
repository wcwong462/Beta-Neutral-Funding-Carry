"""Market betas: each coin's exposure to BTC and ETH, and what is left after hedging.

The strategy uses `joint`: one rolling OLS of a coin's daily return on BTC and ETH together, so
the book can be held neutral to each. The residual variance is the trailing variance of the
hedged daily return: the coin-specific risk the hedge cannot remove, which the optimiser prices.
"""

import numpy as np
import pandas as pd


def rolling_ols(R: pd.DataFrame, X: pd.DataFrame, window_days=30, min_days=15, bars_per_day=1,
                days=None):
    """Rolling OLS of every column of R on all columns of X at once.

    Returns ({factor: beta}, {factor: sampling variance of that beta}, R^2), each a frame indexed
    like R - or, when `days` is given, sampled at each day's last bar and reindexed to `days`.
    Var(beta_k) = s2_e / (n - 1) * [Cxx^-1]_kk, with s2_e the regression's residual variance.
    """
    factors = list(X.columns)
    win, minp = window_days * bars_per_day, min_days * bars_per_day
    both = X.notna().all(axis=1)
    Xm, Y = X.where(both, axis=0), R.where(both, axis=0)
    roll = lambda s: s.rolling(win, min_periods=minp)
    K = len(factors)

    cxx = np.full((len(R), K, K), np.nan)                                 # T x K x K
    for i, a in enumerate(factors):
        for j, b in enumerate(factors):
            cxx[:, i, j] = roll(Xm[a]).cov(Xm[b]).to_numpy()
    cxy = np.stack([roll(Y).cov(Xm[f]).to_numpy() for f in factors], axis=-1)  # T x N x K
    inv = np.full_like(cxx, np.nan)
    ok = ~np.isnan(cxx).any(axis=(1, 2))
    inv[ok] = np.linalg.pinv(cxx[ok])
    beta = np.einsum("tkl,tnl->tnk", inv, cxy)                            # T x N x K

    vy = roll(Y).var().to_numpy()
    s2e = np.clip(vy - np.einsum("tnk,tkl,tnl->tn", beta, cxx, beta), 0, None)
    n = roll(Y).count().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        bvar = s2e[:, :, None] / (n[:, :, None] - 1) * np.diagonal(inv, axis1=1, axis2=2)[:, None, :]
        r2 = 1 - s2e / vy

    def frame(a):
        df = pd.DataFrame(a, index=R.index, columns=R.columns)
        return df if days is None else df.groupby(R.index.floor("D")).last().reindex(days)
    return ({f: frame(beta[:, :, k]) for k, f in enumerate(factors)},
            {f: frame(bvar[:, :, k]) for k, f in enumerate(factors)}, frame(r2))


def residual_variance(ret: pd.DataFrame, betas, factor_returns, window=30, min_days=15):
    """Trailing variance of the daily return left after hedging with `betas`."""
    hedged = ret - sum(betas[f].mul(factor_returns[f], axis=0) for f in betas)
    return hedged.rolling(window, min_periods=min_days).var()


def _factors(ret):
    return {"BTC": ret["BTC"], "ETH": ret["ETH"], "MKT": (ret["BTC"] + ret["ETH"]) / 2}


# joint = one regression on BTC and ETH, neutral to each; the others hedge on a single factor:
# one coin, their average, or per coin whichever fits better.
BETA_TYPES = ("joint", "btc", "eth", "combined", "adaptive")


def market_betas(ret: pd.DataFrame, beta_type="joint", window_days=30, min_days=15,
                 shrink_prior_days=None, clip=None):
    """Daily betas under one market definition, and the residual variance left after hedging.

    joint    : {BTC, ETH} from one 2-factor regression - neutral to each separately
    btc, eth : {BTC} or {ETH}, single-factor
    combined : {MKT}, single factor on the average of BTC and ETH returns
    adaptive : {ADAPT}, per coin and day the single-factor BTC or ETH beta, whichever fits
               better (higher R^2); the book is neutral to that mixed number

    shrink_prior_days pulls each beta toward its longer-window estimate by inverse sampling
    variance; clip bounds it. Row d uses returns up to the close of d.

    Returns (betas, resid, exposures): betas {factor: beta} is what the optimiser holds neutral;
    exposures {BTC | ETH | MKT: beta} is the hedge implied per factor return (for adaptive, the
    chosen factor's beta and 0 on the other) - see hedged_returns.
    """
    f = _factors(ret)

    def est(names):
        X = pd.concat({n: f[n] for n in names}, axis=1)
        b, v, r2 = rolling_ols(ret, X, window_days, min_days)
        if shrink_prior_days:
            prior, pvar, _ = rolling_ols(ret, X, shrink_prior_days, min_days)
            for k in names:
                w_raw, w_pri = 1 / v[k], 1 / pvar[k]
                post = (b[k] * w_raw + prior[k] * w_pri) / (w_raw + w_pri)
                b[k] = post.where(post.notna(), b[k])
        if clip:
            b = {k: x.clip(-clip, clip) for k, x in b.items()}
        return b, r2

    if beta_type == "adaptive":
        (bb, r2b), (be, r2e) = est(["BTC"]), est(["ETH"])
        use_btc = r2b.fillna(-1) >= r2e.fillna(-1)
        betas = {"ADAPT": bb["BTC"].where(use_btc, be["ETH"])}
        exposures = {"BTC": bb["BTC"].where(use_btc, 0), "ETH": be["ETH"].where(~use_btc, 0)}
    else:
        names = {"joint": ["BTC", "ETH"], "btc": ["BTC"], "eth": ["ETH"], "combined": ["MKT"]}
        if beta_type not in names:
            raise ValueError(f"unknown beta type {beta_type!r}; one of {BETA_TYPES}")
        betas, _ = est(names[beta_type])
        exposures = betas
    return betas, residual_variance(ret, exposures, f, window_days, min_days), exposures


def hedged_returns(ret: pd.DataFrame, exposures, lag=1):
    """Daily returns after hedging with the exposures known `lag` days earlier (out of sample)."""
    f = _factors(ret)
    return ret - sum(exposures[k].shift(lag).mul(f[k], axis=0) for k in exposures)
