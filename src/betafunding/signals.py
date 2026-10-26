"""Signals: everything the optimiser reads, computed once over the whole panel from the config.

Every statistic is indexed by the day it was computed *at the close of*. The backtest reads
row d-1 to trade day d, so nothing is shifted inside this module.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .beta import hedged_returns, market_betas
from .data import Panel
from .funding import funding_forecast


# Funding-forecast buckets (APY %) for the price-drift model.
DRIFT_EDGES = [-np.inf, -300, -100, -25, 0, 25, 100, 300, np.inf]


def price_drift(panel: Panel, f_hat, exposures, mask=None, edges=DRIFT_EDGES, prior=200):
    """Expected next-day hedged return of a long, by funding-forecast bucket, learnt on the past.

    For a decision at the close of day t: the mean over every past pair (forecast at s, hedged
    return over s+1) with s+1 <= t, in the bucket of today's forecast, shrunk toward 0 by
    `prior` pseudo-observations. Only name-days in `mask` (tradable) teach the model.
    """
    y = hedged_returns(panel.ret, exposures, lag=1).shift(-1)   # row s: return over s+1, betas at s
    x = f_hat * 365 * 100
    valid = x.notna() & y.notna() & (mask if mask is not None else True)
    code = pd.DataFrame(np.digitize(x.fillna(0).to_numpy(), edges[1:-1]), index=x.index, columns=x.columns)
    out = pd.DataFrame(0.0, index=x.index, columns=x.columns)
    for k in range(len(edges) - 1):
        m = valid & (code == k)
        # pair s becomes known at the close of s+1; usable for the decision at t when s <= t-1
        tot = y.where(m, 0).sum(axis=1).cumsum().shift(1).fillna(0)
        cnt = m.sum(axis=1).cumsum().shift(1).fillna(0)
        mean_k = tot / (cnt + prior)
        out = out.where(code != k, np.broadcast_to(mean_k.to_numpy()[:, None], out.shape))
    return out.where(x.notna())


@dataclass
class Signals:
    betas: dict                  # factor (BTC, ETH, MKT or ADAPT) -> daily DataFrame
    resid_var: pd.DataFrame
    f_hat: pd.DataFrame
    dv7: pd.DataFrame            # trailing mean daily USD volume (risk.volume_days)
    eligible: pd.DataFrame       # bool: tradable at that day's close
    vol: pd.DataFrame = None     # trailing daily return volatility (beta.window_days)
    drift: pd.DataFrame = None   # expected next-day hedged return of a long (price_drift), or None


def build_signals(panel: Panel, cfg) -> Signals:
    bm, b, f = cfg["benchmarks"], cfg["beta"], cfg["funding"]
    betas, resid, exposures = market_betas(panel.ret, b.get("type", "joint"), b["window_days"],
                                           b["min_days"], b.get("shrink_prior_days"), b.get("clip"))
    f_hat = funding_forecast(panel.funding, f.get("forecast", "ar1"), f["ema_halflife"],
                             f.get("ar_window", 30), f.get("ar_min_days", 15))
    vd = cfg["risk"].get("volume_days", 7)
    dv7 = panel.dollar_volume.rolling(vd, min_periods=min(3, vd)).mean()
    eligible = (panel.close.notna() & f_hat.notna() & resid.notna()
                & (dv7 >= cfg["risk"]["min_dollar_volume"]))
    for name in betas:
        eligible &= betas[name].notna()
    # BTC and ETH define the market; trade_benchmarks: false keeps them out of the book, so
    # neutrality has to come from the alts themselves
    trade_bm = cfg.get("trade_benchmarks", True)
    if not trade_bm:
        eligible.loc[:, eligible.columns.isin(bm)] = False
    # keep the top_n by trailing volume each day (traded benchmarks always in): bounds the QP
    # and the liquidity of what it can pick, using only volume known at that close
    top_n = cfg["risk"].get("top_n")
    if top_n:
        rank = dv7.where(eligible).rank(axis=1, ascending=False)
        eligible &= (rank <= top_n) | (eligible.columns.isin(bm) & trade_bm)
    vol = panel.ret.rolling(b["window_days"], min_periods=b["min_days"]).std()
    drift = price_drift(panel, f_hat, exposures, mask=eligible,
                        prior=f.get("drift_prior", 200)) if f.get("drift_adjust") else None
    return Signals(betas, resid, f_hat, dv7, eligible, vol, drift)
