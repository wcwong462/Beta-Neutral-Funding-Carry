"""Walk-forward selection: pick the configuration on past days only, trade it on the next block.

    folds:  train = [start, t)  ->  choose the grid point with the best train Sharpe
            test  = [t, t + test_days)  ->  trade that point; every reported day is out of sample
            t += test_days

Train scoring reads each grid point's single full-period run sliced to the train window. In
expanding mode that is exactly a run started at `start` (the book is path-dependent, and the
path is the same). The test block is re-run from the previous block's closing book, so
switching configuration between folds pays its turnover.

A grid point is a dict of dotted config overrides, e.g.
    {"risk.lambda": 10, "funding.min_apy": 10, "funding.forecast": "ar1", "risk.model": "cov"}
"""

import itertools
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtest import run, score
from .config import get, override
from .data import Panel
from .signals import build_signals

# config fields that change the signals (betas, forecast, eligibility) rather than the solve
_SIGNAL_FIELDS = ("beta", "funding.forecast", "funding.ema_halflife", "funding.ar_window",
                  "funding.ar_min_days", "risk.min_dollar_volume", "risk.top_n",
                  "trade_benchmarks", "funding.drift_adjust", "funding.drift_prior")


def expand(grid):
    """{path: [values]} -> list of points (cartesian product)."""
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]


@dataclass
class WFResult:
    folds: pd.DataFrame          # per fold: train/test window, chosen point, train + test Sharpe
    daily: pd.DataFrame          # out-of-sample daily P&L, folds chained
    weights: pd.DataFrame
    full_runs: dict              # point label -> full-period daily frame (for in-sample views)
    full_weights: dict           # point label -> full-period weights


def walk_forward(panel: Panel, cfg, grid=None, start=None, end=None, verbose=True,
                 points=None) -> WFResult:
    """grid: {path: [values]} expanded to its cartesian product. points: {name: {path: value}},
    a list of complete candidate configurations instead - for comparing a few bundles of
    settings without paying for every combination of them."""
    wf = cfg["walkforward"]
    mode = wf.get("score", "sharpe")
    start = pd.Timestamp(start or cfg["backtest"]["start"])
    if points is not None:
        names, points = list(points), list(points.values())
    else:
        if grid is None:
            raise ValueError("walk_forward needs a grid or named points")
        points = expand(grid)
        names = [", ".join(f"{k.split('.')[-1]}={v}" for k, v in p.items()) for p in points]

    # one signal build per distinct signal-affecting configuration
    sig_cache, full = {}, {}
    for i, (name, p) in enumerate(zip(names, points)):
        c = override(cfg, p)
        key = json.dumps({f: get(c, f) for f in _SIGNAL_FIELDS}, sort_keys=True, default=str)
        if key not in sig_cache:
            sig_cache[key] = build_signals(panel, c)
        r = run(panel, sig_cache[key], c, start, end)
        full[name] = (p, c, sig_cache[key], r.daily, r.weights)
        if verbose:
            print(f"[{i+1}/{len(points)}] {name}")

    days = next(iter(full.values()))[3].index
    t = start + pd.Timedelta(days=wf["min_train_days"])
    rows, oos, wts, book = [], [], [], None
    while t <= days[-1]:
        t_end = min(t + pd.Timedelta(days=wf["test_days"] - 1), days[-1])
        lo = start if wf.get("mode", "expanding") == "expanding" else t - pd.Timedelta(days=wf["min_train_days"])
        scores = {k: score(v[3]["net"].loc[lo:t - pd.Timedelta(days=1)], mode) for k, v in full.items()}
        scores = pd.Series(scores).dropna()
        if scores.empty:
            t = t_end + pd.Timedelta(days=1)
            continue
        best = scores.idxmax()
        p, c, sig, _, _ = full[best]
        res = run(panel, sig, c, t, t_end, w0=book)
        book = res.last_w
        oos.append(res.daily)
        wts.append(res.weights)
        rows.append({"train": f"{lo.date()} → {(t - pd.Timedelta(days=1)).date()}",
                     "test": f"{t.date()} → {t_end.date()}", "chosen": best,
                     f"train_{mode}": scores[best], f"test_{mode}": score(res.daily["net"], mode),
                     "test_net_%": res.daily["net"].sum() * 100,
                     "test_funding_%": res.daily["funding"].sum() * 100})
        t = t_end + pd.Timedelta(days=1)

    daily = pd.concat(oos) if oos else pd.DataFrame()
    weights = pd.concat(wts).fillna(0.0) if wts else pd.DataFrame()
    return WFResult(pd.DataFrame(rows), daily, weights, {k: v[3] for k, v in full.items()},
                    {k: v[4] for k, v in full.items()})
