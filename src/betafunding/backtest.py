"""Daily backtest of the optimiser.

At the close of day d-1 the book is re-solved from row d-1 of the signals, then held through
day d:

    price   = sum w_i * r_i,d
    funding = -sum w_i * F_i,d          (F = funding settled inside day d; long pays if > 0)
    cost    = cost_bps * sum |w_i - w_i,old|

w_old is yesterday's weight *drifted* by day d's moves, so turnover counts what the market
pushed the book off target as well as what the signal changed. Returns are additive fractions of
starting equity.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import Panel
from .optimizer import Problem, solve
from .risk import covariance
from .signals import Signals


class StopBook:
    """Per-name stop state. Once a held name has moved stop_loss against (or take_profit for)
    the side held since entry, it is closed at the next decision and kept out for cooldown_days."""

    def __init__(self, growth=None, side=None, blocked_until=None):
        self.growth, self.side = dict(growth or {}), dict(side or {})
        self.blocked_until = dict(blocked_until or {})

    def blocked(self, day):
        return [k for k, until in self.blocked_until.items() if until >= day]

    def update(self, day, w: pd.Series, r: pd.Series, stop=None, take=None, cooldown_days=3):
        """Apply day `day`'s returns `r` to the book `w` held through it."""
        if not (stop or take):
            return
        for k, wk in w.items():
            if wk == 0:
                self.growth.pop(k, None); self.side.pop(k, None)
                continue
            if self.side.get(k) != np.sign(wk):
                self.growth[k], self.side[k] = 1.0, np.sign(wk)
            self.growth[k] *= 1 + r[k]
            move = self.side[k] * (self.growth[k] - 1)
            if (stop and move <= -stop) or (take and move >= take):
                self.blocked_until[k] = day + pd.Timedelta(days=int(cooldown_days))
                self.growth.pop(k); self.side.pop(k)

    def to_dict(self):
        return {"growth": self.growth, "side": {k: int(v) for k, v in self.side.items()},
                "blocked_until": {k: str(v.date()) for k, v in self.blocked_until.items()}}

    @classmethod
    def from_dict(cls, d):
        return cls(d.get("growth"), {k: float(v) for k, v in d.get("side", {}).items()},
                   {k: pd.Timestamp(v) for k, v in d.get("blocked_until", {}).items()})


def make_inputs(sig: Signals, cfg, day, w_old: pd.Series, universe=None,
                panel: Panel = None, blocked=()) -> Problem:
    """Optimiser inputs from the signals at the close of `day`.

    Tradable = eligible that day (and inside `universe` if given), and not stopped out
    (`blocked`). Names held but no longer tradable stay in the problem with a zero cap, so they
    are closed, not silently dropped.

    The per-day cost of holding a long is the funding forecast, less the expected price drift
    when funding.drift_adjust is on; the entry floors (min_apy_long / min_apy_short, APY %) apply
    to that net figure. risk.vol_cap bounds each name's weight at vol_cap / its daily volatility.
    """
    bm = cfg["benchmarks"]
    elig = sig.eligible.loc[day]
    if universe is not None:
        elig = elig & elig.index.isin(universe)
    sel = cfg.get("selection", {})
    if sel.get("mode", "threshold") == "topk":
        # pre-selection: only the k highest and k lowest forecasts (benchmarks, if traded,
        # stay in as the hedge)
        fc = sig.f_hat.loc[day][elig & ~elig.index.isin(bm)].sort_values()
        k = int(sel.get("k", 5))
        pick = set(fc.index[:k]) | set(fc.index[-k:]) | set(elig.index[elig & elig.index.isin(bm)])
        elig = elig & elig.index.isin(pick)
    held = w_old[w_old.abs() > 0].index
    tick = list(elig[elig].index.union(held))
    tradable = elig.reindex(tick, fill_value=False).to_numpy()

    carry = sig.f_hat.loc[day, tick].fillna(0.0).to_numpy()
    f_hat = carry - (sig.drift.loc[day, tick].fillna(0.0).to_numpy() if sig.drift is not None else 0.0)
    B = np.column_stack([sig.betas[f].loc[day, tick].fillna(0.0).to_numpy() for f in sig.betas])
    rv = sig.resid_var.loc[day, tick].fillna(0.0).to_numpy()

    is_bm = np.isin(tick, bm)
    cap = np.full(len(tick), float(cfg["risk"]["max_weight"]))
    vol_cap = cfg["risk"].get("vol_cap")
    if vol_cap and sig.vol is not None:
        v = sig.vol.loc[day, tick].to_numpy()
        cap = np.where(~is_bm & np.isfinite(v) & (v > 0), np.minimum(cap, vol_cap / np.where(v > 0, v, 1)), cap)
    cap[~tradable | np.isin(tick, list(blocked))] = 0.0
    cap_long, cap_short = cap.copy(), cap.copy()
    fc = cfg["funding"]
    if fc["require_sign"]:
        # a name enters only on the side its funding pays, and only past the floor: a short at
        # +1% APY is idiosyncratic risk with no carry behind it. Separate floors per side.
        base = fc.get("min_apy", 0.0)
        fl = (base if fc.get("min_apy_long") is None else fc["min_apy_long"]) / 100 / 365
        fs = (base if fc.get("min_apy_short") is None else fc["min_apy_short"]) / 100 / 365
        cap_short[(f_hat <= fs) & ~is_bm] = 0.0
        cap_long[(f_hat > -fl) & ~is_bm] = 0.0

    Sigma, cond = None, float("nan")
    model = cfg["risk"].get("model", "diag")
    if model != "diag":
        c = cfg["risk"].get("cov", {})
        Sigma, cond = covariance(panel.ret, day, tick, rv, "sample" if model == "cov" else model,
                                 c.get("window_days", 60), c.get("short_days", 7),
                                 c.get("short_weight", 0.5), c.get("ewma_lambda", 0.94))

    return Problem(tick, f_hat, B, rv, cap_long, cap_short,
                  w_old.reindex(tick, fill_value=0.0).to_numpy(), Sigma, cond, carry)


def solver_args(cfg, lam=None, gamma=None, max_gross=None, horizon=None) -> dict:
    """The optimiser's settings from the config; gamma is spread over horizon_days."""
    risk, to = cfg["risk"], cfg["turnover"]
    gamma = to["gamma"] if gamma is None else gamma
    return {"lam": risk["lambda"] if lam is None else lam,
            "gamma": gamma / (to.get("horizon_days", 1) if horizon is None else horizon),
            "max_gross": risk["max_gross"] if max_gross is None else max_gross,
            "beta_tol": cfg["beta"]["tolerance"], "min_trade": to["min_trade"],
            "kappa": to.get("kappa", 0.0)}


@dataclass
class Result:
    daily: pd.DataFrame          # price, funding, cost, net, gross, turnover, beta_*, status
    weights: pd.DataFrame        # target weights, one row per traded day
    last_w: pd.Series            # drifted book after the last day, to chain the next segment


def run(panel: Panel, sig: Signals, cfg, start=None, end=None, universe=None,
        lam=None, gamma=None, max_gross=None, horizon=None, w0=None) -> Result:
    risk = cfg["risk"]
    every = int(cfg["turnover"].get("rebalance_days", 1))
    args = solver_args(cfg, lam, gamma, max_gross, horizon)
    fee = cfg["cost_bps"] / 1e4
    stops = StopBook()
    stop_args = (risk.get("stop_loss"), risk.get("take_profit"), risk.get("cooldown_days", 3))

    days = panel.days
    start = pd.Timestamp(start or cfg["backtest"]["start"])
    end = end or cfg["backtest"].get("end")
    days = days[(days >= start) & ((days <= pd.Timestamp(end)) if end else True)]
    w_old = pd.Series(dtype=float) if w0 is None else w0.copy()
    rows, wts = [], {}
    for n_day, d in enumerate(days):
        prev = d - pd.Timedelta(days=1)
        if prev not in panel.close.index:
            continue
        inp = make_inputs(sig, cfg, prev, w_old, universe, panel, stops.blocked(prev))
        if not inp.coins:
            w_old = pd.Series(dtype=float)
            continue
        if n_day % every == 0:
            w, status = solve(inp, **args)
        else:
            # calendar rebalancing: between rebalance days the book only drifts; a name that
            # stopped being tradable (delisted, illiquid) is still closed
            live = sig.eligible.loc[prev].reindex(inp.coins, fill_value=False).to_numpy()
            w = np.where(live, inp.w_old, 0.0)
            status = "hold"
        w = pd.Series(w, index=inp.coins)
        r = panel.ret.loc[d, inp.coins].fillna(0.0)
        f = panel.funding.loc[d, inp.coins].fillna(0.0)
        turnover = float((w - pd.Series(inp.w_old, index=inp.coins)).abs().sum())
        price, funding, cost = float(w @ r), float(-(w @ f)), fee * turnover
        net = price + funding - cost
        row = {"date": d, "price": price, "funding": funding, "cost": cost, "net": net,
               "funding_fcst": float(-((inp.carry if inp.carry is not None else inp.f) @ w.to_numpy())),
               "gross": float(w.abs().sum()), "turnover": turnover,
               "n_long": int((w > 0).sum()), "n_short": int((w < 0).sum()), "status": status,
               # concentration: Herfindahl of |w| / gross; 1 / hhi = effective number of names
               "hhi": float(((w.abs() / w.abs().sum()) ** 2).sum()) if w.abs().sum() > 0 else np.nan,
               "max_w": float(w.abs().max()) if len(w) else 0.0, "cond": inp.cond}
        for k, f in enumerate(sig.betas):
            row[f"beta_{f}"] = float(inp.B[:, k] @ w.to_numpy())
        rows.append(row)
        wts[d] = w[w != 0]
        stops.update(d, w, r, *stop_args)
        w_old = (w * (1 + r) / (1 + net))
        w_old = w_old[w_old != 0]
    daily = pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()
    weights = pd.DataFrame(wts).T.sort_index().fillna(0.0) if wts else pd.DataFrame()
    return Result(daily, weights, w_old)


def _hold_days(weights: pd.DataFrame) -> float:
    """Mean length, in days, of an unbroken same-side position in one name."""
    if weights is None or weights.empty:
        return np.nan
    runs = []
    for c in weights:
        side = np.sign(weights[c].to_numpy())
        n = 0
        for i, x in enumerate(side):
            if x != 0 and (i == 0 or x == side[i - 1]):
                n += 1
            elif x != 0:
                runs.append(n); n = 1
            else:
                if n: runs.append(n)
                n = 0
        if n:
            runs.append(n)
    return float(np.mean(runs)) if runs else np.nan


def score(net: pd.Series, mode="sharpe") -> float:
    """Selection score for walk-forward: sharpe, sortino, calmar, or a composite
    (0.4 sortino + 0.3 sharpe + 0.3 calmar)."""
    net = net.dropna()
    if len(net) < 6 or net.std() == 0:
        return np.nan
    ann = net.mean() * 365
    sharpe = ann / (net.std() * np.sqrt(365))
    down = net[net < 0].std() * np.sqrt(365)
    sortino = ann / down if down > 0 else np.nan
    eq = net.cumsum()
    dd = -(eq - eq.cummax()).min()
    calmar = ann / dd if dd > 0 else np.nan
    if mode == "composite":
        return 0.4 * sortino + 0.3 * sharpe + 0.3 * calmar
    return {"sharpe": sharpe, "sortino": sortino, "calmar": calmar}[mode]


def summarize(daily: pd.DataFrame, panel: Panel = None, benchmarks=("BTC", "ETH"),
              weights: pd.DataFrame = None) -> dict:
    """Performance and risk measures of a daily P&L frame from run().

    Returns are additive daily fractions of starting equity; annualised on 365 days (perps
    settle every day). Ex-post betas regress net P&L on the benchmarks' daily returns.
    """
    if daily.empty:
        return {}
    net = daily["net"]
    eq = net.cumsum()
    dd = eq - eq.cummax()
    vol = net.std() * np.sqrt(365)
    ann = net.mean() * 365
    down = net[net < 0].std() * np.sqrt(365)
    # longest stretch below a previous high, in days
    under = (dd < 0).astype(int)
    dd_days = int(under.groupby((under == 0).cumsum()).sum().max()) if len(under) else 0
    monthly = net.groupby(net.index.to_period("M")).sum()
    wins, losses = net[net > 0], net[net < 0]
    var95 = net.quantile(0.05)
    fc = daily["funding_fcst"].sum() if "funding_fcst" in daily else np.nan
    out = {
        "days": len(net),
        "net_apy_%": ann * 100,
        "funding_apy_%": daily["funding"].mean() * 365 * 100,
        "price_apy_%": daily["price"].mean() * 365 * 100,
        "cost_apy_%": daily["cost"].mean() * 365 * 100,
        "vol_%": vol * 100,
        "sharpe": ann / vol if vol > 0 else np.nan,
        "sortino": ann / down if down > 0 else np.nan,
        "calmar": ann / -dd.min() if dd.min() < 0 else np.nan,
        "composite": score(net, "composite"),
        "max_dd_%": dd.min() * 100,
        "max_dd_days": dd_days,
        "hit_rate_%": (net > 0).mean() * 100,
        "monthly_win_%": (monthly > 0).mean() * 100,
        "profit_factor": wins.sum() / -losses.sum() if len(losses) and losses.sum() < 0 else np.nan,
        "win_loss_ratio": wins.mean() / -losses.mean() if len(wins) and len(losses) else np.nan,
        "skew": net.skew(),
        "kurtosis": net.kurt(),
        "var95_%": var95 * 100,
        "cvar95_%": net[net <= var95].mean() * 100,
        "worst_day_%": net.min() * 100,
        "best_day_%": net.max() * 100,
        "funding_capture_%": daily["funding"].sum() / fc * 100 if fc and fc > 0 else np.nan,
        "avg_gross": daily["gross"].mean(),
        "avg_n_long": daily["n_long"].mean(),
        "avg_n_short": daily["n_short"].mean(),
        "avg_turnover_%": daily["turnover"].mean() * 100,
        "hold_days": _hold_days(weights),
        "eff_n": float((1 / daily["hhi"]).mean()) if "hhi" in daily else np.nan,
        "avg_max_w": daily["max_w"].mean() if "max_w" in daily else np.nan,
        "median_cond": daily["cond"].median() if "cond" in daily else np.nan,
        # tail-robust views: the total without its 10 best days, the typical month, and the
        # drawdown of a compounding account (additive sums understate it)
        "ex_best10_%": net.sort_values().iloc[:-10].sum() * 100,
        "median_month_%": monthly.median() * 100,
        "max_dd_compound_%": ((1 + net).cumprod() / (1 + net).cumprod().cummax() - 1).min() * 100,
    }
    if panel is not None:
        X = panel.ret.reindex(net.index)[list(benchmarks)].dropna()
        y = net.reindex(X.index)
        if len(X) > 20:
            A = np.column_stack([np.ones(len(X)), X.to_numpy()])
            coef = np.linalg.lstsq(A, y.to_numpy(), rcond=None)[0]
            for b, c in zip(benchmarks, coef[1:]):
                out[f"expost_beta_{b}"] = c
    return out
