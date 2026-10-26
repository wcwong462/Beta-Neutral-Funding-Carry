"""Today's rebalance plan: target book, trades, and the summary in the strategy's format."""

import json
import os

import numpy as np
import pandas as pd

from .backtest import StopBook, make_inputs, solver_args
from .data import Panel
from .funding import APY
from .optimizer import solve
from .signals import Signals


def load_book(path, equity_usd=10_000.0):
    """Book state from JSON: {"as_of", "equity_usd", "weights", "stops"}. `as_of` is the close
    the weights were set at; an empty book is returned when the file does not exist."""
    if not path or not os.path.exists(path):
        return None, float(equity_usd), pd.Series(dtype=float), StopBook()
    with open(path) as f:
        s = json.load(f)
    as_of = pd.Timestamp(s["as_of"]) if s.get("as_of") else None
    return (as_of, float(s.get("equity_usd", equity_usd)), pd.Series(s.get("weights", {}), dtype=float),
            StopBook.from_dict(s.get("stops", {})))


def save_book(path, as_of, equity_usd, weights: pd.Series, stops: StopBook):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump({"as_of": str(pd.Timestamp(as_of).date()), "equity_usd": equity_usd,
                   "weights": weights[weights != 0].round(6).to_dict(), "stops": stops.to_dict()},
                  f, indent=2)


def roll_forward(panel: Panel, cfg, as_of, weights: pd.Series, stops: StopBook):
    """Carry a book set at the close of `as_of` through every day since: weights drift with
    each day's moves and funding, and the stop state sees each day's return."""
    if as_of is None or weights.empty:
        return weights
    w = weights.copy()
    for d in panel.days[panel.days > as_of]:
        r = panel.ret.loc[d].reindex(w.index).fillna(0.0)
        f = panel.funding.loc[d].reindex(w.index).fillna(0.0)
        risk = cfg["risk"]
        stops.update(d, w, r, risk.get("stop_loss"), risk.get("take_profit"),
                     risk.get("cooldown_days", 3))
        w = w * (1 + r) / (1 + float(w @ r - w @ f))
    return w


def plan(panel: Panel, sig: Signals, cfg, w_old: pd.Series, stops: StopBook,
         equity_usd: float, day=None):
    """Target book for the next day, as (summary dict, trade table)."""
    day = day or panel.days[-1]
    p = make_inputs(sig, cfg, day, w_old, panel=panel, blocked=stops.blocked(day))
    w, status = solve(p, **solver_args(cfg))
    w = pd.Series(w, index=p.coins)
    old = pd.Series(p.w_old, index=p.coins)
    dw = w - old
    keep = (w != 0) | (old != 0)
    table = pd.DataFrame({
        "side": np.where(w > 0, "LONG", np.where(w < 0, "SHORT", "CLOSE")),
        "current_%": old * 100, "target_%": w * 100, "trade_usd": dw * equity_usd,
        **{f"beta_{b}": pd.Series(p.B[:, k], index=p.coins) for k, b in enumerate(sig.betas)},
        "funding_fcst_apy_%": pd.Series(p.f, index=p.coins) * APY,
    })[keep].sort_values("target_%")
    summary = {
        "day": str(day.date()), "status": status, "equity_usd": equity_usd,
        **{f"net_beta_{b}": float(p.B[:, k] @ w.to_numpy()) for k, b in enumerate(sig.betas)},
        "expected_funding_apy_%": float(-(p.f @ w.to_numpy())) * APY,
        "gross": float(w.abs().sum()), "turnover_%": float(dw.abs().sum()) * 100,
        "fee_%": float(dw.abs().sum()) * cfg["cost_bps"] / 100,
        "stopped_out": sorted(stops.blocked(day)),
    }
    return summary, table


def to_markdown(summary, table) -> str:
    s = summary
    betas = ", ".join(f"{k[9:]} {v:+.3f}" for k, v in s.items() if k.startswith("net_beta_"))
    beta_cols = [c for c in table.columns if c.startswith("beta_")]
    out = [f"### Daily rebalance ({s['day']})",
           f"- Equity: ${s['equity_usd']:,.0f}",
           f"- Net beta: {betas} (target 0)",
           f"- Expected funding: {s['expected_funding_apy_%']:.1f}% APY",
           f"- Gross: {s['gross']:.2f}x · turnover {s['turnover_%']:.1f}% (fee {s['fee_%']:.3f}%) · solver {s['status']}",
           f"- Stopped out (cooling down): {', '.join(s['stopped_out']) or 'none'}",
           "",
           "| Coin | Side | Current | Target | Trade | " + " | ".join(c[5:] + " beta" for c in beta_cols)
           + " | Funding fcst APY |",
           "|" + " --- |" * (6 + len(beta_cols))]
    for coin, r in table.iterrows():
        t = r["trade_usd"]
        act = "hold" if abs(t) < 1 else f"{'buy' if t > 0 else 'sell'} ${abs(t):,.0f}"
        out.append(f"| {coin} | {r['side']} | {r['current_%']:.1f}% | {r['target_%']:.1f}% | {act} | "
                   + " | ".join(f"{r[c]:.2f}" for c in beta_cols) + f" | {r['funding_fcst_apy_%']:.0f}% |")
    return "\n".join(out)
