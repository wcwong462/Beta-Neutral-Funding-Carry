"""Funding forecast from each coin's realised daily funding: a rolling AR(1) (the strategy's) or
an EMA (the research comparison).

Tomorrow's funding is forecast as f(t+1) = a + b f(t), fitted by OLS on the trailing `window`
days. The slope is clipped to [-1, 1] so a short noisy window cannot extrapolate; a coin whose
funding did not move in the window forecasts its mean.
"""

import pandas as pd

APY = 365 * 100          # daily rate -> APY %


def ar1_forecast(funding: pd.DataFrame, window=30, min_days=15) -> pd.DataFrame:
    """Per coin, the per-day funding a long is expected to pay tomorrow, known at today's close."""
    y, x = funding, funding.shift(1)
    both = y.notna() & x.notna()
    Y, X = y.where(both), x.where(both)
    roll = lambda d: d.rolling(window, min_periods=min_days)
    vx = roll(X).var()
    slope = (roll(Y).cov(X) / vx).where(vx > 1e-16, 0.0).clip(-1, 1)
    intercept = roll(Y).mean() - slope * roll(X).mean()
    return (intercept + slope * funding).where(roll(Y).count() >= min_days)


def funding_forecast(funding: pd.DataFrame, method="ar1", halflife=7, ar_window=30, ar_min_days=15):
    """Per-day funding a long is expected to pay tomorrow: rolling AR(1), or an EMA."""
    if method == "ar1":
        return ar1_forecast(funding, ar_window, ar_min_days)
    if method == "ema":
        return funding.ewm(halflife=halflife, min_periods=3).mean()
    raise ValueError(f"unknown forecast method {method!r}")
