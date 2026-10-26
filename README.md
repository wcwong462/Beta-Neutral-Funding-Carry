# BetaFunding: Beta-Neutral Crypto Funding Carry

A daily long/short book of Binance perpetual futures that collects **funding**, the payment
between longs and shorts that keeps a perp's price near spot. When a coin's funding is strongly
positive, the book shorts it and collects; when strongly negative, it goes long and collects.
Net beta to BTC and to ETH is held at zero, so what is left is carry plus coin-specific risk.

Research only: public data, notebooks run top to bottom, nothing here places an order.

## Results

Mar 2024 – Sep 2026, coins with ≥ $20m average daily volume, 5 bps fees.

| | Sharpe | Max drawdown | Without the 10 best days |
|---|---|---|---|
| Strategy as first designed | 1.62 | −62% | +97% |
| **Final strategy** (in-sample) | **1.87** | **−23%** | **+68%** |
| **Walk-forward**, 824 out-of-sample days | **1.45** | −53% | +33% |

The walk-forward drawdown comes from its first year: with no track record yet, the selector kept
the original design until Apr 2025. From May 2025 it chose the final strategy, and 8 of 9 blocks
were positive.

![steps](research/figures/steps_equity.png)

**Limitations**
- Concentrated book: about 3 longs and 2 shorts at a time.
- 2024 lost money; the profit came in 2025–26.
- The entry threshold, stop-loss and position cap were chosen on the full sample.
- Daily bars only, and no live record. Sized for a small account (≤ $10k).

## The model

Each day at 00:00 UTC, solve for weights `w` (signed, fraction of equity):

    max_w   −f̂·w  −  λ·Σ σ²_i w_i²  −  γ·Σ|w − w_old|
    s.t.    |Σ w_i β_i| ≤ 0.02 for BTC and for ETH
            Σ|w_i| ≤ 2,   |w_i| ≤ 0.20

- **f̂**: tomorrow's funding from a rolling 30-day AR(1). Funding is persistent (half-life ~1.4 days).
- **β**: rolling joint OLS of each coin on BTC and ETH.
- **σ²**: each coin's residual variance after the hedge.
- **Rules**: trade only coins with |f̂| > 300% APY; skip weight changes under 3%; 20% stop-loss
  with a 3-day cooldown.
- **Solver**: cvxpy. Every solution is checked against the constraints.

## How it got there

**Data.** All 667 Binance USDT perps that traded since Jan 2024, including the 135 that were
delisted. Using only today's survivors picks the winners and makes every short lose.

**Problems solved along the way** (notebook 02):

| Problem | Fix |
|---|---|
| ~150 coins and a 60-day window: the sample covariance is singular, and the optimiser levers noise | per-coin residual variance (Sharpe 1.62); Ledoit–Wolf as the option (1.20). Sample and EWMA lose money once their best days are removed |
| maximising funding alone puts the book in 1–2 coins | risk penalty λ |
| one day of carry never beats a one-off fee, so the book stays empty | fee spread over a ~5-day holding period |
| solver returned a 221× gross book flagged "optimal_inaccurate" | constraint check on every solution |

**Where the money comes from** (03). +210% APY from funding, −86% from price, −10% from fees.
The hedge works: the market part of the price P&L has 1.9% volatility, the coin-specific part
68%. Most of the profit comes from longs entered at |forecast| > 300% APY.

![by forecast bucket](research/figures/attr_buckets.png)

**Improvements** (04). Each step is kept only if it improves the total *without its 10 best
days*, so a few squeezes cannot drive the choice.

| Step | Change | Sharpe | Without best 10 days | Max DD |
|---|---|---|---|---|
| 0 | as first designed | 1.62 | +97% | −62% |
| 1 | only \|forecast\| > 300% APY; BTC/ETH as the hedge | 1.73 | +98% | −34% |
| 2 | + 20% stop-loss | 1.93 | +127% | −40% |
| 3 | + drift-adjusted forecast (rejected) | 1.50 | +43% | −41% |
| 4 | + 20% single-coin cap (**final**) | 1.87 | +68% | −23% |

After step 2 the remaining drawdown came from single-coin squeezes: a +57% move on a 39% short
cost 22% of equity. Capping each coin at 20% halves the drawdown for 0.06 of Sharpe.

**Walk-forward.** Each 60-day block is traded with whichever step scored best on all earlier days.

![walk-forward](research/figures/walkforward.png)

**Liquidity floor** (06, not the main result). A lower floor admits thinner coins with wilder
squeezes. Sharpe stays between 1.6 and 2.1 for floors from $100k to $50m.

## Code

```
config/strategy.yaml    every setting; defaults = final strategy
src/betafunding/        data → signals → optimiser → backtest / daily plan
research/01–06          notebooks: data, research, attribution, improvements, daily plan, liquidity
```

The original design is `override(cfg, ORIGINAL_SPEC)`; each experiment is a few dotted overrides.

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace research/01_fetch_data.ipynb   # data, ~15 min
PYTHONPATH=src python -m betafunding backtest      # final strategy
PYTHONPATH=src python -m betafunding plan          # today's target book
```

**No look-ahead**: every statistic on day d uses data up to the close of d and trades day d+1.
The liquidity filter and the walk-forward use past days only.
