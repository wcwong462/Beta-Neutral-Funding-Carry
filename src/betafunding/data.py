"""Market data: the daily panel the model runs on, and a REST top-up for the latest days.

Store layout (as written by research/01 from the data.binance.vision archive):
    <root>/price/<SYM>USDT-1d-data.parquet     index timestamp (naive UTC), open .. close, volume
    <root>/funding/<SYM>USDT-funding-data.parquet  index fundingTime, fundingRate

Clock: a daily bar is the UTC day. The book is decided at the close of day d (00:00 UTC of
d+1) and held through day d+1. Funding for day d is every settlement stamped inside day d, so
8h, 4h and 1h contracts are comparable without annualising rate by rate.
"""

import glob
import logging
import os
import time
from dataclasses import dataclass

import pandas as pd

log = logging.getLogger("betafunding.data")


@dataclass
class Panel:
    close: pd.DataFrame          # daily close, columns = coin (BTC, ETH, ...)
    ret: pd.DataFrame            # daily simple return
    funding: pd.DataFrame        # funding settled inside the day (a long pays if > 0)
    dollar_volume: pd.DataFrame  # daily USD volume

    @property
    def days(self):
        return self.close.index


def _coin(path, suffix):
    name = os.path.basename(path).replace(suffix, "")
    return name[:-4] if name.endswith("USDT") else name


def load_panel(root, timeframe="1d", drop_today=True) -> Panel:
    """Read the parquet store into aligned daily matrices."""
    closes, vols, funds = {}, {}, {}
    for path in glob.glob(os.path.join(root, "price", f"*USDT-{timeframe}-data.parquet")):
        df = pd.read_parquet(path)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        coin = _coin(path, f"-{timeframe}-data.parquet")
        closes[coin] = pd.to_numeric(df["close"], errors="coerce")
        vols[coin] = closes[coin] * pd.to_numeric(df["volume"], errors="coerce")
    for path in glob.glob(os.path.join(root, "funding", "*USDT-funding-data.parquet")):
        df = pd.read_parquet(path)
        funds[_coin(path, "-funding-data.parquet")] = pd.to_numeric(df["fundingRate"], errors="coerce")
    if not closes:
        raise ValueError(f"no price parquet under {root}/price")

    close = pd.concat(closes, axis=1).sort_index()
    close = close.groupby(close.index.floor("D")).last()
    if drop_today:
        close = close[close.index < pd.Timestamp.utcnow().tz_localize(None).normalize()]
    dv = pd.concat(vols, axis=1).groupby(lambda t: t.floor("D")).sum(min_count=1).reindex(close.index)

    raw = pd.concat(funds, axis=1).sort_index() if funds else pd.DataFrame(index=close.index)
    daily = raw.groupby(raw.index.floor("D")).sum(min_count=1).reindex(close.index)
    funding = daily.reindex(columns=close.columns)
    # a day with a price and no settlement is a zero, once the coin's funding record has begun
    first = raw.apply(pd.Series.first_valid_index)
    for c in funding.columns:
        if pd.notna(first.get(c)):
            live = (funding.index >= first[c].floor("D")) & close[c].notna()
            funding.loc[live, c] = funding.loc[live, c].fillna(0.0)
    return Panel(close, close.pct_change(fill_method=None), funding, dv)


def refresh_recent(root, days=60, coins=None, exchange="binanceusdm"):
    """Top the store up with the last `days` of daily klines and funding from the REST API.

    The archive only has complete months; the daily plan needs yesterday. Active USDT perps
    tagged COIN by default. Existing rows are kept; overlapping rows are replaced.
    """
    import ccxt
    ex = getattr(ccxt, exchange)({"enableRateLimit": True})
    ex.load_markets()
    if coins is None:
        coins = sorted({m["base"] for m in ex.markets.values()
                        if m.get("swap") and m.get("quote") == "USDT" and m.get("active")
                        and (m.get("info") or {}).get("underlyingType") == "COIN"})
    since = ex.milliseconds() - days * 86_400_000
    os.makedirs(os.path.join(root, "price"), exist_ok=True)
    os.makedirs(os.path.join(root, "funding"), exist_ok=True)
    for i, coin in enumerate(coins):
        sym = f"{coin}/USDT:USDT"
        if sym not in ex.markets:
            continue
        try:
            k = ex.fetch_ohlcv(sym, "1d", since=since, limit=days + 1)
            f = ex.fetch_funding_rate_history(sym, since=since, limit=1000)
        except Exception as e:                      # one coin failing should not stop the run
            log.warning(f"{coin}: {str(e)[:100]}")
            continue
        _merge(os.path.join(root, "price", f"{coin}USDT-1d-data.parquet"),
               pd.DataFrame(k, columns=["ts", "open", "high", "low", "close", "volume"])
               .assign(timestamp=lambda x: pd.to_datetime(x.ts, unit="ms")).drop(columns="ts")
               .set_index("timestamp"))
        if f:
            _merge(os.path.join(root, "funding", f"{coin}USDT-funding-data.parquet"),
                   pd.DataFrame({"fundingTime": pd.to_datetime([r["timestamp"] for r in f], unit="ms"),
                                 "fundingRate": [r["fundingRate"] for r in f]}).set_index("fundingTime"))
        if (i + 1) % 50 == 0:
            log.info(f"refreshed {i + 1}/{len(coins)}")
        time.sleep(ex.rateLimit / 1000)


def _merge(path, new):
    old = pd.read_parquet(path) if os.path.exists(path) else None
    df = new if old is None else pd.concat([old[~old.index.isin(new.index)], new])
    df.sort_index().to_parquet(path)
