"""Binance public data archive (data.binance.vision): every USDT-M perp that ever traded.

The REST API only serves contracts listed today, so a universe built from it is survivors
only - coins that collapsed and were delisted (LUNA, OM, ...) are missing, and with them the
shorts that paid. The archive keeps monthly zip files for delisted symbols too.

    from feeds import list_archive_symbols, non_crypto_symbols, fetch_archive
    syms = [s for s in list_archive_symbols() if s.endswith("USDT") and s not in non_crypto_symbols()]
    fetch_archive(syms, "2024-01", "2026-09", "1d", "data/binance_archive")

Writes the parquet layout the model's betafunding.data.load_panel reads:
    <root>/price/<SYM>-<tf>-data.parquet      index timestamp (naive UTC), OHLCV + extras
    <root>/funding/<SYM>-funding-data.parquet  index fundingTime, fundingRate, fundingIntervalHours
Only complete months exist in the archive; the current month is not fetched.
"""

import io
import logging
import os
import re
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

log = logging.getLogger("feeds.archive")

_S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
_DL = "https://data.binance.vision"
_KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_av",
               "trades", "tb_base_av", "tb_quote_av", "ignore"]


def non_crypto_symbols():
    """Binance USDT-M contracts whose underlying is not a coin (equities, commodities, indices),
    from today's market list - the archive itself does not say."""
    import ccxt
    ex = ccxt.binanceusdm({"enableRateLimit": True})
    ex.load_markets()
    return {m["id"] for m in ex.markets.values()
            if m.get("swap") and (m.get("info") or {}).get("underlyingType") not in (None, "COIN")}


def list_archive_symbols(kind="klines"):
    """Every symbol directory under data/futures/um/monthly/<kind>/ (paginated S3 listing)."""
    prefix = f"data/futures/um/monthly/{kind}/"
    out, marker = [], None
    while True:
        params = {"prefix": prefix, "delimiter": "/"}
        if marker:
            params["marker"] = marker
        xml = requests.get(_S3, params=params, timeout=30).text
        found = re.findall(r"<Prefix>([^<]+)</Prefix>", xml)[1:]
        out += [p.rstrip("/").split("/")[-1] for p in found]
        if "<IsTruncated>true</IsTruncated>" not in xml or not found:
            break
        nxt = re.findall(r"<NextMarker>([^<]+)</NextMarker>", xml)
        marker = nxt[0] if nxt else found[-1]
    return sorted(set(out))


def _get(url, tries=5):
    """GET with backoff: the CDN drops TLS connections under parallel load."""
    for k in range(tries):
        try:
            return requests.get(url, timeout=60)
        except requests.RequestException as e:
            if k == tries - 1:
                log.warning(f"giving up on {url}: {str(e)[:80]}")
                return None
            time.sleep(2 * (k + 1))


def _csv(url, names):
    r = _get(url)
    if r is None or r.status_code != 200:
        return None
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        raw = z.read(z.namelist()[0]).decode()
    first = raw.split("\n", 1)[0]
    has_header = not first.split(",")[0].strip().lstrip("-").isdigit()
    return pd.read_csv(io.StringIO(raw), header=0 if has_header else None,
                       names=None if has_header else names)


def _months(start, end):
    return [p.strftime("%Y-%m") for p in pd.period_range(start, end, freq="M")]


def _fetch_one(sym, months, timeframe, root, resume=True):
    kpath = os.path.join(root, "price", f"{sym}-{timeframe}-data.parquet")
    fpath = os.path.join(root, "funding", f"{sym}-funding-data.parquet")
    if resume and os.path.exists(kpath) and os.path.exists(fpath):
        return sym, -1, -1                                   # already fetched
    parts, fparts = [], []
    for m in months:
        k = _csv(f"{_DL}/data/futures/um/monthly/klines/{sym}/{timeframe}/{sym}-{timeframe}-{m}.zip",
                 _KLINE_COLS)
        if k is not None and len(k):
            k.columns = _KLINE_COLS[:len(k.columns)]
            parts.append(k)
        f = _csv(f"{_DL}/data/futures/um/monthly/fundingRate/{sym}/{sym}-fundingRate-{m}.zip",
                 ["calc_time", "funding_interval_hours", "last_funding_rate"])
        if f is not None and len(f):
            fparts.append(f)
    if not parts:
        return sym, 0, 0
    k = pd.concat(parts)
    k["timestamp"] = pd.to_datetime(pd.to_numeric(k["open_time"]), unit="ms")
    k = k.drop(columns=["open_time", "ignore"], errors="ignore").drop_duplicates("timestamp")
    k = k.set_index("timestamp").sort_index().apply(pd.to_numeric, errors="coerce")
    os.makedirs(os.path.join(root, "price"), exist_ok=True)
    k.to_parquet(kpath)
    n_f = 0
    if fparts:
        f = pd.concat(fparts)
        f.columns = ["calc_time", "funding_interval_hours", "last_funding_rate"][:len(f.columns)]
        f = pd.DataFrame({
            "fundingTime": pd.to_datetime(pd.to_numeric(f["calc_time"]), unit="ms"),
            "fundingRate": pd.to_numeric(f["last_funding_rate"], errors="coerce"),
            "fundingIntervalHours": pd.to_numeric(f["funding_interval_hours"], errors="coerce"),
        }).drop_duplicates("fundingTime").set_index("fundingTime").sort_index()
        os.makedirs(os.path.join(root, "funding"), exist_ok=True)
        f.to_parquet(fpath)
        n_f = len(f)
    return sym, len(k), n_f


def fetch_archive(symbols, start, end, timeframe="1d", root="data/binance_archive", workers=8,
                  resume=True):
    """Monthly klines + funding for each symbol over [start, end] (YYYY-MM). Returns a summary.

    A symbol with no kline file in the window (listed after `end`, delisted before `start`)
    writes nothing. resume skips symbols whose two files already exist (bars = -1 in the
    summary); the archive is static, so a rerun only fills what is missing.
    """
    months = _months(start, end)
    rows = []
    with ThreadPoolExecutor(workers) as pool:
        for i, (sym, n_k, n_f) in enumerate(pool.map(
                lambda s: _fetch_one(s, months, timeframe, root, resume), symbols)):
            rows.append({"symbol": sym, "bars": n_k, "funding_rows": n_f})
            if (i + 1) % 100 == 0:
                log.info(f"{i + 1}/{len(symbols)} symbols")
    return pd.DataFrame(rows)
