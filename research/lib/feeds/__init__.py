"""Historical data: every Binance USDT-M perp that ever traded, from data.binance.vision.

    from feeds import list_archive_symbols, non_crypto_symbols, fetch_archive
"""

from .archive import fetch_archive, list_archive_symbols, non_crypto_symbols

__all__ = ["fetch_archive", "list_archive_symbols", "non_crypto_symbols"]
