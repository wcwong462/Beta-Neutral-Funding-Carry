"""Command line.

    python -m betafunding refresh               # top up the store with the last 60 days (REST)
    python -m betafunding plan [--book state/book.json] [--save]
                                                # rolls the saved book forward to the latest
                                                # close (drift + stops), then solves today
    python -m betafunding backtest [--start 2024-03-01] [--end ...]

Run from the BetaFunding directory with src on the path (PYTHONPATH=src).
"""

import argparse
import logging

import pandas as pd

from . import (build_signals, load_book, load_config, load_panel, plan, refresh_recent,
               roll_forward, run, save_book, summarize, to_markdown)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="betafunding")
    ap.add_argument("command", choices=["refresh", "plan", "backtest"])
    ap.add_argument("--config", default="config/strategy.yaml")
    ap.add_argument("--book", default="state/book.json", help="current book + stop state (plan)")
    ap.add_argument("--save", action="store_true", help="write the target book back to --book")
    ap.add_argument("--days", type=int, default=60, help="refresh: days to fetch")
    ap.add_argument("--start")
    ap.add_argument("--end")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)

    if args.command == "refresh":
        refresh_recent(cfg["data"]["root"], args.days)
        return

    panel = load_panel(**cfg["data"])
    sig = build_signals(panel, cfg)

    if args.command == "backtest":
        res = run(panel, sig, cfg, args.start, args.end)
        print(pd.Series(summarize(res.daily)).round(2).to_string())
        return

    as_of, equity, w_old, stops = load_book(args.book, cfg["equity_usd"])
    w_old = roll_forward(panel, cfg, as_of, w_old, stops)
    summary, table = plan(panel, sig, cfg, w_old, stops, equity)
    print(to_markdown(summary, table))
    if args.save:
        save_book(args.book, panel.days[-1], equity, table["target_%"] / 100, stops)
        print(f"\nsaved target book to {args.book} (as of {panel.days[-1].date()})")


if __name__ == "__main__":
    main()
