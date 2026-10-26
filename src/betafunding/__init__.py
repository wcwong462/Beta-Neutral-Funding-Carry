"""BetaFunding: a BTC/ETH-neutral book of Binance crypto perps that trades extreme funding.

    from betafunding import load_config, load_panel, build_signals, run, summarize
    cfg = load_config()
    panel = load_panel(**cfg["data"])
    sig = build_signals(panel, cfg)
    print(summarize(run(panel, sig, cfg).daily))

The research notebooks use the same package; the original design is
override(cfg, ORIGINAL_SPEC).

CLI:  python -m betafunding plan | backtest | refresh
"""

from .backtest import Result, StopBook, make_inputs, run, score, solver_args, summarize
from .beta import BETA_TYPES, hedged_returns, market_betas
from .config import ORIGINAL_SPEC, get, load_config, override
from .data import Panel, load_panel, refresh_recent
from .funding import APY, ar1_forecast, funding_forecast
from .optimizer import Problem, solve
from .report import load_book, plan, roll_forward, save_book, to_markdown
from .risk import METHODS as COV_MODELS, covariance
from .signals import DRIFT_EDGES, Signals, build_signals, price_drift
from .walkforward import WFResult, expand, walk_forward

__all__ = ["Result", "StopBook", "make_inputs", "run", "score", "solver_args", "summarize",
           "BETA_TYPES", "hedged_returns", "market_betas", "ORIGINAL_SPEC", "get", "load_config",
           "override", "Panel", "load_panel", "refresh_recent", "APY", "ar1_forecast",
           "funding_forecast", "Problem", "solve", "load_book", "plan", "roll_forward",
           "save_book", "to_markdown", "COV_MODELS", "covariance", "DRIFT_EDGES", "Signals",
           "build_signals", "price_drift", "WFResult", "expand", "walk_forward"]
