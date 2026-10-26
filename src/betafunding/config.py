"""Settings: config/strategy.yaml as a plain dict, plus dotted-path overrides.

The file holds the final strategy. The design the research started from differs in a handful
of settings, kept here as ORIGINAL_SPEC so every comparison starts from the same place:

    cfg = load_config()
    original = override(cfg, ORIGINAL_SPEC)
    variant = override(cfg, {"risk.max_weight": 0.3, "funding.forecast": "ema"})
"""

import copy

import yaml

# The strategy as first specified: no BTC/ETH positions (alts hedge each other), a 10% APY entry
# floor on both sides, 40% single-coin cap, no stop-loss.
ORIGINAL_SPEC = {
    "trade_benchmarks": False,
    "funding.min_apy_long": None,
    "funding.min_apy_short": None,
    "risk.max_weight": 0.40,
    "risk.stop_loss": None,
}


def load_config(path="config/strategy.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def override(cfg: dict, changes: dict) -> dict:
    """A copy of `cfg` with each dotted path in `changes` set, e.g. {"risk.lambda": 30}."""
    out = copy.deepcopy(cfg)
    for path, val in changes.items():
        node = out
        *parents, leaf = path.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = val
    return out


def get(cfg: dict, path: str):
    """The value at a dotted path, or {} when it is missing."""
    for p in path.split("."):
        cfg = cfg.get(p, {}) if isinstance(cfg, dict) else {}
    return cfg
