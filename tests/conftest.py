import copy

import numpy as np
import pandas as pd
import pytest

from sprint2.contracts import Market
from sprint2.run import load_config


def toy_market(T=520, N=60, seed=0) -> Market:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=T)
    vol = rng.uniform(0.008, 0.03, N)
    r = rng.normal(0.0003, 1, (T, N)) * vol
    price = 100 * np.exp(np.cumsum(r, 0))
    price[rng.random((T, N)) < 0.01] = np.nan                     # scattered missing prices
    price[300:, 5] = np.nan                                       # a delisting (price disappears)
    volume = rng.uniform(1e4, 1e5, (T, N))
    volume[rng.random((T, N)) < 0.01] = 0.0                       # zero-volume days
    member = np.ones((T, N), bool)
    member[:, 50:] = False                                        # 50 members at start ...
    member[200:, 50:55] = True                                    # ... index changes later
    member[200:, 40:45] = False
    member[300:, 5] = False
    codes = pd.Index([f"{i:06d}" for i in range(N)], dtype=str)
    return Market(dates, codes, price, volume, member, {"toy": True, "seed": seed})


def toy_config(market: Market) -> dict:
    cfg = copy.deepcopy(load_config())
    d = market.dates
    cfg["selection_defaults"]["min_valid"] = 20
    cfg["strategies"][0]["params"] = {"window": 120, "min_periods": 100}
    cfg["strategies"][1]["params"] = {"window": 20}
    cfg["periods"] = {"strategy_start": str(d[0].date()), "prep": [str(d[150].date()), str(d[260].date())],
                      "eval": [str(d[261].date()), str(d[-1].date())]}
    cfg["model_b1"]["level"] |= {"n_burn": 30, "n_keep": 60, "min_months": 3}
    cfg["model_b2"] |= {"n_burn": 40, "n_burn_warm": 20, "n_keep": 60, "min_weeks": 10}
    return cfg


@pytest.fixture(scope="session")
def market():
    return toy_market()


@pytest.fixture(scope="session")
def cfg(market):
    return toy_config(market)


@pytest.fixture(scope="session")
def result(market, cfg):
    from sprint2.run import pipeline
    return pipeline(cfg, market)
