"""Toy signals used to prove that Part 3·4 run unchanged when Part 1·2 are swapped via the config."""
import numpy as np
import pandas as pd


def short_reversal(market, window=5):
    r = pd.DataFrame(market.ret)
    return -r.rolling(window, min_periods=window).sum().to_numpy()


def random_score(market, seed=0):
    rng = np.random.default_rng(seed)
    s = rng.random(market.price.shape)                 # row t never depends on later rows
    return np.where(market.valid, s, np.nan)
