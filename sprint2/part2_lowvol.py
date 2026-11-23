"""Part 2 signal: LOWVOL60 -- negative 60-day volatility (kjy branch, commit 02ea9b7).

score[t, i] = -std(r[t-window+1 .. t, i], ddof=1), r = simple return between adjacent valid prices.
All `window` most recent returns must exist (a missing day removes the name until 60 new returns exist).
"""
import numpy as np
import pandas as pd

from .contracts import Market


def lowvol(market: Market, window: int = 60) -> np.ndarray:
    r = pd.DataFrame(market.ret)
    return -r.rolling(window, min_periods=window).std(ddof=1).to_numpy()
