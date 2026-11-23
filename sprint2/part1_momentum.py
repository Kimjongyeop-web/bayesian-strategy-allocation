"""Part 1 signal: HIGH2Y_BASE -- closeness to the 2-year high (hsh branch, commit 8dbc034).

score[t, i] = P[t, i] / max(P[t-window+1 .. t, i]), at least `min_periods` valid prices in the window.

Differences from hsh, on purpose:
- uses the 수정주가 sheet directly (hsh rebuilt prices from cumulated log returns with missing days as 0);
- no 12-1 momentum fallback before the 2-year window is ready. Until then the strategy has no valid
  signal and stays in cash; the prep period (config) covers that span, so no second signal is mixed in.
"""
import numpy as np
import pandas as pd

from .contracts import Market


def high2y(market: Market, window: int = 504, min_periods: int = 403) -> np.ndarray:
    p = pd.DataFrame(np.where(market.valid, market.price, np.nan))
    return (p / p.rolling(window, min_periods=min_periods).max()).to_numpy()
