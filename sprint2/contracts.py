"""Common input/output spec between parts.

Part 1·2 only promise a `SignalFn`. Everything downstream (selection, ledger, Part 3, Part 4) works on
the generic objects below and never imports a strategy module. Swapping a signal = changing the
`strategies:` entry in the config (module:function + params).

Timing convention (all parts): information up to the close of day t -> decision at t -> fill at close t+1.
Arrays are (dates x tickers) in `Market.codes` order; strategy-indexed arrays follow `strategy_ids` order.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np
import pandas as pd


# 1. Market input -----------------------------------------------------------------------------------
@dataclass(frozen=True)
class Market:
    """Prices/volume/membership as stored in the workbook. Row t holds only information known at close t."""
    dates: pd.DatetimeIndex
    codes: pd.Index            # str, leading zeros preserved
    price: np.ndarray          # adjusted close, NaN = no observation
    volume: np.ndarray
    member: np.ndarray         # bool, KOSPI200 member on that date
    source: dict = field(default_factory=dict)

    execution_mask: np.ndarray | None = None

    def __post_init__(self):
        shape = (len(self.dates), len(self.codes))
        for name in ("price", "volume", "member"):
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} shape {getattr(self, name).shape} != {shape}")
        valid = np.isfinite(self.price) & (self.price > 0)
        prev = np.vstack([np.full((1, shape[1]), False), valid[:-1]])
        ret = np.full(shape, np.nan)
        ret[1:][valid[1:] & prev[1:]] = (self.price[1:] / self.price[:-1] - 1)[valid[1:] & prev[1:]]
        idx = np.where(valid, np.arange(shape[0])[:, None], -1)
        last_valid = np.maximum.accumulate(idx, axis=0)
        mark = np.where(last_valid >= 0, self.price[np.maximum(last_valid, 0), np.arange(shape[1])], np.nan)
        object.__setattr__(self, "valid", valid)
        # simple return between ADJACENT trading days, both prices valid (no forward fill)
        object.__setattr__(self, "ret", ret)
        # evaluation mark = last valid price (valuation only; never an execution price)
        object.__setattr__(self, "mark", mark)
        object.__setattr__(self, "last_valid", last_valid)
        # executable at close t: valid price and positive volume
        vol_ok = np.nan_to_num(self.volume, nan=0.0) > 0
        mask = np.ones(shape, bool) if self.execution_mask is None else np.asarray(self.execution_mask)
        if mask.shape != shape or not np.isin(mask, [0, 1]).all():
            raise ValueError("execution_mask must be aligned and binary")
        object.__setattr__(self, "tradable", valid & vol_ok & mask.astype(bool))

    def slice(self, end: int) -> "Market":
        """Rows 0..end (inclusive) -- used to prove signals/decisions do not look ahead."""
        return Market(self.dates[:end + 1], self.codes, self.price[:end + 1], self.volume[:end + 1],
                      self.member[:end + 1], self.source,
                      None if self.execution_mask is None else self.execution_mask[:end + 1])


# 2. Strategy definition ----------------------------------------------------------------------------
class SignalFn(Protocol):
    """score[t, i] may use rows <= t only. Larger = more attractive. NaN = no signal (not eligible)."""
    def __call__(self, market: Market, **params) -> np.ndarray: ...


@dataclass(frozen=True)
class SelectionRule:
    top: float = 0.20          # buy top max(1, round(top*N)), N = members that day
    keep: float = 0.40         # keep held names inside top round(keep*N)
    min_valid: int = 50        # fewer valid member signals -> no reselection
    reselect: str = "month_end"


@dataclass(frozen=True)
class StrategyDef:
    strategy_id: str
    signal: Callable
    params: dict
    rule: SelectionRule

    @property
    def version(self) -> str:
        """Fingerprint of signal code + params + rule. Outputs/posteriors are only valid for this version."""
        try:
            src = inspect.getsource(self.signal)
        except (OSError, TypeError):
            src = repr(self.signal)
        blob = json.dumps([self.strategy_id, f"{self.signal.__module__}.{self.signal.__qualname__}", src,
                           self.params, self.rule.__dict__], sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


# 3. Strategy output --------------------------------------------------------------------------------
STRATEGY_DAILY_COLUMNS = ["nav", "net_return", "gross_return", "cost", "writeoff", "cash", "exposure",
                          "turnover", "live"]


@dataclass
class StrategyOutput:
    """One independent virtual strategy account.

    daily: STRATEGY_DAILY_COLUMNS indexed by date. gross_return = (NAV_t + cost_t)/NAV_{t-1} - 1
           (same-day trading cost restored, write-offs NOT restored). live = held stocks at close t-1.
    held_w[t]:   actual stock weights after fills at close t (fraction of NAV_t)
    target_w[t]: weights decided at close t, to be filled at close t+1 (this is Part 3's Q column)
    scores:      signal panel; downstream reselection of real sleeves uses it with `definition.rule`.
    """
    definition: StrategyDef
    daily: pd.DataFrame
    held_w: np.ndarray
    target_w: np.ndarray
    scores: np.ndarray
    info_cutoff: str = "close t"
    execution: str = "close t+1"

    @property
    def strategy_id(self) -> str:
        return self.definition.strategy_id

    @property
    def version(self) -> str:
        return self.definition.version


# 4. Forecast output (Part 3) -----------------------------------------------------------------------
@dataclass(frozen=True)
class Forecast:
    """Posterior at close t and predictive moments of G = sum_{h=2..H+1} y_{t+h} (fills at t+1)."""
    date: pd.Timestamp
    strategy_ids: tuple
    H: int
    m: np.ndarray          # posterior mean of latent expected log return mu_t
    P: np.ndarray          # posterior covariance of mu_t
    V_next: np.ndarray     # EWMA risk proxy V_{t+1} (conditioned on, not integrated over)
    a: np.ndarray          # predictive mean of log G
    B: np.ndarray          # predictive covariance of log G
    mean: np.ndarray       # H-day simple return mean
    cov: np.ndarray        # H-day simple return covariance
    valid: bool
    reason: str = ""


# 5. Allocation output (Part 3) ---------------------------------------------------------------------
@dataclass
class Decision:
    """Daily review result. HOLD = no capital moved between sleeves; sleeves follow their own rules."""
    date: pd.Timestamp
    action: str                    # "HOLD" | "TRANSFER"
    reason: str
    w_target: np.ndarray           # continuous candidate (QP or fixed weights); NaN if none
    w_exec: np.ndarray             # expected sleeve capital / NAV after the executable plan
    w_hold: np.ndarray             # expected sleeve capital / NAV if holding
    score_cand: float
    score_hold: float
    cost_cand: float               # expected cost / NAV
    cost_hold: float
    plan: object = None            # ledger.Plan actually sent for t+1

    @property
    def gain(self) -> float:
        return self.score_cand - self.score_hold


# 6. Ledger output (Part 4) -------------------------------------------------------------------------
LEDGER_DAILY_COLUMNS = ["nav", "net_return", "gross_return", "cost", "writeoff", "cash", "exposure",
                        "turnover", "buy_notional", "sell_notional", "internal_notional", "unfilled",
                        "stale_positions", "funding", "action_executed"]
ORDER_COLUMNS = ["execution_date", "decision_date", "account", "ticker", "side", "order_qty", "filled_qty",
                 "price", "notional", "cost", "status"]
