"""Shared by every strategy: calendar, 20/40 selection, hold-existing weights, trading-cost rates."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import SelectionRule


def month_end_mask(dates: pd.DatetimeIndex) -> np.ndarray:
    """True on the last trading day of each month in the data calendar.

    The last row of the data is NOT treated as a month end unless the next month is visible: knowing that
    day t is the month's last trading day needs the calendar, which is public in advance (the exchange
    calendar), so this uses calendar information only, never prices after t.
    """
    p = dates.to_period("M")
    out = np.zeros(len(dates), dtype=bool)
    out[:-1] = p[:-1] != p[1:]
    out[-1] = False
    return out


def select(score: np.ndarray, member: np.ndarray, held: np.ndarray, rule: SelectionRule):
    """Indices to hold: top max(1, round(top*N)) + held names inside top round(keep*N).

    N = number of members that day. Ties keep the fixed ticker order (stable sort). Returns None if fewer
    than `min_valid` members have a signal (reselection deferred).
    """
    n_mem = int(member.sum())
    ok = member & np.isfinite(score)
    if ok.sum() < rule.min_valid:
        return None
    cand = np.flatnonzero(ok)
    order = cand[np.argsort(-score[cand], kind="stable")]
    pick = set(order[:max(1, round(rule.top * n_mem))].tolist())
    keep = set(order[:round(rule.keep * n_mem)].tolist())
    pick |= {i for i in np.flatnonzero(held) if i in keep}
    return np.array(sorted(pick), dtype=int)


def hold_existing_weights(pick: np.ndarray, u: np.ndarray, budget: float = 1.0) -> np.ndarray:
    """Target weights (fraction of sleeve capital) for the picked names (hsh rule, reweight=false).

    Kept names keep their current weight u. New names share the available capital equally; if that is less
    than one equal slot each, new names get one slot (budget/len(pick)) and kept names shrink proportionally.
    `budget` = capital that can actually be traded (1 minus holdings that cannot be sold).
    The equal weight of the selection list is NOT the order target.
    """
    W = np.zeros_like(u)
    held = u[pick] > 1e-12
    keep, new = pick[held], pick[~held]
    W[keep] = u[keep]
    if len(new):
        slot = budget / len(pick)
        avail = budget - W[keep].sum()
        if avail >= slot * len(new):
            W[new] = avail / len(new)
        else:
            W[new] = slot
            if len(keep):
                W[keep] *= (budget - slot * len(new)) / W[keep].sum()
    return W


def internal_target(rule: SelectionRule, score: np.ndarray, member: np.ndarray, u: np.ndarray,
                    reselect: bool, locked: np.ndarray | None = None) -> np.ndarray:
    """The strategy's own rule applied to ANY holding u (virtual account or real sleeve).

    reselect day: 20/40 selection with hold-existing weights. Other days: keep current weights and exit
    names that left the index (mandatory exit). Never restores a drifted holding to a target.
    locked: held names without a valid price at the decision (cannot be sold). They keep their weight, get no
    order, and their value is not counted as money available for new names.
    """
    locked = np.zeros(len(u), bool) if locked is None else locked & (u > 1e-12)
    held = u > 1e-12
    W = None
    if reselect:
        pick = select(score, member, held, rule)
        if pick is not None:
            pick = pick[~locked[pick]]
            W = hold_existing_weights(pick, np.where(locked, 0.0, u), 1.0 - u[locked].sum())
    if W is None:
        W = u.copy()
        W[~member] = 0.0
    W[locked] = u[locked]
    return W


def sale_tax(dates: pd.DatetimeIndex, schedule) -> np.ndarray:
    """Sale tax by EXECUTION date (schedule breakpoints are trade dates; see docs)."""
    tax = np.full(len(dates), np.nan)
    for d, rate in schedule:
        tax[dates >= pd.Timestamp(d)] = rate
    if np.isnan(tax).any():
        raise ValueError("sale_tax_schedule does not cover the first date")
    return tax


class Costs:
    """Proportional costs: buy = buy_bps; sell = sell_bps + sale tax of the execution date."""

    def __init__(self, dates: pd.DatetimeIndex, cfg: dict):
        self.buy = cfg["buy_bps"] / 1e4
        self.sell = cfg["sell_bps"] / 1e4 + sale_tax(dates, cfg["sale_tax_schedule"])

    def rates(self, t: int, n: int):
        """(buy, sell) rate vectors for a fill at row t."""
        return np.full(n, self.buy), np.full(n, self.sell[t])
