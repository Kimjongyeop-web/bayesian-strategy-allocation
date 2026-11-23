"""Common quantity ledger, execution and the daily loops.

One account = sleeves (one per strategy) + unallocated cash c0. Each sleeve owns quantities q[s] and cash
c[s]. A virtual strategy account is an account with one sleeve that only follows its own rule.

Daily order inside `run_virtual` / `run_policy`:
  1. value holdings at today's marks           (execute)
  2. fill yesterday's plan at today's close    (sells first, then buys scaled to cash)
  3. record gross/net returns                  (execute)
  4./5. caller-side: posterior update, internal orders, allocation decision
  6. fix next-day quantities                   (make_plan -> Plan)

Order quantities are fixed with decision-day prices and NAV; the fill day only scales buys down to cash.
Sleeve orders are netted per ticker: only the net goes to the market and pays cost once; the matched part
is an internal ownership transfer at the fill price, at no cost.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .contracts import (LEDGER_DAILY_COLUMNS, ORDER_COLUMNS, Market, StrategyDef, StrategyOutput)
from .portfolio import Costs, internal_target, month_end_mask

TOL = 1e-12


@dataclass
class Plan:
    decision_t: int
    action: str                       # INTERNAL (virtual) | HOLD | TRANSFER
    delta: np.ndarray                 # S x N desired sleeve quantity change (target - current)
    order: np.ndarray                 # N net market order quantity, after the minimum-order rule
    cash_target: np.ndarray | None    # S sleeve cash after a capital transfer (TRANSFER only)


class Account:
    def __init__(self, name: str, sleeve_ids, n: int, cash: float, sleeve_cash=None):
        self.name, self.sleeve_ids = name, tuple(sleeve_ids)
        self.q = np.zeros((len(self.sleeve_ids), n))
        self.c = np.zeros(len(self.sleeve_ids)) if sleeve_cash is None else np.asarray(sleeve_cash, float)
        self.c0 = cash - self.c.sum()
        self.pending: Plan | None = None
        self.nav_prev = cash
        self.rows, self.orders, self.sleeve_rows = [], [], []
        self.max_identity_error = 0.0

    def clone(self) -> "Account":
        a = Account.__new__(Account)
        a.name, a.sleeve_ids = self.name, self.sleeve_ids
        a.q, a.c, a.c0 = self.q.copy(), self.c.copy(), self.c0
        return a

    def capital(self, mark) -> np.ndarray:
        return self.c + self.q @ mark

    def nav(self, mark) -> float:
        return float(self.c0 + self.capital(mark).sum())

    @property
    def cash(self) -> float:
        return float(self.c0 + self.c.sum())


def split_fills(delta: np.ndarray, filled: np.ndarray):
    """Give each sleeve its share of a ticker's (partial) net fill.

    delta: S x N desired changes; filled: N net market fill (same sign as the net desire, |filled| <= |net|).
    The side opposite to the net is fully matched internally; the net side receives internal + market fill
    pro rata. Returns (sleeve quantity changes, each sleeve's share of the market order for cost allocation).
    """
    buy, sell = np.clip(delta, 0, None), np.clip(-delta, 0, None)
    B, Sl = buy.sum(0), sell.sum(0)
    up = B - Sl > 0
    recv = np.where(up, Sl + filled, B)
    deliver = np.where(up, Sl, B - filled)
    fb = np.divide(recv, B, out=np.zeros_like(B), where=B > TOL)
    fs = np.divide(deliver, Sl, out=np.zeros_like(Sl), where=Sl > TOL)
    share = np.where(up, np.divide(buy, B, out=np.zeros_like(buy), where=B > TOL),
                     np.divide(sell, Sl, out=np.zeros_like(sell), where=Sl > TOL))
    return buy * fb - sell * fs, share, np.minimum(B, Sl)


def _fill(acc: Account, plan: Plan, price, tradable, member, cb, cs, skip=None, *, block_internal=False) -> dict:
    """Apply a plan at the given prices. Mutates acc. Used for real fills and for decision-time what-ifs."""
    px = np.nan_to_num(price)
    delta, order = plan.delta.copy(), plan.order.copy()
    if block_internal:
        delta[:, ~tradable] = 0.0
    if skip is not None and skip.any():
        delta[:, skip], order[skip] = 0.0, 0.0
    qtot = acc.q.sum(0)
    sells = (order < 0) & tradable
    buys = (order > 0) & tradable & member
    filled = np.zeros_like(order)
    filled[sells] = -np.minimum(-order[sells], qtot[sells])
    proceeds = -(filled * px)[sells].sum()
    sell_cost = -(filled * px * cs)[sells].sum()
    need = (order * px * (1 + cb))[buys].sum()
    avail = acc.cash + proceeds - sell_cost
    scale = min(1.0, max(avail, 0.0) / need) if need > 0 else 0.0
    filled[buys] = order[buys] * scale
    dq, share, matched = split_fills(delta, filled)
    cost_j = np.where(filled > 0, filled * px * cb, -filled * px * cs)
    acc.q += dq
    if acc.q.min() < -1e-9 * max(1.0, acc.q.max()):
        raise AssertionError(f"{acc.name}: negative quantity {acc.q.min()}")
    acc.q[acc.q < TOL] = 0.0
    acc.c -= (dq * px).sum(1) + (share * cost_j).sum(1)
    before = acc.c.copy()
    if plan.cash_target is not None:                       # capital transfer: re-pool all cash
        pool = acc.cash
        tgt = np.clip(plan.cash_target, 0, None)
        f = min(1.0, pool / tgt.sum()) if tgt.sum() > 0 else 0.0
        acc.c = tgt * f
        acc.c0 = pool - acc.c.sum()
    else:                                                  # HOLD: cover price-gap shortfalls, log as funding
        short = np.clip(-acc.c, 0, None)
        donors = np.clip(acc.c, 0, None)
        if short.sum() > 0 and max(acc.c0, 0.0) + donors.sum() > 0:
            take = min(short.sum(), max(acc.c0, 0.0))
            rest = min(short.sum() - take, donors.sum())
            acc.c0 -= take
            acc.c = np.where(acc.c < 0, acc.c + short * (take + rest) / short.sum(), acc.c)
            acc.c -= rest * donors / donors.sum() if rest > 0 else 0.0
    unfilled = (np.abs(order) > TOL) & (np.abs(filled) < np.abs(order) * (1 - 1e-9))
    return dict(cost=float(cost_j.sum()), buy=float((filled * px)[filled > 0].sum()), sell=proceeds,
                internal=float((matched * px).sum()), filled=filled, order=order, cost_j=cost_j,
                unfilled=unfilled, flow=acc.c - before)


def execute(acc: Account, t: int, mk: Market, costs: Costs, stale_cfg: dict) -> None:
    """Steps 1-3 for one account at row t: value, fill yesterday's plan, record."""
    mark = np.nan_to_num(mk.mark[t])
    nav_pre = acc.nav(mark)
    writeoff, skip = 0.0, None
    after = stale_cfg.get("after_days")
    if after is not None:                                  # optional: convert long-stale exits to cash
        stale = (acc.q.sum(0) > 0) & (t - mk.last_valid[t] >= after) & ~mk.member[t]
        if stale.any():
            rate = float(stale_cfg["recovery_rate"])
            val = acc.q[:, stale] * mark[stale]
            acc.c += val.sum(1) * rate
            writeoff = float(val.sum() * (1 - rate))
            for j in np.flatnonzero(stale):
                acc.orders.append([mk.dates[t], None, acc.name, mk.codes[j], "RECOVERY", -acc.q[:, j].sum(),
                                   -acc.q[:, j].sum(), mark[j], acc.q[:, j].sum() * mark[j] * rate, 0.0, "RECOVERY"])
            acc.q[:, stale] = 0.0
            skip = stale
    res, action = None, ""
    if acc.pending is not None:
        plan, action = acc.pending, acc.pending.action
        cb, cs = costs.rates(t, mk.price.shape[1])
        res = _fill(acc, plan, mk.price[t], mk.tradable[t], mk.member[t], cb, cs, skip, block_internal=mk.execution_mask is not None)
        dd = mk.dates[plan.decision_t]
        for j in np.flatnonzero((np.abs(res["order"]) > TOL)):
            status = "UNFILLED" if abs(res["filled"][j]) < TOL else ("PARTIAL" if res["unfilled"][j] else "FILLED")
            acc.orders.append([mk.dates[t], dd, acc.name, mk.codes[j], "BUY" if res["order"][j] > 0 else "SELL",
                               res["order"][j], res["filled"][j], mk.price[t, j],
                               abs(res["filled"][j]) * np.nan_to_num(mk.price[t, j]), res["cost_j"][j], status])
        acc.pending = None
    cost = res["cost"] if res else 0.0
    nav = acc.nav(mark)
    err = abs(nav - (nav_pre - cost - writeoff))
    acc.max_identity_error = max(acc.max_identity_error, err / max(nav_pre, 1.0))
    if err > 1e-7 * max(nav_pre, 1.0) or acc.cash < -1e-7 * max(nav_pre, 1.0) or acc.c.min() < -1e-7 * max(nav_pre, 1.0):
        raise AssertionError(f"{acc.name} {mk.dates[t].date()}: ledger check failed err={err} cash={acc.cash}")
    held = acc.q.sum(0) > 0
    stock = float(acc.q.sum(0) @ mark)
    acc.rows.append([mk.dates[t], nav, nav / acc.nav_prev - 1, (nav + cost) / acc.nav_prev - 1, cost / acc.nav_prev,
                     writeoff / acc.nav_prev, acc.cash, stock / nav if nav > 0 else 0.0,
                     ((res["buy"] + res["sell"]) / acc.nav_prev) if res else 0.0, res["buy"] if res else 0.0,
                     res["sell"] if res else 0.0, res["internal"] if res else 0.0,
                     int(res["unfilled"].sum()) if res else 0, int((held & ~mk.valid[t]).sum()),
                     float(np.abs(res["flow"]).sum()) if res else 0.0, action])
    cap = acc.capital(mark)
    flow = res["flow"] if res else np.zeros(len(cap))
    for s, sid in enumerate(acc.sleeve_ids):
        acc.sleeve_rows.append([mk.dates[t], acc.name, sid, cap[s], float(acc.q[s] @ mark), flow[s], cap[s] / nav])
    acc.nav_prev = nav


def make_plan(acc: Account, t: int, mk: Market, target_q: np.ndarray, min_order: float, action: str,
              costs: Costs, cash_target=None) -> Plan:
    """Net the sleeve targets and apply the minimum order (fraction of NAV); full exits are always sent.

    Small sells are skipped only if the planned buys stay funded without them: many small trims that pay
    for new names (hold-existing rule with few new names) are sent rather than leaving the buys unfunded.
    """
    mark = np.nan_to_num(mk.mark[t])
    nav = acc.nav(mark)
    delta = target_q - acc.q
    delta[np.abs(delta) < TOL] = 0.0
    net = delta.sum(0)
    order = net.copy()
    small = (np.abs(order) * mark < min_order * nav) & (target_q.sum(0) > TOL)
    order[small] = 0.0
    cb, cs = costs.rates(min(t + 1, len(mk.dates) - 1), len(mark))
    buys = (np.clip(order, 0, None) * mark * (1 + cb)).sum()
    sells = (np.clip(-order, 0, None) * mark * (1 - cs)).sum()
    if buys > 0 and buys > acc.cash + sells:
        trims = small & (net < 0) & (-net * mark > 1e-9 * nav)       # ignore float-rounding deltas
        order[trims] = net[trims]
    return Plan(t, action, delta, order, cash_target)


def locked(acc: Account, t: int, mk: Market) -> np.ndarray:
    """Held tickers without a valid price at decision row t: cannot be sold, stay at their last mark."""
    return (acc.q.sum(0) > 0) & ~mk.valid[t]


def simulate(acc: Account, plan: Plan, t: int, mk: Market, costs: Costs) -> dict:
    """Expected result of a plan at decision-day marks and next-day cost rates (no new information).

    stock = sleeve value in tradable names only (locked holdings are not strategy exposure).
    """
    a = acc.clone()
    mark = np.nan_to_num(mk.mark[t])
    nav = a.nav(mark)
    cb, cs = costs.rates(min(t + 1, len(mk.dates) - 1), mk.price.shape[1])
    res = _fill(a, plan, mk.mark[t], mk.tradable[t], mk.member[t], cb, cs, block_internal=mk.execution_mask is not None)
    live = np.where(mk.valid[t], mark, 0.0)
    return dict(capital=a.capital(mark) / nav, stock=(a.q @ live) / nav, cost=res["cost"] / nav)


def _weights_to_qty(W, capital, mark):
    return np.divide(W * capital, mark, out=np.zeros_like(W), where=mark > 0)


def hold_targets(acc: Account, t: int, mk: Market, books: dict, reselect: bool) -> np.ndarray:
    """HOLD_ALLOCATION: each sleeve keeps its own holdings/cash and follows only its own rule."""
    mark = np.nan_to_num(mk.mark[t])
    cap = acc.capital(mark)
    lk = locked(acc, t, mk)
    T = acc.q.copy()
    for s, sid in enumerate(acc.sleeve_ids):
        if cap[s] <= 0:
            continue
        u = acc.q[s] * mark / cap[s]
        b = books[sid]
        W = internal_target(b.definition.rule, b.scores[t], mk.member[t], u, reselect, lk)
        T[s] = _weights_to_qty(W, cap[s], mark)
    return T


def transfer_targets(acc: Account, t: int, mk: Market, books: dict, w: np.ndarray):
    """Candidate Q*w: sleeve s gets w_s of NAV in the virtual strategy's current target composition.

    Locked holdings stay where they are; w is a share of NAV in tradable names (sum(w) <= 1 - locked share).
    """
    mark = np.nan_to_num(mk.mark[t])
    nav = acc.nav(mark)
    lk = locked(acc, t, mk)
    T = np.zeros_like(acc.q)
    for s, sid in enumerate(acc.sleeve_ids):
        T[s] = _weights_to_qty(tradable_target(books[sid], t, mk) * w[s], nav, mark)
    cash_target = np.clip(w * nav - T @ mark, 0, None)
    T[:, lk] = acc.q[:, lk]
    return T, cash_target


def tradable_target(book, t: int, mk: Market) -> np.ndarray:
    """Q column: the virtual strategy's target in names with a valid price at t, rescaled to its total stock
    share. Its own frozen holdings cannot be bought, so their weight goes pro rata to the tradable names
    instead of becoming idle cash in the real sleeve."""
    w = book.target_w[t]
    tw = np.where(mk.valid[t], w, 0.0)
    return tw * (w.sum() / tw.sum()) if tw.sum() > 0 else tw


def locked_share(acc: Account, t: int, mk: Market) -> float:
    mark = np.nan_to_num(mk.mark[t])
    return float(acc.q.sum(0)[locked(acc, t, mk)] @ mark[locked(acc, t, mk)] / acc.nav(mark))


def run_virtual(mk: Market, sdef: StrategyDef, scores: np.ndarray, costs: Costs, ex: dict,
                start: int, end: int) -> StrategyOutput:
    """Independent virtual strategy account, run every day whatever the real allocation is."""
    sid = sdef.strategy_id
    acc = Account(sid, [sid], len(mk.codes), ex["initial_cash"], sleeve_cash=[ex["initial_cash"]])
    reselect = month_end_mask(mk.dates)
    T, N = mk.price.shape
    held_w, target_w = np.zeros((T, N)), np.zeros((T, N))
    live = []
    for t in range(start, end + 1):
        live.append(acc.q.sum() > 0)                  # held stocks at close t-1
        execute(acc, t, mk, costs, ex.get("stale_exit", {}))
        mark = np.nan_to_num(mk.mark[t])
        nav = acc.nav(mark)
        u = acc.q[0] * mark / nav
        held_w[t] = u
        if t < end:
            W = internal_target(sdef.rule, scores[t], mk.member[t], u, bool(reselect[t]), locked(acc, t, mk))
            target_w[t] = W
            acc.pending = make_plan(acc, t, mk, _weights_to_qty(W, nav, mark)[None], ex["min_order_nav"], "INTERNAL",
                                    costs)
    daily = ledger_frame(acc)
    daily["live"] = live
    out = StrategyOutput(sdef, daily, held_w, target_w, scores)
    out.account = acc
    return out


def run_policy(mk: Market, books: dict, policy, costs: Costs, ex: dict, start: int, end: int, name: str):
    """Real allocation account. `policy.decide(acc, t, ctx)` returns a contracts.Decision with a Plan."""
    ids = tuple(books)
    acc = Account(name, ids, len(mk.codes), ex["initial_cash"])
    ctx = dict(market=mk, books=books, costs=costs, ex=ex, reselect=month_end_mask(mk.dates), start=start)
    decisions = []
    for t in range(start, end + 1):
        execute(acc, t, mk, costs, ex.get("stale_exit", {}))
        if t < end:
            d = policy.decide(acc, t, ctx)
            acc.pending = d.plan
            decisions.append(d)
    return acc, decisions


def ledger_frame(acc: Account) -> pd.DataFrame:
    return pd.DataFrame(acc.rows, columns=["date", *LEDGER_DAILY_COLUMNS]).set_index("date")


def orders_frame(acc: Account) -> pd.DataFrame:
    return pd.DataFrame(acc.orders, columns=ORDER_COLUMNS)


def sleeves_frame(acc: Account) -> pd.DataFrame:
    return pd.DataFrame(acc.sleeve_rows, columns=["date", "account", "sleeve", "capital", "stock_value", "flow",
                                                  "allocation"])
