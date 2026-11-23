"""Part 3b: allocation review between strategy sleeves and cash (shared by B0, B1, B2).

Rolling posterior-predictive mean-variance rule (not a global or multi-period optimum):
    J(w) = w'mean - gamma/2 w'cov w - TC(w),  w >= 0, sum(w) <= 1
    TC(w) = sum_i cb_i max(z_i, 0) + cs_i max(-z_i, 0),  z = Q w - x_old
Holdings without a valid price (locked) stay in place: sum(w) <= 1 - locked share.
A small continuous QP gives the candidate; the executable plan (min order, cash, tradability, netting) is
then re-scored with its own expected weights and costs and compared with HOLD by the SAME evaluator.
Capital moves only if the candidate beats HOLD by more than `score_tol`; ties, invalid forecasts and solver
failures HOLD with a recorded reason. A candidate whose plan sends no market order (only moves ownership of
shared holdings between sleeves) is a HOLD (`no_market_trade`): the portfolio would not change.
No extra lambda, no allocation band, no probability threshold. Review daily (B0, B1) or only on the first day and
month-ends (B2, `review="monthly"`).
"""
from __future__ import annotations

import cvxpy as cp
import numpy as np

from .contracts import Decision
from .ledger import hold_targets, locked, locked_share, make_plan, simulate, tradable_target, transfer_targets


def utility(w, mean, cov, cost, gamma):
    return float(w @ mean - gamma / 2 * w @ cov @ w - cost)


def exposure_weights(sim: dict, Q: np.ndarray) -> np.ndarray:
    """Strategy exposure of a plan = sleeve stock value / (NAV * stock share of the virtual strategy).

    Sleeve cash beyond the virtual strategy's own cash ratio (e.g. orders skipped by the minimum-order
    rule) is treated as cash, not as strategy exposure. The virtual strategy is a proxy for the sleeve.
    """
    e = Q.sum(0)
    return np.divide(sim["stock"], e, out=np.zeros_like(e), where=e > 1e-9)


def _moves_market(plan) -> bool:
    return bool(np.any(np.abs(plan.order) > 0))


def _sim(acc, plan, t, ctx, Q):
    s = simulate(acc, plan, t, ctx["market"], ctx["costs"])
    s["w"] = exposure_weights(s, Q)
    return s


class AllocationQP:
    """DPP-parameterised QP, compiled once and re-solved daily."""

    def __init__(self, n: int, S: int, gamma: float, solver: str = "CLARABEL"):
        self.w, self.z = cp.Variable(S, nonneg=True), cp.Variable(n)
        self.Q, self.x = cp.Parameter((n, S)), cp.Parameter(n)
        self.cb, self.cs = cp.Parameter(n, nonneg=True), cp.Parameter(n, nonneg=True)
        self.mu, self.L, self.ub = cp.Parameter(S), cp.Parameter((S, S)), cp.Parameter(S, nonneg=True)
        self.cap = cp.Parameter(nonneg=True)
        obj = (self.mu @ self.w - gamma / 2 * cp.sum_squares(self.L.T @ self.w)
               - self.cb @ cp.pos(self.z) - self.cs @ cp.neg(self.z))
        cons = [self.z == self.Q @ self.w - self.x, cp.sum(self.w) <= self.cap, self.w <= self.ub]
        self.prob, self.solver = cp.Problem(cp.Maximize(obj), cons), solver

    def solve(self, Q, x_old, cb, cs, mean, cov, ub, cap=1.0):
        self.Q.value, self.x.value, self.cb.value, self.cs.value = Q, x_old, cb, cs
        self.mu.value, self.ub.value, self.cap.value = mean, ub, max(cap, 0.0)
        self.L.value = np.linalg.cholesky(cov + 1e-14 * np.eye(len(cov)))
        try:
            self.prob.solve(solver=self.solver)
        except cp.error.SolverError as e:
            return None, f"solver_error: {e}"
        if self.prob.status not in ("optimal", "optimal_inaccurate") or self.w.value is None:
            return None, f"solver_status: {self.prob.status}"
        w = np.clip(self.w.value, 0, None)
        if w.sum() > cap:
            w = w * cap / w.sum()
        return w, self.prob.status


def _context(acc, t, ctx):
    mk, books = ctx["market"], ctx["books"]
    mark = np.nan_to_num(mk.mark[t])
    nav = acc.nav(mark)
    Q = np.column_stack([tradable_target(books[s], t, mk) for s in acc.sleeve_ids])
    return mk, books, mark, nav, Q


def _hold(acc, t, ctx):
    mk, books, *_ = _context(acc, t, ctx)
    T = hold_targets(acc, t, mk, books, bool(ctx["reselect"][t]))
    return make_plan(acc, t, mk, T, ctx["ex"]["min_order_nav"], "HOLD", ctx["costs"])


def _transfer(acc, t, ctx, w):
    mk, books, *_ = _context(acc, t, ctx)
    T, cash_target = transfer_targets(acc, t, mk, books, w)
    return make_plan(acc, t, mk, T, ctx["ex"]["min_order_nav"], "TRANSFER", ctx["costs"], cash_target)


class FixedMonthly:
    """Benchmark: move to fixed weights on the first day and every month end; HOLD otherwise."""

    def __init__(self, weights: dict):
        self.weights = weights

    def decide(self, acc, t, ctx):
        w = np.array([self.weights[s] for s in acc.sleeve_ids], float)
        w *= min(1.0, (1 - locked_share(acc, t, ctx["market"])) / w.sum())
        Q = _context(acc, t, ctx)[-1]
        hold = _hold(acc, t, ctx)
        sh = _sim(acc, hold, t, ctx, Q)
        nan = float("nan")
        if t == ctx["start"] or ctx["reselect"][t]:
            plan = _transfer(acc, t, ctx, w)
            sc = _sim(acc, plan, t, ctx, Q)
            return Decision(ctx["market"].dates[t], "TRANSFER", "schedule", w, sc["w"], sh["w"], nan, nan,
                            sc["cost"], sh["cost"], plan)
        return Decision(ctx["market"].dates[t], "HOLD", "schedule", w, sh["w"], sh["w"], nan, nan,
                        sh["cost"], sh["cost"], hold)


class BayesPolicy:
    """Bayesian allocation rule for B0, B1, B2 (cash allowed, sum w <= 1 - locked share)."""

    def __init__(self, forecasts: dict, gamma: float, score_tol: float, n: int, S: int, solver="CLARABEL",
                 review: str = "daily"):
        if review not in ("daily", "monthly"):
            raise ValueError(f"review must be daily or monthly, not {review}")
        self.forecasts, self.gamma, self.tol, self.review = forecasts, gamma, score_tol, review
        self.qp = AllocationQP(n, S, gamma, solver)

    def decide(self, acc, t, ctx):
        mk, books, mark, nav, Q = _context(acc, t, ctx)
        date = mk.dates[t]
        hold = _hold(acc, t, ctx)
        sh = _sim(acc, hold, t, ctx, Q)
        f = self.forecasts.get(t)
        nan_w = np.full(len(acc.sleeve_ids), np.nan)

        def keep(reason, w_target=nan_w, sc=None, score_hold=np.nan):
            score_c = np.nan if sc is None else sc["score"]
            cost_c = np.nan if sc is None else sc["cost"]
            w_exec = sh["w"] if sc is None else sc["w"]
            return Decision(date, "HOLD", reason, w_target, w_exec, sh["w"], score_c, score_hold,
                            cost_c, sh["cost"], hold)

        if self.review == "monthly" and not (t == ctx["start"] or ctx["reselect"][t]):
            return keep("schedule")
        if f is None or not f.valid:
            return keep("forecast_invalid" if f else "no_forecast")
        if tuple(f.strategy_ids) != tuple(acc.sleeve_ids):
            raise ValueError("forecast strategy order differs from account sleeves")
        score_hold = utility(sh["w"], f.mean, f.cov, sh["cost"], self.gamma)
        cb, cs = ctx["costs"].rates(t + 1, len(mark))
        x_old = np.where(locked(acc, t, mk), 0.0, acc.q.sum(0) * mark / nav)
        ub = (Q.sum(0) > 1e-9).astype(float)             # a strategy with no target portfolio gets no capital
        w, status = self.qp.solve(Q, x_old, cb, cs, f.mean, f.cov, ub, 1 - locked_share(acc, t, mk))
        if w is None:
            return keep(status, score_hold=score_hold)
        plan = _transfer(acc, t, ctx, w)
        sc = _sim(acc, plan, t, ctx, Q)
        sc["score"] = utility(sc["w"], f.mean, f.cov, sc["cost"], self.gamma)
        gain = sc["score"] - score_hold
        if gain > self.tol and _moves_market(plan):
            return Decision(date, "TRANSFER", f"gain>{self.tol:g}", w, sc["w"], sh["w"], sc["score"], score_hold,
                            sc["cost"], sh["cost"], plan)
        return keep("no_market_trade" if gain > self.tol else "no_gain", w, sc, score_hold)
