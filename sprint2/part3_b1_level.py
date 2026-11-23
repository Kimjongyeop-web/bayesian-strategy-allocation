"""Part 3 model B1, part 2: Bayesian forecast of the COMMON level c = (y_M + y_L)/2, so that cash can be chosen.

B1's spread model (part3_b1_spread) forecasts only d = y_M − y_L. To decide between the strategies AND cash, the
allocation also needs the expected return and risk of being invested at all. This module gives c its own forecast:

    monthly c_m (sum of daily log gross returns on days both strategies are live)
    c_{m+1} = μ + e_{m+1},  e ~ N(0, exp(h_{m+1}))          prevailing mean + stochastic volatility (PM-SV)
    h_{m+1} = λ0 + λ1 h_m + ξ,  ξ ~ N(0, σ²_ξ)

Priors (Pettenuzzo & Ravazzolo 2016, data-based on the current sample, t = n + 1):
    μ ~ N(mean c, ψ² s²/n)       h_1 ~ N(ln s², k_h)
    (λ0, λ1) ~ N((m_λ0, m_λ1), diag(V_λ0, V_λ1)), λ1 ∈ (−1, 1)       σ⁻²_ξ ~ G(1/k_ξ, 1)  (Poirier: mean, dof)
Gibbs: μ | ·, then SV with the Kim–Shephard–Chib (1998) 7-component mixture, h | · by a banded precision sampler.
Re-estimated at every month-end on an expanding window. Predictive: one simulated c_{m+1} per kept draw.

Joint moments for the allocation (cash allowed): spread mean/variance from B1's spread forecast, E[c], Var(c) from
here, corr(c, d) from the EWMA risk proxy V. zero_spread=True gives the Cash_timing benchmark (no M-vs-L view).
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_solve_banded, cholesky_banded, solve_banded

from .contracts import Forecast
from .portfolio import month_end_mask

# Kim, Shephard & Chib (1998) mixture approximation of log χ²_1 (means already shifted by −1.2704)
KSC_P = np.array([0.00730, 0.10556, 0.00002, 0.04395, 0.34001, 0.24566, 0.25750])
KSC_M = np.array([-10.12999, -3.97281, -8.56686, 2.77786, 0.61942, 1.79518, -1.08819]) - 1.2704
KSC_V = np.array([5.79596, 2.61369, 5.17950, 0.16735, 0.64009, 0.34023, 1.26261])


def _sample_gmrf(ab: np.ndarray, c: np.ndarray, rng) -> np.ndarray:
    """Draw x ~ N(P⁻¹c, P⁻¹) for P given in upper band form."""
    U = cholesky_banded(ab)
    mean = cho_solve_banded((U, False), c)
    return mean + solve_banded((0, ab.shape[0] - 1), U, rng.standard_normal(len(c)))


def pm_sv_predictive(y: np.ndarray, hp: dict, n_burn: int, n_keep: int, rng) -> np.ndarray:
    """One-step predictive draws of the next value of y under PM-SV."""
    T = len(y)
    t = T + 1
    X = np.ones((T, 1))
    rbar = y.mean()
    s2 = ((y - rbar) ** 2).sum() / max(t - 2, 1)
    b0 = np.array([rbar])
    Vi = np.linalg.inv(hp["psi"] ** 2 * s2 * np.linalg.inv(X.T @ X))
    lam_mean = np.array([hp["m_lambda0"], hp["m_lambda1"]])
    lam_var = np.array([hp["V_lambda0"], hp["V_lambda1"]])
    xi_shape, xi_rate = 0.5, hp["k_xi"] / 2
    h1_mean, h1_var = np.log(s2), hp["k_h"]
    h = np.full(T, np.log(s2))
    lam = np.array([np.log(s2) * (1 - hp["m_lambda1"]), hp["m_lambda1"]])
    s2xi = 1.0 / (xi_shape / xi_rate)
    pred = np.empty(n_keep)
    for it in range(n_burn + n_keep):
        iv = np.exp(-h)
        P = Vi + (X * iv[:, None]).T @ X
        c = Vi @ b0 + (X * iv[:, None]).T @ y
        L = np.linalg.cholesky(P)
        b = np.linalg.solve(P, c) + np.linalg.solve(L.T, rng.standard_normal(1))
        e = y - X @ b
        ys = np.log(e ** 2 + 1e-12)
        lp = np.log(KSC_P) - 0.5 * np.log(KSC_V) - 0.5 * (ys[:, None] - h[:, None] - KSC_M) ** 2 / KSC_V
        pr_s = np.exp(lp - lp.max(1, keepdims=True))
        cum = np.cumsum(pr_s / pr_s.sum(1, keepdims=True), 1)
        s = (rng.random(T)[:, None] > cum).sum(1)
        mo, vo = KSC_M[s], KSC_V[s]
        l0, l1 = lam
        dg = np.full(T, (1 + l1 ** 2) / s2xi)
        dg[0] = 1 / h1_var + l1 ** 2 / s2xi
        dg[-1] = 1 / s2xi
        cv = np.full(T, (l0 - l1 * l0) / s2xi)
        cv[0] = h1_mean / h1_var - l1 * l0 / s2xi
        cv[-1] = l0 / s2xi
        ab = np.zeros((2, T))
        ab[1] = dg + 1 / vo
        ab[0, 1:] = -l1 / s2xi
        h = _sample_gmrf(ab, cv + (ys - mo) / vo, rng)
        Z = np.column_stack([np.ones(T - 1), h[:-1]])
        Pl = np.diag(1 / lam_var) + Z.T @ Z / s2xi
        cl = lam_mean / lam_var + Z.T @ h[1:] / s2xi
        ml = np.linalg.solve(Pl, cl)
        Ll = np.linalg.cholesky(np.linalg.inv(Pl))
        for _ in range(50):
            cand = ml + Ll @ rng.standard_normal(2)
            if -1 < cand[1] < 1:
                lam = cand
                break
        res = h[1:] - Z @ lam
        s2xi = 1 / rng.gamma(xi_shape + (T - 1) / 2, 1.0 / (xi_rate + res @ res / 2))
        if it >= n_burn:
            h_next = lam[0] + lam[1] * h[-1] + rng.normal(0, np.sqrt(s2xi))
            pred[it - n_burn] = np.array([1.0]) @ (b + 0.0) + np.exp(h_next / 2) * rng.standard_normal()
    return pred


def level_forecasts(y: np.ndarray, dates, prep, start: int, end: int, cfg: dict) -> dict:
    """{month-end row t: (mean, var) of next month's c}, for month-ends from the one before `start` to `end`."""
    me = np.flatnonzero(month_end_mask(dates))
    both = np.isfinite(y).all(1)
    c = np.full(len(me), np.nan)
    prev = -1
    for k, t in enumerate(me):
        blk = both[prev + 1:t + 1]
        if blk.any():
            c[k] = y[prev + 1:t + 1][blk].mean(1).sum()
        prev = t
    first = int(np.searchsorted(me, start, side="right")) - 1
    last = int(np.searchsorted(me, end, side="right")) - 1
    out = {}
    for k in range(first, last + 1):
        ys = c[:k + 1]
        ys = ys[np.isfinite(ys)]
        if len(ys) < cfg["min_months"]:
            raise ValueError("not enough months for the level model before the evaluation start")
        rng = np.random.default_rng(cfg["seed"] + 500_000 + k)
        dr = pm_sv_predictive(ys, cfg["priors"], cfg["n_burn"], cfg["n_keep"], rng)
        out[int(me[k])] = (float(dr.mean()), float(dr.var(ddof=1)))
    return out


def add_level(fc: dict, level: dict, ids, *, zero_spread: bool = False):
    """Return new forecasts (same days) whose common level comes from `level`, plus table columns."""
    keys = np.array(sorted(level))
    uc, ud = np.array([0.5, 0.5]), np.array([1.0, -1.0])
    Amat = np.array([[1.0, 0.5], [1.0, -0.5]])
    out, rows = {}, {}
    for t, f in fc.items():
        k = keys[keys <= t]
        if not len(k) or not f.valid:
            out[t] = f
            continue
        mc, vc = level[int(k[-1])]
        V, H = f.V_next, f.H
        corr = (uc @ V @ ud) / np.sqrt((uc @ V @ uc) * (ud @ V @ ud))
        if zero_spread:
            md, vd = 0.0, H * (ud @ V @ ud)
        else:
            md, vd = float(f.a[0] - f.a[1]), float(f.B[0, 0] + f.B[1, 1] - 2 * f.B[0, 1])
        cv = corr * np.sqrt(vc * vd)
        S = np.array([[vc, cv], [cv, vd]])
        a = Amat @ np.array([mc, md])
        B = Amat @ S @ Amat.T
        B = (B + B.T) / 2
        dg = np.diag(B)
        mean = np.exp(a + dg / 2) - 1
        cov = np.exp(a[:, None] + a[None, :] + (dg[:, None] + dg[None, :]) / 2) * np.expm1(B)
        ok = bool(np.isfinite(mean).all() and np.isfinite(cov).all() and np.all(np.linalg.eigvalsh(B) > 0))
        out[t] = Forecast(f.date, f.strategy_ids, H, f.m, f.P, V, a, B, mean, cov, ok,
                          "" if ok else "non-finite or non-PD predictive")
        rows[f.date] = {"level_mean": mc, "level_var": vc, "spread_mean": md, "spread_var": vd,
                        f"cash_mean_{ids[0]}": mean[0], f"cash_mean_{ids[1]}": mean[1]}
    return out, rows
