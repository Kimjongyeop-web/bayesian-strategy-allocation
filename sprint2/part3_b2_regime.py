"""Part 3 model B2: Bayesian two-regime Markov-switching model PER STRATEGY (calm / turbulent), cash allowed.

For each strategy s separately (weekly log returns, rows <= t only):

    y_t | S_t = k ~ N(μ_k, σ²_k),   k ∈ {0 calm, 1 turbulent},   S_t Markov with P = [[p00, 1−p00], [1−p11, p11]]

Priors (fixed before the first run, see docs/methodology.md):
    μ_k ~ N(ȳ, (sigma_mu_annual / 52)²)     both regimes centred on the strategy's prevailing weekly mean
    σ²_k ~ IG(var_shape, (var_shape − 1) s²)   both centred on the sample variance
    p_kk ~ Beta(p_stay_mean · n, (1 − p_stay_mean) · n),  n = p_stay_strength
Identification: calm = lower variance (labels swapped when a draw violates σ²_0 < σ²_1).
Gibbs (Kim & Nelson 1999, ch. 9): FFBS for S, conjugate μ, σ², p. Re-estimated at each month-end on an expanding
window; the chain is warm-started from the previous month-end's last draw (shorter burn-in after the first fit).

Predictive (next month, horizon `weeks_ahead`): per kept draw, start from the filtered regime probability at the
origin, simulate regime paths and returns, sum to a monthly log return. Regime and parameter uncertainty both enter.
Joint moments for the allocation: each strategy's predictive mean/variance from its own model, correlation from the
EWMA risk proxy V (beta 0.97). Allocation: the shared cash-allowed rule (as B0), reviewed at month-ends.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .contracts import Forecast
from .portfolio import month_end_mask

H_DAYS = 21


def weekly_returns(y: np.ndarray, dates: pd.DatetimeIndex, upto: int) -> list[np.ndarray]:
    """Per strategy: weekly sums of daily log returns on live days, rows <= upto (last week may be partial)."""
    wk = dates[:upto + 1].to_period("W-FRI")
    out = []
    for i in range(y.shape[1]):
        s = pd.Series(y[:upto + 1, i], index=wk).dropna()
        out.append(s.groupby(level=0).sum().to_numpy())
    return out


def _ffbs(yv, mu, s2, P, rng):
    """Forward filter, backward sample for a 2-state chain (python floats: fast for scalars)."""
    T = len(yv)
    p00, p11 = P
    pi1 = (1 - p00) / (2 - p00 - p11)
    f1 = np.empty(T)
    a0, a1 = 1 - pi1, pi1
    c0, c1 = 1 / math.sqrt(2 * math.pi * s2[0]), 1 / math.sqrt(2 * math.pi * s2[1])
    m0, m1, v0, v1 = mu[0], mu[1], 2 * s2[0], 2 * s2[1]
    for t in range(T):
        x = yv[t]
        l0 = c0 * math.exp(-(x - m0) ** 2 / v0) * a0
        l1 = c1 * math.exp(-(x - m1) ** 2 / v1) * a1
        tot = l0 + l1
        q1 = l1 / tot if tot > 0 else a1
        f1[t] = q1
        a1 = (1 - q1) * (1 - p00) + q1 * p11
        a0 = 1 - a1
    S = np.empty(T, dtype=np.int8)
    u = rng.random(T)
    S[-1] = u[-1] < f1[-1]
    for t in range(T - 2, -1, -1):
        q1 = f1[t]
        if S[t + 1]:
            w0, w1 = (1 - q1) * (1 - p00), q1 * p11
        else:
            w0, w1 = (1 - q1) * p00, q1 * (1 - p11)
        S[t] = u[t] < w1 / (w0 + w1)
    return S, f1[-1]


def gibbs(yv: np.ndarray, cfg: dict, n_burn: int, n_keep: int, rng, init=None):
    ybar, s2 = float(yv.mean()), float(yv.var(ddof=1))
    tau2 = (cfg["sigma_mu_annual"] / 52) ** 2
    a0, b0 = cfg["var_shape"], (cfg["var_shape"] - 1) * s2
    pa, pb = cfg["p_stay_mean"] * cfg["p_stay_strength"], (1 - cfg["p_stay_mean"]) * cfg["p_stay_strength"]
    if init is None:
        mu, sig2, P = np.array([ybar, ybar]), np.array([0.6 * s2, 2.0 * s2]), np.array([0.95, 0.9])
    else:
        mu, sig2, P = (np.array(v, float) for v in init)
    keep = dict(mu=np.empty((n_keep, 2)), s2=np.empty((n_keep, 2)), P=np.empty((n_keep, 2)), q1=np.empty(n_keep))
    for it in range(n_burn + n_keep):
        S, q1 = _ffbs(yv, mu, sig2, P, rng)
        for k in (0, 1):
            yk = yv[S == k]
            prec = 1 / tau2 + len(yk) / sig2[k]
            mu[k] = rng.normal((ybar / tau2 + yk.sum() / sig2[k]) / prec, prec ** -0.5)
            sig2[k] = 1 / rng.gamma(a0 + len(yk) / 2, 1 / (b0 + ((yk - mu[k]) ** 2).sum() / 2))
        n00 = int(((S[:-1] == 0) & (S[1:] == 0)).sum())
        n01 = int(((S[:-1] == 0) & (S[1:] == 1)).sum())
        n11 = int(((S[:-1] == 1) & (S[1:] == 1)).sum())
        n10 = int(((S[:-1] == 1) & (S[1:] == 0)).sum())
        P = np.array([rng.beta(pa + n00, pb + n01), rng.beta(pa + n11, pb + n10)])
        if sig2[0] > sig2[1]:                                 # identification: regime 0 is the calm one
            mu, sig2, P, q1 = mu[::-1].copy(), sig2[::-1].copy(), P[::-1].copy(), 1 - q1
        if it >= n_burn:
            k = it - n_burn
            keep["mu"][k], keep["s2"][k], keep["P"][k], keep["q1"][k] = mu, sig2, P, q1
    return keep, (mu.copy(), sig2.copy(), P.copy())


def predictive(keep: dict, weeks: int, paths: int, rng) -> np.ndarray:
    n = len(keep["q1"])
    q1 = np.repeat(keep["q1"], paths)
    mu, s2, P = (np.repeat(keep[k], paths, 0) for k in ("mu", "s2", "P"))
    st = (rng.random(n * paths) < q1).astype(int)
    total = np.zeros(n * paths)
    for _ in range(weeks):
        stay = np.where(st == 1, P[:, 1], P[:, 0])
        st = np.where(rng.random(n * paths) < stay, st, 1 - st)
        idx = np.arange(n * paths)
        total += mu[idx, st] + np.sqrt(s2[idx, st]) * rng.standard_normal(n * paths)
    return total


def _ewma_V(y, prep, end, beta, jitter):
    rows = y[prep[0]:prep[1] + 1]
    rows = rows[np.isfinite(rows).all(1)]
    V = np.cov(rows, rowvar=False, ddof=1) + jitter * np.eye(y.shape[1])
    out = {}
    for t in range(prep[0], end + 1):
        if np.isfinite(y[t]).all():
            V = beta * V + (1 - beta) * np.outer(y[t], y[t])
        out[t] = V.copy()
    return out


def run_filter(y: np.ndarray, dates: pd.DatetimeIndex, ids, cfg: dict, *, prep, start: int, end: int, beta: float,
               jitter: float, sampler=None):
    me = np.flatnonzero(month_end_mask(dates))
    first = int(np.searchsorted(me, start, side="right")) - 1
    last = int(np.searchsorted(me, end, side="right")) - 1
    if first < 0:
        raise ValueError("no month-end before the evaluation start for B2")
    Vs = _ewma_V(y, prep, end, beta, jitter)
    scale = H_DAYS / (5 * cfg["weeks_ahead"])
    state = [None] * len(ids)
    fc, rows = {}, []
    for j, k in enumerate(range(first, last + 1)):
        t = int(me[k])
        if cfg.get("progress", False) and j % 12 == 0:
            print(f"Regime fit {dates[t].date()}", flush=True)
        wk = weekly_returns(y, dates, t)
        mean_s, var_s, row = np.empty(len(ids)), np.empty(len(ids)), {}
        for i, sid in enumerate(ids):
            yv = wk[i]
            if len(yv) < cfg["min_weeks"]:
                raise ValueError(f"B2: only {len(yv)} weeks for {sid} before {dates[t].date()}")
            rng = np.random.default_rng(cfg["seed"] + 1000 * i + k)
            burn = cfg["n_burn"] if state[i] is None else cfg["n_burn_warm"]
            keep, state[i] = (gibbs if sampler is None else sampler)(yv, cfg, burn, cfg["n_keep"], rng, state[i])
            dr = predictive(keep, cfg["weeks_ahead"], cfg["paths_per_draw"], rng) * scale ** 0.5
            mean_s[i] = dr.mean() / scale ** 0.5 * scale        # mean scales with time, sd with sqrt(time)
            var_s[i] = dr.var(ddof=1)
            row |= {f"p_turbulent_{sid}": float(keep["q1"].mean()), f"mu_calm_{sid}": float(keep["mu"][:, 0].mean()),
                    f"mu_turb_{sid}": float(keep["mu"][:, 1].mean()),
                    f"vol_calm_{sid}": float(np.sqrt(keep["s2"][:, 0] * 52).mean()),
                    f"vol_turb_{sid}": float(np.sqrt(keep["s2"][:, 1] * 52).mean()),
                    f"p_stay_calm_{sid}": float(keep["P"][:, 0].mean()),
                    f"p_stay_turb_{sid}": float(keep["P"][:, 1].mean()), f"n_weeks_{sid}": len(yv)}
        V = Vs[min(max(t, prep[0]), end)]
        rho = V[0, 1] / np.sqrt(V[0, 0] * V[1, 1])
        B = np.array([[var_s[0], rho * np.sqrt(var_s[0] * var_s[1])], [rho * np.sqrt(var_s[0] * var_s[1]), var_s[1]]])
        a = mean_s.copy()
        dg = np.diag(B)
        mean = np.exp(a + dg / 2) - 1
        cov = np.exp(a[:, None] + a[None, :] + (dg[:, None] + dg[None, :]) / 2) * np.expm1(B)
        ok = bool(np.isfinite(mean).all() and np.isfinite(cov).all() and np.all(np.linalg.eigvalsh(B) > 0))
        day = start if j == 0 else t
        fc[day] = Forecast(dates[day], tuple(ids), H_DAYS, a.copy(), B.copy(), V.copy(), a, B, mean, cov, ok,
                           "" if ok else "non-finite or non-PD predictive")
        row = {"date": dates[day], "origin_month_end": dates[t], "valid": ok, "rho": rho,
               **{f"mean_{s}": mean[i] for i, s in enumerate(ids)}, **{f"var_{s}": cov[i, i] for i, s in enumerate(ids)},
               f"cov_{ids[0]}_{ids[1]}": cov[0, 1], **row}
        rows.append(row)
    return fc, pd.DataFrame(rows).set_index("date")
