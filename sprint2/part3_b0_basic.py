"""Part 3 model B0 (basic Bayesian, the plain baseline): per-strategy latent expected log return + H-day predictive.

    y_t  = mu_t + eps_t,      eps_t ~ N(0, V_t)        y_t = log(1 + g_t), g = virtual gross return
    mu_t = mu_{t-1} + eta_t,  eta_t ~ N(0, W),  W = q * V_ref

V_t is an EWMA second moment (zero-mean) risk proxy, V_{t+1} = beta V_t + (1-beta) y_t y_t'. The model is a
Gaussian state-space model CONDITIONAL on this risk estimate; V uncertainty is not integrated out.
Inputs are generic (strategy ids + a returns matrix); no strategy module is imported here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import Forecast


def to_log(g: np.ndarray) -> np.ndarray:
    if np.any(g[np.isfinite(g)] <= -1):
        raise ValueError("gross return <= -100%: log return undefined (not clipped on purpose)")
    return np.log1p(g)


def reference_cov(y: np.ndarray, jitter: float = 1e-8) -> np.ndarray:
    """Covariance (ddof=1) of rows where every strategy is observed, + jitter*I."""
    rows = y[np.isfinite(y).all(1)]
    if len(rows) < y.shape[1] + 20:
        raise ValueError(f"only {len(rows)} common observations to build V_ref")
    return np.cov(rows, rowvar=False, ddof=1).reshape(y.shape[1], y.shape[1]) + jitter * np.eye(y.shape[1])


class LocalLevelFilter:
    def __init__(self, V_ref: np.ndarray, q: float, beta: float, m0: float = 0.0, prior_obs: float = 60.0):
        S = len(V_ref)
        self.m = np.full(S, float(m0))
        self.P = V_ref / prior_obs
        self.V = V_ref.copy()
        self.W = q * V_ref
        self.beta = beta

    def update(self, y: np.ndarray) -> None:
        """Kalman step with V_t from yesterday; only observed components enter; then the EWMA update."""
        a, R = self.m, self.P + self.W
        o = np.isfinite(y)
        if o.any():
            Vo = self.V[np.ix_(o, o)]
            K = np.linalg.solve(R[np.ix_(o, o)] + Vo, R[o, :]).T          # S x |o|
            Hs = np.eye(len(a))[o]
            IKH = np.eye(len(a)) - K @ Hs
            self.m = a + K @ (y[o] - a[o])
            self.P = IKH @ R @ IKH.T + K @ Vo @ K.T
        else:
            self.m, self.P = a, R
        if o.all():                                                       # joint EWMA on full observations
            self.V = self.beta * self.V + (1 - self.beta) * np.outer(y, y)

    def predict(self, H: int):
        """Moments of G = sum_{h=2}^{H+1} y_{t+h}: decision at close t, fill at close t+1.

        Future V frozen at V_{t+1} (conditional approximation). Returns log moments (a, B) and the simple
        H-day return mean/cov of exp(G)-1 under lognormality.
        """
        cW = H ** 2 + H * (H + 1) * (2 * H + 1) / 6
        a = H * self.m
        B = H ** 2 * self.P + cW * self.W + H * self.V
        B = (B + B.T) / 2
        d = np.diag(B)
        mean = np.exp(a + d / 2) - 1
        cov = np.exp(a[:, None] + a[None, :] + (d[:, None] + d[None, :]) / 2) * np.expm1(B)
        return a, B, mean, cov


def run_filter(y: np.ndarray, dates: pd.DatetimeIndex, ids, *, q: float, beta: float, H: int, m0: float,
               prior_obs: float, jitter: float, prep: tuple[int, int], end: int):
    """Filter from prep start to `end`. V_ref uses the prep window only (all <= prep end, before any decision).

    Returns {t: Forecast} for every row in [prep start, end] and a tidy table for storage.
    """
    p0, p1 = prep
    V_ref = reference_cov(y[p0:p1 + 1], jitter)
    f = LocalLevelFilter(V_ref, q, beta, m0, prior_obs)
    out, rows = {}, []
    S = len(ids)
    for t in range(p0, end + 1):
        f.update(y[t])
        a, B, mean, cov = f.predict(H)
        ok = bool(np.all(np.isfinite(mean)) and np.all(np.isfinite(cov)) and np.all(np.linalg.eigvalsh(B) > 0))
        out[t] = Forecast(dates[t], tuple(ids), H, f.m.copy(), f.P.copy(), f.V.copy(), a, B, mean, cov, ok,
                          "" if ok else "non-finite or non-PD predictive")
        row = {"date": dates[t], "H": H, "valid": ok}
        for i, s in enumerate(ids):
            row |= {f"y_{s}": y[t, i], f"m_{s}": f.m[i], f"P_{s}": f.P[i, i], f"V_{s}": f.V[i, i],
                    f"a_{s}": a[i], f"B_{s}": B[i, i], f"mean_{s}": mean[i], f"var_{s}": cov[i, i]}
        for i in range(S):
            for k in range(i + 1, S):
                p = f"{ids[i]}_{ids[k]}"
                row |= {f"P_{p}": f.P[i, k], f"V_{p}": f.V[i, k], f"B_{p}": B[i, k], f"cov_{p}": cov[i, k]}
        rows.append(row)
    return out, pd.DataFrame(rows).set_index("date"), V_ref
