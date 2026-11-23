"""Part 3 model B1, part 1: Bayesian learning of the expected M-L return SPREAD, shrunk toward "no difference".

    d_t     = y_1,t - y_2,t = delta_t + eps_t,   eps_t ~ N(0, v_t),  v_t = u'V_t u,  u = (1, -1)
    delta_t = phi * delta_{t-1} + eta_t,          eta_t ~ N(0, W),    W = (1 - phi^2) sigma^2
    delta_0 ~ N(0, sigma^2)                       (stationary prior centred on "no difference")

The common level (being invested at all vs cash) is modelled separately in part3_b1_level; B1 combines both.
The AR(1) pull toward 0 keeps the shrinkage active over time (half-life `half_life` days), not only at the start.

V_t is the same kind of joint EWMA risk proxy as B0 (zero mean, updated on full observations); the model is
conditional on it. Pre-specified, not tuned: half_life 252, sigma_annual 0.05, beta 0.97, H 21.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import Forecast


def reference_cov(y: np.ndarray, jitter: float) -> np.ndarray:
    rows = y[np.isfinite(y).all(1)]
    if len(rows) < y.shape[1] + 20:
        raise ValueError(f"only {len(rows)} common observations to build V_ref")
    return np.cov(rows, rowvar=False, ddof=1) + jitter * np.eye(y.shape[1])


def horizon_weights(phi: float, H: int):
    """For G = sum_{h=2}^{H+1} d_{t+h}: weight of delta_t, and sum of squared weights of eta_{t+1..t+H+1}."""
    hs = np.arange(2, H + 2)
    c = float((phi ** hs).sum())
    e = [float((phi ** (hs[hs >= j] - j)).sum()) for j in range(1, H + 2)]
    return c, float(np.square(e).sum())


def spread_moments(a_d: float, s_state: float, V: np.ndarray, H: int):
    """Joint log moments of the two H-day returns implied by a spread forecast (common level set to 0).

    a = a_d * (1/2, -1/2);  B = H * V + s_state * u u',  u = (1/2, -1/2)  ->  Var(G_1 - G_2) = H u2'V u2 + s_state.
    """
    u = np.array([0.5, -0.5])
    a = a_d * u
    B = H * V + s_state * np.outer(u, u)
    B = (B + B.T) / 2
    d = np.diag(B)
    mean = np.exp(a + d / 2) - 1
    cov = np.exp(a[:, None] + a[None, :] + (d[:, None] + d[None, :]) / 2) * np.expm1(B)
    return a, B, mean, cov


class SpreadFilter:
    def __init__(self, V_ref: np.ndarray, phi: float, sigma: float, beta: float):
        self.phi, self.beta = phi, beta
        self.m, self.P = 0.0, sigma ** 2
        self.W = (1 - phi ** 2) * sigma ** 2
        self.V = V_ref.copy()

    def update(self, y: np.ndarray) -> None:
        a, R = self.phi * self.m, self.phi ** 2 * self.P + self.W
        if np.isfinite(y).all():
            d = y[0] - y[1]
            v = self.V[0, 0] + self.V[1, 1] - 2 * self.V[0, 1]           # V_t from yesterday
            K = R / (R + v)
            self.m = a + K * (d - a)
            self.P = (1 - K) ** 2 * R + K ** 2 * v
            self.V = self.beta * self.V + (1 - self.beta) * np.outer(y, y)
        else:
            self.m, self.P = a, R

    def predict(self, H: int):
        c, e2 = horizon_weights(self.phi, H)
        a_d, s = c * self.m, c ** 2 * self.P + e2 * self.W
        return a_d, s, *spread_moments(a_d, s, self.V, H)


def run_filter(y: np.ndarray, dates: pd.DatetimeIndex, ids, *, half_life: float, sigma_annual: float, beta: float,
               H: int, jitter: float, prep: tuple[int, int], end: int):
    if len(ids) != 2:
        raise ValueError("B1 models the spread of exactly two strategies")
    p0, p1 = prep
    V_ref = reference_cov(y[p0:p1 + 1], jitter)
    f = SpreadFilter(V_ref, 0.5 ** (1 / half_life), sigma_annual / 252, beta)
    out, rows = {}, []
    for t in range(p0, end + 1):
        f.update(y[t])
        a_d, s, a, B, mean, cov = f.predict(H)
        ok = bool(np.isfinite(mean).all() and np.isfinite(cov).all() and np.all(np.linalg.eigvalsh(B) > 0))
        out[t] = Forecast(dates[t], tuple(ids), H, np.array([f.m]), np.array([[f.P]]), f.V.copy(), a, B, mean, cov,
                          ok, "" if ok else "non-finite or non-PD predictive")
        row = {"date": dates[t], "H": H, "valid": ok, "state_mean": f.m, "state_var": f.P, "a_spread": a_d,
               "B_spread": B[0, 0] + B[1, 1] - 2 * B[0, 1]}
        for i, sid in enumerate(ids):
            row |= {f"y_{sid}": y[t, i], f"V_{sid}": f.V[i, i], f"a_{sid}": a[i], f"B_{sid}": B[i, i],
                    f"mean_{sid}": mean[i], f"var_{sid}": cov[i, i]}
        p = f"{ids[0]}_{ids[1]}"
        row |= {f"V_{p}": f.V[0, 1], f"B_{p}": B[0, 1], f"cov_{p}": cov[0, 1]}
        rows.append(row)
    return out, pd.DataFrame(rows).set_index("date"), V_ref
