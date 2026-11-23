"""Part 4: performance tables and figures from saved run files (no strategy or model imports).

Tables and figures are separate functions so later material (slides, papers) can reuse them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def power_cer_monthly(daily: pd.DataFrame, A: float) -> dict:
    """Pettenuzzo–Ravazzolo eq. 30: annualised CER of monthly gross returns under power utility."""
    nav = daily["nav"]
    me = nav.groupby(nav.index.to_period("M")).last()
    W = (me / me.shift(1)).dropna().to_numpy()
    if len(W) < 2:
        return {"n_months": len(W), "power_cer": np.nan}
    U = W ** (1 - A) / (1 - A)
    return {"n_months": len(W), "power_cer": float(12 * (((1 - A) * U.mean()) ** (1 / (1 - A)) - 1))}


def metrics(daily: pd.DataFrame, gamma: float, ann: int = 252, ddof: int = 1) -> dict:
    """Daily net-return metrics. CER = ann*mean - gamma/2 * ann*var (ddof=1), risk-free 0."""
    r = daily["net_return"].to_numpy(float)
    if not np.isfinite(r).all():
        raise ValueError("missing evaluation returns must not be dropped silently")
    wealth = np.r_[1.0, np.cumprod(1 + r)]
    sd = r.std(ddof=ddof) if len(r) > ddof else np.nan
    out = dict(start=str(daily.index[0].date()), end=str(daily.index[-1].date()), days=len(r),
               total_return=wealth[-1] - 1, cagr=wealth[-1] ** (ann / len(r)) - 1,
               volatility=sd * np.sqrt(ann), sharpe=r.mean() / sd * np.sqrt(ann) if sd > 0 else np.nan,
               cer=ann * r.mean() - gamma / 2 * ann * sd ** 2,
               max_drawdown=float(np.min(wealth / np.maximum.accumulate(wealth) - 1)),
               annual_turnover=float(daily["turnover"].mean() * ann),
               cost_total=float(daily["cost"].sum()), writeoff_total=float(daily["writeoff"].sum()),
               mean_cash=float((1 - daily["exposure"]).mean()),
               trade_days=int(((daily["buy_notional"] + daily["sell_notional"]) > 0).sum()),
               unfilled_orders=int(daily["unfilled"].sum()))
    if "action_executed" in daily:
        acts = daily["action_executed"]
        out |= dict(transfer_fills=int((acts == "TRANSFER").sum()))
    return out


def decision_counts(decisions: pd.DataFrame) -> dict:
    if decisions.empty:
        return {}
    gaps = decisions.index[decisions["action"] == "TRANSFER"].to_series().diff().dt.days.dropna()
    return dict(hold=int((decisions["action"] == "HOLD").sum()), transfer=int((decisions["action"] == "TRANSFER").sum()),
                mean_calendar_days_between_transfers=float(gaps.mean()) if len(gaps) else np.nan,
                hold_reasons="; ".join(f"{k}={v}" for k, v in decisions.loc[decisions.action == "HOLD", "reason"]
                                       .value_counts().items()))


def summary_table(dailies: dict, decisions: dict, gamma: float, ann: int, ddof: int, start=None) -> pd.DataFrame:
    rows = []
    for name, d in dailies.items():
        d = d if start is None else d.loc[d.index > pd.Timestamp(start)]
        rows.append({"policy": name, **metrics(d, gamma, ann, ddof), **decision_counts(decisions.get(name, pd.DataFrame()))})
    return pd.DataFrame(rows).set_index("policy")


def forecast_errors(fc: pd.DataFrame, ids, H: int) -> pd.DataFrame:
    """Realised G_{t,H} = sum_{h=2}^{H+1} y_{t+h} next to the forecast made at t.

    Consecutive targets overlap by H-1 days: errors are serially dependent, not independent samples.
    """
    out = pd.DataFrame(index=fc.index)
    for s in ids:
        y = fc[f"y_{s}"]
        G = sum(y.shift(-h) for h in range(2, H + 2))      # NaN if any day missing or beyond the sample
        out[f"G_{s}"] = G
        out[f"a_{s}"] = fc[f"a_{s}"]
        out[f"z_{s}"] = (G - fc[f"a_{s}"]) / np.sqrt(fc[f"B_{s}"])
        out[f"in90_{s}"] = (out[f"z_{s}"].abs() <= 1.6449).where(G.notna())
    return out


def forecast_summary(err: pd.DataFrame, ids) -> pd.DataFrame:
    rows = []
    for s in ids:
        z = err[f"z_{s}"].dropna()
        rows.append(dict(strategy=s, n=len(z), mean_error=float((err[f"G_{s}"] - err[f"a_{s}"]).mean()),
                         z_mean=float(z.mean()), z_sd=float(z.std(ddof=1)),
                         coverage90=float(err[f"in90_{s}"].dropna().astype(float).mean())))
    return pd.DataFrame(rows).set_index("strategy")


def _newey_west_var(x: np.ndarray, lags: int) -> float:
    x = x - x.mean()
    n = len(x)
    v = x @ x / n
    for k in range(1, lags + 1):
        v += 2 * (1 - k / (lags + 1)) * (x[k:] @ x[:-k]) / n
    return v


def spread_forecast_eval(fc: pd.DataFrame, ids, H: int) -> dict:
    """Skill of the predicted H-day log-return spread (strategy 1 - strategy 2) vs a 'zero difference' forecast.

    Same columns exist for every model (a_*, B_*, B_pair, y_*). Targets overlap (H days): the Diebold-Mariano
    statistic uses Newey-West with H lags; it is still an approximate test, not independent samples.
    """
    s1, s2 = ids
    d = fc[f"y_{s1}"] - fc[f"y_{s2}"]
    G = sum(d.shift(-h) for h in range(2, H + 2))
    a = fc[f"a_{s1}"] - fc[f"a_{s2}"]
    B = fc[f"B_{s1}"] + fc[f"B_{s2}"] - 2 * fc[f"B_{s1}_{s2}"]
    ok = G.notna() & a.notna() & B.notna()
    G, a, B = G[ok].to_numpy(), a[ok].to_numpy(), B[ok].to_numpy()
    if len(G) < 3 * H:
        return {"n": len(G)}
    e = G - a
    loss = e ** 2 - G ** 2                                 # < 0 means the model beats the zero forecast
    dm = loss.mean() / np.sqrt(_newey_west_var(loss, H) / len(loss))
    z = e / np.sqrt(B)
    nz = a != 0
    return dict(n=len(G), r2_vs_zero=1 - (e ** 2).sum() / (G ** 2).sum(), dm_stat=dm,
                sign_hit=float((np.sign(a[nz]) == np.sign(G[nz])).mean()) if nz.any() else np.nan,
                mean_forecast=float(a.mean()), mean_realised=float(G.mean()),
                z_sd=float(z.std(ddof=1)), coverage90=float((np.abs(z) <= 1.6449).mean()))


def _boot_index(n: int, block: int, draws: int, seed: int) -> np.ndarray:
    """Circular block bootstrap indices (draws x n)."""
    rng = np.random.default_rng(seed)
    k = -(-n // block)
    starts = rng.integers(0, n, (draws, k))
    return ((starts[:, :, None] + np.arange(block)) % n).reshape(draws, -1)[:, :n]


def _cer_sharpe(R: np.ndarray, gamma: float, ann: int, ddof: int):
    m, sd = R.mean(-1), R.std(-1, ddof=ddof)
    return ann * m - gamma / 2 * ann * sd ** 2, m / sd * np.sqrt(ann)


def significance(dailies: dict, policies, benchmarks, periods: dict, gamma: float, ann: int, ddof: int,
                 boot: dict) -> pd.DataFrame:
    """Policy - benchmark differences in CER and Sharpe with circular block bootstrap CIs (same days resampled
    for both series, so their correlation is kept). p_boot = share of resamples with difference <= 0
    (one-sided, percentile method). Many comparisons are reported together: read them as a family, not one
    by one (no multiple-testing correction applied here)."""
    rows = []
    for period, (start, end) in periods.items():
        for b in benchmarks:
            for p in policies:
                if p == b or b not in dailies:
                    continue
                df = pd.concat([dailies[p]["net_return"], dailies[b]["net_return"]], axis=1, keys=["p", "b"])
                df = df.loc[(df.index > pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))].dropna()
                if len(df) < 2 * boot["block"]:
                    continue
                X = df.to_numpy()
                idx = _boot_index(len(X), boot["block"], boot["draws"], boot["seed"])
                cp_, sp_ = _cer_sharpe(X[:, 0], gamma, ann, ddof)
                cb_, sb_ = _cer_sharpe(X[:, 1], gamma, ann, ddof)
                cP, sP = _cer_sharpe(X[idx, 0], gamma, ann, ddof)
                cB, sB = _cer_sharpe(X[idx, 1], gamma, ann, ddof)
                for stat, point, dist in (("cer", cp_ - cb_, cP - cB), ("sharpe", sp_ - sb_, sP - sB)):
                    lo, hi = np.percentile(dist, [2.5, 97.5])
                    rows.append(dict(period=period, policy=p, benchmark=b, stat=stat, diff=point, ci95_low=lo,
                                     ci95_high=hi, p_boot=float((dist <= 0).mean()), days=len(X)))
    return pd.DataFrame(rows)


def plot_all(dailies: dict, sleeves: pd.DataFrame, out: Path, start=None) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out.mkdir(parents=True, exist_ok=True)
    files = []
    fig, ax = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True, layout="constrained")
    for name, d in dailies.items():
        r = d["net_return"] if start is None else d.loc[d.index > pd.Timestamp(start), "net_return"]
        w = (1 + r).cumprod()
        ax[0].plot(w.index, w, label=name, lw=1.2)
        ax[1].plot(w.index, 100 * (w / w.cummax() - 1), lw=1.0)
    ax[0].set(ylabel="Net wealth (start = 1)", title="Cumulative net wealth")
    ax[1].set(ylabel="Drawdown (%)")
    ax[0].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=0.25)
    files.append(out / "wealth_drawdown.png")
    fig.savefig(files[-1], dpi=130)
    plt.close(fig)
    if sleeves is not None and not sleeves.empty:
        accts = list(sleeves["account"].unique())
        fig, axes = plt.subplots(len(accts), 1, figsize=(11, 2.6 * len(accts)), sharex=True, squeeze=False,
                                 layout="constrained")
        for a, name in zip(axes[:, 0], accts):
            p = sleeves[sleeves.account == name].pivot(index="date", columns="sleeve", values="allocation")
            a.stackplot(p.index, *[p[c].to_numpy() for c in p.columns], labels=list(p.columns), alpha=0.8)
            a.set(ylim=(0, 1), ylabel="Capital share", title=f"{name}: realised allocation (rest = cash)")
            a.legend(loc="upper left", fontsize=8)
            a.grid(alpha=0.25)
        files.append(out / "allocation.png")
        fig.savefig(files[-1], dpi=130)
        plt.close(fig)
    return files
