"""Thin runner: config -> market -> virtual strategies -> posterior -> policies -> files.

    python -m sprint2 check-data [--data PATH]
    python -m sprint2 run [--config FILE] [--data PATH] [--start D] [--end D] [--run-id ID]
    python -m sprint2 report RUN_DIR
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import platform
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import data as data_mod
from .contracts import Market, SelectionRule, StrategyDef
from .ledger import ledger_frame, orders_frame, run_policy, run_virtual, sleeves_frame
from .part3_allocation import BayesPolicy, FixedMonthly
from . import part3_b0_basic, part3_b1_level, part3_b1_spread, part3_b2_regime
from .part3_b0_basic import to_log
from .part4_evaluation import (forecast_errors, forecast_summary, plot_all, power_cer_monthly, significance,
                               spread_forecast_eval, summary_table)
from .portfolio import Costs, month_end_mask

ROOT = Path(__file__).resolve().parent.parent
_LEVEL_CACHE: dict = {}


def load_config(path=None) -> dict:
    path = Path(path or ROOT / "config/kospi200.yaml")
    with path.open(encoding="utf-8") as stream:
        market = yaml.safe_load(stream)
    if "strategies" in market:  # fully resolved saved/custom config
        return market
    if set(market) != {"data", "execution", "periods"}:
        raise ValueError("Market config requires exactly data/execution/periods")
    with (ROOT / "config/common.yaml").open(encoding="utf-8") as stream:
        common = yaml.safe_load(stream)
    return {**common, **market}


def apply_overrides(cfg: dict, items) -> dict:
    """`a.b.c=value` (value parsed as YAML). Only existing keys, so a typo cannot add a silent new key."""
    for item in items:
        key, _, raw = item.partition("=")
        *path, last = key.strip().split(".")
        node = cfg
        for k in path:
            node = node[k]
        if not isinstance(node, dict) or last not in node:
            raise KeyError(f"--set {key}: no such config key")
        node[last] = yaml.safe_load(raw)
    return cfg


def build_strategies(cfg: dict) -> list[StrategyDef]:
    """Registry/adapter: the only place that resolves strategy modules."""
    out = []
    for s in cfg["strategies"]:
        mod, fn = s["signal"].split(":")
        rule = SelectionRule(**{**cfg["selection_defaults"], **s.get("selection", {})})
        out.append(StrategyDef(s["id"], getattr(importlib.import_module(mod), fn), dict(s.get("params", {})), rule))
    return out


def _rows(dates, cfg):
    p = cfg["periods"]
    first = lambda d: int(dates.searchsorted(pd.Timestamp(d)))
    last = lambda d: int(dates.searchsorted(pd.Timestamp(d), side="right")) - 1
    r = dict(strategy_start=first(p["strategy_start"]), prep=(first(p["prep"][0]), last(p["prep"][1])),
             eval=(first(p["eval"][0]), last(p["eval"][1])))
    if not (r["strategy_start"] <= r["prep"][0] < r["prep"][1] < r["eval"][0] < r["eval"][1] < len(dates)):
        raise ValueError(f"periods must be ordered strategy_start <= prep < eval inside the data: {r}")
    return r


def pipeline(cfg: dict, mk: Market) -> dict:
    """Everything in memory. Strategy outputs and posteriors are recomputed every run (never reused)."""
    for key in ("prep", "eval"):
        a, b = map(pd.Timestamp, cfg["periods"][key])
        if a < mk.dates[0] or b > mk.dates[-1]:
            raise ValueError(f"Requested {key} interval {a.date()}..{b.date()} outside source calendar")
    rows = _rows(mk.dates, cfg)
    ex, md = cfg["execution"], cfg["model"]
    costs = Costs(mk.dates, ex)
    e0, e1 = rows["eval"]
    defs = build_strategies(cfg)
    ids = tuple(d.strategy_id for d in defs)
    books = {}
    for d in defs:
        print(f"Building strategy {d.strategy_id}", flush=True)
        scores = np.asarray(d.signal(mk, **d.params), dtype=float)
        if scores.shape != mk.price.shape:
            raise ValueError(f"{d.strategy_id}: signal shape {scores.shape} != {mk.price.shape}")
        books[d.strategy_id] = run_virtual(mk, d, scores, costs, ex, rows["strategy_start"], e1)
    y = np.full((len(mk.dates), len(ids)), np.nan)
    for i, s in enumerate(ids):
        dly = books[s].daily
        pos = mk.dates.get_indexer(dly.index)
        y[pos, i] = np.where(dly["live"], to_log(dly["gross_return"].to_numpy()), np.nan)
    res = dict(rows=rows, ids=ids, books=books, y=y, policies={}, forecasts={}, vref={})
    for p in cfg["policies"]:
        name = p["id"]
        print(f"Running policy {name}", flush=True)
        if p["type"] == "virtual":
            res["policies"][name] = dict(daily=books[p["strategy"]].daily, decisions=pd.DataFrame(), account=None)
            continue
        H = np.nan                                         # forecast horizon, set by Bayesian models
        if p["type"] == "fixed_monthly":
            policy = FixedMonthly(p["weights"])
        elif p["type"] == "bayes":
            fc, table, vref, H = fit_model(p, cfg, y, mk, ids, rows["prep"], e0, e1)
            res["forecasts"][name] = table
            if vref is not None:
                res["vref"][name] = vref
            policy = BayesPolicy(fc, md["gamma"], md["score_tol"], len(mk.codes), len(ids), md.get("solver", "CLARABEL"),
                                 review=p.get("review", "daily"))
        else:
            raise ValueError(f"unknown policy type {p['type']}")
        acc, decs = run_policy(mk, books, policy, costs, ex, e0, e1, name)
        res["policies"][name] = dict(daily=ledger_frame(acc), decisions=decisions_frame(decs, ids, H),
                                     account=acc)
    return res


def fit_model(p: dict, cfg: dict, y, mk: Market, ids, prep, start, end):
    """Model registry/adapter: each Part 3 model lives in its own module; nothing else picks between them."""
    md, model = cfg["model"], p.get("model", "b0")
    if model == "b0":
        m0 = cfg["model_b0"]
        fc, table, vref = part3_b0_basic.run_filter(y, mk.dates, ids, q=p.get("q", m0["q"]), beta=md["beta"], H=m0["H"],
                                                    m0=m0["m0"], prior_obs=m0["prior_obs"], jitter=md["vref_jitter"],
                                                    prep=prep, end=end)
        return fc, table, vref, m0["H"]
    if model == "b1":                                      # spread forecast + common-level forecast (cash)
        m1 = cfg["model_b1"]
        fc, table, vref = part3_b1_spread.run_filter(y, mk.dates, ids, jitter=md["vref_jitter"], prep=prep, end=end,
                                                     **m1["spread"])
        key = (start, end, hashlib.sha256(y.tobytes()).hexdigest(),
               hashlib.sha256(mk.dates.values.tobytes()).hexdigest(),
               json.dumps(m1["level"], sort_keys=True), mk.source.get("market", "kospi200"))
        if key not in _LEVEL_CACHE:                        # B1 and Cash_timing share the level forecast
            _LEVEL_CACHE.clear()
            _LEVEL_CACHE[key] = part3_b1_level.level_forecasts(y, mk.dates, prep, start, end, m1["level"])
        fc, extra = part3_b1_level.add_level(fc, _LEVEL_CACHE[key], ids, zero_spread=p.get("zero_spread", False))
        table = table.join(pd.DataFrame.from_dict(extra, orient="index"), how="left")
        return fc, table, vref, m1["spread"]["H"]
    if model in ("b2", "b2_1", "b2_2"):
        if model == "b2":
            module = part3_b2_regime
        elif model == "b2_1":
            from . import part3_b2_1_shared_mean as module
        else:
            from . import part3_b2_2_joint as module
        fc, table = module.run_filter(y, mk.dates, ids, cfg["model_b2"], prep=prep, start=start, end=end,
                                               beta=md["beta"], jitter=md["vref_jitter"])
        return fc, table, None, part3_b2_regime.H_DAYS
    raise ValueError(f"unknown model {model}")


def decisions_frame(decs, ids, H) -> pd.DataFrame:
    rows, last = [], None
    for k, d in enumerate(decs):
        row = {"date": d.date, "action": d.action, "reason": d.reason, "score_cand": d.score_cand,
               "score_hold": d.score_hold, "gain": d.gain, "cost_cand": d.cost_cand, "cost_hold": d.cost_hold,
               "forecast_H": H, "days_since_last_transfer": np.nan if last is None else k - last}
        for i, s in enumerate(ids):
            row |= {f"w_target_{s}": d.w_target[i], f"w_exec_{s}": d.w_exec[i], f"w_hold_{s}": d.w_hold[i]}
        rows.append(row)
        if d.action == "TRANSFER":
            last = k
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()


def _git():
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "sprint2", "config"], cwd=ROOT,
                                    capture_output=True, text=True).stdout.strip())
        return dict(commit=sha, code_dirty=dirty)
    except OSError:
        return dict(commit=None, code_dirty=None)


def save(res: dict, cfg: dict, mk: Market, out: Path, elapsed: float) -> None:
    out.mkdir(parents=True, exist_ok=False)
    (out / "strategies").mkdir()
    (out / "policies").mkdir()
    rows, ids, dates = res["rows"], res["ids"], mk.dates
    for s, b in res["books"].items():
        b.daily.to_csv(out / "strategies" / f"{s}_daily.csv")
        me = np.flatnonzero(month_end_mask(dates) & (b.target_w.sum(1) > 0))
        tw = [(dates[t], mk.codes[j], b.held_w[t, j], b.target_w[t, j]) for t in me
              for j in np.flatnonzero((b.target_w[t] > 0) | (b.held_w[t] > 0))]
        pd.DataFrame(tw, columns=["decision_date", "ticker", "held_weight", "target_weight"]).to_csv(
            out / "strategies" / f"{s}_month_end_targets.csv", index=False)
        orders_frame(b.account).to_csv(out / "strategies" / f"{s}_orders.csv.gz", index=False)
    sleeves = []
    for name, p in res["policies"].items():
        p["daily"].to_csv(out / "policies" / f"{name}_daily.csv")
        if p["account"] is not None:
            p["decisions"].to_csv(out / "policies" / f"{name}_decisions.csv")
            orders_frame(p["account"]).to_csv(out / "policies" / f"{name}_orders.csv.gz", index=False)
            sleeves.append(sleeves_frame(p["account"]))
    sleeves = pd.concat(sleeves, ignore_index=True) if sleeves else pd.DataFrame()
    sleeves.to_csv(out / "policies" / "sleeves.csv", index=False)
    for name, table in res["forecasts"].items():
        table.to_csv(out / "policies" / f"{name}_forecasts.csv")
    pd.DataFrame(res["y"], index=dates, columns=[f"y_{s}" for s in ids]).dropna(how="all").to_csv(
        out / "strategies" / "observations_log_gross.csv")
    e0, e1 = rows["eval"]
    manifest = {
        "status": "completed", "elapsed_seconds": round(elapsed, 1), "git": _git(), "data": mk.source,
        "rows": {"strategy_start": str(dates[rows["strategy_start"]].date()),
                 "prep": [str(dates[i].date()) for i in rows["prep"]],
                 "eval_actually_run": [str(dates[e0].date()), str(dates[e1].date())]},
        "strategies": {s: {"version": b.version, "signal": f"{b.definition.signal.__module__}:{b.definition.signal.__name__}",
                           "params": b.definition.params, "rule": b.definition.rule.__dict__}
                       for s, b in res["books"].items()},
        "requested_periods": cfg["periods"], "execution_assumptions": cfg["execution"],
        "return_basis": "adjusted_price_excluding_dividends",
        "vref": {k: v.tolist() for k, v in res["vref"].items()},
        "max_ledger_identity_error_rel": {**{s: b.account.max_identity_error for s, b in res["books"].items()},
                                          **{n: p["account"].max_identity_error for n, p in res["policies"].items()
                                             if p["account"] is not None}},
        "python": platform.python_version(),
        "packages": {p: version(p) for p in ["numpy", "pandas", "scipy", "cvxpy", "clarabel", "openpyxl", "matplotlib", "PyYAML"]},
    }
    (out / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    report(out)


def report(out: Path) -> pd.DataFrame:
    """(Re)build summary tables and figures from saved files only."""
    out = Path(out)
    if (out / "daily.csv.gz").exists():
        from .publication import unpacked
        with unpacked(out) as expanded:
            result = report(expanded)
            for file in expanded.glob("*.csv"):
                import shutil
                shutil.copyfile(file, out / file.name)
            shutil.copytree(expanded / "figures", out / "figures", dirs_exist_ok=True)
            return result
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    ev = cfg["evaluation"]
    e0 = man["rows"]["eval_actually_run"][0]
    rd = lambda f: pd.read_csv(f, index_col=0, parse_dates=True)
    dailies, decisions = {}, {}
    for p in cfg["policies"]:
        dailies[p["id"]] = rd(out / "policies" / f"{p['id']}_daily.csv")
        f = out / "policies" / f"{p['id']}_decisions.csv"
        decisions[p["id"]] = rd(f) if f.exists() else pd.DataFrame()
    summary = summary_table(dailies, decisions, ev["gamma"], ev["annualization"], ev["ddof"], start=e0)
    summary.to_csv(out / "summary.csv")
    e1 = man["rows"]["eval_actually_run"][1]
    periods = {"full": [e0, e1], **ev.get("periods", {})}
    by_period = []
    for name, (a, b) in periods.items():
        cut = {k: d.loc[(d.index > pd.Timestamp(a)) & (d.index <= pd.Timestamp(b))] for k, d in dailies.items()}
        if all(len(d) > 2 for d in cut.values()):
            by_period.append(summary_table(cut, {}, ev["gamma"], ev["annualization"], ev["ddof"]).assign(period=name))
    if by_period:
        pd.concat(by_period).to_csv(out / "summary_by_period.csv")
    if ev.get("benchmarks") and ev.get("bootstrap"):
        pols = [p["id"] for p in cfg["policies"]]
        significance(dailies, pols, ev["benchmarks"], periods, ev["gamma"], ev["annualization"], ev["ddof"],
                     ev["bootstrap"]).to_csv(out / "significance.csv", index=False)
    ids = [s["id"] for s in cfg["strategies"]]
    fsum, spread = [], []
    for p in cfg["policies"]:
        f = out / "policies" / f"{p['id']}_forecasts.csv"
        model = p.get("model", "b0")
        if not f.exists() or model in ("b2", "b2_1", "b2_2"):                # B2's monthly regime table is saved as is
            continue
        fc = rd(f).loc[e0:]
        if model == "b0":                                  # per-strategy levels: B0 forecast errors
            Hp = cfg["model_b0"]["H"]
            err = forecast_errors(fc, ids, Hp)
            err.to_csv(out / "policies" / f"{p['id']}_forecast_errors.csv")
            fsum.append(forecast_summary(err, ids).assign(policy=p["id"]))
        elif model == "b1" and len(ids) == 2 and not p.get("zero_spread", False):
            Hp = cfg["model_b1"]["spread"]["H"]
            for name, (a, b) in periods.items():
                spread.append({"policy": p["id"], "model": model, "H": Hp, "period": name,
                               **spread_forecast_eval(fc.loc[a:b], ids, Hp)})
    if fsum:
        pd.concat(fsum).to_csv(out / "forecast_summary.csv")
    if spread:
        pd.DataFrame(spread).to_csv(out / "forecast_spread.csv", index=False)
    if "power_cer_A" in ev:
        pd.DataFrame([{"policy": k, "period": name, **power_cer_monthly(d.loc[(d.index > pd.Timestamp(a))
                                                                               & (d.index <= pd.Timestamp(b))],
                                                                        ev["power_cer_A"])}
                      for name, (a, b) in periods.items() for k, d in dailies.items()]).to_csv(
            out / "summary_power_cer.csv", index=False)
    sleeves = pd.read_csv(out / "policies" / "sleeves.csv", parse_dates=["date"]) \
        if (out / "policies" / "sleeves.csv").stat().st_size > 2 else None
    plot_all(dailies, sleeves, out / "figures", start=e0)
    return summary


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(prog="python -m sprint2", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check-data", help="schema/unit report of the original workbook")
    r = sub.add_parser("run", help="run strategies + allocation policies")
    for p in (c, r):
        p.add_argument("--market", choices=["kospi200", "sp500", "all"])
        p.add_argument("--config", type=Path)
        p.add_argument("--data", type=Path, help="override data.path")
    r.add_argument("--start", help="override periods.eval[0]")
    r.add_argument("--end", help="override periods.eval[1]")
    r.add_argument("--run-id")
    r.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override an existing config key, e.g. execution.stale_exit.recovery_rate=1.0 (repeatable)")
    g = sub.add_parser("report", help="rebuild summary/figures of a saved run")
    g.add_argument("run_dir", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "report":
        if (a.run_dir / "comparison.json").exists():
            from .markets import compare
            runs = json.loads((a.run_dir / "comparison.json").read_text(encoding="utf-8"))
            for child in runs.values():
                report(a.run_dir / child)
            print(compare(a.run_dir).to_string())
        else:
            print(report(a.run_dir).to_string())
        return
    if a.market == "all":
        from .markets import run_all
        return run_all(a)
    cfg = load_config(a.config or (ROOT / "config/sp500.yaml" if a.market == "sp500" else None))
    market = a.market or cfg["data"].get("market", "kospi200")
    if cfg["data"].get("market", "kospi200") != market:
        raise ValueError("Config market does not match --market")
    cfg["data"]["market"] = market
    if a.data:
        cfg["data"]["path"] = str(a.data)
    data_path = Path(cfg["data"]["path"])
    cfg["data"]["path"] = str(data_path if data_path.is_absolute() else ROOT / data_path)
    if a.cmd == "check-data":
        data_mod.print_check(data_mod.check(cfg["data"]["path"], cfg["data"].get("sha256"), market=market))
        return
    cfg = apply_overrides(copy.deepcopy(cfg), a.set)
    if a.start:
        cfg["periods"]["eval"][0] = a.start
    if a.end:
        cfg["periods"]["eval"][1] = a.end
    run_id = a.run_id or time.strftime("%Y%m%d_%H%M%S") + f"_{cfg['periods']['eval'][0]}_{cfg['periods']['eval'][1]}"
    out = ROOT / cfg["output"]["results_dir"] / ("sp500" if market == "sp500" else "") / run_id
    if out.exists():
        raise FileExistsError(f"{out} exists; results are never overwritten")
    t0 = time.time()
    print("loading market", flush=True)
    mk = data_mod.load_market(cfg["data"]["path"], cfg["data"].get("sha256"), ROOT / cfg["data"]["cache_dir"], market=market)
    print("running pipeline", flush=True)
    res = pipeline(cfg, mk)
    save(res, cfg, mk, out, time.time() - t0)
    print(f"saved {out}")
    cols = ["start", "end", "cagr", "volatility", "sharpe", "cer", "max_drawdown", "annual_turnover",
            "cost_total", "mean_cash", "hold", "transfer"]
    s = pd.read_csv(out / "summary.csv", index_col=0)
    print(s[[c for c in cols if c in s]].to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
