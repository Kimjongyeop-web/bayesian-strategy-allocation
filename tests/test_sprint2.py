import copy

import numpy as np
import pandas as pd
import pytest

from sprint2.contracts import Market, SelectionRule
from sprint2.ledger import Account, Plan, _fill, execute, hold_targets, make_plan, split_fills
from sprint2.part1_momentum import high2y
from sprint2.part2_lowvol import lowvol
from sprint2.part3_b0_basic import LocalLevelFilter
from sprint2.portfolio import Costs, hold_existing_weights, internal_target, select
from sprint2.run import pipeline, report, save

from conftest import toy_config, toy_market


def perturbed(mk: Market, k: int, seed=1) -> Market:
    """Same market up to row k, different afterwards (prices, volume, membership)."""
    rng = np.random.default_rng(seed)
    price, volume, member = mk.price.copy(), mk.volume.copy(), mk.member.copy()
    price[k + 1:] *= np.exp(np.cumsum(rng.normal(0, 0.05, price[k + 1:].shape), 0))
    volume[k + 1:] = rng.uniform(0, 1e5, volume[k + 1:].shape)
    member[k + 1:, :10] = ~member[k + 1:, :10]
    return Market(mk.dates, mk.codes, price, volume, member, mk.source)


def same(a, b):
    return np.array_equal(np.asarray(a, float), np.asarray(b, float), equal_nan=True)


# --- data / signals ------------------------------------------------------------------------------------
def test_returns_use_adjacent_valid_prices_only():
    dates = pd.bdate_range("2020-01-01", periods=4)
    p = np.array([[100.0], [np.nan], [110.0], [121.0]])
    mk = Market(dates, pd.Index(["000001"]), p, np.ones_like(p), np.ones_like(p, bool))
    assert np.isnan(mk.ret[1, 0]) and np.isnan(mk.ret[2, 0])          # no return across the gap, no ffill
    assert mk.ret[3, 0] == pytest.approx(0.1)
    assert mk.mark[1, 0] == 100.0 and not mk.tradable[1, 0]            # valuation mark != executable price


@pytest.mark.parametrize("fn,params", [(high2y, {"window": 120, "min_periods": 100}), (lowvol, {"window": 20})])
def test_signals_do_not_look_ahead(market, fn, params):
    k = 350
    a, b = fn(market, **params), fn(perturbed(market, k), **params)
    assert same(a[:k + 1], b[:k + 1]) and not same(a, b)


def test_selection_and_hold_existing_weights():
    rule = SelectionRule(top=0.2, keep=0.4, min_valid=5)
    score = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0], float)
    member = np.ones(10, bool)
    held = np.zeros(10, bool)
    held[[3, 9]] = True                                       # 3 is inside top 40%, 9 is not
    assert select(score, member, held, rule).tolist() == [0, 1, 3]
    assert select(np.r_[score[:4], [np.nan] * 6], member, held, rule) is None
    tie = np.ones(10)
    assert select(tie, member, np.zeros(10, bool), rule).tolist() == [0, 1]   # ties: fixed ticker order
    u = np.zeros(10)
    u[3] = 0.2                                                # kept name keeps its weight
    W = hold_existing_weights(np.array([0, 1, 3]), u)
    assert W[3] == 0.2 and W[0] == W[1] == pytest.approx(0.4)
    u[3] = 0.9                                                # not enough left: one slot each, shrink kept
    W = hold_existing_weights(np.array([0, 1, 3]), u)
    assert W[0] == pytest.approx(1 / 3) and W[3] == pytest.approx(1 / 3) and W.sum() == pytest.approx(1)


def test_non_reselect_day_keeps_drifted_weights_and_exits_nonmembers():
    u = np.array([0.3, 0.2, 0.1, 0.0])
    member = np.array([True, True, False, True])
    W = internal_target(SelectionRule(min_valid=1), np.arange(4.0), member, u, reselect=False)
    assert W.tolist() == [0.3, 0.2, 0.0, 0.0]


def test_locked_holding_is_not_buying_budget():
    rule = SelectionRule(top=0.2, keep=0.4, min_valid=5)
    score = np.r_[np.arange(9.0, 0, -1), np.nan]               # ticker 9: no price -> no score
    member = np.r_[np.ones(9, bool), False]
    u = np.zeros(10)
    u[9] = 0.1                                                  # delisted, cannot be sold
    lk = np.zeros(10, bool)
    lk[9] = True
    W = internal_target(rule, score, member, u, reselect=True, locked=lk)
    assert W[9] == 0.1 and W.sum() == pytest.approx(1.0)        # new names share only the 0.9 that exists
    W = internal_target(rule, score, member, u, reselect=False, locked=lk)
    assert W[9] == 0.1                                          # no futile exit order while there is no price
    W = internal_target(rule, score, member, u, reselect=True)  # old behaviour: budget counted 1.0
    assert W.sum() == pytest.approx(1.0) and W[9] == 0.0


def test_transfer_keeps_locked_holdings_in_place():
    from types import SimpleNamespace
    from sprint2.ledger import transfer_targets, locked_share
    mk = _toy3()
    price = mk.price.copy()
    price[1, 2] = np.nan                                        # ticker 2 has no price at the decision (t=1)
    mk = Market(mk.dates, mk.codes, price, mk.volume, mk.member)
    acc = Account("A", ["M", "L"], 3, 10_000.0)
    acc.q[0, 2] = 10.0                                          # 1,000 at its last mark (100)
    acc.c0 = 9_000.0
    tw = np.zeros((4, 3))
    tw[1] = [0.5, 0.0, 0.5]                                     # virtual target includes the unpriced name
    books = {"M": SimpleNamespace(target_w=tw), "L": SimpleNamespace(target_w=tw)}
    assert locked_share(acc, 1, mk) == pytest.approx(0.1)
    T, cash = transfer_targets(acc, 1, mk, books, np.array([0.6, 0.3]))
    assert T[0, 2] == 10.0 and T[1, 2] == 0.0                   # untouched, no buy of an unpriced name
    # the unpriced name's 0.5 goes pro rata to the tradable names: ticker 0 gets the whole 1.0
    assert T[0, 0] == pytest.approx(0.6 * 10_000 / 100) and T[1, 0] == pytest.approx(0.3 * 10_000 / 100)


# --- Part 3 --------------------------------------------------------------------------------------------
def test_kalman_q0_matches_analytic_posterior_with_missing():
    rng = np.random.default_rng(3)
    V = np.array([[2e-4, 5e-5], [5e-5, 1e-4]])
    f = LocalLevelFilter(V, q=0.0, beta=1.0, m0=0.001, prior_obs=10)  # beta=1: V fixed
    P0inv = np.linalg.inv(V / 10)
    info, vec = P0inv.copy(), P0inv @ np.full(2, 0.001)
    for t in range(300):
        y = rng.multivariate_normal([0.0005, -0.0002], V)
        if t % 7 == 0:
            y[1] = np.nan
        if t % 11 == 0:
            y[:] = np.nan
        f.update(y)
        o = np.isfinite(y)
        if o.any():
            Hs = np.eye(2)[o]
            Vi = np.linalg.inv(V[np.ix_(o, o)])
            info += Hs.T @ Vi @ Hs
            vec += Hs.T @ Vi @ y[o]
    P = np.linalg.inv(info)
    np.testing.assert_allclose(f.P, P, rtol=1e-9)
    np.testing.assert_allclose(f.m, P @ vec, rtol=1e-8, atol=1e-12)


def test_predictive_lag_and_horizon_sum():
    V = np.array([[2e-4, 6e-5], [6e-5, 1.5e-4]])
    f = LocalLevelFilter(V, q=0.05, beta=0.9, prior_obs=4)
    f.update(np.array([0.01, -0.005]))
    P, W, Vn = f.P, f.W, f.V
    _, B1, _, _ = f.predict(1)
    _, B5, _, _ = f.predict(5)
    np.testing.assert_allclose(B1, P + 2 * W + Vn)
    np.testing.assert_allclose(B5, 25 * P + 80 * W + 5 * Vn)
    # Monte Carlo of G = y_{t+2} + ... + y_{t+H+1} with mu_t ~ N(m, P), fills at t+1
    rng = np.random.default_rng(0)
    n, H = 400_000, 5
    mu = rng.multivariate_normal(f.m, P, n)
    G = np.zeros((n, 2))
    for h in range(1, H + 2):
        mu = mu + rng.multivariate_normal([0, 0], W, n)
        if h >= 2:
            G += mu + rng.multivariate_normal([0, 0], Vn, n)
    a, B, mean, cov = f.predict(H)
    np.testing.assert_allclose(G.mean(0), a, atol=4 * np.sqrt(np.diag(B) / n).max())
    np.testing.assert_allclose(np.cov(G.T), B, rtol=0.02)
    R = np.expm1(G)                                            # log -> simple H-day return
    np.testing.assert_allclose(R.mean(0), mean, atol=4 * np.sqrt(np.diag(cov) / n).max())
    np.testing.assert_allclose(np.cov(R.T), cov, rtol=0.03)


# --- ledger --------------------------------------------------------------------------------------------
def _toy3(T=4):
    dates = pd.bdate_range("2021-01-04", periods=T)
    price = np.full((T, 3), 100.0)
    return Market(dates, pd.Index(["000001", "000002", "000003"]), price, np.full((T, 3), 1e5),
                  np.ones((T, 3), bool))


EX = {"buy_bps": 10.0, "sell_bps": 10.0, "sale_tax_schedule": [["2000-01-01", 0.002]]}


def test_internal_transfer_is_netted_and_costed_once():
    mk = _toy3()
    costs = Costs(mk.dates, EX)
    acc = Account("A", ["M", "L"], 3, 10_000.0)
    acc.q[0, 0] = 10.0                                        # sleeve M owns 10 of ticker 0
    acc.c0 = 10_000.0 - 1_000.0
    target = acc.q.copy()
    target[0, 0], target[1, 0] = 0.0, 4.0                     # M sells 10, L buys 4 -> market sells 6
    target[1, 1] = 5.0                                        # L buys 5 of ticker 1
    acc.pending = make_plan(acc, 0, mk, target, 0.0, "TRANSFER", costs, cash_target=np.array([0.0, 0.0]))
    nav0 = acc.nav(mk.mark[1])
    execute(acc, 1, mk, costs, {})
    expected_cost = 6 * 100 * (0.001 + 0.002) + 5 * 100 * 0.001
    row = acc.rows[-1]
    assert row[4] * nav0 == pytest.approx(expected_cost)      # cost column = cost / NAV_prev
    assert acc.nav(mk.mark[1]) == pytest.approx(nav0 - expected_cost)
    assert acc.q[0, 0] == 0 and acc.q[1, 0] == pytest.approx(4) and acc.q[1, 1] == pytest.approx(5)
    assert row[11] == pytest.approx(400.0)                    # internal transfer notional, no cost
    sells = [o for o in acc.orders if o[4] == "SELL"]
    assert len(sells) == 1 and sells[0][6] == pytest.approx(-6)


def test_split_fills_partial():
    delta = np.array([[-10.0], [4.0]])
    dq, share, matched = split_fills(delta, np.array([-3.0]))          # only 3 of the 6 net sold
    assert dq[:, 0].tolist() == pytest.approx([-7.0, 4.0]) and matched[0] == 4
    dq, _, _ = split_fills(np.array([[2.0], [6.0]]), np.array([4.0]))  # net buy 8, half filled
    assert dq.sum() == pytest.approx(4.0) and dq[0, 0] == pytest.approx(1.0)


def test_gap_unfilled_min_order_nonnegative():
    mk = _toy3(3)
    price = mk.price.copy()
    price[1, 0] *= 1.3                                         # 30% gap up on the fill day
    price[1, 2] = np.nan                                       # no price on fill day
    vol = mk.volume.copy()
    vol[1, 1] = 0.0                                            # no volume -> not tradable
    mk = Market(mk.dates, mk.codes, price, vol, mk.member)
    costs = Costs(mk.dates, EX)
    acc = Account("A", ["M"], 3, 10_000.0, sleeve_cash=[10_000.0])
    acc.q[0, 2] = 1.0
    acc.c[0] -= 100.0
    acc.c0 = 0.0
    target = np.array([[99.0, 0.01, 0.0]])                     # all cash into ticker 0; tiny order; exit 2
    acc.pending = make_plan(acc, 0, mk, target, 0.0025, "INTERNAL", costs)
    assert acc.pending.order[1] == 0.0                          # below min order -> skipped
    assert acc.pending.order[2] == -1.0                         # full exit sent despite size
    execute(acc, 1, mk, costs, {})
    assert acc.cash >= -1e-9 and acc.q.min() >= 0
    assert acc.q[0, 0] < 99.0                                   # buys scaled to cash after the gap
    assert acc.q[0, 2] == 1.0                                   # no price -> unfilled, kept at last mark
    status = {o[3]: o[10] for o in acc.orders}
    assert status["000003"] == "UNFILLED" and status["000001"] == "PARTIAL"
    assert acc.nav(np.nan_to_num(mk.mark[1])) == pytest.approx(acc.rows[-1][1])


def test_small_trims_are_sent_when_they_fund_buys():
    mk = _toy3()
    costs = Costs(mk.dates, EX)
    acc = Account("A", ["M"], 3, 10_000.0, sleeve_cash=[0.0])
    acc.q[0, :2] = 50.0                                        # fully invested, no cash
    acc.c0 = 0.0
    target = np.array([[49.8, 49.8, 0.39]])                    # trims of 20 (< 25 = 0.25% NAV) pay for 39
    plan = make_plan(acc, 0, mk, target, 0.0025, "INTERNAL", costs)
    assert plan.order.tolist() == pytest.approx([-0.2, -0.2, 0.39])
    acc.c0 = 1_000.0                                           # with spare cash the trims are skipped again
    plan = make_plan(acc, 0, mk, target, 0.0025, "INTERNAL", costs)
    assert plan.order.tolist() == pytest.approx([0.0, 0.0, 0.39])


def test_stale_exit_only_when_configured(market, cfg):
    acc = Account("A", ["M"], len(market.codes), 1000.0, sleeve_cash=[900.0])
    acc.c0 = 0.0
    acc.q[0, 5] = 1.0
    costs = Costs(market.dates, cfg["execution"])
    execute(acc, 320, market, costs, {})
    assert acc.q[0, 5] == 1.0                                   # default: no cash invented
    execute(acc, 321, market, costs, {"after_days": 10, "recovery_rate": 0.5})
    assert acc.q[0, 5] == 0.0 and acc.rows[-1][5] > 0           # write-off recorded


# --- whole pipeline on toy data --------------------------------------------------------------------------
def test_pipeline_ledgers_and_hold_semantics(result, market):
    for name, p in result["policies"].items():
        d = p["daily"]
        assert np.isfinite(d["net_return"]).all() and (d["cash"] >= -1e-6).all()
    reselect = __import__("sprint2.portfolio", fromlist=["x"]).month_end_mask(market.dates)
    allocators = [n for n, p in result["policies"].items() if p["account"] is not None]
    assert {"EW_monthly", "Cash_timing", "B0", "B1", "B2"} <= set(allocators)
    for name in allocators:
        p = result["policies"][name]
        acc, dec = p["account"], p["decisions"]
        assert acc.max_identity_error < 1e-9
        orders = pd.DataFrame(acc.orders, columns=["execution_date", "decision_date", "account", "ticker", "side",
                                                   "order_qty", "filled_qty", "price", "notional", "cost", "status"])
        hold_days = dec.index[(dec.action == "HOLD") & ~reselect[market.dates.get_indexer(dec.index)]]
        # every TRANSFER sends at least one market order (no relabel-only transfers)
        transfer_days = dec.index[dec.action == "TRANSFER"]
        assert set(transfer_days) <= set(orders.decision_date)
        o = orders[orders.decision_date.isin(hold_days)]
        # HOLD on a normal day: no buys, no restoring of drift; only exits of names that left the index
        assert (o.side == "SELL").all()
        t = market.dates.get_indexer(o.decision_date)
        j = market.codes.get_indexer(o.ticker)
        assert (~market.member[t, j]).all()
    assert (result["policies"]["EW_monthly"]["decisions"].action == "TRANSFER").sum() >= 5
    gross = result["books"]["M"].daily
    nav_prev = gross["nav"].shift(1)
    g = (gross["nav"] + gross["cost"] * nav_prev) / nav_prev - 1
    np.testing.assert_allclose(g.iloc[1:], gross["gross_return"].iloc[1:], rtol=1e-10)


def test_pipeline_does_not_look_ahead(market, cfg, result):
    k = 400
    other = pipeline(cfg, perturbed(market, k))
    cut = market.dates[k]
    for s in result["books"]:
        a, b = result["books"][s], other["books"][s]
        assert same(a.target_w[:k + 1], b.target_w[:k + 1])
        pd.testing.assert_frame_equal(a.daily.loc[:cut], b.daily.loc[:cut])
    for name in [n for n, p in result["policies"].items() if p["account"] is not None]:
        da, db = result["policies"][name]["decisions"].loc[:cut], other["policies"][name]["decisions"].loc[:cut]
        pd.testing.assert_frame_equal(da, db)
    for name in result["forecasts"]:
        pd.testing.assert_frame_equal(result["forecasts"][name].loc[:cut], other["forecasts"][name].loc[:cut])
    for name in ("B1", "B2"):
        assert not result["forecasts"][name].equals(other["forecasts"][name])


def test_config_overrides_existing_keys_only():
    from sprint2.run import apply_overrides, load_config
    cfg = apply_overrides(load_config(), ["execution.stale_exit.after_days=1",
                                          "execution.stale_exit.recovery_rate=1.0"])
    assert cfg["execution"]["stale_exit"] == {"after_days": 1, "recovery_rate": 1.0}
    with pytest.raises(KeyError):
        apply_overrides(load_config(), ["execution.stale_exit.recovry_rate=1.0"])


def test_swap_signals_without_touching_part3_part4(tmp_path):
    mk = toy_market(seed=5)
    cfg = toy_config(mk)
    cfg["strategies"] = [
        {"id": "REV", "signal": "toy_signals:short_reversal", "params": {"window": 5}},
        {"id": "RND", "signal": "toy_signals:random_score", "params": {"seed": 1}},
        {"id": "LV", "signal": "sprint2.part2_lowvol:lowvol", "params": {"window": 20}},
    ]
    cfg["policies"] = [{"id": "REV_only", "type": "virtual", "strategy": "REV"},
                       {"id": "EQ", "type": "fixed_monthly", "weights": {"REV": 0.3, "RND": 0.3, "LV": 0.4}},
                       {"id": "Bayes", "type": "bayes", "model": "b0", "q": 1e-4}]
    res = pipeline(cfg, mk)
    assert res["ids"] == ("REV", "RND", "LV")
    save(res, cfg, mk, tmp_path / "run", 0.0)
    summary = report(tmp_path / "run")
    assert set(summary.index) == {"REV_only", "EQ", "Bayes"}
    assert (tmp_path / "run" / "figures" / "allocation.png").stat().st_size > 0
    with pytest.raises(FileExistsError):
        save(res, cfg, mk, tmp_path / "run", 0.0)


# --- B1 spread model and significance -------------------------------------------------------------------------
def test_b1_static_spread_matches_conjugate_posterior():
    from sprint2.part3_b1_spread import SpreadFilter
    rng = np.random.default_rng(7)
    V = np.array([[2e-4, 1.2e-4], [1.2e-4, 1.5e-4]])
    sigma = 3e-4
    f = SpreadFilter(V, phi=1.0, sigma=sigma, beta=1.0)          # phi=1 -> W=0; beta=1 -> V fixed
    v = V[0, 0] + V[1, 1] - 2 * V[0, 1]
    prec, num = 1 / sigma ** 2, 0.0
    for t in range(400):
        y = rng.multivariate_normal([4e-4, 1e-4], V)
        if t % 9 == 0:
            y[0] = np.nan                                         # spread unobserved -> prediction step only
        f.update(y)
        if np.isfinite(y).all():
            prec += 1 / v
            num += (y[0] - y[1]) / v
    assert f.P == pytest.approx(1 / prec, rel=1e-9) and f.m == pytest.approx(num / prec, rel=1e-8)


def test_b1_spread_predictive_ar1_monte_carlo():
    from sprint2.part3_b1_spread import SpreadFilter
    V = np.array([[2e-4, 1.2e-4], [1.2e-4, 1.5e-4]])
    f = SpreadFilter(V, phi=0.9, sigma=1e-3, beta=1.0)
    f.update(np.array([0.01, -0.004]))
    H, n = 4, 400_000
    a_d, s, a, B, mean, cov = f.predict(H)
    rng = np.random.default_rng(1)
    delta = rng.normal(f.m, np.sqrt(f.P), n)
    v = V[0, 0] + V[1, 1] - 2 * V[0, 1]
    G = np.zeros(n)
    for h in range(1, H + 2):
        delta = f.phi * delta + rng.normal(0, np.sqrt(f.W), n)
        if h >= 2:
            G += delta + rng.normal(0, np.sqrt(v), n)
    assert G.mean() == pytest.approx(a_d, abs=4 * G.std() / np.sqrt(n))
    assert G.var() == pytest.approx(B[0, 0] + B[1, 1] - 2 * B[0, 1], rel=0.02)
    assert a[0] - a[1] == pytest.approx(a_d)


def test_significance_and_spread_eval():
    from sprint2.part4_evaluation import significance, spread_forecast_eval
    idx = pd.bdate_range("2020-01-01", periods=400)
    r = pd.Series(np.random.default_rng(0).normal(3e-4, 0.01, 400), index=idx)
    daily = {"A": pd.DataFrame({"net_return": r}), "B": pd.DataFrame({"net_return": r})}
    sig = significance(daily, ["A"], ["B"], {"full": ["2019-12-31", "2021-12-31"]}, 5.0, 252, 1,
                       {"block": 21, "draws": 200, "seed": 0})
    assert (sig[["diff", "ci95_low", "ci95_high"]].abs().to_numpy() < 1e-12).all()
    rng = np.random.default_rng(1)
    fc = pd.DataFrame({"y_M": rng.normal(0, 0.01, 300), "y_L": rng.normal(0, 0.01, 300)}, index=idx[:300])
    H = 5
    d = fc.y_M - fc.y_L
    fc["a_M"], fc["a_L"] = sum(d.shift(-h) for h in range(2, H + 2)), 0.0   # a perfect spread forecast
    fc["B_M"], fc["B_L"], fc["B_M_L"] = 1e-4, 1e-4, 0.0
    ev = spread_forecast_eval(fc, ["M", "L"], H)
    assert ev["r2_vs_zero"] == pytest.approx(1.0) and ev["sign_hit"] == 1.0 and ev["dm_stat"] < 0


# ---- B1 level model and cash -------------------------------------------------------------------------------

def test_b1_level_keeps_the_spread_and_sets_the_common_level():
    from sprint2.contracts import Forecast
    from sprint2.part3_b1_level import add_level
    V = np.array([[4e-4, 1e-4], [1e-4, 2e-4]])
    a = np.array([0.003, -0.003])
    B = 21 * V + 1e-4 * np.outer([0.5, -0.5], [0.5, -0.5])
    f = Forecast(pd.Timestamp("2020-01-31"), ("M", "L"), 21, np.zeros(1), np.zeros((1, 1)), V, a, B,
                 np.zeros(2), np.eye(2), True, "")
    out, _ = add_level({5: f}, {4: (0.01, 0.002)}, ("M", "L"))
    g = out[5]
    assert abs((g.a[0] + g.a[1]) / 2 - 0.01) < 1e-12 and abs((g.a[0] - g.a[1]) - 0.006) < 1e-12
    assert abs(g.B[0, 0] + g.B[1, 1] - 2 * g.B[0, 1] - (B[0, 0] + B[1, 1] - 2 * B[0, 1])) < 1e-12
    assert abs((g.B.sum() / 4) - 0.002) < 1e-12                       # Var(c)
    z, _ = add_level({5: f}, {4: (0.01, 0.002)}, ("M", "L"), zero_spread=True)
    assert abs(z[5].a[0] - z[5].a[1]) < 1e-12


def test_b1_and_cash_timing_can_hold_cash_and_never_exceed_full_investment(result):
    for name in ("B0", "B1", "Cash_timing"):
        d = result["policies"][name]["daily"].iloc[2:]
        assert (d["exposure"] <= 1 + 1e-9).all()
        dec = result["policies"][name]["decisions"]
        assert set(dec.reason) <= {"no_gain", "gain>1e-10", "no_market_trade", "forecast_invalid", "no_forecast"}
    t = result["forecasts"]["Cash_timing"]
    assert (t["spread_mean"].dropna() == 0).all()


# ---- B2: per-strategy Bayesian 2-regime model ----------------------------------------------------------------

def test_b2_gibbs_recovers_regimes_and_orders_them():
    from sprint2.part3_b2_regime import gibbs, predictive
    rng = np.random.default_rng(0)
    T = 900
    S = np.zeros(T, int)
    for t in range(1, T):
        S[t] = S[t - 1] if rng.random() < (0.97 if S[t - 1] == 0 else 0.9) else 1 - S[t - 1]
    yv = np.where(S == 0, rng.normal(0.004, 0.02, T), rng.normal(-0.006, 0.05, T))
    cfg = dict(sigma_mu_annual=0.2, var_shape=3.0, p_stay_mean=0.96, p_stay_strength=20.0)
    k, _ = gibbs(yv, cfg, 200, 400, rng)
    vol = np.sqrt(k["s2"].mean(0))
    assert abs(vol[0] - 0.02) < 0.003 and abs(vol[1] - 0.05) < 0.008       # calm = lower variance
    assert abs(k["P"].mean(0)[0] - 0.97) < 0.02
    assert k["mu"].mean(0)[0] > k["mu"].mean(0)[1]
    d = predictive(k, 4, 2, rng)
    assert len(d) == 800 and np.isfinite(d).all()


def test_b2_filter_flags_turbulence():
    from sprint2.part3_b2_regime import gibbs
    rng = np.random.default_rng(1)
    calm = rng.normal(0.002, 0.015, 300)
    cfg = dict(sigma_mu_annual=0.2, var_shape=3.0, p_stay_mean=0.96, p_stay_strength=20.0)
    k1, _ = gibbs(np.r_[calm, rng.normal(0.0, 0.06, 15)], cfg, 200, 300, rng)
    k0, _ = gibbs(np.r_[calm, rng.normal(0.0, 0.06, 15), rng.normal(0.002, 0.015, 40)], cfg, 200, 300, rng)
    assert k1["q1"].mean() > 0.8 > 0.2 > k0["q1"].mean()


def test_b2_policy_reviews_monthly_with_cash(result, market):
    from sprint2.portfolio import month_end_mask
    me = set(market.dates[month_end_mask(market.dates)])
    dec = result["policies"]["B2"]["decisions"]
    assert all(d in me or d == dec.index[0] for d in dec[dec.action == "TRANSFER"].index)
    assert (result["policies"]["B2"]["daily"]["exposure"] <= 1 + 1e-9).all()
    tab = result["forecasts"]["B2"]
    p = tab.filter(like="p_turbulent_").to_numpy()
    assert tab["valid"].all() and ((p >= 0) & (p <= 1)).all()
