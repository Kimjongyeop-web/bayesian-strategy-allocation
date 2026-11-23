import copy
import numpy as np
import pandas as pd
import pytest
from openpyxl import Workbook
from sprint2.contracts import Market
from sprint2.data import load_market, sha256
from sprint2.portfolio import Costs
from sprint2.run import ROOT, load_config
from sprint2.ledger import Account, Plan, _fill


def workbook(path, member='SP500편입여부', mismatch=False):
    wb = Workbook(); wb.remove(wb.active)
    for name in ['수정주가', '거래량', member, '거래가능여부']:
        ws = wb.create_sheet(name); ws.append(['날짜', 'P001', 'P001_segment2'])
        for i, date in enumerate(pd.bdate_range('2020-01-01', periods=4)):
            ws.append([date.to_pydatetime(), *({'수정주가':[100+i,None if i==1 else 50],
                       '거래량':[0 if i==0 else 10,10],member:[1,int(i<2)],
                       '거래가능여부':[int(i!=2),1]}[name])])
        if mismatch and name == '거래가능여부': ws['B1']='wrong'
    wb.save(path)


def test_adapter_cache_slice_missing_and_mask(tmp_path):
    path=tmp_path/'us.xlsx'; workbook(path)
    mk=load_market(path,sha256(path),tmp_path/'cache',market='sp500')
    cached=load_market(path,sha256(path),tmp_path/'cache',market='sp500')
    assert mk.codes.tolist()==['P001','P001_segment2']
    assert mk.volume[0,0]==0 and np.isnan(mk.price[1,1])
    assert not mk.tradable[2,0] and mk.valid[2,0]
    assert not mk.member[2,1] and mk.member[1,1]
    assert np.array_equal(mk.tradable,cached.tradable)
    assert np.array_equal(mk.slice(2).execution_mask,mk.execution_mask[:3])
    assert np.isnan(mk.ret[2,1])
    with pytest.raises(ValueError, match='SHA'): load_market(path,'wrong',tmp_path)
    with pytest.raises(KeyError): load_market(path,cache_dir=tmp_path,market='kospi200')
    workbook(path,mismatch=True)
    with pytest.raises(ValueError,match='mask'):load_market(path,cache_dir=tmp_path,market='sp500')


def test_us_costs_no_korean_tax():
    us=load_config(ROOT/'config/sp500.yaml'); kr=load_config()
    dates=pd.bdate_range('2020-01-01',periods=3)
    a,b=Costs(dates,us['execution']),Costs(dates,kr['execution'])
    assert np.allclose(a.sell,0.00065) and np.all(b.sell>a.sell)
    assert us['strategies']==kr['strategies']
    assert us['model_b2']==kr['model_b2']


def test_no_market_or_internal_fill_when_blocked():
    acc=Account('x',['a','b'],1,1000,sleeve_cash=[0,500]);acc.q[0,0]=5
    plan=Plan(0,'TRANSFER',np.array([[-5.],[10.]]),np.array([5.]),None)
    before=acc.q.copy()
    result=_fill(acc,plan,np.array([100.]),np.array([False]),np.array([True]),np.zeros(1),np.zeros(1),block_internal=True)
    assert not result['filled'].any() and not result['internal']
    assert np.array_equal(before,acc.q)


def test_calendar_bounds_and_legacy_regression(market,cfg,result):
    from sprint2.run import pipeline
    bad=copy.deepcopy(cfg);bad['periods']['eval'][1]='2099-01-01'
    with pytest.raises(ValueError,match='outside'):pipeline(bad,market)
    # Explicitly supplying no execution mask keeps all legacy policy NAV paths unchanged.
    same=Market(market.dates,market.codes,market.price,market.volume,market.member,market.source,None)
    rerun=pipeline(cfg,same)
    for name in result['policies']:
        np.testing.assert_array_equal(result['policies'][name]['daily'].nav,rerun['policies'][name]['daily'].nav)


def test_comparison_saved_files_different_calendars(tmp_path):
    import json
    from sprint2.markets import compare
    for name,dates in [('kospi200',['2020-01-02','2020-01-03']),('sp500',['2020-01-03','2020-01-06'])]:
        out=tmp_path/name;(out/'policies').mkdir(parents=True)
        (out/'manifest.json').write_text(json.dumps({'data':{'currency':'USD' if name=='sp500' else 'KRW'},
            'return_basis':'adjusted_price_excluding_dividends','rows':{'eval_actually_run':dates}}))
        (out/'config.json').write_text(json.dumps({'policies':[{'id':'B0'}]}))
        pd.DataFrame([{'policy':'B0','start':dates[0],'end':dates[1],'period':'full'}]).set_index('policy').to_csv(out/'summary_by_period.csv')
        pd.DataFrame({'nav':[100,110]},index=pd.to_datetime(dates)).to_csv(out/'policies/B0_daily.csv')
        pd.DataFrame({'action':['HOLD','TRANSFER']},index=pd.to_datetime(dates)).to_csv(out/'policies/B0_decisions.csv')
    (tmp_path/'comparison.json').write_text(json.dumps({'kospi200':'kospi200','sp500':'sp500'}))
    table=compare(tmp_path)
    assert len(table)==2 and table.hold.eq(1).all() and table.transfer.eq(1).all()
    assert (tmp_path/'comparison.png').is_file()


@pytest.mark.parametrize("damage", ["truncated", "missing", "shape"])
def test_corrupt_cache_recovers_from_original(tmp_path, damage):
    path = tmp_path / "us.xlsx"
    workbook(path)
    baseline = load_market(path, cache_dir=tmp_path / "cache", market="sp500")
    cache = next((tmp_path / "cache").rglob("market.npz"))
    if damage == "truncated":
        cache.write_bytes(b"PK incomplete")
    elif damage == "missing":
        np.savez(cache, dates=baseline.dates.values)
    else:
        np.savez(cache, dates=baseline.dates.values, codes=baseline.codes.values.astype(str),
                 price=np.zeros((1, 1)), volume=baseline.volume, member=baseline.member,
                 execution_mask=baseline.execution_mask)
    with pytest.warns(UserWarning, match="rebuilding"):
        recovered = load_market(path, cache_dir=tmp_path / "cache", market="sp500")
    np.testing.assert_array_equal(recovered.price, baseline.price)
    np.testing.assert_array_equal(recovered.execution_mask, baseline.execution_mask)
    with np.load(cache, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["price"], baseline.price)


def test_interrupted_cache_write_preserves_previous_file(tmp_path, monkeypatch):
    from sprint2.data import _write_cache
    cache = tmp_path / "market.npz"
    _write_cache(cache, price=np.array([1.0]))
    original = cache.read_bytes()
    def interrupted(stream, **arrays):
        stream.write(b"incomplete")
        raise KeyboardInterrupt()
    monkeypatch.setattr(np, "savez_compressed", interrupted)
    with pytest.raises(KeyboardInterrupt):
        _write_cache(cache, price=np.array([2.0]))
    assert cache.read_bytes() == original
    assert list(tmp_path.iterdir()) == [cache]


def test_repeated_run_names_preserve_existing_results(tmp_path):
    from sprint2.markets import _reserve_folder
    first = _reserve_folder(tmp_path / "dual_full")
    (first / "evidence.txt").write_text("preserve")
    second = _reserve_folder(tmp_path / "dual_full")
    third = _reserve_folder(tmp_path / "dual_full")
    assert second.name == "dual_full_2" and third.name == "dual_full_3"
    assert (first / "evidence.txt").read_text() == "preserve"


def test_all_new_holdings_roundoff_does_not_divide_empty_weights():
    from sprint2.portfolio import hold_existing_weights
    with np.errstate(divide="raise", invalid="raise"):
        weights = hold_existing_weights(np.arange(11), np.zeros(11), budget=0.1)
    np.testing.assert_array_equal(weights, np.full(11, 0.1 / 11))
    assert weights.sum() == pytest.approx(0.1)
