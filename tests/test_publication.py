from pathlib import Path
import copy
import pandas as pd
import numpy as np
from sprint2.run import save, report, load_config, ROOT
from sprint2.publication import pack, read_daily, read_decisions


def test_configs_share_model_and_independent_costs():
    a, b = load_config(), load_config(ROOT / "config/sp500.yaml")
    assert a["strategies"] == b["strategies"] and a["model_b2"] == b["model_b2"]
    assert a["execution"]["sale_tax_schedule"] != b["execution"]["sale_tax_schedule"]
    assert a["data"]["market"] == "kospi200" and b["data"]["market"] == "sp500"


def test_published_report_without_workbook(tmp_path, result, cfg, market):
    source, target = tmp_path / "run", tmp_path / "published"
    save(result, cfg, market, source, 0)
    pack(source, target)
    expected = pd.read_csv(source / "summary.csv", index_col=0)
    actual = report(target)
    pd.testing.assert_frame_equal(expected, actual, check_dtype=False, atol=1e-12, rtol=1e-12)
    for policy, values in result["policies"].items():
        np.testing.assert_allclose(read_daily(target, policy).nav, values["daily"].nav, rtol=1e-14)


def test_parent_report_regenerates_both_compact_markets(tmp_path, result, cfg, market):
    import json
    from sprint2.contracts import Market
    from sprint2.run import main
    target = tmp_path / "final"
    target.mkdir()
    for name, currency in [("kospi200", "KRW"), ("sp500", "USD")]:
        mk = Market(market.dates, market.codes, market.price, market.volume, market.member, {"currency": currency})
        source = tmp_path / name
        save(result, cfg, mk, source, 0)
        pack(source, target / name)
        (target / name / "summary.csv").unlink()
    (target / "comparison.json").write_text(json.dumps({m: m for m in ["kospi200", "sp500"]}))
    main(["report", str(target)])
    for name in ["kospi200", "sp500"]:
        assert (target / name / "summary.csv").exists()
    assert set(pd.read_csv(target / "comparison.csv").market) == {"kospi200", "sp500"}
