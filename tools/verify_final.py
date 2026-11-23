"""Verify final selection, complete two-market evidence and local document links."""
from pathlib import Path
import re,json,sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def main():
    c=pd.read_csv(ROOT/"docs/evidence/momentum/final_candidates.csv",index_col="strategy")
    chosen=c[c.in_mcs90 & ~c.index.str.contains("~") & (c.annual_turnover<=3) & c.rebalance_basis.isin(["monthly","offset_avg"])].sort_values("sharpe",ascending=False)
    assert chosen.index[0]=="high52w504|top20|keep2x|m|drift"
    table=pd.read_csv(ROOT/"reports/final/comparison.csv")
    assert len(table)==54 and set(table.market)=={"kospi200","sp500"}
    assert set(table.policy)=={"M_only","L_only","EW_monthly","Cash_timing","B0","B1","B2","B2_1","B2_2"}
    for market in ["kospi200","sp500"]:
        folder=ROOT/"reports/final"/market
        for name in ["daily.csv.gz","decisions.csv.gz","allocation.csv.gz","forecasts.csv.gz","schema.json","manifest.json","config.json"]:
            assert (folder/name).is_file(),name
        manifest=json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
        assert manifest["status"]=="completed"
        assert set(json.loads((folder/"schema.json").read_text(encoding="utf-8"))["daily"])==set(table.policy)
        assert manifest["selected_extensions"]["manifest"]["git"]["code_dirty"] is False
    for file in [ROOT/"README.md", ROOT/"reports/final/README.md", *ROOT.glob("docs/**/*.md")]:
        text=file.read_text(encoding="utf-8")
        assert not re.search(r"github.com/minjunxjang/bayesian-strategy-allocation/(?:tree|blob)/(?:minjun|hsh|kjy|e547|ed775)",text),file
        for link in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)",file.read_text(encoding="utf-8")):
            if "://" not in link and not link.startswith("#"):
                assert (file.parent/link.split("#")[0]).exists(),(file,link)
    print("PASS: selection, two markets, 54 rows, compact inputs and document links")
if __name__=="__main__":main()
