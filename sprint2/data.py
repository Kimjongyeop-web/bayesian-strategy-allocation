"""Market input: read the original workbook once (read-only), cache by SHA-256, check the schema.

The workbook is never modified. Only 수정주가 / 거래량 / KOSPI200편입여부 enter the model; the log-return
sheet is read by `check` only to confirm units. QC sheets are not used as signal inputs.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import warnings
import zipfile
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook

from .contracts import Market

PRICE, VOLUME, MEMBER, LOGRET = "수정주가", "거래량", "KOSPI200편입여부", "일간로그수익률"
EXPECTED_SHEETS = ["파일안내", PRICE, VOLUME, LOGRET, MEMBER, "기업별편입이력", "편입편출변경",
                   "QC기업별요약", "QC이벤트상세", "QC일별현황"]


def sha256(path: Path) -> str:
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _read_matrix(wb, sheet: str):
    print(f"Reading sheet: {sheet}", flush=True)
    rows = wb[sheet].iter_rows(values_only=True)
    header = next(rows)
    if header[0] != "날짜" or not all(isinstance(c, str) and c for c in header[1:]):
        raise ValueError(f"{sheet}: expected '날짜' + text ticker codes in row 1")
    codes = list(header[1:])
    if len(set(codes)) != len(codes):
        raise ValueError(f"{sheet}: duplicate ticker codes")
    dates, values = [], []
    for row in rows:
        if row[0] is None:
            if any(v is not None for v in row[1:]):
                raise ValueError(f"{sheet}: observations without a date")
            continue
        dates.append(row[0])
        values.append([np.nan if v is None else float(v) for v in row[1:]])
    dates = pd.DatetimeIndex(pd.to_datetime(dates, errors="raise"), name="date")
    if not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError(f"{sheet}: dates must be unique and increasing")
    return dates, codes, np.asarray(values, dtype=float)


def load_market(path, expected_sha256: str | None = None, cache_dir="cache", *, market="kospi200") -> Market:
    """Load prices/volume/membership. Cached as npz keyed by the workbook hash (cache is Git-ignored)."""
    spec = market_spec(market)
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Workbook not found: {path} (set data.path in the config or pass --data)")
    digest = sha256(path)
    if expected_sha256 and digest != expected_sha256:
        raise ValueError(f"SHA-256 mismatch for {path.name}: {digest} != {expected_sha256}")
    cache = Path(cache_dir) / market / (digest + "_loader3_price_unfiltered") / "market.npz"
    execution_mask = None
    cached = False
    if cache.is_file():
        try:
            with np.load(cache, allow_pickle=False) as z:
                dates, codes = pd.DatetimeIndex(z["dates"], name="date"), list(z["codes"])
                price, volume, member = z["price"], z["volume"], z["member"]
                execution_mask = z["execution_mask"] if spec.get("tradable") else None
            if (len(dates) == 0 or dates.hasnans or not dates.is_unique
                    or not dates.is_monotonic_increasing or len(set(codes)) != len(codes)
                    or not all(isinstance(c, str) and c for c in codes)
                    or not np.isin(member, [0, 1]).all()):
                raise ValueError("Invalid cache dates/codes/membership")
            Market(dates, pd.Index(codes), price, volume, member, {}, execution_mask)
            cached = True
            print(f"[{market}] loaded validated cache", flush=True)
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile, zlib.error) as exc:
            warnings.warn(f"[{market}] invalid cache; rebuilding from source: {exc}")
    if not cached:
        print(f"[{market}] reading original workbook (first load may take several minutes)", flush=True)
        wb = load_workbook(path, read_only=True, data_only=True)
        try:
            dates, codes, price = _read_matrix(wb, PRICE)
            d2, c2, volume = _read_matrix(wb, VOLUME)
            d3, c3, member = _read_matrix(wb, spec["membership"])
            if spec.get("tradable"):
                d4, c4, execution_mask = _read_matrix(wb, spec["tradable"])
                if not dates.equals(d4) or codes != c4 or not np.isin(execution_mask, [0, 1]).all():
                    raise ValueError("Trading mask must be aligned and binary")
        finally:
            wb.close()
        if not (dates.equals(d2) and dates.equals(d3) and codes == c2 == c3):
            raise ValueError("Price/volume/membership matrices are not aligned")
        if not np.isin(member, [0, 1]).all():
            raise ValueError("Membership must be 0/1 with no blanks")
        member = member.astype(bool)
        cache.parent.mkdir(parents=True, exist_ok=True)
        _write_cache(cache, dates=dates.values.astype("datetime64[ns]"), codes=np.array(codes, dtype=str),
                     price=price, volume=volume, member=member,
                     **({"execution_mask": execution_mask} if execution_mask is not None else {}))
    source = {"path": str(path.resolve()), "sha256": digest, "rows": len(dates), "tickers": len(codes),
              "first_date": str(dates[0].date()), "last_date": str(dates[-1].date())}
    source.update(market=market, currency=spec["currency"], return_basis="adjusted_price_excluding_dividends",
                  analysis_filter="none", loader_version=3,
                  missing_price_cells=int((~np.isfinite(price)).sum()),
                  execution_blocked_cells=int((execution_mask == 0).sum()) if execution_mask is not None else 0)
    return Market(dates, pd.Index(codes, dtype=str, name="ticker"), price, volume, member, source, execution_mask)


def check(path, expected_sha256: str | None = None, *, market="kospi200") -> dict:
    """Schema/unit report read directly from the workbook (slow: reads four matrices)."""
    spec = market_spec(market)
    member_sheet = spec["membership"]
    path = Path(path)
    digest = sha256(path)
    if expected_sha256 and digest != expected_sha256:
        raise ValueError("SHA-256 mismatch")
    wb = load_workbook(path, read_only=True, data_only=True)
    try:
        sheets = wb.sheetnames
        dims = {s: [wb[s].max_row, wb[s].max_column] for s in (PRICE, VOLUME, LOGRET, member_sheet)}
        dates, codes, price = _read_matrix(wb, PRICE)
        d_lr, c_lr, logret = _read_matrix(wb, LOGRET)
        d_m, c_m, member = _read_matrix(wb, member_sheet)
        d_v, c_v, volume = _read_matrix(wb, VOLUME)
    finally:
        wb.close()
    if not (dates.equals(d_lr) and dates.equals(d_m) and dates.equals(d_v) and codes == c_lr == c_m == c_v):
        raise ValueError("Matrix dates/codes are not aligned")
    if not np.isin(member, [0, 1]).all():
        raise ValueError("Membership must be binary")
    valid = np.isfinite(price) & (price > 0)
    both = valid[1:] & valid[:-1]
    own = np.log(price[1:] / price[:-1])
    diff = np.abs(own[both] - logret[1:][both])
    diff = diff[np.isfinite(diff)]
    report = {
        "file": path.name, "sha256": digest, "sha256_matches": expected_sha256 in (None, digest),
        "sheets": sheets, "sheets_as_expected": sheets == EXPECTED_SHEETS if market == "kospi200" else all(s in sheets for s in (PRICE, VOLUME, LOGRET, member_sheet, spec["tradable"])), "matrix_dims": dims,
        "dates": [str(dates[0].date()), str(dates[-1].date())], "trading_days": len(dates),
        "dates_unique_sorted": bool(dates.is_unique and dates.is_monotonic_increasing),
        "tickers": len(codes), "tickers_unique": len(set(codes)) == len(codes),
        "same_ticker_order_all_matrices": codes == c_lr == c_m == c_v,
        "ticker_lengths": sorted({len(c) for c in codes}),
        "tickers_with_leading_zero": sum(c.startswith("0") for c in codes),
        "membership_values": sorted(np.unique(member[np.isfinite(member)]).tolist()),
        "members_per_day_min_max": [int(np.nansum(member, 1).min()), int(np.nansum(member, 1).max())],
        "price_cells_missing": int((~np.isfinite(price)).sum()),
        "price_cells_nonpositive": int((np.isfinite(price) & (price <= 0)).sum()),
        "member_days_without_price": int(((member == 1) & ~valid).sum()),
        "zero_volume_member_days": int(((member == 1) & (volume == 0)).sum()),
        "logret_sheet_vs_log_price_diff_max": float(diff.max()) if diff.size else None,
        "logret_sheet_vs_log_price_cells_compared": int(diff.size),
        "logret_sheet_cells_without_both_prices": int((np.isfinite(logret[1:]) & ~both).sum()),
        "return_unit": "decimal (0.01 = 1%)" if diff.size and diff.max() < 1e-6 else "CHECK",
    }
    mk = load_market(path, expected_sha256, market=market)
    report.update(market=market, currency=spec["currency"], source=mk.source,
                  tradable_cells=int(mk.tradable.sum()))
    return report


def print_check(report: dict) -> None:
    print(json.dumps(report, ensure_ascii=False, indent=2))


def market_spec(market):
    specs = {"kospi200": {"membership": "KOSPI200편입여부", "currency": "KRW"},
             "sp500": {"membership": "SP500편입여부", "currency": "USD", "tradable": "거래가능여부"}}
    if market not in specs:
        raise ValueError(f"Unknown market: {market}")
    return specs[market]


def _write_cache(cache, **arrays):
    """Publish only a fully written NPZ; interruption never exposes a partial cache."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=cache.parent, suffix=".npz.tmp", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, cache)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
