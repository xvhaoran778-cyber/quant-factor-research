from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import requests


BAR_COLUMNS = ["date", "symbol", "open", "high", "low", "close", "adj_open", "adj_high", "adj_low", "adj_close", "volume", "amount", "suspended"]
BAR_COLUMNS_WITH_RESUMPTION = BAR_COLUMNS + ["is_resumption"]
TIMELINE_COLUMNS = ["symbol", "valid_from", "valid_to", "name", "is_st", "is_listed", "source"]
EXCLUDED_NAME_MARKERS = ("ST", "退市", "摘牌")


@dataclass(frozen=True)
class SyncResult:
    rows: int
    symbols: int
    failed: int
    checksum: str
    quality_report: dict


def normalize_symbol(code: str) -> str:
    code = str(code).strip().zfill(6)
    exchange = "BJ" if code.startswith(("4", "8", "92")) else "SH" if code.startswith(("5", "6", "9")) else "SZ"
    return f"{code}.{exchange}"


def is_excluded_security_name(name: str) -> bool:
    normalized = str(name or "").strip().upper().replace(" ", "")
    return any(marker in normalized for marker in EXCLUDED_NAME_MARKERS)


def is_a_share_symbol(symbol: str) -> bool:
    code, _, exchange = str(symbol).partition(".")
    if exchange == "SH":
        return code.startswith(("600", "601", "603", "605", "688"))
    if exchange == "SZ":
        return code.startswith(("000", "001", "002", "003", "300", "301"))
    return False  # Exclude .BJ (北交所) — Tencent data source does not cover it


def compress_security_status(
    frame: pd.DataFrame,
    symbol: str,
    name: str,
    valid_from: date,
    valid_to: date,
    source: str = "baostock",
) -> pd.DataFrame:
    """Compress daily ST observations into point-in-time eligibility intervals."""
    if frame.empty or valid_from > valid_to:
        return pd.DataFrame(columns=TIMELINE_COLUMNS)
    status = frame.copy()
    status["date"] = pd.to_datetime(status["date"], errors="coerce").dt.date
    st_column = "isST" if "isST" in status else "is_st"
    raw_st = status[st_column] if st_column in status else pd.Series(False, index=status.index)
    status["is_st"] = raw_st.astype(str).str.strip().str.lower().isin({"1", "true", "yes"})
    status = status.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")
    if status.empty:
        return pd.DataFrame(columns=TIMELINE_COLUMNS)
    status = status[(status["date"] >= valid_from) & (status["date"] <= valid_to)]
    if status.empty:
        return pd.DataFrame(columns=TIMELINE_COLUMNS)
    changes = status["is_st"].ne(status["is_st"].shift()).cumsum()
    groups = list(status.groupby(changes, sort=True))
    rows: list[dict] = []
    for index, (_, group) in enumerate(groups):
        interval_start = valid_from if index == 0 else group["date"].iloc[0]
        interval_end = valid_to if index == len(groups) - 1 else groups[index + 1][1]["date"].iloc[0] - timedelta(days=1)
        rows.append({
            "symbol": symbol,
            "valid_from": interval_start,
            "valid_to": interval_end,
            "name": name,
            "is_st": bool(group["is_st"].iloc[0]),
            "is_listed": True,
            "source": source,
        })
    return pd.DataFrame(rows, columns=TIMELINE_COLUMNS)


def _fetch_exchange_delistings() -> pd.DataFrame:
    """Fetch official exchange delisting dates through the AKShare adapters."""
    import akshare as ak

    frames: list[pd.DataFrame] = []
    for exchange, fetcher, code_column, name_column, list_column, delist_column in (
        ("SH", lambda: ak.stock_info_sh_delist(symbol="全部"), "公司代码", "公司简称", "上市日期", "暂停上市日期"),
        ("SZ", lambda: ak.stock_info_sz_delist(symbol="终止上市公司"), "证券代码", "证券简称", "上市日期", "终止上市日期"),
    ):
        try:
            raw = fetcher()
        except Exception:
            continue
        if raw.empty or code_column not in raw:
            continue
        normalized = pd.DataFrame({
            "symbol": raw[code_column].map(normalize_symbol),
            "name": raw.get(name_column, ""),
            "list_date": pd.to_datetime(raw.get(list_column), errors="coerce").dt.date,
            "delist_date": pd.to_datetime(raw.get(delist_column), errors="coerce").dt.date,
            "delist_source": f"{exchange.lower()}_exchange_via_akshare",
        })
        frames.append(normalized)
    return pd.concat(frames, ignore_index=True).drop_duplicates("symbol", keep="last") if frames else pd.DataFrame(
        columns=["symbol", "name", "list_date", "delist_date", "delist_source"]
    )


def sync_security_timeline(root: str | Path, start: date, end: date, interval: float = 0.02) -> dict:
    """Persist historical listing, delisting and ST intervals for the A-share universe."""
    import baostock as bs

    store = ParquetMarketStore(root)
    login = None
    for attempt in range(3):
        login = bs.login()
        if login.error_code == "0":
            break
        time.sleep(2)
    if login is None or login.error_code != "0":
        raise RuntimeError(f"BaoStock 登录失败（重试 3 次）: {login.error_msg if login else 'no response'}")
    failures: list[dict] = []
    try:
        query = bs.query_stock_basic()
        rows: list[list[str]] = []
        while query.error_code == "0" and query.next():
            rows.append(query.get_row_data())
        if query.error_code != "0":
            raise RuntimeError(f"证券主表查询失败: {query.error_msg}")
        master = pd.DataFrame(rows, columns=query.fields)
        if master.empty or "code" not in master:
            raise RuntimeError("证券主表为空")
        master = master.rename(columns={"code_name": "name", "ipoDate": "list_date", "outDate": "delist_date"})
        master["symbol"] = master["code"].astype(str).str.rsplit(".", n=1).str[-1].map(normalize_symbol)
        master["list_date"] = pd.to_datetime(master.get("list_date"), errors="coerce").dt.date
        master["delist_date"] = pd.to_datetime(master.get("delist_date"), errors="coerce").dt.date
        if "type" in master:
            master = master[master["type"].astype(str).eq("1")]
        master = master[master["symbol"].map(is_a_share_symbol)]

        delistings = _fetch_exchange_delistings()
        if not delistings.empty:
            master = master.merge(delistings, on="symbol", how="outer", suffixes=("", "_exchange"))
            master["name"] = master.get("name", pd.Series(index=master.index, dtype=str)).fillna(master.get("name_exchange"))
            master["list_date"] = master.get("list_date").fillna(master.get("list_date_exchange"))
            master["delist_date"] = master.get("delist_date_exchange").fillna(master.get("delist_date"))
        master = master[
            master["list_date"].notna()
            & (master["list_date"] <= end)
            & (master["delist_date"].isna() | (master["delist_date"] >= start))
        ].drop_duplicates("symbol", keep="last").sort_values("symbol")

        timeline_frames: list[pd.DataFrame] = []
        for row in master.itertuples(index=False):
            code, exchange = row.symbol.split(".")
            provider_code = f"{exchange.lower()}.{code}"
            interval_start = max(start, row.list_date)
            interval_end = min(end, row.delist_date) if pd.notna(row.delist_date) else end
            query = bs.query_history_k_data_plus(
                provider_code, "date,code,isST",
                start_date=interval_start.isoformat(), end_date=interval_end.isoformat(),
                frequency="d", adjustflag="3",
            )
            status_rows: list[list[str]] = []
            while query.error_code == "0" and query.next():
                status_rows.append(query.get_row_data())
            if query.error_code != "0" or not status_rows:
                failures.append({"symbol": row.symbol, "error": query.error_msg or "无历史状态记录"})
                continue
            status = pd.DataFrame(status_rows, columns=query.fields)
            timeline_frames.append(compress_security_status(status, row.symbol, str(row.name or row.symbol), interval_start, interval_end))
            if interval:
                time.sleep(interval)
    finally:
        bs.logout()

    timeline = pd.concat(timeline_frames, ignore_index=True) if timeline_frames else pd.DataFrame(columns=TIMELINE_COLUMNS)
    for column in ("valid_from", "valid_to"):
        if column in timeline:
            timeline[column] = pd.to_datetime(timeline[column]).dt.date
    master_output = master[["symbol", "name", "list_date", "delist_date"]].copy()
    master_tmp = store.security_master_path.with_suffix(".tmp.parquet")
    timeline_tmp = store.timeline_path.with_suffix(".tmp.parquet")
    master_output.to_parquet(master_tmp, index=False, compression="zstd")
    timeline.to_parquet(timeline_tmp, index=False, compression="zstd")
    master_tmp.replace(store.security_master_path)
    timeline_tmp.replace(store.timeline_path)
    metadata = {
        "status": "ready" if not failures and len(timeline) else "partial",
        "provider": "baostock_status_with_exchange_delistings",
        "point_in_time": True,
        "requested_start": str(start), "requested_end": str(end),
        "total_symbols": len(master_output), "completed_symbols": len(master_output) - len(failures),
        "failed_symbols": len(failures), "failure_samples": failures[:100],
        "intervals": len(timeline), "st_intervals": int(timeline.get("is_st", pd.Series(dtype=bool)).sum()),
        "delisted_symbols": int(master_output["delist_date"].notna().sum()),
    }
    store.write_timeline_metadata(metadata)
    return metadata


def clean_daily_bars(frame: pd.DataFrame, symbol: str) -> tuple[pd.DataFrame, dict]:
    source_rows = len(frame)
    if frame.empty:
        return pd.DataFrame(columns=BAR_COLUMNS), {"source_rows": 0, "clean_rows": 0, "dropped": 0, "duplicates": 0, "invalid_ohlc": 0}
    result = frame.rename(columns={"trade_date": "date", "vol": "volume"}).copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.date
    for column in ["open", "high", "low", "close", "volume", "amount"]:
        if column not in result:
            result[column] = np.nan if column != "amount" else 0.0
        result[column] = pd.to_numeric(result[column], errors="coerce")
    for column in ["adj_open", "adj_high", "adj_low", "adj_close"]:
        if column not in result:
            result[column] = result[column.removeprefix("adj_")]
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result["amount"].isna().all() or (result["amount"].fillna(0) == 0).all():
        result["amount"] = result["close"] * result["volume"]
    result["symbol"] = symbol
    result["suspended"] = result["volume"].fillna(0) <= 0
    duplicate_count = int(result.duplicated(["date", "symbol"], keep="last").sum())
    result = result.drop_duplicates(["date", "symbol"], keep="last")
    invalid_ohlc = (
        result[["open", "high", "low", "close"]].isna().any(axis=1)
        | (result[["open", "high", "low", "close"]] <= 0).any(axis=1)
        | (result["high"] < result[["open", "close", "low"]].max(axis=1))
        | (result["low"] > result[["open", "close", "high"]].min(axis=1))
        | (result["volume"] < 0)
        | (result["amount"] < 0)
        | result[["adj_open", "adj_high", "adj_low", "adj_close"]].isna().any(axis=1)
        | (result[["adj_open", "adj_high", "adj_low", "adj_close"]] <= 0).any(axis=1)
    )
    invalid_count = int(invalid_ohlc.sum())
    result = result.loc[~invalid_ohlc & result["date"].notna(), BAR_COLUMNS]
    result = result.sort_values(["date", "symbol"]).reset_index(drop=True)
    return result, {
        "source_rows": source_rows,
        "clean_rows": len(result),
        "dropped": source_rows - len(result),
        "duplicates": duplicate_count,
        "invalid_ohlc": invalid_count,
    }


class TencentDailyClient:
    """Tencent raw bars for execution plus positive hfq bars for return calculations."""

    raw_endpoint = "https://web.ifzq.gtimg.cn/appstock/app/kline/kline"
    adjusted_endpoint = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"

    def __init__(self, interval: float = 0.08):
        self.interval = interval
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0", "Referer": "https://gu.qq.com/"})

    def fetch(self, symbol: str, start: date, end: date, allow_unadjusted: bool = False) -> pd.DataFrame:
        code, exchange = symbol.split(".")
        prefix = "sh" if exchange == "SH" else "bj" if exchange == "BJ" else "sz"
        frames: list[pd.DataFrame] = []
        cursor = start
        while cursor <= end:
            chunk_end = min(cursor + timedelta(days=900), end)
            base_param = f"{prefix}{code},day,{cursor.isoformat()},{chunk_end.isoformat()},800"
            raw_response = self.session.get(self.raw_endpoint, params={"param": base_param}, timeout=15)
            adjusted_response = self.session.get(self.adjusted_endpoint, params={"param": f"{base_param},hfq"}, timeout=15)
            raw_response.raise_for_status()
            adjusted_response.raise_for_status()
            raw_rows = raw_response.json().get("data", {}).get(f"{prefix}{code}", {}).get("day") or []
            adjusted_rows = adjusted_response.json().get("data", {}).get(f"{prefix}{code}", {}).get("hfqday") or []
            adjusted_by_date = {row[0]: row for row in adjusted_rows if len(row) >= 6}
            if raw_rows:
                normalized_rows = []
                for row in raw_rows:
                    if len(row) < 6:
                        continue
                    adjusted = adjusted_by_date.get(row[0])
                    if not adjusted:
                        if not allow_unadjusted:
                            continue
                        adjusted = row
                    normalized_rows.append({
                        "date": row[0], "open": row[1], "close": row[2],
                        "high": row[3], "low": row[4], "volume": row[5],
                        "amount": row[6] if len(row) > 6 else None,
                        "adj_open": adjusted[1], "adj_close": adjusted[2],
                        "adj_high": adjusted[3], "adj_low": adjusted[4],
                    })
                if not normalized_rows:
                    cursor = chunk_end + timedelta(days=1)
                    time.sleep(self.interval)
                    continue
                frame = pd.DataFrame(normalized_rows)
                # Tencent daily K-line volume is reported in board lots (100 shares).
                frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce") * 100
                reported_amount = pd.to_numeric(frame["amount"], errors="coerce")
                frame["amount"] = reported_amount.where(
                    reported_amount.notna() & (reported_amount > 0),
                    pd.to_numeric(frame["close"], errors="coerce") * frame["volume"],
                )
                frames.append(frame)
            cursor = chunk_end + timedelta(days=1)
            time.sleep(self.interval)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_a_share_universe() -> pd.DataFrame:
    # CNINFO publishes the official stock-to-org mapping in one response and is
    # considerably safer for bulk universe discovery than quote-list scraping.
    try:
        response = requests.get(
            "https://www.cninfo.com.cn/new/data/szse_stock.json",
            headers={"User-Agent": "Mozilla/5.0", "Referer": "https://www.cninfo.com.cn/"},
            timeout=30,
        )
        response.raise_for_status()
        stock_list = response.json().get("stockList", [])
        frame = pd.DataFrame([
            {"symbol": normalize_symbol(row["code"]), "name": row.get("zwjc", "")}
            for row in stock_list
            if row.get("code") and row.get("category") == "A股"
        ])
        if len(frame) >= 5000:
            return frame.drop_duplicates("symbol").sort_values("symbol").reset_index(drop=True)
    except Exception:
        pass

    # Fallback adapted from the Eastmoney source used by a-stock-data. It is
    # deliberately serialized and throttled because this endpoint rate-limits.
    url = "https://82.push2.eastmoney.com/api/qt/clist/get"
    params = {
        "pn": 1, "pz": 100, "po": 1, "np": 1, "fltt": 2, "invt": 2,
        "fid": "f3", "fs": "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81+s:2048",
        "fields": "f12,f14",
    }
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"})
    rows: list[dict] = []
    page = 1
    while True:
        params["pn"] = page
        for attempt in range(4):
            try:
                response = session.get(url, params=params, timeout=30)
                response.raise_for_status()
                break
            except requests.RequestException:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        data = response.json().get("data") or {}
        batch = data.get("diff") or []
        rows.extend(batch)
        if not batch or len(rows) >= int(data.get("total") or 0):
            break
        page += 1
        time.sleep(1.1)
    frame = pd.DataFrame([{"symbol": normalize_symbol(row["f12"]), "name": row.get("f14", "")} for row in rows if row.get("f12")])
    return frame.drop_duplicates("symbol").sort_values("symbol").reset_index(drop=True)


def _fetch_industries_eastmoney(interval: float) -> tuple[pd.DataFrame, list[dict]]:
    import akshare as ak

    boards = ak.stock_board_industry_name_em()
    name_column = next((column for column in ("板块名称", "行业名称", "名称") if column in boards), None)
    if not name_column:
        raise RuntimeError("行业板块列表缺少名称字段")
    rows: list[dict] = []
    failures: list[dict] = []
    for industry in boards[name_column].dropna().astype(str).drop_duplicates():
        try:
            members = ak.stock_board_industry_cons_em(symbol=industry)
            code_column = next((column for column in ("代码", "股票代码", "证券代码") if column in members), None)
            if not code_column:
                raise RuntimeError("成份列表缺少股票代码")
            rows.extend({"symbol": normalize_symbol(code), "industry": industry} for code in members[code_column].dropna())
        except Exception as exc:
            failures.append({"industry": industry, "error": str(exc)[:160]})
        time.sleep(interval)
    snapshot = pd.DataFrame(rows).drop_duplicates("symbol") if rows else pd.DataFrame(columns=["symbol", "industry"])
    return snapshot, failures


def _fetch_industries_baostock() -> pd.DataFrame:
    import baostock as bs

    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"BaoStock 登录失败: {login.error_msg}")
    rows: list[list[str]] = []
    try:
        result = bs.query_stock_industry()
        while result.error_code == "0" and result.next():
            rows.append(result.get_row_data())
        if result.error_code != "0":
            raise RuntimeError(f"BaoStock 行业查询失败: {result.error_msg}")
        frame = pd.DataFrame(rows, columns=result.fields)
    finally:
        bs.logout()
    if frame.empty or not {"code", "industry"}.issubset(frame.columns):
        return pd.DataFrame(columns=["symbol", "industry"])
    frame = frame[frame["industry"].fillna("").str.strip().ne("")].copy()
    frame["symbol"] = frame["code"].str.rsplit(".", n=1).str[-1].map(normalize_symbol)
    return frame[["symbol", "industry"]].drop_duplicates("symbol")


def sync_industry_snapshot(root: str | Path, interval: float = 0.15) -> dict:
    """Persist a current industry snapshot with an automatic backup provider."""
    store = ParquetMarketStore(root)
    provider = "eastmoney_via_akshare"
    failures: list[dict] = []
    try:
        snapshot, failures = _fetch_industries_eastmoney(interval)
    except Exception as exc:
        failures = [{"industry": "全部", "error": str(exc)[:160]}]
        snapshot = pd.DataFrame(columns=["symbol", "industry"])
    if len(snapshot) < 3000:
        snapshot = _fetch_industries_baostock()
        provider = "baostock"
    if len(snapshot) < 3000:
        raise RuntimeError(f"行业映射覆盖不足，仅获得 {len(snapshot)} 只股票")
    snapshot_path = store.root / "industry_snapshot.parquet"
    temporary = snapshot_path.with_suffix(".tmp.parquet")
    snapshot.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(snapshot_path)
    metadata = {
        "provider": provider,
        "as_of": str(date.today()),
        "symbols": len(snapshot),
        "industries": int(snapshot["industry"].nunique()),
        "failed_industries": failures,
        "point_in_time": False,
        "warning": "当前行业快照并非历史时点分类，仅用于行业集中度约束。",
    }
    metadata_path = store.root / "industry_snapshot.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


class ParquetMarketStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.bars_dir = self.root / "daily"
        self.root.mkdir(parents=True, exist_ok=True)
        self.bars_dir.mkdir(parents=True, exist_ok=True)

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def timeline_path(self) -> Path:
        return self.root / "security_timeline.parquet"

    @property
    def timeline_metadata_path(self) -> Path:
        return self.root / "security_timeline.json"

    @property
    def security_master_path(self) -> Path:
        return self.root / "security_master.parquet"

    def write_manifest(self, payload: dict) -> None:
        temporary = self.manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        temporary.replace(self.manifest_path)

    def manifest(self) -> dict:
        return json.loads(self.manifest_path.read_text(encoding="utf-8")) if self.manifest_path.exists() else {"status": "empty"}

    def write_timeline_metadata(self, payload: dict) -> None:
        temporary = self.timeline_metadata_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        temporary.replace(self.timeline_metadata_path)

    def timeline_metadata(self) -> dict:
        if not self.timeline_metadata_path.exists():
            return {"status": "missing", "point_in_time": False}
        return json.loads(self.timeline_metadata_path.read_text(encoding="utf-8"))

    def timeline_ready(self) -> bool:
        return self.timeline_path.exists() and self.timeline_metadata().get("status") == "ready"

    def filter_point_in_time_universe(self, frame: pd.DataFrame, date_column: str = "date") -> pd.DataFrame:
        """Keep securities that were listed and non-ST on each historical date."""
        if frame.empty or not self.timeline_ready() or not {"symbol", date_column}.issubset(frame.columns):
            return frame
        timeline = pd.read_parquet(self.timeline_path)
        timeline["valid_from"] = pd.to_datetime(timeline["valid_from"], errors="coerce")
        timeline["valid_to"] = pd.to_datetime(timeline["valid_to"], errors="coerce")
        eligible = timeline[timeline["is_listed"].astype(bool) & ~timeline["is_st"].astype(bool)]
        dates = pd.to_datetime(frame[date_column], errors="coerce")
        mask = pd.Series(False, index=frame.index)
        for symbol, intervals in eligible.groupby("symbol", sort=False):
            symbol_mask = frame["symbol"].astype(str).eq(str(symbol))
            if not symbol_mask.any():
                continue
            symbol_dates = dates[symbol_mask]
            covered = pd.Series(False, index=symbol_dates.index)
            for row in intervals.itertuples(index=False):
                covered |= symbol_dates.between(row.valid_from, row.valid_to, inclusive="both")
            mask.loc[covered.index] = covered
        return frame.loc[mask].copy()

    def timeline_audit(self, start: date | None = None, end: date | None = None) -> dict:
        metadata = self.timeline_metadata()
        warnings: list[str] = []
        checks = {
            "schema_errors": 0, "invalid_intervals": 0, "overlapping_intervals": 0,
            "bar_symbols_without_timeline": 0, "universe_symbols_without_bars": 0,
            "delisted_symbols_without_bars": 0,
        }
        if not self.timeline_path.exists() or not self.security_master_path.exists():
            warnings.append("缺少历史退市/ST/摘牌时间线或证券主表")
            return {
                **metadata, "status": "missing", "point_in_time_universe": False,
                "survivorship_bias": True, "checks": checks, "warnings": warnings,
            }
        timeline = pd.read_parquet(self.timeline_path)
        master = pd.read_parquet(self.security_master_path)
        missing_columns = set(TIMELINE_COLUMNS) - set(timeline.columns)
        checks["schema_errors"] = len(missing_columns)
        if missing_columns:
            warnings.append(f"时间线缺少字段: {', '.join(sorted(missing_columns))}")
        else:
            timeline["valid_from"] = pd.to_datetime(timeline["valid_from"], errors="coerce").dt.date
            timeline["valid_to"] = pd.to_datetime(timeline["valid_to"], errors="coerce").dt.date
            checks["invalid_intervals"] = int((timeline["valid_from"].isna() | timeline["valid_to"].isna() | (timeline["valid_from"] > timeline["valid_to"])).sum())
            overlaps = 0
            for _, group in timeline.sort_values(["symbol", "valid_from"]).groupby("symbol"):
                previous_end = group["valid_to"].shift()
                overlaps += int((group["valid_from"] <= previous_end).fillna(False).sum())
            checks["overlapping_intervals"] = overlaps
            bar_symbols: set[str] = set()
            for path in self.bars_dir.glob("*.parquet"):
                if not start or not end:
                    bar_symbols.add(path.stem)
                    continue
                dates = pd.to_datetime(pd.read_parquet(path, columns=["date"])["date"], errors="coerce")
                if not dates.empty and dates.max() >= pd.Timestamp(start) and dates.min() <= pd.Timestamp(end):
                    bar_symbols.add(path.stem)
            timeline_symbols = set(timeline["symbol"].astype(str))
            checks["bar_symbols_without_timeline"] = len(bar_symbols - timeline_symbols)
            if start and end and {"symbol", "list_date", "delist_date"}.issubset(master.columns):
                list_dates = pd.to_datetime(master["list_date"], errors="coerce")
                delist_dates = pd.to_datetime(master["delist_date"], errors="coerce")
                relevant_mask = list_dates.notna() & (list_dates <= pd.Timestamp(end)) & (delist_dates.isna() | (delist_dates >= pd.Timestamp(start)))
                relevant = set(master.loc[relevant_mask, "symbol"].astype(str))
                relevant_delisted = set(master.loc[relevant_mask & delist_dates.notna() & (delist_dates <= pd.Timestamp(end)), "symbol"].astype(str))
                checks["universe_symbols_without_bars"] = len(relevant - bar_symbols)
                checks["delisted_symbols_without_bars"] = len(relevant_delisted - bar_symbols)
        requested_start = pd.to_datetime(metadata.get("requested_start"), errors="coerce")
        requested_end = pd.to_datetime(metadata.get("requested_end"), errors="coerce")
        coverage_ok = bool(
            start and end and pd.notna(requested_start) and pd.notna(requested_end)
            and requested_start.date() <= start and requested_end.date() >= end
        )
        if not coverage_ok:
            warnings.append("时间线未完整覆盖回测区间")
        if metadata.get("status") != "ready" or int(metadata.get("failed_symbols", 0)):
            warnings.append("部分证券缺少历史 ST 状态")
        if checks["delisted_symbols_without_bars"]:
            warnings.append("回测区间内存在已退市证券但缺少对应行情")
        elif checks["universe_symbols_without_bars"]:
            warnings.append("历史可交易股票名单中存在缺少行情的证券")
        passed = metadata.get("status") == "ready" and coverage_ok and not any(checks.values())
        return {
            "status": "passed" if passed else "failed",
            "point_in_time_universe": passed,
            "survivorship_bias": not passed,
            "coverage_start": metadata.get("requested_start"), "coverage_end": metadata.get("requested_end"),
            "symbols": int(metadata.get("total_symbols", 0)), "intervals": int(metadata.get("intervals", len(timeline))),
            "st_intervals": int(metadata.get("st_intervals", 0)), "delisted_symbols": int(metadata.get("delisted_symbols", 0)),
            "checks": checks, "warnings": warnings,
        }

    def verification(self, start: date, end: date) -> dict:
        audit = json.loads(self.audit_path.read_text(encoding="utf-8")) if self.audit_path.exists() else {}
        timeline = self.timeline_audit(start, end)
        market_ok = audit.get("status") in {"passed", "passed_with_quarantine"}
        manifest_ok = self.manifest().get("status") == "ready"
        verified = market_ok and manifest_ok and timeline.get("status") == "passed"
        if verified:
            status, label = "verified", "已验证"
            warnings: list[str] = []
        elif not market_ok or not manifest_ok:
            status, label = "data_quality_failed", "数据校验未通过"
            warnings = ["行情同步或结构审计未通过，结果不可视为已验证"]
        else:
            status, label = "survivorship_bias", "存在幸存者偏差"
            warnings = timeline.get("warnings", []) or ["缺少历史可交易股票名单"]
        return {
            "status": status, "label": label, "verified": verified,
            "survivorship_bias": status == "survivorship_bias",
            "timeline_status": timeline.get("status"), "warnings": warnings,
        }

    def path_for(self, symbol: str) -> Path:
        return self.bars_dir / f"{symbol}.parquet"

    def benchmark_path(self, symbol: str = "000001.SH") -> Path:
        directory = self.root / "benchmarks"
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{symbol}.parquet"

    def benchmark(self, start: date, end: date, symbol: str = "000001.SH") -> pd.DataFrame:
        """Read a cached benchmark. Read-only: does not fetch or write during backtests."""
        path = self.benchmark_path(symbol)
        cached = pd.read_parquet(path) if path.exists() else pd.DataFrame()
        if cached.empty:
            return pd.DataFrame(columns=BAR_COLUMNS)
        dates = pd.to_datetime(cached["date"]).dt.date
        return cached[(dates >= start) & (dates <= end)].sort_values("date").reset_index(drop=True)

    def save_symbol(self, symbol: str, bars: pd.DataFrame) -> None:
        path = self.path_for(symbol)
        if path.exists():
            bars = pd.concat([pd.read_parquet(path), bars], ignore_index=True)
            bars, _ = clean_daily_bars(bars, symbol)
        temporary = path.with_suffix(".tmp.parquet")
        bars.to_parquet(temporary, index=False, compression="zstd")
        temporary.replace(path)

    def latest_date(self, symbol: str) -> date | None:
        path = self.path_for(symbol)
        if not path.exists():
            return None
        dates = pd.read_parquet(path, columns=["date"])["date"]
        return pd.to_datetime(dates).dt.date.max() if not dates.empty else None

    def date_bounds(self, symbol: str) -> tuple[date | None, date | None]:
        path = self.path_for(symbol)
        if not path.exists():
            return None, None
        dates = pd.to_datetime(pd.read_parquet(path, columns=["date"])["date"], errors="coerce").dropna()
        if dates.empty:
            return None, None
        return dates.dt.date.min(), dates.dt.date.max()

    def trading_calendar(self, start: date, end: date) -> list[date]:
        """Extract trading days from benchmark index data."""
        bench_path = self.benchmark_path("000001.SH")
        if not bench_path.exists():
            return []
        bench = pd.read_parquet(bench_path)
        if bench.empty:
            return []
        dates = pd.to_datetime(bench["date"]).dt.date
        return sorted(d for d in dates if start <= d <= end)

    def read(self, start: date, end: date, symbols: list[str] | None = None, fill_suspensions: bool = True) -> pd.DataFrame:
        paths = [self.path_for(symbol) for symbol in symbols] if symbols else sorted(self.bars_dir.glob("*.parquet"))
        frames = []
        for path in paths:
            if not path.exists():
                continue
            frame = pd.read_parquet(path)
            dates = pd.to_datetime(frame["date"]).dt.date
            frame = frame[(dates >= start) & (dates <= end)].copy()
            if fill_suspensions and not frame.empty:
                frame = self._fill_suspension_rows(frame, start, end)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=BAR_COLUMNS_WITH_RESUMPTION)

    def _fill_suspension_rows(self, frame: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
        """Insert suspended placeholder rows for missing trading days and flag resumption days."""
        symbol = str(frame["symbol"].iloc[0])
        existing_dates = set(pd.to_datetime(frame["date"]).dt.date)
        calendar = self.trading_calendar(start, end)
        if not calendar:
            frame["is_resumption"] = False
            return frame
        missing = [d for d in calendar if d not in existing_dates]
        if not missing:
            frame["is_resumption"] = False
            return frame
        last_close = float(frame["close"].iloc[-1])
        last_adj = float(frame["adj_close"].iloc[-1]) if "adj_close" in frame.columns else last_close
        susp_rows = pd.DataFrame([{
            "date": d, "symbol": symbol,
            "open": last_close, "high": last_close, "low": last_close, "close": last_close,
            "adj_open": last_adj, "adj_high": last_adj, "adj_low": last_adj, "adj_close": last_adj,
            "volume": 0, "amount": 0, "suspended": True, "is_resumption": False,
        } for d in missing])
        frame["is_resumption"] = False
        combined = pd.concat([frame, susp_rows], ignore_index=True)
        combined = combined.sort_values("date").reset_index(drop=True)
        combined["prev_suspended"] = combined["suspended"].shift(1, fill_value=False)
        combined.loc[(~combined["suspended"]) & (combined["prev_suspended"]), "is_resumption"] = True
        combined = combined.drop(columns=["prev_suspended"])
        return combined

    @property
    def audit_path(self) -> Path:
        return self.root / "audit_report.json"

    def audit(self) -> dict:
        universe_path = self.root / "universe.parquet"
        universe = pd.read_parquet(universe_path) if universe_path.exists() else pd.DataFrame(columns=["symbol", "name"])
        paths = sorted(self.bars_dir.glob("*.parquet"))
        totals = {
            "files": len(paths), "rows": 0, "duplicate_rows": 0, "invalid_ohlc_rows": 0,
            "null_price_rows": 0, "symbol_mismatch_files": 0, "unsorted_files": 0,
            "extreme_return_rows": 0, "negative_volume_rows": 0,
            "extreme_return_after_first_5_rows": 0,
        }
        latest_dates: dict[str, int] = {}
        earliest: date | None = None
        latest: date | None = None
        samples: list[dict] = []
        quarantined_symbols: set[str] = set()
        for path in paths:
            frame = pd.read_parquet(path)
            if frame.empty:
                samples.append({"symbol": path.stem, "issue": "empty_file"})
                continue
            dates = pd.to_datetime(frame["date"], errors="coerce")
            prices = frame[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
            expected_symbol = path.stem
            mismatch = bool((frame["symbol"].astype(str) != expected_symbol).any())
            duplicates = int(frame.duplicated(["date", "symbol"]).sum())
            null_prices = int(prices.isna().any(axis=1).sum())
            invalid = int(((prices <= 0).any(axis=1) | (prices["high"] < prices[["open", "close", "low"]].max(axis=1)) | (prices["low"] > prices[["open", "close", "high"]].min(axis=1))).sum())
            negative_volume = int((pd.to_numeric(frame["volume"], errors="coerce") < 0).sum())
            adjusted_close = pd.to_numeric(frame.get("adj_close", frame["close"]), errors="coerce")
            return_series = adjusted_close.pct_change()
            extreme_mask = return_series.abs() > 0.35
            extreme_returns = int(extreme_mask.sum())
            mature_extreme_mask = extreme_mask & (np.arange(len(frame)) >= 5)
            mature_extreme_returns = int(mature_extreme_mask.sum())
            if mature_extreme_returns:
                quarantined_symbols.add(expected_symbol)
            unsorted = not dates.is_monotonic_increasing
            file_latest = dates.max().date()
            file_earliest = dates.min().date()
            latest_dates[str(file_latest)] = latest_dates.get(str(file_latest), 0) + 1
            earliest = file_earliest if earliest is None else min(earliest, file_earliest)
            latest = file_latest if latest is None else max(latest, file_latest)
            totals["rows"] += len(frame)
            totals["duplicate_rows"] += duplicates
            totals["invalid_ohlc_rows"] += invalid
            totals["null_price_rows"] += null_prices
            totals["symbol_mismatch_files"] += int(mismatch)
            totals["unsorted_files"] += int(unsorted)
            totals["extreme_return_rows"] += extreme_returns
            totals["extreme_return_after_first_5_rows"] += mature_extreme_returns
            totals["negative_volume_rows"] += negative_volume
            if len(samples) < 100 and any((mismatch, duplicates, null_prices, invalid, negative_volume, unsorted)):
                samples.append({"symbol": expected_symbol, "mismatch": mismatch, "duplicates": duplicates, "null_prices": null_prices, "invalid_ohlc": invalid, "negative_volume": negative_volume, "unsorted": unsorted})
            if len(samples) < 100 and mature_extreme_returns:
                for index in np.flatnonzero(mature_extreme_mask)[:3]:
                    samples.append({"symbol": expected_symbol, "issue": "extreme_adjusted_return", "date": str(dates.iloc[index].date()), "return": round(float(return_series.iloc[index]), 6)})
        file_symbols = {path.stem for path in paths}
        universe_symbols = set(universe.get("symbol", pd.Series(dtype=str)).astype(str))
        excluded = universe[universe.get("name", pd.Series(index=universe.index, dtype=str)).map(is_excluded_security_name)] if not universe.empty else universe
        structural_failures = any(totals[key] for key in ("duplicate_rows", "invalid_ohlc_rows", "null_price_rows", "symbol_mismatch_files", "unsorted_files", "negative_volume_rows"))
        universe_without_bars_count = len(universe_symbols - file_symbols)
        effective_universe = universe_symbols - set(excluded.get("symbol", pd.Series(dtype=str)).astype(str)) if not universe.empty else universe_symbols
        coverage_ratio = len(file_symbols & effective_universe) / max(len(effective_universe), 1)
        coverage_warning = coverage_ratio < 0.95 and len(effective_universe - file_symbols) > 0
        timeline = self.timeline_audit(earliest, latest) if earliest and latest else self.timeline_audit()
        if structural_failures:
            audit_status = "failed"
        elif quarantined_symbols and coverage_warning:
            audit_status = "passed_with_warnings"
        elif quarantined_symbols:
            audit_status = "passed_with_quarantine"
        elif coverage_warning:
            audit_status = "passed_with_warnings"
        else:
            audit_status = "passed"
        report = {
            "status": audit_status,
            "provider": self.manifest().get("provider"),
            "adjustment": "raw_execution_hfq_signal" if any("adj_close" in pd.read_parquet(path, columns=None).columns for path in paths[:1]) else "legacy_qfq",
            "earliest_date": str(earliest) if earliest else None, "latest_date": str(latest) if latest else None,
            "universe_symbols": len(universe_symbols), "symbols_with_bars": len(file_symbols),
            "universe_without_bars": universe_without_bars_count, "bars_outside_universe": len(file_symbols - universe_symbols),
            "coverage_ratio": round(coverage_ratio, 4),
            "excluded_name_symbols": len(excluded), "latest_date_distribution": dict(sorted(latest_dates.items(), reverse=True)[:10]),
            "point_in_time_fundamentals_available": (self.root / "fundamentals.parquet").exists(),
            "quarantined_symbols": len(quarantined_symbols), "quarantine": sorted(quarantined_symbols),
            "timeline": timeline,
            "point_in_time_universe": timeline.get("point_in_time_universe", False),
            "survivorship_bias": timeline.get("survivorship_bias", True),
            "checks": totals, "issue_samples": samples,
        }
        temporary = self.audit_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.audit_path)
        return report


def sync_market_store(root: str | Path, start: date, end: date, interval: float = 0.08, progress: Callable[[dict], None] | None = None) -> SyncResult:
    store = ParquetMarketStore(root)
    existing_paths = sorted(store.bars_dir.glob("*.parquet"))
    if existing_paths and "adj_close" not in pd.read_parquet(existing_paths[0]).columns:
        raise RuntimeError("检测到旧版前复权行情库；为避免混写，必须先重建 raw+hfq 数据目录")
    store.write_manifest({
        "status": "running", "phase": "security_timeline", "provider": "tencent_raw_hfq",
        "completed_symbols": 0, "total_symbols": 0, "start": start, "end": end,
    })
    try:
        timeline_metadata = sync_security_timeline(root, start, end, min(interval, 0.02))
    except Exception as exc:
        timeline_metadata = {
            "status": "failed", "point_in_time": False,
            "requested_start": str(start), "requested_end": str(end),
            "failed_symbols": 1, "error": str(exc)[:300],
        }
        store.write_timeline_metadata(timeline_metadata)
    universe = fetch_a_share_universe()
    if store.security_master_path.exists():
        historical = pd.read_parquet(store.security_master_path)
        historical = historical[["symbol", "name"]]
        universe = pd.concat([universe, historical], ignore_index=True)
        universe = universe.drop_duplicates("symbol", keep="first").sort_values("symbol").reset_index(drop=True)
    universe.to_parquet(store.root / "universe.parquet", index=False, compression="zstd")
    client = TencentDailyClient(interval)
    totals = {"source_rows": 0, "clean_rows": 0, "dropped": 0, "duplicates": 0, "invalid_ohlc": 0}
    failures: list[dict] = []
    completed = 0
    store.write_manifest({"status": "running", "provider": "tencent_raw_hfq", "total_symbols": len(universe), "completed_symbols": 0, "start": start, "end": end})
    for row in universe.itertuples(index=False):
        symbol = row.symbol
        earliest, latest = store.date_bounds(symbol)
        missing_ranges: list[tuple[date, date]] = []
        if earliest is None or latest is None:
            missing_ranges.append((start, end))
        else:
            if start < earliest:
                missing_ranges.append((start, earliest - timedelta(days=1)))
            if latest < end:
                missing_ranges.append((latest + timedelta(days=1), end))
        try:
            fetched = [client.fetch(symbol, range_start, range_end) for range_start, range_end in missing_ranges if range_start <= range_end]
            raw = pd.concat(fetched, ignore_index=True) if fetched else pd.DataFrame()
            bars, report = clean_daily_bars(raw, symbol)
            if not bars.empty:
                store.save_symbol(symbol, bars)
            for key in totals:
                totals[key] += report[key]
        except Exception as exc:
            failures.append({"symbol": symbol, "error": str(exc)[:200]})
        completed += 1
        if completed % 25 == 0 or completed == len(universe):
            state = {"status": "running", "provider": "tencent_raw_hfq", "total_symbols": len(universe), "completed_symbols": completed, "failed_symbols": len(failures), "start": start, "end": end, "quality": totals}
            store.write_manifest(state)
            if progress:
                progress(state)
    # Sync benchmark indices (000001.SH and 000905.SH) so backtests never need to fetch online
    for bench_symbol in ("000001.SH", "000905.SH"):
        try:
            bench_path = store.benchmark_path(bench_symbol)
            cached = pd.read_parquet(bench_path) if bench_path.exists() else pd.DataFrame()
            fetched = client.fetch(bench_symbol, start, end, allow_unadjusted=True)
            cleaned, _ = clean_daily_bars(fetched, bench_symbol)
            if not cleaned.empty:
                combined = pd.concat([cached, cleaned], ignore_index=True) if not cached.empty else cleaned
                combined, _ = clean_daily_bars(combined, bench_symbol)
                combined.to_parquet(bench_path, index=False, compression="zstd")
        except Exception:
            pass
    digest = hashlib.sha256()
    for path in sorted(store.bars_dir.glob("*.parquet")):
        digest.update(path.name.encode())
        digest.update(str(path.stat().st_size).encode())
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode())
    audit = store.audit()
    report = {
        **totals,
        "failed_symbols": failures[:200],
        "universe_size": len(universe),
        "adjustment": "raw_execution_hfq_signal",
        "source": "Tencent Finance; architecture inspired by simonlin1212/a-stock-data",
        "security_timeline": timeline_metadata,
        "survivorship_bias_warning": None if audit.get("point_in_time_universe") else "历史可交易股票时间线未通过校验，回测结果必须标记为存在幸存者偏差。",
        "audit": audit,
    }
    usable_audit = audit["status"] in {"passed", "passed_with_quarantine"}
    manifest_status = "ready" if usable_audit else "failed"
    if audit["status"] == "passed_with_warnings":
        manifest_status = "ready_with_warnings"
    final = {"status": manifest_status, "provider": "tencent_raw_hfq", "total_symbols": len(universe), "completed_symbols": completed, "failed_symbols": len(failures), "start": start, "end": end, "quality": report, "checksum": digest.hexdigest()}
    store.write_manifest(final)
    return SyncResult(totals["clean_rows"], len(universe) - len(failures), len(failures), digest.hexdigest(), report)
