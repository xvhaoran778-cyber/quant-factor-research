#!/usr/bin/env python3
"""Search academic OHLCV factors that do not need fundamentals."""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.services.market_store import ParquetMarketStore  # noqa: E402
from app.services.research_backtest import build_weekly_feature_panel, run_scored_backtest  # noqa: E402
from app.services.strategies.v8_canonical import v8_candidate_filter  # noqa: E402


MARKET_DIR = Path("/Volumes/xhrrrrr_macmini副盘/quantlab/market")
OUT = ROOT / "reports" / "academic_ohlcv_factor_search.json"

FACTOR_NAMES = (
    "near_high_252",
    "maxret20_low",
    "ivol60_low",
    "beta60_low",
    "amihud20",
    "overnight20",
    "intraday20",
    "tail20_safe",
    "skew60_low",
    "gapvol20_low",
)

WINDOWS = {
    "2012-2018": (date(2012, 1, 1), date(2018, 12, 31)),
    "2019-2023": (date(2019, 1, 1), date(2023, 12, 31)),
    "2024-2026": (date(2024, 1, 1), date(2026, 7, 5)),
}

COMBOS = [
    {"near_high_252": 0.40, "ivol60_low": 0.30, "maxret20_low": 0.30},
    {"near_high_252": 0.35, "tail20_safe": 0.25, "ivol60_low": 0.25, "gapvol20_low": 0.15},
    {"ivol60_low": 0.35, "maxret20_low": 0.30, "skew60_low": 0.20, "beta60_low": 0.15},
    {"amihud20": 0.35, "near_high_252": 0.25, "ivol60_low": 0.25, "tail20_safe": 0.15},
    {"overnight20": 0.35, "near_high_252": 0.25, "ivol60_low": 0.20, "gapvol20_low": 0.20},
    {"intraday20": 0.35, "near_high_252": 0.25, "maxret20_low": 0.20, "tail20_safe": 0.20},
    {"near_high_252": 0.30, "overnight20": 0.25, "intraday20": 0.20, "ivol60_low": 0.25},
    {"tail20_safe": 0.35, "gapvol20_low": 0.25, "beta60_low": 0.20, "ivol60_low": 0.20},
    {"near_high_252": 0.25, "maxret20_low": 0.25, "amihud20": 0.25, "overnight20": 0.25},
    {"ivol60_low": 0.30, "tail20_safe": 0.25, "intraday20": 0.25, "skew60_low": 0.20},
]


def factor_panel(store: ParquetMarketStore, start: date, end: date) -> pd.DataFrame:
    warmup = start - timedelta(days=420)
    daily = store.read(warmup, end, symbols=None, fill_suspensions=False)
    daily["date"] = pd.to_datetime(daily["date"])
    daily = store.filter_point_in_time_universe(daily).drop_duplicates(["date", "symbol"], keep="last")
    close = daily.pivot(index="date", columns="symbol", values="adj_close").sort_index()
    open_ = daily.pivot(index="date", columns="symbol", values="adj_open").reindex(close.index)
    amount = daily.pivot(index="date", columns="symbol", values="amount").reindex(close.index)
    ret = close.pct_change()

    benchmark = store.benchmark(warmup, end).sort_values("date")
    benchmark["date"] = pd.to_datetime(benchmark["date"])
    market_ret = benchmark.set_index("date")["close"].pct_change().reindex(close.index).fillna(0)
    market_var = market_ret.rolling(60).var()
    beta = ret.rolling(60).cov(market_ret).div(market_var, axis=0)
    residual = ret.sub(beta.mul(market_ret, axis=0))

    overnight = open_ / close.shift(1) - 1
    intraday = close / open_ - 1
    amihud = ret.abs() / amount.replace(0, np.nan)

    factors = {
        "near_high_252": close / close.rolling(252).max(),
        "maxret20_low": -ret.rolling(20).max(),
        "ivol60_low": -residual.rolling(60).std(),
        "beta60_low": -beta,
        "amihud20": amihud.rolling(20).mean(),
        "overnight20": overnight.rolling(20).mean(),
        "intraday20": intraday.rolling(20).mean(),
        "tail20_safe": ret.rolling(20).quantile(0.05),
        "skew60_low": -ret.rolling(60).skew(),
        "gapvol20_low": -overnight.rolling(20).std(),
    }
    stacked = []
    for name, values in factors.items():
        item = values.stack().rename(name).reset_index()
        item.columns = ["date", "symbol", name]
        stacked.append(item)
    out = stacked[0]
    for item in stacked[1:]:
        out = out.merge(item, on=["date", "symbol"], how="outer")
    return out[out["date"] >= pd.Timestamp(start)]


def build_panel(store: ParquetMarketStore, start: date, end: date) -> pd.DataFrame:
    base = build_weekly_feature_panel(store, start, end, prefer_materialized=True)
    return base.merge(factor_panel(store, start, end), on=["date", "symbol"], how="left")


def rank_score(group: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    score = pd.Series(0.0, index=group.index)
    for name, weight in weights.items():
        values = group[name].fillna(group[name].median())
        score += values.rank(pct=True, ascending=True, method="average").fillna(0.5) * weight
    return score


def candidate_filter(group: pd.DataFrame) -> pd.DataFrame:
    base = v8_candidate_filter(group)
    if base.empty:
        return base
    return base[
        (base["vol20"] <= base["vol20"].quantile(0.90))
        & (base["ret20"] > -0.12)
        & (base["close"] >= 3)
    ]


def run_one(panel: pd.DataFrame, weights: dict[str, float], top_n: int, retention: int) -> dict:
    return run_scored_backtest(
        panel,
        lambda group: rank_score(group, weights),
        top_n=top_n,
        initial_cash=100_000,
        market_filter=True,
        retention_multiple=retention,
        universe_size=1200,
        candidate_filter=candidate_filter,
    )["metrics"]


def main() -> None:
    store = ParquetMarketStore(MARKET_DIR)
    panels = {
        label: build_panel(store, start, end)
        for label, (start, end) in WINDOWS.items()
    }
    rows = []
    for weights in COMBOS:
        for top_n in (3, 5):
            for retention in (1, 3):
                metrics = {
                    label: run_one(panel, weights, top_n, retention)
                    for label, panel in panels.items()
                }
                annuals = [m["annual_return"] for m in metrics.values()]
                sharpes = [m["sharpe"] for m in metrics.values()]
                drawdowns = [abs(m["max_drawdown"]) for m in metrics.values()]
                row = {
                    "weights": weights,
                    "top_n": top_n,
                    "retention": retention,
                    "passes_20pct_all_windows": all(value >= 0.20 for value in annuals),
                    "score": min(annuals) + min(sharpes) + sum(annuals) / len(annuals) - max(drawdowns) * 0.25,
                    "metrics": metrics,
                }
                rows.append(row)
                print("tested", weights, top_n, retention, row["passes_20pct_all_windows"], flush=True)
    rows.sort(key=lambda row: row["score"], reverse=True)
    OUT.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "target": "annual_return >= 20% in every validation window",
                "factor_sources": [
                    "52-week high momentum",
                    "low idiosyncratic volatility / low lottery MAX",
                    "Amihud illiquidity",
                    "overnight/intraday return decomposition",
                ],
                "top": rows[:25],
                "passing": [row for row in rows if row["passes_20pct_all_windows"]],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    for row in rows[:10]:
        print(row)


if __name__ == "__main__":
    main()
