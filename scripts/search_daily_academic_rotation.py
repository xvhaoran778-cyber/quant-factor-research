#!/usr/bin/env python3
"""Daily next-open rotation for academic OHLCV factors."""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from app.services.market_store import ParquetMarketStore  # noqa: E402
from scripts.search_academic_ohlcv_factors import COMBOS, MARKET_DIR, WINDOWS, factor_panel, rank_score  # noqa: E402


OUT = ROOT / "reports" / "daily_academic_rotation_search.json"


def metrics(returns: pd.Series) -> dict:
    returns = returns.fillna(0)
    equity = (1 + returns).cumprod()
    total = float(equity.iloc[-1] - 1) if len(equity) else 0.0
    annual = float((1 + total) ** (252 / max(len(returns), 1)) - 1) if total > -1 else -1.0
    drawdown = equity / equity.cummax() - 1
    return {
        "total_return": round(total, 6),
        "annual_return": round(annual, 6),
        "sharpe": round(float(returns.mean() / returns.std() * np.sqrt(252)), 4) if returns.std() else 0,
        "max_drawdown": round(float(drawdown.min()), 6),
        "positive_days": round(float((returns > 0).mean()), 4),
    }


def build_daily_panel(store: ParquetMarketStore, start: date, end: date) -> pd.DataFrame:
    warmup = start - timedelta(days=420)
    daily = store.read(warmup, end, symbols=None, fill_suspensions=False)
    daily["date"] = pd.to_datetime(daily["date"])
    daily = store.filter_point_in_time_universe(daily).sort_values(["symbol", "date"])
    daily["ret20"] = daily.groupby("symbol")["adj_close"].pct_change(20)
    daily["vol20"] = daily.groupby("symbol")["adj_close"].pct_change().rolling(20).std().reset_index(level=0, drop=True) * np.sqrt(252)
    daily["liquidity"] = daily.groupby("symbol")["amount"].rolling(20).mean().reset_index(level=0, drop=True)
    daily["listed_days"] = daily.groupby("symbol").cumcount() + 1
    daily["next_open"] = daily.groupby("symbol")["open"].shift(-1)
    daily["next2_open"] = daily.groupby("symbol")["open"].shift(-2)
    daily["future_return"] = daily["next2_open"] / daily["next_open"] - 1

    benchmark = store.benchmark(warmup, end).sort_values("date")
    benchmark["date"] = pd.to_datetime(benchmark["date"])
    benchmark["ma20"] = benchmark["close"].rolling(20).mean()
    benchmark["ma60"] = benchmark["close"].rolling(60).mean()
    benchmark["risk_on"] = ~((benchmark["close"] < benchmark["ma20"]) & (benchmark["ma20"] < benchmark["ma60"]))
    breadth = daily.groupby("date")["ret20"].agg(
        market_breadth20=lambda values: float((values > 0).mean()),
        market_median_ret20="median",
    )
    panel = daily.merge(factor_panel(store, start, end), on=["date", "symbol"], how="left")
    panel = panel.merge(benchmark[["date", "risk_on"]], on="date", how="left")
    panel = panel.merge(breadth, on="date", how="left")
    return panel[(panel["date"].dt.date >= start) & (panel["date"].dt.date <= end)]


def backtest(panel: pd.DataFrame, weights: dict[str, float], top_n: int, cost: float, breadth: float) -> dict:
    returns = []
    last = set()
    for _, group in panel.groupby("date", sort=True):
        group = group.dropna(subset=["next_open", "future_return", "ret20", "vol20", "liquidity"]).copy()
        group = group[
            (group["risk_on"].fillna(False))
            & (group["market_breadth20"] >= breadth)
            & (group["listed_days"] >= 120)
            & (group["close"] >= 3)
            & (group["liquidity"] > 0)
            & (group["ret20"] > -0.12)
        ]
        if group.empty:
            current = set()
            daily_return = 0.0
        else:
            liquid = group.nlargest(min(1200, len(group)), "liquidity")
            liquid = liquid[liquid["vol20"] <= liquid["vol20"].quantile(0.90)]
            chosen = liquid.assign(score=rank_score(liquid, weights)).nlargest(top_n, "score")
            current = set(chosen["symbol"])
            daily_return = float(chosen["future_return"].clip(-0.12, 0.12).mean()) if len(chosen) else 0.0
        turnover = len(current.symmetric_difference(last)) / max(top_n, 1)
        returns.append(daily_return - turnover * cost)
        last = current
    return metrics(pd.Series(returns))


def main() -> None:
    store = ParquetMarketStore(MARKET_DIR)
    panels = {
        label: build_daily_panel(store, start, end)
        for label, (start, end) in WINDOWS.items()
    }
    rows = []
    for weights in COMBOS:
        for top_n in (3, 5, 8):
            for breadth in (0.35, 0.45, 0.55):
                metrics_by_window = {
                    label: backtest(panel, weights, top_n, cost=0.0015, breadth=breadth)
                    for label, panel in panels.items()
                }
                annuals = [m["annual_return"] for m in metrics_by_window.values()]
                sharpes = [m["sharpe"] for m in metrics_by_window.values()]
                drawdowns = [abs(m["max_drawdown"]) for m in metrics_by_window.values()]
                row = {
                    "weights": weights,
                    "top_n": top_n,
                    "breadth": breadth,
                    "passes_20pct_all_windows": all(value >= 0.20 for value in annuals),
                    "score": min(annuals) + min(sharpes) + sum(annuals) / len(annuals) - max(drawdowns) * 0.25,
                    "metrics": metrics_by_window,
                }
                rows.append(row)
                print("tested", top_n, breadth, row["passes_20pct_all_windows"], flush=True)
    rows.sort(key=lambda row: row["score"], reverse=True)
    OUT.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "target": "annual_return >= 20% in every validation window",
                "execution": "daily signal, next-open to next-open, approximate 15bp turnover cost",
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
