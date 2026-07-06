#!/usr/bin/env python3
"""Analyze drawdowns and test simple regime filters for the return strategy."""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from app.services.market_store import ParquetMarketStore
from app.services.research_backtest import run_scored_backtest
from app.services.strategies.v8_canonical import v8_candidate_filter
from scripts.run_multifactor_combination import build_panel, ranks, score_from_ranks


MARKET_DIR = "/Volumes/xhrrrrr_macmini副盘/quantlab/market"
WEIGHT_SETS = {
    "top_recent_return": {
        "correlation_breakdown": 0.0,
        "low_volatility_20": 0.2,
        "liquidity_strength_20": 0.4,
        "downside_volatility_20": 0.0,
        "breakout_position_60": 0.2,
        "distance_to_high_20": 0.0,
        "liquidity_acceleration_20": 0.2,
        "low_drawdown_momentum_60": 0.0,
    },
    "return_guarded": {
        "correlation_breakdown": 0.0,
        "low_volatility_20": 0.0,
        "liquidity_strength_20": 0.6,
        "downside_volatility_20": 0.2,
        "breakout_position_60": 0.2,
        "distance_to_high_20": 0.0,
        "liquidity_acceleration_20": 0.0,
        "low_drawdown_momentum_60": 0.0,
    },
    "sharpe_guarded": {
        "correlation_breakdown": 0.1,
        "low_volatility_20": 0.0,
        "liquidity_strength_20": 0.6,
        "downside_volatility_20": 0.3,
        "breakout_position_60": 0.0,
        "distance_to_high_20": 0.0,
        "liquidity_acceleration_20": 0.0,
        "low_drawdown_momentum_60": 0.0,
    },
}


def scorer(weights: dict[str, float]):
    def wrapped(group: pd.DataFrame) -> pd.Series:
        return score_from_ranks(ranks(group), weights)

    return wrapped


def regime_filter(name: str):
    def wrapped(group: pd.DataFrame) -> pd.DataFrame:
        base = v8_candidate_filter(group)
        if base.empty or name == "base":
            return base
        row = base.iloc[0]
        if name == "breadth45" and row.get("market_breadth20", 1) < 0.45:
            return base.iloc[0:0]
        if name == "median_ret20_positive" and row.get("market_median_ret20", 0) <= 0:
            return base.iloc[0:0]
        if name == "breadth45_or_median_ret20_positive" and not (
            row.get("market_breadth20", 1) >= 0.45 or row.get("market_median_ret20", 0) > 0
        ):
            return base.iloc[0:0]
        if name == "breadth45_and_median_ret20_gt_neg2" and not (
            row.get("market_breadth20", 1) >= 0.45 and row.get("market_median_ret20", 0) > -0.02
        ):
            return base.iloc[0:0]
        return base

    return wrapped


def drawdown_periods(equity_points: list[dict], limit: int = 5) -> list[dict]:
    equity = pd.DataFrame(equity_points)
    equity["date"] = pd.to_datetime(equity["date"])
    equity["peak"] = equity["equity"].cummax()
    equity["drawdown"] = equity["equity"] / equity["peak"] - 1
    troughs = equity.nsmallest(limit, "drawdown")
    return [
        {"date": str(row.date.date()), "equity": float(row.equity), "drawdown": round(float(row.drawdown), 6)}
        for row in troughs.itertuples(index=False)
    ]


def run_window(panel: pd.DataFrame, weights: dict[str, float], filter_name: str) -> dict:
    result = run_scored_backtest(
        panel,
        scorer(weights),
        top_n=5,
        initial_cash=100_000,
        market_filter=True,
        retention_multiple=3,
        universe_size=1000,
        candidate_filter=regime_filter(filter_name),
    )
    return {
        "filter": filter_name,
        "metrics": result["metrics"],
        "worst_drawdowns": drawdown_periods(result["equity"]),
    }


def main() -> None:
    store = ParquetMarketStore(MARKET_DIR)
    windows = {
        "2019-2023": build_panel(store, date(2019, 1, 1), date(2023, 12, 31)),
        "2024-2026": build_panel(store, date(2024, 1, 1), date(2026, 7, 5)),
    }
    filters = [
        "base",
        "breadth45",
        "median_ret20_positive",
        "breadth45_or_median_ret20_positive",
        "breadth45_and_median_ret20_gt_neg2",
    ]
    output = {
        "weight_sets": WEIGHT_SETS,
        "windows": {
            label: {
                weight_name: [run_window(panel, weights, filter_name) for filter_name in filters]
                for weight_name, weights in WEIGHT_SETS.items()
            }
            for label, panel in windows.items()
        },
    }
    path = ROOT / "reports" / "return_strategy_drawdown_analysis.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    for label, rows in output["windows"].items():
        print(label)
        for weight_name, weight_rows in rows.items():
            print(" ", weight_name)
            for row in weight_rows:
                metrics = row["metrics"]
                print("  ", row["filter"], metrics["total_return"], metrics["sharpe"], metrics["max_drawdown"])


if __name__ == "__main__":
    main()
