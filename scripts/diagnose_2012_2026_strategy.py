#!/usr/bin/env python3
"""Diagnose and minimally optimize the 2012-2026 loss of the recent-return strategy."""

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
WEIGHTS = {
    "correlation_breakdown": 0.0,
    "low_volatility_20": 0.2,
    "liquidity_strength_20": 0.0,
    "downside_volatility_20": 0.0,
    "breakout_position_60": 0.0,
    "distance_to_high_20": 0.0,
    "liquidity_acceleration_20": 0.0,
    "low_drawdown_momentum_60": 0.0,
    "market_relative_strength_60": 0.6,
    "volatility_squeeze_20": 0.0,
    "dry_up_breakout_60": 0.2,
    "money_flow_persistence_20": 0.0,
}


def scorer(group: pd.DataFrame) -> pd.Series:
    return score_from_ranks(ranks(group), WEIGHTS)


def filter_for(name: str):
    def wrapped(group: pd.DataFrame) -> pd.DataFrame:
        base = v8_candidate_filter(group)
        if base.empty or name == "base":
            return base
        row = base.iloc[0]
        if name == "risk_on" and not bool(row.get("market_risk_on", False)):
            return base.iloc[0:0]
        if name == "breadth45" and row.get("market_breadth20", 0) < 0.45:
            return base.iloc[0:0]
        if name == "risk_on_breadth45" and not (bool(row.get("market_risk_on", False)) and row.get("market_breadth20", 0) >= 0.45):
            return base.iloc[0:0]
        if name == "risk_on_breadth45_median_positive" and not (
            bool(row.get("market_risk_on", False))
            and row.get("market_breadth20", 0) >= 0.45
            and row.get("market_median_ret20", 0) > 0
        ):
            return base.iloc[0:0]
        return base

    return wrapped


def drawdowns(equity_points: list[dict], limit: int = 8) -> list[dict]:
    equity = pd.DataFrame(equity_points)
    equity["date"] = pd.to_datetime(equity["date"])
    equity["peak"] = equity["equity"].cummax()
    equity["drawdown"] = equity["equity"] / equity["peak"] - 1
    return [
        {"date": str(row.date.date()), "equity": float(row.equity), "drawdown": round(float(row.drawdown), 6)}
        for row in equity.nsmallest(limit, "drawdown").itertuples(index=False)
    ]


def backtest(panel: pd.DataFrame, filter_name: str) -> dict:
    result = run_scored_backtest(
        panel,
        scorer,
        top_n=5,
        initial_cash=100_000,
        market_filter=False,
        retention_multiple=3,
        universe_size=1000,
        candidate_filter=filter_for(filter_name),
    )
    return {
        "filter": filter_name,
        "metrics": result["metrics"],
        "worst_drawdowns": drawdowns(result["equity"]),
    }


def main() -> None:
    store = ParquetMarketStore(MARKET_DIR)
    panel = build_panel(store, date(2012, 1, 1), date(2026, 7, 5))
    filters = ["base", "risk_on", "breadth45", "risk_on_breadth45", "risk_on_breadth45_median_positive"]
    rows = [backtest(panel, name) for name in filters]
    output = {"weights": WEIGHTS, "filters": rows}
    path = ROOT / "reports" / "strategy_2012_2026_diagnosis.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in rows:
        metrics = row["metrics"]
        print(row["filter"], metrics["total_return"], metrics["annual_return"], metrics["sharpe"], metrics["max_drawdown"], metrics.get("yearly_returns"))


if __name__ == "__main__":
    main()
