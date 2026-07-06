#!/usr/bin/env python3
"""Backtest regime switching between aggressive and defensive factor weights."""

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

AGGRESSIVE_WEIGHTS = {
    "correlation_breakdown": 0.0,
    "low_volatility_20": 0.2,
    "liquidity_strength_20": 0.4,
    "downside_volatility_20": 0.0,
    "breakout_position_60": 0.2,
    "distance_to_high_20": 0.0,
    "liquidity_acceleration_20": 0.2,
    "low_drawdown_momentum_60": 0.0,
}
DEFENSIVE_WEIGHTS = {
    "correlation_breakdown": 0.1,
    "low_volatility_20": 0.0,
    "liquidity_strength_20": 0.6,
    "downside_volatility_20": 0.3,
    "breakout_position_60": 0.0,
    "distance_to_high_20": 0.0,
    "liquidity_acceleration_20": 0.0,
    "low_drawdown_momentum_60": 0.0,
}


def regime(row: pd.Series) -> str:
    breadth = row.get("market_breadth20", 0)
    median_ret = row.get("market_median_ret20", 0)
    risk_on = bool(row.get("market_risk_on", False))
    if risk_on and breadth >= 0.45 and median_ret > -0.02:
        return "aggressive"
    if risk_on and breadth >= 0.40 and median_ret > -0.04:
        return "defensive"
    return "cash"


def switched_scorer(group: pd.DataFrame) -> pd.Series:
    mode = regime(group.iloc[0])
    if mode == "cash":
        return pd.Series(0.0, index=group.index)
    weights = AGGRESSIVE_WEIGHTS if mode == "aggressive" else DEFENSIVE_WEIGHTS
    return score_from_ranks(ranks(group), weights)


def switched_filter(group: pd.DataFrame) -> pd.DataFrame:
    base = v8_candidate_filter(group)
    if base.empty:
        return base
    mode = regime(base.iloc[0])
    if mode == "cash":
        return base.iloc[0:0]
    if mode == "aggressive":
        return base
    return base[base["downside_vol20"] < base["downside_vol20"].quantile(0.8)]


def drawdown_periods(equity_points: list[dict], limit: int = 5) -> list[dict]:
    equity = pd.DataFrame(equity_points)
    equity["date"] = pd.to_datetime(equity["date"])
    equity["peak"] = equity["equity"].cummax()
    equity["drawdown"] = equity["equity"] / equity["peak"] - 1
    return [
        {"date": str(row.date.date()), "equity": float(row.equity), "drawdown": round(float(row.drawdown), 6)}
        for row in equity.nsmallest(limit, "drawdown").itertuples(index=False)
    ]


def regime_counts(panel: pd.DataFrame) -> dict[str, int]:
    weekly = panel.sort_values("date").groupby("week", sort=True).first()
    counts = weekly.apply(regime, axis=1).value_counts()
    return {str(key): int(value) for key, value in counts.items()}


def backtest_window(label: str, start: date, end: date) -> dict:
    store = ParquetMarketStore(MARKET_DIR)
    panel = build_panel(store, start, end)
    result = run_scored_backtest(
        panel,
        switched_scorer,
        top_n=5,
        initial_cash=100_000,
        market_filter=True,
        retention_multiple=3,
        universe_size=1000,
        candidate_filter=switched_filter,
    )
    return {
        "label": label,
        "start": str(start),
        "end": str(end),
        "metrics": result["metrics"],
        "regime_counts": regime_counts(panel),
        "worst_drawdowns": drawdown_periods(result["equity"]),
    }


def main() -> None:
    windows = [
        ("2019-2023", date(2019, 1, 1), date(2023, 12, 31)),
        ("2024-2026", date(2024, 1, 1), date(2026, 7, 5)),
    ]
    output = {
        "strategy": "regime_switch_aggressive_defensive",
        "rules": {
            "aggressive": "market_risk_on and market_breadth20 >= 0.45 and market_median_ret20 > -0.02",
            "defensive": "market_risk_on and market_breadth20 >= 0.40 and market_median_ret20 > -0.04",
            "cash": "otherwise",
        },
        "aggressive_weights": AGGRESSIVE_WEIGHTS,
        "defensive_weights": DEFENSIVE_WEIGHTS,
        "windows": [backtest_window(label, start, end) for label, start, end in windows],
    }
    path = ROOT / "reports" / "regime_switch_strategy_results.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in output["windows"]:
        metrics = row["metrics"]
        print(row["label"], metrics["total_return"], metrics["annual_return"], metrics["sharpe"], metrics["max_drawdown"], row["regime_counts"])


if __name__ == "__main__":
    main()
