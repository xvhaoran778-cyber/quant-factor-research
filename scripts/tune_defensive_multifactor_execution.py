#!/usr/bin/env python3
"""Tune execution knobs for the best defensive multi-factor combo."""

from __future__ import annotations

import json
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from app.services.market_store import ParquetMarketStore
from app.services.research_backtest import run_scored_backtest
from app.services.strategies.v8_canonical import v8_candidate_filter
from scripts.run_multifactor_combination import MARKET_DIR, build_panel, ranks, score_from_ranks


FACTORS = ["low_volatility_20", "downside_volatility_20", "distance_to_high_20"]
WEIGHTS = {"low_volatility_20": 0.5, "downside_volatility_20": 0.3, "distance_to_high_20": 0.2}


def scorer(group: pd.DataFrame) -> pd.Series:
    return score_from_ranks(ranks(group, FACTORS), WEIGHTS)


def main() -> None:
    store = ParquetMarketStore(MARKET_DIR)
    windows = {
        "2012-2018": build_panel(store, date(2012, 1, 1), date(2018, 12, 31), FACTORS),
        "2019-2023": build_panel(store, date(2019, 1, 1), date(2023, 12, 31), FACTORS),
        "2024-2026": build_panel(store, date(2024, 1, 1), date(2026, 7, 5), FACTORS),
    }
    rows = []
    for top_n in (3, 5, 8, 10):
        for universe_size in (600, 1000, 1600):
            for retention_multiple in (2, 3, 4):
                metrics = {
                    label: run_scored_backtest(
                        panel,
                        scorer,
                        top_n=top_n,
                        initial_cash=100_000,
                        market_filter=True,
                        retention_multiple=retention_multiple,
                        universe_size=universe_size,
                        candidate_filter=v8_candidate_filter,
                    )["metrics"]
                    for label, panel in windows.items()
                }
                sharpes = [m["sharpe"] for m in metrics.values()]
                annuals = [m["annual_return"] for m in metrics.values()]
                drawdowns = [abs(m["max_drawdown"]) for m in metrics.values()]
                rows.append({
                    "top_n": top_n,
                    "universe_size": universe_size,
                    "retention_multiple": retention_multiple,
                    "score": min(sharpes) + sum(sharpes) / len(sharpes) + min(annuals) - max(drawdowns) * 0.35,
                    "metrics": metrics,
                })
                print("tested", top_n, universe_size, retention_multiple, flush=True)
    rows.sort(key=lambda row: row["score"], reverse=True)
    path = ROOT / "reports" / "defensive_multifactor_execution_tuning.json"
    path.write_text(json.dumps({"generated_at": datetime.now().isoformat(timespec="seconds"), "top": rows[:20]}, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in rows[:10]:
        print(row)


if __name__ == "__main__":
    main()
