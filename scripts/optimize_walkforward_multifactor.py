#!/usr/bin/env python3
"""Walk-forward search for a small robust multi-factor set."""

from __future__ import annotations

import json
import sys
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from app.services.market_store import ParquetMarketStore
from scripts.run_multifactor_combination import MARKET_DIR, backtest, build_panel


FACTORS = [
    "low_volatility_20",
    "downside_volatility_20",
    "distance_to_high_20",
    "market_relative_strength_60",
    "dry_up_breakout_60",
    "money_flow_persistence_20",
    "short_reversal_5",
    "calm_pullback_20",
]


def candidate_weights() -> list[dict[str, float]]:
    raw = [
        {"low_volatility_20": 0.6, "distance_to_high_20": 0.4},
        {"low_volatility_20": 0.5, "downside_volatility_20": 0.3, "distance_to_high_20": 0.2},
        {"low_volatility_20": 0.4, "distance_to_high_20": 0.3, "market_relative_strength_60": 0.3},
        {"low_volatility_20": 0.4, "distance_to_high_20": 0.3, "dry_up_breakout_60": 0.3},
        {"downside_volatility_20": 0.4, "money_flow_persistence_20": 0.3, "distance_to_high_20": 0.3},
        {"short_reversal_5": 0.4, "low_volatility_20": 0.4, "distance_to_high_20": 0.2},
        {"calm_pullback_20": 0.4, "low_volatility_20": 0.4, "money_flow_persistence_20": 0.2},
        {"market_relative_strength_60": 0.4, "low_volatility_20": 0.3, "dry_up_breakout_60": 0.3},
    ]
    return [{name: row.get(name, 0.0) for name in FACTORS} for row in raw]


def score(metrics_by_window: dict[str, dict]) -> float:
    sharpes = [m["sharpe"] for m in metrics_by_window.values()]
    annuals = [m["annual_return"] for m in metrics_by_window.values()]
    drawdowns = [abs(m["max_drawdown"]) for m in metrics_by_window.values()]
    return min(sharpes) + sum(sharpes) / len(sharpes) + min(annuals) - max(drawdowns) * 0.5


def main() -> None:
    store = ParquetMarketStore(MARKET_DIR)
    windows = {
        "2012-2018": build_panel(store, date(2012, 1, 1), date(2018, 12, 31), FACTORS),
        "2019-2023": build_panel(store, date(2019, 1, 1), date(2023, 12, 31), FACTORS),
        "2024-2026": build_panel(store, date(2024, 1, 1), date(2026, 7, 5), FACTORS),
    }
    rows = []
    for weights in candidate_weights():
        print("testing", {k: v for k, v in weights.items() if v}, flush=True)
        window_results = {label: backtest(panel, weights) for label, panel in windows.items()}
        metrics = {label: result["metrics"] for label, result in window_results.items()}
        rows.append({"weights": weights, "score": score(metrics), "metrics": metrics})
    rows.sort(key=lambda row: row["score"], reverse=True)
    output = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "objective": "maximize worst-window Sharpe plus average Sharpe, penalize drawdown",
        "top": rows[:20],
    }
    path = ROOT / "reports" / "walkforward_multifactor_optimization.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in rows[:10]:
        active = {k: v for k, v in row["weights"].items() if v}
        print(round(row["score"], 4), active, row["metrics"])


if __name__ == "__main__":
    main()
