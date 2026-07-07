#!/usr/bin/env python3
"""Search small regime gates for a 20% annual-return candidate."""

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


AGGRESSIVE = {
    "low_volatility_20": 0.2,
    "market_relative_strength_60": 0.4,
    "dry_up_breakout_60": 0.3,
    "money_flow_persistence_20": 0.1,
}
DEFENSIVE = {
    "low_volatility_20": 0.5,
    "downside_volatility_20": 0.3,
    "distance_to_high_20": 0.2,
}
FACTORS = sorted(set(AGGRESSIVE) | set(DEFENSIVE))


def mode(row: pd.Series, breadth: float, median_ret: float, vol_ceiling: float) -> str:
    if not bool(row.get("market_risk_on", False)):
        return "cash"
    if (
        row.get("market_breadth20", 0) >= breadth
        and row.get("market_median_ret20", 0) >= median_ret
        and row.get("market_median_vol20", 1) <= vol_ceiling
    ):
        return "aggressive"
    if row.get("market_breadth20", 0) >= 0.35 and row.get("market_median_ret20", 0) > -0.05:
        return "defensive"
    return "cash"


def make_runner(breadth: float, median_ret: float, vol_ceiling: float):
    def scorer(group: pd.DataFrame) -> pd.Series:
        current = mode(group.iloc[0], breadth, median_ret, vol_ceiling)
        if current == "cash":
            return pd.Series(0.0, index=group.index)
        weights = AGGRESSIVE if current == "aggressive" else DEFENSIVE
        return score_from_ranks(ranks(group, list(weights)), weights)

    def candidate_filter(group: pd.DataFrame) -> pd.DataFrame:
        base = v8_candidate_filter(group)
        if base.empty:
            return base
        current = mode(base.iloc[0], breadth, median_ret, vol_ceiling)
        if current == "cash":
            return base.iloc[0:0]
        if current == "defensive":
            return base[base["downside_vol20"] <= base["downside_vol20"].quantile(0.75)]
        return base

    return scorer, candidate_filter


def test(panel: pd.DataFrame, breadth: float, median_ret: float, vol_ceiling: float) -> dict:
    scorer, candidate_filter = make_runner(breadth, median_ret, vol_ceiling)
    return run_scored_backtest(
        panel,
        scorer,
        top_n=3,
        initial_cash=100_000,
        market_filter=True,
        retention_multiple=4,
        universe_size=600,
        candidate_filter=candidate_filter,
    )["metrics"]


def main() -> None:
    store = ParquetMarketStore(MARKET_DIR)
    windows = {
        "2012-2018": build_panel(store, date(2012, 1, 1), date(2018, 12, 31), FACTORS),
        "2019-2023": build_panel(store, date(2019, 1, 1), date(2023, 12, 31), FACTORS),
        "2024-2026": build_panel(store, date(2024, 1, 1), date(2026, 7, 5), FACTORS),
    }
    rows = []
    for breadth in (0.45, 0.50, 0.55, 0.60):
        for median_ret in (-0.01, 0.00, 0.01, 0.02):
            for vol_ceiling in (0.35, 0.45, 0.60):
                metrics = {label: test(panel, breadth, median_ret, vol_ceiling) for label, panel in windows.items()}
                annuals = [m["annual_return"] for m in metrics.values()]
                sharpes = [m["sharpe"] for m in metrics.values()]
                drawdowns = [abs(m["max_drawdown"]) for m in metrics.values()]
                rows.append({
                    "breadth": breadth,
                    "median_ret": median_ret,
                    "vol_ceiling": vol_ceiling,
                    "passes_20pct_all_windows": all(a >= 0.20 for a in annuals),
                    "score": min(annuals) + min(sharpes) + sum(annuals) / len(annuals) - max(drawdowns) * 0.25,
                    "metrics": metrics,
                })
                print("tested", breadth, median_ret, vol_ceiling, flush=True)
    rows.sort(key=lambda row: row["score"], reverse=True)
    output = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "target": "annual_return >= 20% in every validation window",
        "aggressive_weights": AGGRESSIVE,
        "defensive_weights": DEFENSIVE,
        "top": rows[:20],
        "passing": [row for row in rows if row["passes_20pct_all_windows"]],
    }
    path = ROOT / "reports" / "regime_20pct_candidate_search.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in rows[:10]:
        print(row)


if __name__ == "__main__":
    main()
