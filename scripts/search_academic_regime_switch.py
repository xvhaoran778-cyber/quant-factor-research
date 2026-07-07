#!/usr/bin/env python3
"""Regime switch search using academic OHLCV factors."""

from __future__ import annotations

import json
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from app.services.market_store import ParquetMarketStore  # noqa: E402
from app.services.research_backtest import run_scored_backtest  # noqa: E402
from scripts.search_academic_ohlcv_factors import MARKET_DIR, WINDOWS, build_panel, candidate_filter, rank_score  # noqa: E402


OUT = ROOT / "reports" / "academic_regime_switch_search.json"

AGGRESSIVE = (
    {"amihud20": 0.35, "near_high_252": 0.25, "ivol60_low": 0.25, "tail20_safe": 0.15},
    {"near_high_252": 0.25, "maxret20_low": 0.25, "amihud20": 0.25, "overnight20": 0.25},
)
DEFENSIVE = (
    {"near_high_252": 0.35, "tail20_safe": 0.25, "ivol60_low": 0.25, "gapvol20_low": 0.15},
    {"tail20_safe": 0.35, "gapvol20_low": 0.25, "beta60_low": 0.20, "ivol60_low": 0.20},
)


def mode(row: pd.Series, breadth: float, median_ret: float, weak_breadth: float) -> str:
    if not bool(row.get("market_risk_on", False)):
        return "cash"
    if row.get("market_breadth20", 0) >= breadth and row.get("market_median_ret20", 0) >= median_ret:
        return "aggressive"
    if row.get("market_breadth20", 0) >= weak_breadth and row.get("market_median_ret20", 0) > -0.04:
        return "defensive"
    return "cash"


def make_runner(aggressive: dict[str, float], defensive: dict[str, float], breadth: float, median_ret: float, weak_breadth: float):
    def scorer(group: pd.DataFrame) -> pd.Series:
        current = mode(group.iloc[0], breadth, median_ret, weak_breadth)
        if current == "cash":
            return pd.Series(0.0, index=group.index)
        return rank_score(group, aggressive if current == "aggressive" else defensive)

    def filtered(group: pd.DataFrame) -> pd.DataFrame:
        current = mode(group.iloc[0], breadth, median_ret, weak_breadth) if not group.empty else "cash"
        if current == "cash":
            return group.iloc[0:0]
        base = candidate_filter(group)
        if current == "defensive" and not base.empty:
            base = base[base["vol20"] <= base["vol20"].quantile(0.75)]
        return base

    return scorer, filtered


def run_one(panel: pd.DataFrame, aggressive: dict[str, float], defensive: dict[str, float], breadth: float, median_ret: float, weak_breadth: float) -> dict:
    scorer, filtered = make_runner(aggressive, defensive, breadth, median_ret, weak_breadth)
    return run_scored_backtest(
        panel,
        scorer,
        top_n=3,
        initial_cash=100_000,
        market_filter=True,
        retention_multiple=2,
        universe_size=1200,
        candidate_filter=filtered,
    )["metrics"]


def main() -> None:
    store = ParquetMarketStore(MARKET_DIR)
    panels = {
        label: build_panel(store, start, end)
        for label, (start, end) in WINDOWS.items()
    }
    rows = []
    for aggressive in AGGRESSIVE:
        for defensive in DEFENSIVE:
            for breadth in (0.45, 0.55, 0.65):
                for median_ret in (-0.01, 0.00, 0.01):
                    for weak_breadth in (0.30, 0.40, 0.50):
                        metrics = {
                            label: run_one(panel, aggressive, defensive, breadth, median_ret, weak_breadth)
                            for label, panel in panels.items()
                        }
                        annuals = [m["annual_return"] for m in metrics.values()]
                        sharpes = [m["sharpe"] for m in metrics.values()]
                        drawdowns = [abs(m["max_drawdown"]) for m in metrics.values()]
                        row = {
                            "aggressive": aggressive,
                            "defensive": defensive,
                            "breadth": breadth,
                            "median_ret": median_ret,
                            "weak_breadth": weak_breadth,
                            "passes_20pct_all_windows": all(value >= 0.20 for value in annuals),
                            "score": min(annuals) + min(sharpes) + sum(annuals) / len(annuals) - max(drawdowns) * 0.25,
                            "metrics": metrics,
                        }
                        rows.append(row)
                        print("tested", breadth, median_ret, weak_breadth, row["passes_20pct_all_windows"], flush=True)
    rows.sort(key=lambda row: row["score"], reverse=True)
    OUT.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "target": "annual_return >= 20% in every validation window",
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
