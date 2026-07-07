#!/usr/bin/env python3
"""Search strict market gates for existing A-share presets."""

from __future__ import annotations

import json
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.services.market_store import ParquetMarketStore  # noqa: E402
from app.services.research_backtest import (  # noqa: E402
    PRESETS,
    _candidate_filter,
    _score,
    build_weekly_feature_panel,
    run_scored_backtest,
)


MARKET_DIR = Path("/Volumes/xhrrrrr_macmini副盘/quantlab/market")
OUT = ROOT / "reports" / "strict_market_gate_search.json"

PRESET_IDS = (
    "low_vol_high_momentum_social",
    "low_vol_style_rotation",
)

WINDOWS = {
    "2012-2018": (date(2012, 1, 1), date(2018, 12, 31)),
    "2019-2023": (date(2019, 1, 1), date(2023, 12, 31)),
    "2024-2026": (date(2024, 1, 1), date(2026, 7, 5)),
}


def gate_filter(base_filter, breadth20: float, median_ret20: float, vol_ceiling: float):
    def wrapped(group: pd.DataFrame) -> pd.DataFrame:
        if group.empty:
            return group
        row = group.iloc[0]
        if row.get("market_breadth20", 0) < breadth20:
            return group.iloc[0:0]
        if row.get("market_median_ret20", 0) < median_ret20:
            return group.iloc[0:0]
        if row.get("market_median_vol20", 9) > vol_ceiling:
            return group.iloc[0:0]
        return base_filter(group)

    return wrapped


def run_one(panel: pd.DataFrame, preset_id: str, breadth20: float, median_ret20: float, vol_ceiling: float) -> dict:
    preset = PRESETS[preset_id]
    result = run_scored_backtest(
        panel,
        lambda group: _score(group, preset_id),
        top_n=preset.get("default_top_n", 5),
        initial_cash=100_000,
        market_filter=True,
        retention_multiple=int(preset.get("retention_multiple", 2)),
        universe_size=int(preset.get("universe_size", 1000)),
        candidate_filter=gate_filter(lambda group: _candidate_filter(group, preset_id), breadth20, median_ret20, vol_ceiling),
    )
    return result["metrics"]


def main() -> None:
    store = ParquetMarketStore(MARKET_DIR)
    panels = {
        label: build_weekly_feature_panel(store, start, end, prefer_materialized=True)
        for label, (start, end) in WINDOWS.items()
    }
    rows = []
    for preset_id in PRESET_IDS:
        for breadth20 in (0.45, 0.55, 0.65):
            for median_ret20 in (-0.01, 0.00, 0.01):
                for vol_ceiling in (0.35, 0.50):
                    metrics = {
                        label: run_one(panel, preset_id, breadth20, median_ret20, vol_ceiling)
                        for label, panel in panels.items()
                    }
                    annuals = [m["annual_return"] for m in metrics.values()]
                    sharpes = [m["sharpe"] for m in metrics.values()]
                    drawdowns = [abs(m["max_drawdown"]) for m in metrics.values()]
                    rows.append(
                        {
                            "preset_id": preset_id,
                            "breadth20": breadth20,
                            "median_ret20": median_ret20,
                            "vol_ceiling": vol_ceiling,
                            "passes_20pct_all_windows": all(value >= 0.20 for value in annuals),
                            "score": min(annuals) + min(sharpes) + sum(annuals) / len(annuals) - max(drawdowns) * 0.25,
                            "metrics": metrics,
                        }
                    )
                    print("tested", preset_id, breadth20, median_ret20, vol_ceiling, flush=True)
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
