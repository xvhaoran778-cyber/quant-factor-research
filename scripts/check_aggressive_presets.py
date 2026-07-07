#!/usr/bin/env python3
"""Validate aggressive A-share presets across fixed windows."""

from __future__ import annotations

import json
import sys
from datetime import date, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.services.market_store import ParquetMarketStore  # noqa: E402
from app.services.research_backtest import PRESETS, build_weekly_feature_panel, run_preset_backtest  # noqa: E402


MARKET_DIR = Path("/Volumes/xhrrrrr_macmini副盘/quantlab/market")
OUT = ROOT / "reports" / "aggressive_preset_20pct_check.json"

PRESET_IDS = (
    "low_price_breakout_proxy",
    "limit_up_continuation_proxy",
    "small_cap_reversal_guarded",
    "social_small_cap_quality_momentum",
    "low_vol_high_momentum_social",
    "small_large_style_rotation",
    "breakout_acceleration",
    "oversold_rebound_aggressive",
)

WINDOWS = {
    "2012-2018": (date(2012, 1, 1), date(2018, 12, 31)),
    "2019-2023": (date(2019, 1, 1), date(2023, 12, 31)),
    "2024-2026": (date(2024, 1, 1), date(2026, 7, 5)),
}


def main() -> None:
    store = ParquetMarketStore(MARKET_DIR)
    panels = {
        label: build_weekly_feature_panel(store, start, end, prefer_materialized=True)
        for label, (start, end) in WINDOWS.items()
    }
    rows = []
    for preset_id in PRESET_IDS:
        if preset_id not in PRESETS:
            continue
        item = {
            "id": preset_id,
            "name": PRESETS[preset_id]["name"],
            "metrics": {},
            "passes_20pct_all_windows": True,
        }
        for label, panel in panels.items():
            result = run_preset_backtest(
                panel,
                preset_id,
                top_n=PRESETS[preset_id].get("default_top_n", 5),
                initial_cash=100_000,
            )
            metrics = result["metrics"]
            item["metrics"][label] = metrics
            if metrics.get("annual_return", -1) < 0.20:
                item["passes_20pct_all_windows"] = False
        rows.append(item)
        print(
            preset_id,
            {
                label: (
                    values.get("annual_return"),
                    values.get("sharpe"),
                    values.get("max_drawdown"),
                )
                for label, values in item["metrics"].items()
            },
        )

    OUT.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "target": "annual_return >= 20% in every validation window",
                "windows": {label: [str(start), str(end)] for label, (start, end) in WINDOWS.items()},
                "rows": rows,
                "passing": [row for row in rows if row["passes_20pct_all_windows"]],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
