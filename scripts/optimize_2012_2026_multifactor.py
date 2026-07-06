#!/usr/bin/env python3
"""Optimize the 12-factor set on 2012-2026, then full-backtest the best few."""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from app.services.market_store import ParquetMarketStore
from scripts.run_multifactor_combination import MARKET_DIR, backtest, build_panel, quick_screen, weight_grid


def main() -> None:
    store = ParquetMarketStore(MARKET_DIR)
    panel = build_panel(store, date(2012, 1, 1), date(2026, 7, 5))
    screened = quick_screen(panel, weight_grid(step=4), top_n=12)
    full_backtests = [backtest(panel, row["weights"]) for row in screened[:5]]
    full_backtests.sort(key=lambda row: row["metrics"]["total_return"], reverse=True)
    output = {
        "window": "2012-2026",
        "screened": screened,
        "full_backtests": full_backtests,
    }
    path = ROOT / "reports" / "multifactor_2012_2026_optimization.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    for i, row in enumerate(full_backtests, 1):
        metrics = row["metrics"]
        print(i, row["weights"], metrics["total_return"], metrics["annual_return"], metrics["sharpe"], metrics["max_drawdown"])


if __name__ == "__main__":
    main()
