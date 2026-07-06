#!/usr/bin/env python3
"""Mine factor combinations that survive older A-share regimes."""

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


WINDOWS = {
    "2012-2018": (date(2012, 1, 1), date(2018, 12, 31)),
    "2019-2023": (date(2019, 1, 1), date(2023, 12, 31)),
    "2024-2026": (date(2024, 1, 1), date(2026, 7, 5)),
}


def main() -> None:
    store = ParquetMarketStore(MARKET_DIR)
    panels = {label: build_panel(store, start, end) for label, (start, end) in WINDOWS.items()}
    combos = weight_grid(step=4)

    studies = {}
    for train_label in ("2012-2018", "2019-2023"):
        screened = quick_screen(panels[train_label], combos, top_n=12)
        candidates = screened[:4]
        studies[train_label] = {
            "screened": screened,
            "validation": {
                label: [backtest(panel, row["weights"]) for row in candidates]
                for label, panel in panels.items()
            },
        }

    output = {
        "objective": "old_period_survival",
        "grid_step": 0.25,
        "studies": studies,
    }
    path = ROOT / "reports" / "old_period_factor_mining.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    for train_label, study in studies.items():
        print("train", train_label)
        for label, rows in study["validation"].items():
            print(" ", label)
            for i, row in enumerate(rows, 1):
                metrics = row["metrics"]
                print("  ", i, metrics["total_return"], metrics["annual_return"], metrics["sharpe"], metrics["max_drawdown"], row["weights"])


if __name__ == "__main__":
    main()
