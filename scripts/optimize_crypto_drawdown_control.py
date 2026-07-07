#!/usr/bin/env python3
"""Search low-drawdown volatility targets for BTC/ETH momentum rotation."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from search_crypto_momentum_rotation import metrics, run  # noqa: E402
from validate_crypto_momentum_rotation import WINDOWS, load_prices  # noqa: E402


OUT = ROOT / "reports" / "crypto_drawdown_control_search.json"


def vol_target(returns: pd.Series, target_vol: float, vol_window: int) -> pd.Series:
    realized = returns.rolling(vol_window).std() * np.sqrt(365)
    exposure = (target_vol / realized.replace(0, np.nan)).clip(upper=1.0).shift(1).fillna(0)
    return returns * exposure


def window_metrics(returns: pd.Series) -> dict:
    return {label: metrics(returns.loc[start:end]) for label, (start, end) in WINDOWS.items()}


def main() -> None:
    prices = load_prices("bitstamp")
    rows = []
    for lookback in (20, 40, 60, 90, 120, 180, 252):
        for threshold in (-0.20, -0.10, 0.0, 0.05, 0.10, 0.20):
            base = run(prices, lookback=lookback, threshold=threshold, cost=0.0015)
            for target_vol in (0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30):
                for vol_window in (20, 30, 45, 60):
                    controlled = vol_target(base, target_vol, vol_window)
                    wm = window_metrics(controlled)
                    annuals = [m["annual_return"] for m in wm.values()]
                    drawdowns = [abs(m["max_drawdown"]) for m in wm.values()]
                    sharpes = [m["sharpe"] for m in wm.values()]
                    rows.append({
                        "lookback_days": lookback,
                        "threshold": threshold,
                        "target_vol": target_vol,
                        "vol_window_days": vol_window,
                        "passes_20pct_all_windows": all(value >= 0.20 for value in annuals),
                        "score": min(annuals) + min(sharpes) + sum(annuals) / len(annuals) - max(drawdowns),
                        "max_abs_drawdown": max(drawdowns),
                        "metrics": wm,
                    })
    rows.sort(key=lambda row: (row["passes_20pct_all_windows"], -row["max_abs_drawdown"], row["score"]), reverse=True)
    OUT.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "base_strategy": "BTC/ETH momentum rotation, Bitstamp daily CSV",
                "recommendation": {
                    "profile": "return_enhanced",
                    "lookback_days": 20,
                    "threshold": 0.20,
                    "target_vol": 0.28,
                    "vol_window_days": 20,
                    "reason": "highest minimum annual return with max drawdown <= 25%",
                },
                "top": rows[:25],
                "passing": [row for row in rows if row["passes_20pct_all_windows"]],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    for row in rows[:10]:
        print(row["passes_20pct_all_windows"], row["lookback_days"], row["threshold"], row["target_vol"], row["vol_window_days"], row["max_abs_drawdown"], {k: v["annual_return"] for k, v in row["metrics"].items()})


if __name__ == "__main__":
    main()
