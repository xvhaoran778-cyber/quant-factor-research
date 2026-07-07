#!/usr/bin/env python3
"""BTC/ETH daily momentum rotation using free CryptoDataDownload CSVs."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "crypto_momentum_rotation_search.json"
BASE_URL = "https://www.cryptodatadownload.com/cdd"
ASSETS = {
    "BTC": "Bitstamp_BTCUSD_d.csv",
    "ETH": "Bitstamp_ETHUSD_d.csv",
}
WINDOWS = {
    "2014-2018": ("2014-11-28", "2018-12-31"),
    "2019-2023": ("2019-01-01", "2023-12-31"),
    "2024-2026": ("2024-01-01", "2026-07-07"),
}


def load_prices() -> pd.DataFrame:
    prices = {}
    for symbol, filename in ASSETS.items():
        df = pd.read_csv(f"{BASE_URL}/{filename}", skiprows=1)
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        prices[symbol] = df.sort_values("date").set_index("date")["close"].astype(float)
    return pd.concat(prices, axis=1).sort_index().ffill()


def metrics(returns: pd.Series) -> dict:
    returns = returns.fillna(0)
    equity = (1 + returns).cumprod()
    total = float(equity.iloc[-1] - 1) if len(equity) else 0.0
    annual = float((1 + total) ** (365 / max(len(returns), 1)) - 1) if total > -1 else -1.0
    drawdown = equity / equity.cummax() - 1
    return {
        "total_return": round(total, 6),
        "annual_return": round(annual, 6),
        "sharpe": round(float(returns.mean() / returns.std() * np.sqrt(365)), 4) if returns.std() else 0,
        "max_drawdown": round(float(drawdown.min()), 6),
    }


def run(prices: pd.DataFrame, lookback: int, threshold: float, cost: float) -> pd.Series:
    returns = prices.pct_change().fillna(0)
    momentum = prices.pct_change(lookback).fillna(-999)
    choice = momentum.idxmax(axis=1)
    best = momentum.max(axis=1)
    signal = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for column in prices.columns:
        signal.loc[(choice == column) & (best > threshold), column] = 1.0
    position = signal.shift(1).fillna(0)
    turnover = signal.diff().abs().sum(axis=1).fillna(signal.abs().sum(axis=1))
    return (position * returns).sum(axis=1) - turnover * cost


def main() -> None:
    prices = load_prices()
    rows = []
    for lookback in (20, 40, 60, 90, 120, 180, 252):
        for threshold in (-0.20, -0.10, 0.0, 0.05, 0.10, 0.20):
            returns = run(prices, lookback, threshold, cost=0.0015)
            window_metrics = {
                label: metrics(returns.loc[start:end])
                for label, (start, end) in WINDOWS.items()
            }
            annuals = [m["annual_return"] for m in window_metrics.values()]
            sharpes = [m["sharpe"] for m in window_metrics.values()]
            drawdowns = [abs(m["max_drawdown"]) for m in window_metrics.values()]
            rows.append({
                "lookback_days": lookback,
                "threshold": threshold,
                "passes_20pct_all_windows": all(value >= 0.20 for value in annuals),
                "score": min(annuals) + min(sharpes) + sum(annuals) / len(annuals) - max(drawdowns) * 0.20,
                "metrics": window_metrics,
            })
    rows.sort(key=lambda row: row["score"], reverse=True)
    OUT.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "data_source": BASE_URL,
                "assets": ASSETS,
                "target": "annual_return >= 20% in every available validation window",
                "note": "Bitstamp BTC starts 2014-11-28; this crypto validation cannot cover 2012-2014.",
                "top": rows[:20],
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
