#!/usr/bin/env python3
"""Validate BTC/ETH momentum rotation across data sources and risk controls."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from search_crypto_momentum_rotation import metrics, run


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "crypto_momentum_validation.json"
BASE_URL = "https://www.cryptodatadownload.com/cdd"
WINDOWS = {
    "2014-2018": ("2014-11-28", "2018-12-31"),
    "2019-2023": ("2019-01-01", "2023-12-31"),
    "2024-2026": ("2024-01-01", "2026-07-05"),
}
SOURCES = {
    "bitstamp": {"BTC": "Bitstamp_BTCUSD_d.csv", "ETH": "Bitstamp_ETHUSD_d.csv"},
    "binance": {"BTC": "Binance_BTCUSDT_d.csv", "ETH": "Binance_ETHUSDT_d.csv"},
}


def load_prices(source: str) -> pd.DataFrame:
    prices = {}
    for symbol, filename in SOURCES[source].items():
        df = pd.read_csv(f"{BASE_URL}/{filename}", skiprows=1)
        date_col = "date" if "date" in df.columns else "Date"
        close_col = "close" if "close" in df.columns else "Close"
        df["date"] = pd.to_datetime(df[date_col]).dt.normalize()
        prices[symbol] = df.sort_values("date").set_index("date")[close_col].astype(float)
    return pd.concat(prices, axis=1).sort_index().ffill()


def window_metrics(returns: pd.Series, windows: dict[str, tuple[str, str]] = WINDOWS) -> dict:
    return {label: metrics(returns.loc[start:end]) for label, (start, end) in windows.items()}


def vol_target(returns: pd.Series, target_vol: float, max_leverage: float = 1.0) -> pd.Series:
    realized = returns.rolling(30).std() * np.sqrt(365)
    leverage = (target_vol / realized.replace(0, np.nan)).clip(upper=max_leverage).shift(1).fillna(0)
    return returns * leverage


def main() -> None:
    rows = []
    for source in SOURCES:
        prices = load_prices(source)
        base = run(prices, lookback=60, threshold=0.05, cost=0.0015)
        rows.append({
            "name": f"{source}_base_60d_5pct",
            "source": source,
            "lookback_days": 60,
            "threshold": 0.05,
            "risk_control": "none",
            "metrics": window_metrics(base),
        })
        for target in (0.35, 0.50, 0.70):
            controlled = vol_target(base, target_vol=target)
            rows.append({
                "name": f"{source}_vol_target_{target:.0%}",
                "source": source,
                "lookback_days": 60,
                "threshold": 0.05,
                "risk_control": f"30d volatility target {target:.0%}, no leverage above 1x",
                "metrics": window_metrics(controlled),
            })

    for row in rows:
        annuals = [m["annual_return"] for m in row["metrics"].values()]
        row["passes_20pct_all_windows"] = all(value >= 0.20 for value in annuals if value != 0)
        row["min_annual_return"] = min(annuals)
        row["max_drawdown"] = min(m["max_drawdown"] for m in row["metrics"].values())

    OUT.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "data_source": BASE_URL,
                "note": "Binance daily CSV begins in 2017, so 2014-2018 is partial/not comparable for that source.",
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    for row in rows:
        print(row["name"], row["passes_20pct_all_windows"], {k: v["annual_return"] for k, v in row["metrics"].items()}, row["max_drawdown"])


if __name__ == "__main__":
    main()
