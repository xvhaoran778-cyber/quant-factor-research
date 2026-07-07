#!/usr/bin/env python3
"""Search daily futures time-series momentum candidates."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MARKET_DIR = Path("/Volumes/xhrrrrr_macmini副盘/quantlab/market")
OUT = ROOT / "reports" / "futures_daily_tsmom_search.json"


def metrics(returns: pd.Series, periods: int = 252) -> dict:
    returns = returns.fillna(0)
    equity = (1 + returns).cumprod()
    total = float(equity.iloc[-1] - 1) if len(equity) else 0.0
    annual = float((1 + total) ** (periods / max(len(returns), 1)) - 1) if total > -1 else -1.0
    drawdown = equity / equity.cummax() - 1
    return {
        "total_return": round(total, 6),
        "annual_return": round(annual, 6),
        "sharpe": round(float(returns.mean() / returns.std() * np.sqrt(periods)), 4) if returns.std() else 0,
        "max_drawdown": round(float(drawdown.min()), 6),
        "positive_days": round(float((returns > 0).mean()), 4),
    }


def load_close() -> pd.DataFrame:
    frames = []
    for path in sorted((MARKET_DIR / "futures" / "daily").glob("*.parquet")):
        frame = pd.read_parquet(path).sort_values("trading_day")
        if len(frame) < 260:
            continue
        rows = frame[["trading_day", "close"]].copy()
        rows["date"] = pd.to_datetime(rows["trading_day"])
        rows[path.stem] = rows["close"].astype(float)
        frames.append(rows[["date", path.stem]])
    close = frames[0]
    for frame in frames[1:]:
        close = close.merge(frame, on="date", how="outer")
    return close.sort_values("date").set_index("date").ffill()


def run(close: pd.DataFrame, lookback: int, vol_window: int, threshold: float, leverage: float, cost: float) -> pd.Series:
    ret = close.pct_change()
    momentum = close.pct_change(lookback)
    vol = ret.rolling(vol_window).std() * np.sqrt(252)
    raw_signal = np.sign(momentum).where(momentum.abs() >= threshold, 0.0)
    weights = raw_signal.div(vol.replace(0, np.nan))
    gross = weights.abs().sum(axis=1).replace(0, np.nan)
    weights = weights.div(gross, axis=0).fillna(0) * leverage
    position = weights.shift(1).fillna(0)
    turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
    return (position * ret).sum(axis=1).fillna(0) - turnover * cost


def window(series: pd.Series, start: str, end: str) -> pd.Series:
    return series[(series.index >= pd.Timestamp(start)) & (series.index <= pd.Timestamp(end))]


def main() -> None:
    close = load_close()
    rows = []
    for lookback in (20, 40, 60, 120, 180):
        for vol_window in (20, 60):
            for threshold in (0.0, 0.03, 0.06, 0.10):
                for leverage in (1.0, 1.5, 2.0):
                    returns = run(close, lookback, vol_window, threshold, leverage, cost=0.0002)
                    windows = {
                        "2012-2018": window(returns, "2012-01-01", "2018-12-31"),
                        "2019-2023": window(returns, "2019-01-01", "2023-12-31"),
                        "2024-2026": window(returns, "2024-01-01", "2026-12-31"),
                    }
                    window_metrics = {label: metrics(value) for label, value in windows.items()}
                    annuals = [m["annual_return"] for m in window_metrics.values()]
                    sharpes = [m["sharpe"] for m in window_metrics.values()]
                    drawdowns = [abs(m["max_drawdown"]) for m in window_metrics.values()]
                    rows.append({
                        "lookback_days": lookback,
                        "vol_window_days": vol_window,
                        "threshold": threshold,
                        "leverage": leverage,
                        "passes_20pct_all_windows": all(value >= 0.20 for value in annuals),
                        "score": min(annuals) + min(sharpes) * 0.2 + sum(annuals) / len(annuals) - max(drawdowns) * 0.25,
                        "metrics": window_metrics,
                    })
    rows.sort(key=lambda row: row["score"], reverse=True)
    OUT.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "target": "annual_return >= 20% in every validation window",
        "instruments": close.columns.tolist(),
        "top": rows[:20],
        "passing": [row for row in rows if row["passes_20pct_all_windows"]],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in rows[:10]:
        print(row)


if __name__ == "__main__":
    main()
