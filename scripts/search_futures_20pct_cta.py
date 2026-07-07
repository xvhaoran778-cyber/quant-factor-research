#!/usr/bin/env python3
"""Search simple futures cross-sectional CTA candidates."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MARKET_DIR = Path("/Volumes/xhrrrrr_macmini副盘/quantlab/market")
OUT = ROOT / "reports" / "futures_20pct_cta_search.json"


def metrics(returns: pd.Series) -> dict:
    returns = returns.fillna(0)
    equity = (1 + returns).cumprod()
    total = float(equity.iloc[-1] - 1) if len(equity) else 0.0
    annual = float((1 + total) ** (52 / max(len(returns), 1)) - 1) if total > -1 else -1.0
    drawdown = equity / equity.cummax() - 1
    return {
        "total_return": round(total, 6),
        "annual_return": round(annual, 6),
        "sharpe": round(float(returns.mean() / returns.std() * np.sqrt(52)), 4) if returns.std() else 0,
        "max_drawdown": round(float(drawdown.min()), 6),
        "positive_weeks": round(float((returns > 0).mean()), 4),
    }


def load_weekly() -> pd.DataFrame:
    frames = []
    for path in sorted((MARKET_DIR / "futures" / "daily").glob("*.parquet")):
        frame = pd.read_parquet(path).sort_values("trading_day").copy()
        frame["date"] = pd.to_datetime(frame["trading_day"])
        frame = frame[frame["date"] >= pd.Timestamp("2010-01-01")]
        if len(frame) < 260:
            continue
        frame["symbol"] = path.stem
        frame["ret1"] = frame["close"].astype(float).pct_change()
        weekly = frame.groupby(frame["date"].dt.to_period("W-FRI")).tail(1).copy()
        weekly["week"] = weekly["date"].dt.to_period("W-FRI").astype(str)
        frames.append(weekly[["week", "date", "symbol", "close", "ret1"]])
    panel = pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"])
    panel["next_ret"] = panel.groupby("symbol")["close"].pct_change().shift(-1)
    return panel


def add_features(panel: pd.DataFrame, lookback: int) -> pd.DataFrame:
    frame = panel.copy()
    grouped = frame.groupby("symbol", sort=False)
    frame["mom"] = grouped["close"].pct_change(lookback)
    frame["vol"] = grouped["ret1"].transform(lambda x: x.rolling(lookback).std() * np.sqrt(252))
    frame["score"] = frame["mom"] / frame["vol"].replace(0, np.nan)
    return frame.dropna(subset=["score", "next_ret", "vol"])


def run(panel: pd.DataFrame, lookback: int, legs: int, threshold: float, vol_target: float, cost: float) -> pd.Series:
    frame = add_features(panel, lookback)
    rows = []
    for week, group in frame.groupby("week", sort=True):
        if len(group) < legs * 2:
            continue
        ranked = group.sort_values("score")
        short = ranked.head(legs)
        long = ranked.tail(legs)
        selected = pd.concat([long.assign(side=1), short.assign(side=-1)])
        selected = selected[selected["score"].abs() >= threshold]
        if selected.empty:
            rows.append((week, 0.0))
            continue
        raw = float((selected["side"] * selected["next_ret"]).mean())
        realized_vol = float((selected["side"] * selected["next_ret"]).std() * np.sqrt(52))
        scale = min(2.0, vol_target / realized_vol) if realized_vol > 0 else 1.0
        rows.append((week, raw * scale - cost * len(selected) / max(len(group), 1)))
    return pd.Series(dict(rows)).sort_index()


def main() -> None:
    panel = load_weekly()
    configs = []
    for lookback in (8, 13, 26, 39):
        for legs in (2, 3, 4):
            for threshold in (0.0, 0.2, 0.4):
                for vol_target in (0.15, 0.25, 0.35):
                    returns = run(panel, lookback, legs, threshold, vol_target, cost=0.0008)
                    years = pd.Index([int(str(index)[:4]) for index in returns.index])
                    windows = {
                        "2012-2018": returns[(years >= 2012) & (years <= 2018)],
                        "2019-2023": returns[(years >= 2019) & (years <= 2023)],
                        "2024-2026": returns[years >= 2024],
                    }
                    window_metrics = {label: metrics(series) for label, series in windows.items()}
                    annuals = [m["annual_return"] for m in window_metrics.values()]
                    sharpes = [m["sharpe"] for m in window_metrics.values()]
                    drawdowns = [abs(m["max_drawdown"]) for m in window_metrics.values()]
                    configs.append({
                        "lookback_weeks": lookback,
                        "legs": legs,
                        "threshold": threshold,
                        "vol_target": vol_target,
                        "passes_20pct_all_windows": all(value >= 0.20 for value in annuals),
                        "score": min(annuals) + min(sharpes) * 0.2 + sum(annuals) / len(annuals) - max(drawdowns) * 0.25,
                        "metrics": window_metrics,
                    })
    configs.sort(key=lambda row: row["score"], reverse=True)
    OUT.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "target": "annual_return >= 20% in every validation window",
        "instruments": sorted(panel["symbol"].unique().tolist()),
        "top": configs[:20],
        "passing": [row for row in configs if row["passes_20pct_all_windows"]],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in configs[:10]:
        print(row)


if __name__ == "__main__":
    main()
