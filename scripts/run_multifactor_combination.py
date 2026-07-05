#!/usr/bin/env python3
"""Try many simple multi-factor weights, then backtest the best few."""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.services.market_store import ParquetMarketStore
from app.services.new_factors_fast import compute_factors_vectorized
from app.services.research_backtest import build_weekly_feature_panel, run_scored_backtest
from app.services.strategies.v8_canonical import v8_candidate_filter


MARKET_DIR = "/Volumes/xhrrrrr_macmini副盘/quantlab/market"
FACTORS = [
    "correlation_breakdown",
    "low_volatility_20",
    "liquidity_strength_20",
    "downside_volatility_20",
    "breakout_position_60",
    "distance_to_high_20",
    "liquidity_acceleration_20",
    "low_drawdown_momentum_60",
]
DIRECTIONS = {
    "correlation_breakdown": 1,
    "low_volatility_20": -1,
    "liquidity_strength_20": 1,
    "downside_volatility_20": -1,
    "breakout_position_60": 1,
    "distance_to_high_20": 1,
    "liquidity_acceleration_20": 1,
    "low_drawdown_momentum_60": 1,
}


def weight_grid(step: int = 5) -> list[dict[str, float]]:
    combos: list[dict[str, float]] = []

    def walk(prefix: list[int], remaining: int, slots: int) -> None:
        if slots == 1:
            weights = [*prefix, remaining]
            if sum(w > 0 for w in weights) >= 2:
                combos.append(dict(zip(FACTORS, [w / step for w in weights], strict=True)))
            return
        for value in range(remaining + 1):
            walk([*prefix, value], remaining - value, slots - 1)

    walk([], step, len(FACTORS))
    return combos


def factor_panel(store: ParquetMarketStore, start: date, end: date) -> pd.DataFrame:
    warmup = start - timedelta(days=365)
    daily = store.read(warmup, end, symbols=None, fill_suspensions=False)
    close = daily.pivot(index="date", columns="symbol", values="close")
    data = {
        "close": close,
        "open": daily.pivot(index="date", columns="symbol", values="open"),
        "high": daily.pivot(index="date", columns="symbol", values="high"),
        "low": daily.pivot(index="date", columns="symbol", values="low"),
        "volume": daily.pivot(index="date", columns="symbol", values="volume"),
        "amount": daily.pivot(index="date", columns="symbol", values="amount"),
        "returns": close.pct_change(),
    }
    factors = compute_factors_vectorized(data, FACTORS).reset_index()
    factors["date"] = pd.to_datetime(factors["date"])
    return factors[factors["date"] >= pd.Timestamp(start)]


def build_panel(store: ParquetMarketStore, start: date, end: date) -> pd.DataFrame:
    panel = build_weekly_feature_panel(store, start, end, prefer_materialized=True)
    return panel.merge(factor_panel(store, start, end), on=["date", "symbol"], how="left")


def ranks(group: pd.DataFrame) -> dict[str, pd.Series]:
    out = {}
    for name in FACTORS:
        values = group[name].fillna(group[name].median())
        out[name] = values.rank(pct=True, ascending=DIRECTIONS[name] > 0, method="average").fillna(0.5)
    return out


def score_from_ranks(rank_values: dict[str, pd.Series], weights: dict[str, float]) -> pd.Series:
    score = next(iter(rank_values.values())) * 0
    for name, weight in weights.items():
        score += rank_values[name] * weight
    return score


def quick_screen(panel: pd.DataFrame, combos: list[dict[str, float]], top_n: int = 40) -> list[dict]:
    frame = panel.dropna(subset=["next_open", *FACTORS]).sort_values(["symbol", "date"]).copy()
    frame["future_return"] = frame.groupby("symbol")["next_open"].shift(-1) / frame["next_open"] - 1
    rows = []
    for weights in combos:
        returns = []
        for _, group in frame.groupby("week", sort=True):
            group = v8_candidate_filter(group.nlargest(min(1000, len(group)), "liquidity"))
            group = group.dropna(subset=["future_return"])
            if len(group) < 20:
                continue
            group = group.copy()
            group["score"] = score_from_ranks(ranks(group), weights)
            returns.append(float(group.nlargest(5, "score")["future_return"].mean()))
        if returns:
            series = pd.Series(returns)
            total = float(series.add(1).prod() - 1)
            sharpe = float(series.mean() / series.std() * (52 ** 0.5)) if series.std() else 0.0
            rows.append({"weights": weights, "quick_total_return": total, "quick_sharpe": sharpe})
    return sorted(rows, key=lambda row: row["quick_total_return"], reverse=True)[:top_n]


def backtest(panel: pd.DataFrame, weights: dict[str, float]) -> dict:
    def scorer(group: pd.DataFrame) -> pd.Series:
        return score_from_ranks(ranks(group), weights)

    result = run_scored_backtest(
        panel,
        scorer,
        top_n=5,
        initial_cash=100_000,
        market_filter=True,
        retention_multiple=3,
        universe_size=1000,
        candidate_filter=v8_candidate_filter,
    )
    return {"weights": weights, "metrics": result["metrics"], "equity": result["equity"]}


def main() -> None:
    store = ParquetMarketStore(MARKET_DIR)
    combos = weight_grid()
    recent_panel = build_panel(store, date(2024, 1, 1), date(2026, 7, 5))
    screened = quick_screen(recent_panel, combos, top_n=40)
    recent_backtests = [backtest(recent_panel, row["weights"]) for row in screened[:30]]
    recent_backtests.sort(key=lambda row: row["metrics"]["total_return"], reverse=True)

    long_panel = build_panel(store, date(2019, 1, 1), date(2023, 12, 31))
    validated = [backtest(long_panel, row["weights"]) for row in recent_backtests[:8]]

    output = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "market_dir": MARKET_DIR,
        "factors": FACTORS,
        "directions": DIRECTIONS,
        "formula": "sum(weight_i * cross_section_rank(adjusted_factor_i))",
        "objective": "total_return",
        "grid_step": 0.2,
        "screened_combinations": len(combos),
        "quick_screen_top": screened,
        "recent_backtests": recent_backtests,
        "long_validation": validated,
    }
    (ROOT / "reports" / "multifactor_return_optimization_results.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
