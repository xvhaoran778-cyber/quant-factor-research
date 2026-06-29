import inspect
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

os.environ.setdefault("TA_CN_MODE", "LONG")

from ta_cn.alphas import alpha191
from app.services.backtest import TradingRules, performance_metrics, trade_cost


FIELD_MAP = {
    "OPEN": "open", "HIGH": "high", "LOW": "low", "CLOSE": "close",
    "VOLUME": "volume", "AMOUNT": "amount", "VWAP": "vwap", "RET": "ret",
    "DTM": "dtm", "DBM": "dbm", "MKT": "mkt", "SMB": "smb", "HML": "hml",
    "BANCHMARKINDEXOPEN": "benchmark_open", "BANCHMARKINDEXCLOSE": "benchmark_close",
}


@dataclass(frozen=True)
class FactorChoice:
    factor_id: str
    direction: str = "normal"
    weight: float = 1.0


def prepare_inputs(bars: pd.DataFrame) -> dict[str, pd.Series]:
    frame = factor_signal_bars(bars).sort_values(["date", "symbol"])
    frame["ret"] = frame.groupby("symbol")["close"].pct_change()
    frame["amount"] = frame["close"] * frame["volume"] if "amount" not in frame else frame["amount"]
    frame["vwap"] = frame["amount"] / frame["volume"].replace(0, np.nan) if "vwap" not in frame else frame["vwap"]
    market = frame.groupby("date").agg(benchmark_open=("open", "mean"), benchmark_close=("close", "mean"))
    frame = frame.join(market, on="date")
    frame["mkt"] = frame["date"].map(market["benchmark_close"].pct_change())
    frame["smb"] = 0.0
    frame["hml"] = 0.0
    previous_open = frame.groupby("symbol")["open"].shift(1)
    frame["dtm"] = np.where(frame["open"] <= previous_open, 0, np.maximum(frame["high"] - frame["open"], frame["open"] - previous_open))
    frame["dbm"] = np.where(frame["open"] >= previous_open, 0, np.maximum(frame["open"] - frame["low"], previous_open - frame["open"]))
    indexed = frame.set_index(["date", "symbol"])
    indexed.index.names = ["date", "asset"]
    return {target: indexed[source].astype(float) for target, source in FIELD_MAP.items()}


def factor_signal_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """Use adjusted prices for factor signals while retaining raw turnover."""
    frame = bars.copy()
    if "adj_close" not in frame:
        return frame
    raw_close = pd.to_numeric(frame["close"], errors="coerce")
    adjusted_close = pd.to_numeric(frame["adj_close"], errors="coerce")
    adjustment = adjusted_close / raw_close.replace(0, np.nan)
    for column in ("open", "high", "low", "close"):
        adjusted_column = f"adj_{column}"
        if adjusted_column in frame:
            values = pd.to_numeric(frame[adjusted_column], errors="coerce")
            frame[column] = values.where(values > 0, pd.to_numeric(frame[column], errors="coerce"))
    volume = pd.to_numeric(frame["volume"], errors="coerce").replace(0, np.nan)
    amount = pd.to_numeric(frame["amount"], errors="coerce") if "amount" in frame else raw_close * pd.to_numeric(frame["volume"], errors="coerce")
    frame["vwap"] = (amount / volume) * adjustment
    return frame


def compute_factor_panel(bars: pd.DataFrame, choices: list[FactorChoice]) -> tuple[pd.DataFrame, dict[str, str]]:
    inputs = prepare_inputs(bars)
    values: dict[str, pd.Series] = {}
    errors: dict[str, str] = {}
    for choice in choices:
        func = getattr(alpha191, choice.factor_id, None)
        if not func:
            errors[choice.factor_id] = "未知因子"
            continue
        required = [name for name in inspect.signature(func).parameters if name != "kwargs"]
        try:
            result = func(**{name: inputs[name] for name in required})
            if not isinstance(result, pd.Series):
                result = pd.Series(result, index=inputs["CLOSE"].index)
            values[choice.factor_id] = result.replace([np.inf, -np.inf], np.nan)
        except Exception as exc:
            errors[choice.factor_id] = str(exc)[:240]
    return pd.DataFrame(values), errors


def weekly_scores(panel: pd.DataFrame, choices: list[FactorChoice]) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame(columns=["date", "symbol", "score"])
    frame = panel.reset_index().rename(columns={"asset": "symbol"})
    frame["week"] = pd.to_datetime(frame["date"]).dt.to_period("W-FRI")
    weekly = frame.sort_values("date").groupby(["week", "symbol"], as_index=False).tail(1).copy()
    score_columns: list[str] = []
    total_weight = 0.0
    for choice in choices:
        if choice.factor_id not in weekly:
            continue
        percentile = weekly.groupby("date")[choice.factor_id].rank(pct=True, method="average")
        if choice.direction == "reverse":
            percentile = 1 - percentile
        column = f"score_{choice.factor_id}"
        weekly[column] = percentile * max(choice.weight, 0)
        score_columns.append(column)
        total_weight += max(choice.weight, 0)
    weekly["score"] = weekly[score_columns].sum(axis=1, min_count=1) / (total_weight or 1)
    detail = [choice.factor_id for choice in choices if choice.factor_id in weekly]
    return weekly[["date", "symbol", "score", *detail]].sort_values(["date", "score", "symbol"], ascending=[True, False, True])


def latest_selection(bars: pd.DataFrame, choices: list[FactorChoice], top_n: int) -> dict:
    panel, errors = compute_factor_panel(bars, choices)
    scores = weekly_scores(panel, choices)
    if scores.empty:
        return {"as_of": None, "stocks": [], "errors": errors}
    as_of = scores["date"].max()
    selected = scores[scores["date"] == as_of].dropna(subset=["score"]).head(top_n)
    return {"as_of": str(as_of), "stocks": selected.to_dict(orient="records"), "errors": errors}


def run_weekly_factor_backtest(bars: pd.DataFrame, choices: list[FactorChoice], top_n: int, initial_cash: float, rules: TradingRules | None = None) -> dict:
    rules = rules or TradingRules(rebalance_days=5, max_positions=top_n)
    bars = bars.sort_values(["date", "symbol"]).copy()
    panel, errors = compute_factor_panel(bars, choices)
    scores = weekly_scores(panel, choices)
    dates = sorted(bars["date"].unique())
    next_day = {dates[index]: dates[index + 1] for index in range(len(dates) - 1)}
    targets_by_day: dict = {}
    for score_date, group in scores.dropna(subset=["score"]).groupby("date"):
        execution_day = next_day.get(score_date)
        if execution_day is not None:
            targets_by_day[execution_day] = group.head(top_n)["symbol"].tolist()

    cash = initial_cash
    positions: dict[str, dict] = {}
    trades: list[dict] = []
    equity_points: list[dict] = []
    turnover_notional = 0.0
    for day_index, trading_day in enumerate(dates):
        today = bars[bars["date"] == trading_day].set_index("symbol")
        targets = targets_by_day.get(trading_day)
        if targets is not None:
            for symbol in sorted(set(positions) - set(targets)):
                if symbol not in today.index or bool(today.loc[symbol, "suspended"]):
                    continue
                position = positions.pop(symbol)
                price = float(today.loc[symbol, "open"]) * (1 - rules.slippage_rate)
                notional = price * position["quantity"]
                commission, tax = trade_cost("sell", notional, rules)
                cash += notional - commission - tax
                turnover_notional += notional
                trades.append({"date": str(trading_day), "symbol": symbol, "side": "sell", "quantity": position["quantity"], "price": round(price, 2), "cost": round(commission + tax, 2)})
            market_value = sum(position["quantity"] * float(today.loc[symbol, "close"]) for symbol, position in positions.items() if symbol in today.index)
            target_value = (cash + market_value) / max(len(targets), 1)
            for symbol in targets:
                if symbol in positions or symbol not in today.index or bool(today.loc[symbol, "suspended"]):
                    continue
                price = float(today.loc[symbol, "open"]) * (1 + rules.slippage_rate)
                quantity = int(min(target_value, cash) / price / rules.lot_size) * rules.lot_size
                if quantity <= 0:
                    continue
                notional = price * quantity
                commission, _ = trade_cost("buy", notional, rules)
                if notional + commission > cash:
                    continue
                cash -= notional + commission
                turnover_notional += notional
                positions[symbol] = {"quantity": quantity, "average_cost": price, "acquired_index": day_index}
                trades.append({"date": str(trading_day), "symbol": symbol, "side": "buy", "quantity": quantity, "price": round(price, 2), "cost": round(commission, 2)})
        market_value = sum(position["quantity"] * float(today.loc[symbol, "close"]) for symbol, position in positions.items() if symbol in today.index)
        equity_points.append({"date": str(trading_day), "equity": round(cash + market_value, 2)})
    equity = pd.Series([point["equity"] for point in equity_points], dtype=float)
    return {"metrics": performance_metrics(equity, turnover_notional / initial_cash), "equity": equity_points, "trades": trades, "errors": errors}
