from dataclasses import asdict, dataclass
from math import sqrt

import numpy as np
import pandas as pd

from app.services.strategy import rank_multifactor


@dataclass(frozen=True)
class TradingRules:
    commission_rate: float = 0.0003
    min_commission: float = 5.0
    stamp_tax_rate: float = 0.0005
    slippage_rate: float = 0.0005
    lot_size: int = 100
    max_positions: int = 5
    rebalance_days: int = 20


def trade_cost(side: str, notional: float, rules: TradingRules) -> tuple[float, float]:
    commission = max(rules.min_commission, notional * rules.commission_rate)
    tax = notional * rules.stamp_tax_rate if side == "sell" else 0.0
    return commission, tax


def performance_metrics(equity: pd.Series, turnover: float) -> dict[str, float]:
    returns = equity.pct_change().dropna()
    if returns.empty:
        return {"total_return": 0, "annual_return": 0, "volatility": 0, "sharpe": 0, "max_drawdown": 0, "turnover": turnover}
    total = equity.iloc[-1] / equity.iloc[0] - 1
    annual = (1 + total) ** (252 / max(len(returns), 1)) - 1 if total > -1 else -1
    volatility = returns.std() * sqrt(252)
    sharpe = returns.mean() / returns.std() * sqrt(252) if returns.std() else 0
    drawdown = equity / equity.cummax() - 1
    return {"total_return": round(float(total), 6), "annual_return": round(float(annual), 6), "volatility": round(float(volatility), 6), "sharpe": round(float(sharpe), 4), "max_drawdown": round(float(drawdown.min()), 6), "turnover": round(turnover, 4)}


def run_backtest(bars: pd.DataFrame, initial_cash: float, rules: TradingRules | None = None) -> dict:
    rules = rules or TradingRules()
    bars = bars.sort_values(["date", "symbol"]).copy()
    dates = sorted(bars["date"].unique())
    cash = initial_cash
    positions: dict[str, dict] = {}
    equity_points: list[dict] = []
    trades: list[dict] = []
    turnover_notional = 0.0

    for day_index, trading_day in enumerate(dates):
        today = bars[bars["date"] == trading_day].set_index("symbol")
        # Orders from completed data through t-1 execute at today's open, preventing look-ahead.
        if day_index > 20 and day_index % rules.rebalance_days == 0:
            signal_day = dates[day_index - 1]
            signal_frame = bars[bars["date"] == signal_day]
            lookback_days = dates[max(0, day_index - 21):day_index]
            lookback = bars[bars["date"].isin(lookback_days)]
            ranked = rank_multifactor(signal_frame, lookback)
            targets = ranked.head(rules.max_positions)["symbol"].tolist()

            for symbol in sorted(set(positions) - set(targets)):
                if symbol not in today.index or bool(today.loc[symbol, "suspended"]):
                    continue
                holding = positions[symbol]
                if holding["acquired_index"] >= day_index:  # T+1 guard
                    continue
                price = float(today.loc[symbol, "open"]) * (1 - rules.slippage_rate)
                quantity = holding["quantity"]
                notional = price * quantity
                commission, tax = trade_cost("sell", notional, rules)
                cash += notional - commission - tax
                turnover_notional += notional
                trades.append({"date": str(trading_day), "symbol": symbol, "side": "sell", "quantity": quantity, "price": round(price, 2), "cost": round(commission + tax, 2)})
                del positions[symbol]

            total_equity = cash + sum(position["quantity"] * float(today.loc[symbol, "close"]) for symbol, position in positions.items() if symbol in today.index)
            target_value = total_equity / max(len(targets), 1)
            for symbol in targets:
                if symbol in positions or symbol not in today.index or bool(today.loc[symbol, "suspended"]):
                    continue
                row = today.loc[symbol]
                open_price = float(row["open"])
                previous_close = float(bars[(bars["symbol"] == symbol) & (bars["date"] < trading_day)].iloc[-1]["close"])
                if open_price >= previous_close * 1.099:  # limit-up cannot be bought
                    continue
                price = open_price * (1 + rules.slippage_rate)
                quantity = int(min(target_value, cash) / price / rules.lot_size) * rules.lot_size
                if quantity <= 0:
                    continue
                notional = price * quantity
                commission, tax = trade_cost("buy", notional, rules)
                if notional + commission > cash:
                    continue
                cash -= notional + commission
                turnover_notional += notional
                positions[symbol] = {"quantity": quantity, "average_cost": price, "acquired_index": day_index}
                trades.append({"date": str(trading_day), "symbol": symbol, "side": "buy", "quantity": quantity, "price": round(price, 2), "cost": round(commission, 2)})

        market_value = sum(position["quantity"] * float(today.loc[symbol, "close"]) for symbol, position in positions.items() if symbol in today.index)
        equity_points.append({"date": str(trading_day), "equity": round(cash + market_value, 2)})

    equity = pd.Series([point["equity"] for point in equity_points], dtype=float)
    turnover = turnover_notional / initial_cash
    return {"metrics": performance_metrics(equity, turnover), "equity": equity_points, "trades": trades, "positions": positions, "rules": asdict(rules)}

