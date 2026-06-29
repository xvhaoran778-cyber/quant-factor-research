"""回测引擎 - 重构版"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable
from datetime import datetime
from loguru import logger
from dataclasses import dataclass, field


@dataclass
class Trade:
    """交易记录"""
    date: str
    code: str
    action: str  # 'buy' or 'sell'
    price: float
    volume: int
    amount: float
    commission: float = 0
    slippage: float = 0
    reason: str = ""


@dataclass
class Position:
    """持仓"""
    code: str
    volume: int
    avg_cost: float
    current_price: float = 0
    market_value: float = 0
    profit: float = 0
    profit_pct: float = 0


class BacktestEngine:
    """回测引擎"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # 回测参数
        self.initial_capital = self.config.get('initial_capital', 1000000)
        self.commission_rate = self.config.get('commission_rate', 0.0003)
        self.slippage_rate = self.config.get('slippage_rate', 0.001)
        self.stamp_tax = self.config.get('stamp_tax', 0.001)
        
        # 状态
        self.reset()
    
    def reset(self):
        """重置回测状态"""
        self.capital = self.initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.daily_equity: List[Dict] = []
    
    def run(self, signal_func: Callable, price_data: Dict[str, pd.DataFrame],
            start_date: str = None, end_date: str = None) -> Dict:
        """运行回测"""
        logger.info("开始回测...")
        self.reset()
        
        # 获取所有交易日期
        all_dates = set()
        for code, df in price_data.items():
            if 'date' in df.columns:
                all_dates.update(df['date'].tolist())
        
        all_dates = sorted(list(all_dates))
        
        if start_date:
            all_dates = [d for d in all_dates if str(d) >= start_date]
        if end_date:
            all_dates = [d for d in all_dates if str(d) <= end_date]
        
        # 逐日回测
        for date in all_dates:
            self._update_positions(date, price_data)
            
            signals = signal_func(date, self.positions, price_data)
            
            if signals:
                self._execute_signals(signals, date, price_data)
            
            self._record_daily_equity(date)
        
        logger.info(f"回测完成，总交易次数: {len(self.trades)}")
        
        return self._generate_report()
    
    def _update_positions(self, date, price_data: Dict[str, pd.DataFrame]):
        """更新持仓价格"""
        for code, position in self.positions.items():
            if code in price_data:
                df = price_data[code]
                if 'date' in df.columns:
                    row = df[df['date'] == date]
                    if not row.empty:
                        position.current_price = float(row.iloc[0]['close'])
                        position.market_value = position.volume * position.current_price
                        position.profit = (position.current_price - position.avg_cost) * position.volume
                        position.profit_pct = (position.current_price / position.avg_cost - 1) * 100
    
    def _execute_signals(self, signals: List[Dict], date, price_data: Dict[str, pd.DataFrame]):
        """执行交易信号"""
        for signal in signals:
            code = signal.get('code')
            action = signal.get('action')
            target_pct = signal.get('target_pct', 0)
            reason = signal.get('reason', '')
            
            if code not in price_data:
                continue
            
            df = price_data[code]
            if 'date' not in df.columns:
                continue
            
            row = df[df['date'] == date]
            if row.empty:
                continue
            
            price = float(row.iloc[0]['close'])
            
            if action == 'buy':
                self._execute_buy(code, price, target_pct, date, reason)
            elif action == 'sell':
                self._execute_sell(code, price, target_pct, date, reason)
    
    def _execute_buy(self, code: str, price: float, target_pct: float, date, reason: str):
        """执行买入"""
        total_equity = self._get_total_equity()
        target_amount = total_equity * target_pct
        
        actual_price = price * (1 + self.slippage_rate)
        max_volume = int(target_amount / actual_price / 100) * 100
        
        if max_volume <= 0:
            return
        
        amount = max_volume * actual_price
        commission = max(amount * self.commission_rate, 5)
        total_cost = amount + commission
        
        if total_cost > self.capital:
            max_volume = int(self.capital / actual_price / 100) * 100
            if max_volume <= 0:
                return
            amount = max_volume * actual_price
            commission = max(amount * self.commission_rate, 5)
            total_cost = amount + commission
        
        self.capital -= total_cost
        
        if code in self.positions:
            pos = self.positions[code]
            total_volume = pos.volume + max_volume
            pos.avg_cost = (pos.avg_cost * pos.volume + actual_price * max_volume) / total_volume
            pos.volume = total_volume
        else:
            self.positions[code] = Position(
                code=code,
                volume=max_volume,
                avg_cost=actual_price,
                current_price=actual_price,
                market_value=max_volume * actual_price
            )
        
        self.trades.append(Trade(
            date=date, code=code, action='buy', price=actual_price,
            volume=max_volume, amount=amount, commission=commission,
            slippage=(actual_price - price) * max_volume, reason=reason
        ))
    
    def _execute_sell(self, code: str, price: float, target_pct: float, date, reason: str):
        """执行卖出"""
        if code not in self.positions:
            return
        
        pos = self.positions[code]
        actual_price = price * (1 - self.slippage_rate)
        
        if target_pct >= 1:
            sell_volume = pos.volume
        else:
            sell_volume = int(pos.volume * target_pct / 100) * 100
        
        if sell_volume <= 0:
            return
        
        amount = sell_volume * actual_price
        commission = max(amount * self.commission_rate, 5)
        stamp_tax = amount * self.stamp_tax
        
        self.capital += amount - commission - stamp_tax
        
        if sell_volume >= pos.volume:
            del self.positions[code]
        else:
            pos.volume -= sell_volume
        
        self.trades.append(Trade(
            date=date, code=code, action='sell', price=actual_price,
            volume=sell_volume, amount=amount, commission=commission + stamp_tax,
            slippage=(price - actual_price) * sell_volume, reason=reason
        ))
    
    def _get_total_equity(self) -> float:
        """获取总权益"""
        position_value = sum(pos.market_value for pos in self.positions.values())
        return self.capital + position_value
    
    def _record_daily_equity(self, date):
        """记录每日权益"""
        total_equity = self._get_total_equity()
        self.daily_equity.append({
            'date': date,
            'equity': total_equity,
            'capital': self.capital,
            'position_value': total_equity - self.capital,
            'position_count': len(self.positions)
        })
    
    def _generate_report(self) -> Dict:
        """生成回测报告"""
        if not self.daily_equity:
            return {}
        
        equity_df = pd.DataFrame(self.daily_equity)
        
        initial_equity = self.initial_capital
        final_equity = equity_df.iloc[-1]['equity']
        total_return = (final_equity / initial_equity - 1) * 100
        
        equity_df['daily_return'] = equity_df['equity'].pct_change()
        
        equity_df['cummax'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax']
        max_drawdown = equity_df['drawdown'].min() * 100
        
        days = len(equity_df)
        annual_return = ((1 + total_return / 100) ** (252 / days) - 1) * 100 if days > 0 else 0
        
        daily_returns = equity_df['daily_return'].dropna()
        sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0
        
        return {
            'initial_capital': initial_equity,
            'final_equity': final_equity,
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
            'total_trades': len(self.trades),
            'buy_trades': len([t for t in self.trades if t.action == 'buy']),
            'sell_trades': len([t for t in self.trades if t.action == 'sell']),
            'total_commission': sum(t.commission for t in self.trades),
            'daily_equity': equity_df,
            'trades': self.trades
        }


class PerformanceMetrics:
    """绩效指标计算器"""
    
    @staticmethod
    def calculate_all(equity_df: pd.DataFrame, trades: List = None) -> Dict:
        """计算所有绩效指标"""
        metrics = {}
        
        metrics.update(PerformanceMetrics._calc_return_metrics(equity_df))
        metrics.update(PerformanceMetrics._calc_risk_metrics(equity_df))
        
        if trades:
            metrics.update(PerformanceMetrics._calc_trade_metrics(trades))
        
        return metrics
    
    @staticmethod
    def _calc_return_metrics(equity_df: pd.DataFrame) -> Dict:
        """计算收益指标"""
        initial = equity_df.iloc[0]['equity']
        final = equity_df.iloc[-1]['equity']
        
        total_return = (final / initial - 1) * 100
        days = len(equity_df)
        annual_return = ((1 + total_return / 100) ** (252 / days) - 1) * 100 if days > 0 else 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return
        }
    
    @staticmethod
    def _calc_risk_metrics(equity_df: pd.DataFrame) -> Dict:
        """计算风险指标"""
        equity_df = equity_df.copy()
        equity_df['daily_return'] = equity_df['equity'].pct_change()
        
        daily_returns = equity_df['daily_return'].dropna()
        
        equity_df['cummax'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax']
        max_drawdown = equity_df['drawdown'].min() * 100
        
        annual_volatility = daily_returns.std() * np.sqrt(252) * 100 if len(daily_returns) > 0 else 0
        sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0
        
        return {
            'max_drawdown': max_drawdown,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe
        }
    
    @staticmethod
    def _calc_trade_metrics(trades: List) -> Dict:
        """计算交易统计"""
        if not trades:
            return {}
        
        return {
            'total_trades': len(trades),
            'buy_trades': len([t for t in trades if t.action == 'buy']),
            'sell_trades': len([t for t in trades if t.action == 'sell']),
            'total_commission': sum(t.commission for t in trades)
        }
