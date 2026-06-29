"""组合级回测引擎 - 同时回测多只股票"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable
from loguru import logger
from dataclasses import dataclass, field


@dataclass
class PortfolioTrade:
    """组合交易记录"""
    date: str
    code: str
    action: str
    price: float
    shares: int
    amount: float
    commission: float = 0
    reason: str = ""


class PortfolioBacktest:
    """组合回测引擎 - 支持多股票同时回测"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.initial_capital = self.config.get('initial_capital', 1000000)
        self.commission_rate = self.config.get('commission_rate', 0.0003)
        self.slippage_rate = self.config.get('slippage_rate', 0.001)
        self.stamp_tax = self.config.get('stamp_tax', 0.001)
        self.max_positions = self.config.get('max_positions', 10)
        self.max_position_pct = self.config.get('max_position_pct', 0.2)
        
        self.reset()
    
    def reset(self):
        """重置状态"""
        self.capital = self.initial_capital
        self.positions: Dict[str, Dict] = {}  # {code: {shares, avg_cost}}
        self.trades: List[PortfolioTrade] = []
        self.daily_equity: List[Dict] = []
    
    def run(self, signal_func: Callable, price_data: Dict[str, pd.DataFrame],
            stock_weights: Dict[str, float] = None) -> Dict:
        """运行组合回测
        
        Args:
            signal_func: 信号生成函数 (date, positions, price_data) -> [{code, action, weight}]
            price_data: {code: DataFrame with date, close}
            stock_weights: {code: weight} 初始权重配置
        
        Returns:
            回测报告
        """
        logger.info(f"开始组合回测，股票数: {len(price_data)}")
        self.reset()
        
        # 获取所有日期
        all_dates = set()
        for df in price_data.values():
            if 'date' in df.columns:
                all_dates.update(df['date'].tolist())
        all_dates = sorted(list(all_dates))
        
        for date in all_dates:
            # 更新持仓市值
            self._update_positions(date, price_data)
            
            # 获取信号
            signals = signal_func(date, self.positions, price_data)
            
            # 执行信号
            if signals:
                self._execute_signals(signals, date, price_data)
            
            # 记录权益
            self._record_equity(date)
        
        return self._generate_report()
    
    def _update_positions(self, date, price_data: Dict[str, pd.DataFrame]):
        """更新持仓价格"""
        for code, pos in list(self.positions.items()):
            if code in price_data:
                df = price_data[code]
                row = df[df['date'] == date]
                if not row.empty:
                    price = float(row.iloc[0]['close'])
                    pos['current_price'] = price
                    pos['market_value'] = pos['shares'] * price
                    pos['profit'] = (price - pos['avg_cost']) * pos['shares']
                    pos['profit_pct'] = (price / pos['avg_cost'] - 1) * 100
    
    def _execute_signals(self, signals: List[Dict], date, price_data: Dict[str, pd.DataFrame]):
        """执行交易信号"""
        total_equity = self._get_total_equity()
        
        for signal in signals:
            code = signal.get('code')
            action = signal.get('action')
            weight = signal.get('weight', 0.1)
            reason = signal.get('reason', '')
            
            if code not in price_data:
                continue
            
            df = price_data[code]
            row = df[df['date'] == date]
            if row.empty:
                continue
            
            price = float(row.iloc[0]['close'])
            
            if action == 'buy':
                self._buy(code, price, weight, total_equity, date, reason)
            elif action == 'sell':
                self._sell(code, price, weight, date, reason)
    
    def _buy(self, code: str, price: float, weight: float, total_equity: float,
             date, reason: str):
        """组合买入"""
        # 检查持仓数量限制
        if len(self.positions) >= self.max_positions and code not in self.positions:
            return
        
        # 检查仓位限制
        current_pct = sum(p.get('market_value', 0) for p in self.positions.values()) / total_equity
        if current_pct + weight > self.max_position_pct * self.max_positions:
            weight = max(0, self.max_position_pct * self.max_positions - current_pct)
        
        target_amount = total_equity * weight
        actual_price = price * (1 + self.slippage_rate)
        shares = int(target_amount / actual_price / 100) * 100
        
        if shares <= 0:
            return
        
        amount = shares * actual_price
        commission = max(amount * self.commission_rate, 5)
        cost = amount + commission
        
        if cost > self.capital:
            shares = int(self.capital / actual_price / 100) * 100
            if shares <= 0:
                return
            amount = shares * actual_price
            commission = max(amount * self.commission_rate, 5)
            cost = amount + commission
        
        self.capital -= cost
        
        if code in self.positions:
            pos = self.positions[code]
            total_shares = pos['shares'] + shares
            pos['avg_cost'] = (pos['avg_cost'] * pos['shares'] + actual_price * shares) / total_shares
            pos['shares'] = total_shares
        else:
            self.positions[code] = {
                'shares': shares,
                'avg_cost': actual_price,
                'current_price': actual_price,
                'market_value': amount
            }
        
        self.trades.append(PortfolioTrade(
            date=date, code=code, action='buy', price=actual_price,
            shares=shares, amount=amount, commission=commission, reason=reason
        ))
    
    def _sell(self, code: str, price: float, weight: float, date, reason: str):
        """组合卖出"""
        if code not in self.positions:
            return
        
        pos = self.positions[code]
        actual_price = price * (1 - self.slippage_rate)
        
        if weight >= 1:
            sell_shares = pos['shares']
        else:
            sell_shares = int(pos['shares'] * weight / 100) * 100
        
        if sell_shares <= 0:
            return
        
        amount = sell_shares * actual_price
        commission = max(amount * self.commission_rate, 5)
        stamp_tax = amount * self.stamp_tax
        
        self.capital += amount - commission - stamp_tax
        
        if sell_shares >= pos['shares']:
            del self.positions[code]
        else:
            pos['shares'] -= sell_shares
        
        self.trades.append(PortfolioTrade(
            date=date, code=code, action='sell', price=actual_price,
            shares=sell_shares, amount=amount, commission=commission + stamp_tax, reason=reason
        ))
    
    def _get_total_equity(self) -> float:
        """获取总权益"""
        position_value = sum(p.get('market_value', 0) for p in self.positions.values())
        return self.capital + position_value
    
    def _record_equity(self, date):
        """记录每日权益"""
        self.daily_equity.append({
            'date': date,
            'equity': self._get_total_equity(),
            'capital': self.capital,
            'positions': len(self.positions)
        })
    
    def _generate_report(self) -> Dict:
        """生成回测报告"""
        if not self.daily_equity:
            return {}
        
        equity_df = pd.DataFrame(self.daily_equity)
        equity_df['daily_return'] = equity_df['equity'].pct_change()
        
        initial = self.initial_capital
        final = equity_df.iloc[-1]['equity']
        total_return = (final / initial - 1) * 100
        
        # 年化收益
        days = len(equity_df)
        annual_return = ((1 + total_return / 100) ** (252 / days) - 1) * 100
        
        # 夏普
        daily_ret = equity_df['daily_return'].dropna()
        sharpe = np.sqrt(252) * daily_ret.mean() / daily_ret.std() if daily_ret.std() > 0 else 0
        
        # 最大回撤
        equity_df['cummax'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax']
        max_dd = equity_df['drawdown'].min() * 100
        
        # 卡尔玛比率
        calmar = -annual_return / max_dd if max_dd != 0 else 0
        
        # 胜率
        buy_trades = [t for t in self.trades if t.action == 'buy']
        sell_trades = [t for t in self.trades if t.action == 'sell']
        
        # 交易统计
        code_trades = {}
        for t in self.trades:
            code_trades.setdefault(t.code, []).append(t)
        
        return {
            'initial_capital': initial,
            'final_equity': final,
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'calmar_ratio': calmar,
            'total_trades': len(self.trades),
            'buy_trades': len(buy_trades),
            'sell_trades': len(sell_trades),
            'total_commission': sum(t.commission for t in self.trades),
            'daily_equity': equity_df,
            'trades': self.trades,
            'code_stats': code_trades
        }
