"""均线交叉策略"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from loguru import logger


class MACrossStrategy:
    """均线交叉策略
    
    策略逻辑：
    - 短期均线上穿长期均线时买入
    - 短期均线下穿长期均线时卖出
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.short_period = self.config.get('short_period', 5)
        self.long_period = self.config.get('long_period', 20)
        self.name = f"MA{self.short_period}_MA{self.long_period}交叉策略"
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号
        
        Args:
            df: 包含OHLCV数据的DataFrame
        
        Returns:
            添加了signal列的DataFrame (1=买入, -1=卖出, 0=持有)
        """
        df = df.copy()
        
        # 计算均线
        df['ma_short'] = df['close'].rolling(self.short_period).mean()
        df['ma_long'] = df['close'].rolling(self.long_period).mean()
        
        # 生成信号
        df['signal'] = 0
        
        # 金叉：短期均线上穿长期均线
        df.loc[
            (df['ma_short'] > df['ma_long']) & 
            (df['ma_short'].shift(1) <= df['ma_long'].shift(1)),
            'signal'
        ] = 1
        
        # 死叉：短期均线下穿长期均线
        df.loc[
            (df['ma_short'] < df['ma_long']) & 
            (df['ma_short'].shift(1) >= df['ma_long'].shift(1)),
            'signal'
        ] = -1
        
        return df
    
    def get_description(self) -> str:
        """获取策略描述"""
        return f"""
【{self.name}】

策略逻辑：
- 当MA{self.short_period}上穿MA{self.long_period}时，产生买入信号
- 当MA{self.short_period}下穿MA{self.long_period}时，产生卖出信号

参数设置：
- 短期均线周期: {self.short_period}日
- 长期均线周期: {self.long_period}日

适用场景：
- 趋势行情中表现较好
- 震荡行情中可能产生较多假信号

风险提示：
- 滞后性：均线是滞后指标，可能错过最佳买卖点
- 震荡市：在横盘震荡时可能频繁止损
"""
    
    def backtest(self, df: pd.DataFrame, initial_capital: float = 1000000) -> Dict:
        """回测策略"""
        df = self.generate_signals(df)
        
        capital = initial_capital
        position = 0
        trades = []
        equity_curve = []
        
        for i, row in df.iterrows():
            if row['signal'] == 1 and position == 0:
                # 买入
                buy_price = row['close']
                shares = int(capital * 0.95 / buy_price / 100) * 100
                if shares > 0:
                    cost = shares * buy_price * 1.0003  # 手续费
                    capital -= cost
                    position = shares
                    trades.append({'type': 'buy', 'price': buy_price, 'shares': shares, 'date': row.get('date', i)})
            
            elif row['signal'] == -1 and position > 0:
                # 卖出
                sell_price = row['close']
                revenue = position * sell_price * 0.9997  # 手续费
                capital += revenue
                trades.append({'type': 'sell', 'price': sell_price, 'shares': position, 'date': row.get('date', i)})
                position = 0
            
            # 记录权益
            equity = capital + position * row['close']
            equity_curve.append(equity)
        
        # 计算绩效指标
        equity_series = pd.Series(equity_curve)
        total_return = (equity_series.iloc[-1] / initial_capital - 1) * 100
        
        return {
            'strategy_name': self.name,
            'initial_capital': initial_capital,
            'final_equity': equity_series.iloc[-1],
            'total_return': total_return,
            'total_trades': len(trades),
            'trades': trades,
            'equity_curve': equity_curve
        }
