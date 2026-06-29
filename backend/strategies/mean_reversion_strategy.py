"""均值回归策略"""

import pandas as pd
import numpy as np
from typing import Dict
from loguru import logger


class MeanReversionStrategy:
    """均值回归策略
    
    策略逻辑：
    - 价格偏离均值过多时反向操作
    - 价格回归均值时平仓
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.ma_period = self.config.get('ma_period', 20)
        self.std_multiplier = self.config.get('std_multiplier', 2.0)
        self.name = f"均值回归策略({self.ma_period}日)"
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号"""
        df = df.copy()
        
        # 计算布林带
        df['ma'] = df['close'].rolling(self.ma_period).mean()
        df['std'] = df['close'].rolling(self.ma_period).std()
        df['upper'] = df['ma'] + self.std_multiplier * df['std']
        df['lower'] = df['ma'] - self.std_multiplier * df['std']
        
        # 计算Z-Score
        df['zscore'] = (df['close'] - df['ma']) / df['std']
        
        # 生成信号
        df['signal'] = 0
        
        # 价格触及下轨且Z-Score < -2时买入（超卖）
        df.loc[
            (df['close'] < df['lower']) & 
            (df['zscore'] < -self.std_multiplier),
            'signal'
        ] = 1
        
        # 价格触及上轨且Z-Score > 2时卖出（超买）
        df.loc[
            (df['close'] > df['upper']) & 
            (df['zscore'] > self.std_multiplier),
            'signal'
        ] = -1
        
        return df
    
    def get_description(self) -> str:
        """获取策略描述"""
        return f"""
【{self.name}】

策略逻辑：
- 基于布林带和Z-Score判断超买超卖
- 价格跌破下轨（超卖）时买入
- 价格突破上轨（超买）时卖出

参数设置：
- 均线周期: {self.ma_period}日
- 标准差倍数: {self.std_multiplier}

适用场景：
- 震荡行情中表现较好
- 有明确支撑阻力的股票

风险提示：
- 趋势行情：在强趋势中可能逆势操作导致亏损
- 假突破：价格可能短暂突破后继续沿原方向运行
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
                buy_price = row['close']
                shares = int(capital * 0.95 / buy_price / 100) * 100
                if shares > 0:
                    cost = shares * buy_price * 1.0003
                    capital -= cost
                    position = shares
                    trades.append({'type': 'buy', 'price': buy_price, 'shares': shares})
            
            elif row['signal'] == -1 and position > 0:
                sell_price = row['close']
                revenue = position * sell_price * 0.9997
                capital += revenue
                trades.append({'type': 'sell', 'price': sell_price, 'shares': position})
                position = 0
            
            equity = capital + position * row['close']
            equity_curve.append(equity)
        
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
