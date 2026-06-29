"""动量策略"""

import pandas as pd
import numpy as np
from typing import Dict, List
from loguru import logger


class MomentumStrategy:
    """动量策略
    
    策略逻辑：
    - 选择过去N日涨幅最大的股票
    - 定期调仓（如每周/每月）
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.lookback_period = self.config.get('lookback_period', 20)
        self.holding_period = self.config.get('holding_period', 5)
        self.top_n = self.config.get('top_n', 3)
        self.name = f"动量策略({self.lookback_period}日)"
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号"""
        df = df.copy()
        
        # 计算动量（过去N日收益率）
        df['momentum'] = df['close'].pct_change(self.lookback_period)
        
        # 计算动量排名
        df['momentum_rank'] = df['momentum'].rank(ascending=False, pct=True)
        
        # 生成信号
        df['signal'] = 0
        
        # 动量排名前20%且动量为正时买入
        df.loc[
            (df['momentum_rank'] <= 0.2) & 
            (df['momentum'] > 0) &
            (df['momentum'].shift(1) <= 0),
            'signal'
        ] = 1
        
        # 动量转负时卖出
        df.loc[
            (df['momentum'] < 0) & 
            (df['momentum'].shift(1) >= 0),
            'signal'
        ] = -1
        
        return df
    
    def get_description(self) -> str:
        """获取策略描述"""
        return f"""
【{self.name}】

策略逻辑：
- 追踪过去{self.lookback_period}日的涨幅（动量）
- 动量转正且排名靠前时买入
- 动量转负时卖出

参数设置：
- 回看周期: {self.lookback_period}日
- 持仓周期: {self.holding_period}日
- 选股数量: {self.top_n}只

适用场景：
- 趋势延续性较强的市场
- 板块轮动行情

风险提示：
- 动量反转：趋势突然反转时可能大幅亏损
- 追高风险：可能在高位买入
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
