"""多因子策略"""

import pandas as pd
import numpy as np
from typing import Dict, List
from loguru import logger


class MultiFactorStrategy:
    """多因子策略
    
    策略逻辑：
    - 综合多个因子评分选股
    - 定期调仓
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.momentum_weight = self.config.get('momentum_weight', 0.3)
        self.value_weight = self.config.get('value_weight', 0.3)
        self.quality_weight = self.config.get('quality_weight', 0.2)
        self.volatility_weight = self.config.get('volatility_weight', 0.2)
        self.name = "多因子策略"
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号"""
        df = df.copy()
        
        # 计算各因子
        df['momentum'] = df['close'].pct_change(20)  # 20日动量
        df['volatility'] = df['close'].pct_change().rolling(20).std()  # 20日波动率
        df['rsi'] = self._calc_rsi(df['close'], 14)  # RSI
        
        # 计算综合评分
        df['momentum_score'] = df['momentum'].rank(ascending=True, pct=True)
        df['volatility_score'] = 1 - df['volatility'].rank(ascending=True, pct=True)  # 低波动得分高
        df['rsi_score'] = np.where(df['rsi'] < 30, 1, np.where(df['rsi'] > 70, 0, 0.5))  # RSI超卖得分高
        
        # 综合得分
        df['composite_score'] = (
            df['momentum_score'] * self.momentum_weight +
            df['volatility_score'] * self.volatility_weight +
            df['rsi_score'] * (self.value_weight + self.quality_weight)
        )
        
        # 生成信号
        df['signal'] = 0
        
        # 综合得分高且动量转正时买入
        df.loc[
            (df['composite_score'] > 0.7) & 
            (df['momentum'] > 0) &
            (df['momentum'].shift(1) <= 0),
            'signal'
        ] = 1
        
        # 综合得分低或动量转负时卖出
        df.loc[
            ((df['composite_score'] < 0.3) | (df['momentum'] < -0.05)) &
            (df['momentum'].shift(1) >= 0),
            'signal'
        ] = -1
        
        return df
    
    def _calc_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))
    
    def get_description(self) -> str:
        """获取策略描述"""
        return f"""
【{self.name}】

策略逻辑：
- 综合动量、波动率、RSI等多个因子
- 因子加权评分，选择得分最高的股票
- 定期调仓

因子权重：
- 动量因子: {self.momentum_weight*100:.0f}%
- 波动率因子: {self.volatility_weight*100:.0f}%
- 价值因子: {self.value_weight*100:.0f}%
- 质量因子: {self.quality_weight*100:.0f}%

适用场景：
- 多因子选股
- 中长期持有

风险提示：
- 因子失效：市场风格切换时因子可能失效
- 过拟合：历史表现不代表未来
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
