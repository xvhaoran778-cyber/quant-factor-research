"""因子策略转换器 - 将因子转换为可回测的策略"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from loguru import logger


class FactorStrategy:
    """因子策略 - 基于单因子或多因子生成交易信号"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.factor_name = self.config.get('factor_name', '')
        self.factor_direction = self.config.get('direction', 'positive')
        self.threshold_buy = self.config.get('threshold_buy', 0.7)
        self.threshold_sell = self.config.get('threshold_sell', 0.3)
        self.name = f"因子策略: {self.factor_name}"
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """基于因子生成交易信号"""
        df = df.copy()
        
        if self.factor_name not in df.columns:
            logger.warning(f"因子 {self.factor_name} 不存在")
            df['signal'] = 0
            return df
        
        factor_values = df[self.factor_name].dropna()
        
        if len(factor_values) == 0:
            df['signal'] = 0
            return df
        
        window = min(60, len(factor_values))
        df['factor_rank'] = df[self.factor_name].rolling(window).rank(pct=True)
        
        df['signal'] = 0
        
        if self.factor_direction == 'positive':
            df.loc[df['factor_rank'] > self.threshold_buy, 'signal'] = 1
            df.loc[df['factor_rank'] < self.threshold_sell, 'signal'] = -1
        else:
            df.loc[df['factor_rank'] < (1 - self.threshold_buy), 'signal'] = 1
            df.loc[df['factor_rank'] > (1 - self.threshold_sell), 'signal'] = -1
        
        return df
    
    def get_description(self) -> str:
        """获取策略描述"""
        direction_text = "因子值越大越好" if self.factor_direction == 'positive' else "因子值越小越好"
        
        return f"""
【{self.name}】

策略逻辑：
- 基于因子 {self.factor_name} 的排名生成交易信号
- {direction_text}
- 因子排名高于 {self.threshold_buy*100:.0f}% 时买入
- 因子排名低于 {self.threshold_sell*100:.0f}% 时卖出

参数设置：
- 因子名称: {self.factor_name}
- 因子方向: {self.factor_direction}
- 买入阈值: {self.threshold_buy*100:.0f}%
- 卖出阈值: {self.threshold_sell*100:.0f}%

适用场景：
- 因子有效性验证
- 单因子策略回测
- 因子组合优化
"""
    
    def backtest(self, df: pd.DataFrame, initial_capital: float = 1000000) -> Dict:
        """回测策略"""
        df = self.generate_signals(df)
        
        capital = initial_capital
        position = 0
        trades = []
        equity_curve = []
        
        for i, (idx, row) in enumerate(df.iterrows()):
            if row['signal'] == 1 and position == 0:
                buy_price = row['close']
                shares = int(capital * 0.95 / buy_price / 100) * 100
                if shares > 0:
                    cost = shares * buy_price * 1.0003
                    capital -= cost
                    position = shares
                    trades.append({'type': 'buy', 'price': buy_price, 'shares': shares, 'idx': i,
                                  'date': str(row.get('date', i))})
            
            elif row['signal'] == -1 and position > 0:
                sell_price = row['close']
                revenue = position * sell_price * 0.9997
                capital += revenue
                trades.append({'type': 'sell', 'price': sell_price, 'shares': position, 'idx': i,
                              'date': str(row.get('date', i))})
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


class MultiFactorStrategy:
    """多因子策略 - 基于多个因子的加权得分生成交易信号"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.factors = self.config.get('factors', {})  # {factor_name: weight}
        self.threshold_buy = self.config.get('threshold_buy', 0.7)
        self.threshold_sell = self.config.get('threshold_sell', 0.3)
        self.name = "多因子策略"
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """基于多因子加权得分生成交易信号"""
        df = df.copy()
        
        # 检查因子是否存在
        available_factors = {k: v for k, v in self.factors.items() if k in df.columns}
        
        if not available_factors:
            logger.warning("没有可用的因子")
            df['signal'] = 0
            return df
        
        # 计算每个因子的排名
        factor_ranks = pd.DataFrame()
        for factor_name, weight in available_factors.items():
            window = min(60, len(df))
            factor_ranks[factor_name] = df[factor_name].rolling(window).rank(pct=True)
        
        # 计算加权得分
        total_weight = sum(available_factors.values())
        df['composite_score'] = 0
        
        for factor_name, weight in available_factors.items():
            df['composite_score'] += factor_ranks[factor_name] * (weight / total_weight)
        
        # 生成信号
        df['signal'] = 0
        df.loc[df['composite_score'] > self.threshold_buy, 'signal'] = 1
        df.loc[df['composite_score'] < self.threshold_sell, 'signal'] = -1
        
        return df
    
    def get_description(self) -> str:
        """获取策略描述"""
        factors_text = "\n".join([f"- {name}: 权重 {weight:.2f}" for name, weight in self.factors.items()])
        
        return f"""
【{self.name}】

策略逻辑：
- 基于多个因子的加权综合得分
- 综合得分高于 {self.threshold_buy*100:.0f}% 时买入
- 综合得分低于 {self.threshold_sell*100:.0f}% 时卖出

因子权重：
{factors_text}

适用场景：
- 多因子选股
- 因子组合优化
- 降低单因子风险
"""
    
    def backtest(self, df: pd.DataFrame, initial_capital: float = 1000000) -> Dict:
        """回测策略"""
        df = self.generate_signals(df)
        
        capital = initial_capital
        position = 0
        trades = []
        equity_curve = []
        
        for i, (idx, row) in enumerate(df.iterrows()):
            if row['signal'] == 1 and position == 0:
                buy_price = row['close']
                shares = int(capital * 0.95 / buy_price / 100) * 100
                if shares > 0:
                    cost = shares * buy_price * 1.0003
                    capital -= cost
                    position = shares
                    trades.append({'type': 'buy', 'price': buy_price, 'shares': shares, 'idx': i,
                                  'date': str(row.get('date', i))})
            
            elif row['signal'] == -1 and position > 0:
                sell_price = row['close']
                revenue = position * sell_price * 0.9997
                capital += revenue
                trades.append({'type': 'sell', 'price': sell_price, 'shares': position, 'idx': i,
                              'date': str(row.get('date', i))})
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


class FactorStrategyConverter:
    """因子策略转换器 - 将IC分析结果或挖掘出的因子转换为策略"""
    
    def __init__(self):
        self.alpha158_factors = [
            'alpha_return_5d', 'alpha_return_10d', 'alpha_return_20d',
            'alpha_volatility_5d', 'alpha_volatility_20d',
            'alpha_rsi_6', 'alpha_rsi_14',
            'alpha_volume_ratio_5d', 'alpha_volume_ratio_10d',
            'alpha_macd_hist', 'alpha_macd_dif',
            'alpha_kdj_k', 'alpha_kdj_d', 'alpha_kdj_j',
            'alpha_boll_position', 'alpha_boll_width',
            'alpha_price_ma_ratio_5', 'alpha_price_ma_ratio_20'
        ]
        
        # 因子方向（正向=越大越好，负向=越小越好）
        self.factor_directions = {
            'alpha_return_5d': 'positive',
            'alpha_return_10d': 'positive',
            'alpha_return_20d': 'positive',
            'alpha_volatility_5d': 'negative',  # 低波动更好
            'alpha_volatility_20d': 'negative',
            'alpha_rsi_6': 'negative',  # 超卖更好
            'alpha_rsi_14': 'negative',
            'alpha_volume_ratio_5d': 'positive',
            'alpha_volume_ratio_10d': 'positive',
            'alpha_macd_hist': 'positive',
            'alpha_macd_dif': 'positive',
            'alpha_kdj_k': 'negative',
            'alpha_kdj_d': 'negative',
            'alpha_kdj_j': 'negative',
            'alpha_boll_position': 'negative',  # 低位更好
            'alpha_boll_width': 'negative',
            'alpha_price_ma_ratio_5': 'positive',
            'alpha_price_ma_ratio_20': 'positive'
        }
    
    def convert_single_factor(self, factor_name: str, direction: str = None) -> FactorStrategy:
        """将单个因子转换为策略"""
        if direction is None:
            direction = self.factor_directions.get(factor_name, 'positive')
        
        return FactorStrategy({
            'factor_name': factor_name,
            'direction': direction,
            'threshold_buy': 0.7,
            'threshold_sell': 0.3
        })
    
    def convert_top_factors(self, ic_results: pd.DataFrame, top_n: int = 5) -> MultiFactorStrategy:
        """将IC排名靠前的因子转换为多因子策略"""
        # 获取有效因子
        valid_factors = ic_results[ic_results['有效性'] == '有效'].head(top_n)
        
        if valid_factors.empty:
            logger.warning("没有有效因子")
            return MultiFactorStrategy()
        
        # 构建因子权重（基于IC绝对值）
        factors = {}
        for _, row in valid_factors.iterrows():
            factor_name = row['因子']
            ic_value = abs(row['IC值'])
            factors[factor_name] = ic_value
        
        return MultiFactorStrategy({
            'factors': factors,
            'threshold_buy': 0.7,
            'threshold_sell': 0.3
        })
    
    def convert_mined_factor(self, factor_expression: str, factor_name: str = None) -> FactorStrategy:
        """将挖掘出的因子转换为策略"""
        if factor_name is None:
            factor_name = f"mined_{hash(factor_expression) % 10000}"
        
        return FactorStrategy({
            'factor_name': factor_name,
            'direction': 'positive',  # 默认正向
            'threshold_buy': 0.7,
            'threshold_sell': 0.3
        })
    
    def get_all_factor_strategies(self) -> List[Dict]:
        """获取所有可用的因子策略"""
        strategies = []
        
        for factor_name in self.alpha158_factors:
            direction = self.factor_directions.get(factor_name, 'positive')
            strategies.append({
                'name': factor_name,
                'direction': direction,
                'description': f"基于{factor_name}的单因子策略"
            })
        
        return strategies
