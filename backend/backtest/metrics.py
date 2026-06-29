"""Performance Metrics - 绩效指标计算"""

import pandas as pd
import numpy as np
from typing import Dict, List
from loguru import logger


class PerformanceMetrics:
    """绩效指标计算器"""
    
    @staticmethod
    def calculate_all(equity_df: pd.DataFrame, trades: List = None) -> Dict:
        """计算所有绩效指标"""
        metrics = {}
        
        # 基础收益指标
        metrics.update(PerformanceMetrics._calc_return_metrics(equity_df))
        
        # 风险指标
        metrics.update(PerformanceMetrics._calc_risk_metrics(equity_df))
        
        # 交易统计
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
        
        # 月度收益
        equity_df['month'] = pd.to_datetime(equity_df['date']).dt.to_period('M')
        monthly_returns = equity_df.groupby('month')['equity'].last().pct_change().dropna()
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'monthly_return_mean': monthly_returns.mean() * 100 if len(monthly_returns) > 0 else 0,
            'monthly_return_std': monthly_returns.std() * 100 if len(monthly_returns) > 0 else 0,
            'best_month': monthly_returns.max() * 100 if len(monthly_returns) > 0 else 0,
            'worst_month': monthly_returns.min() * 100 if len(monthly_returns) > 0 else 0,
            'positive_months': (monthly_returns > 0).sum() if len(monthly_returns) > 0 else 0,
            'negative_months': (monthly_returns < 0).sum() if len(monthly_returns) > 0 else 0
        }
    
    @staticmethod
    def _calc_risk_metrics(equity_df: pd.DataFrame) -> Dict:
        """计算风险指标"""
        equity_df = equity_df.copy()
        equity_df['daily_return'] = equity_df['equity'].pct_change()
        
        daily_returns = equity_df['daily_return'].dropna()
        
        # 最大回撤
        equity_df['cummax'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax']
        max_drawdown = equity_df['drawdown'].min() * 100
        
        # 波动率
        annual_volatility = daily_returns.std() * np.sqrt(252) * 100 if len(daily_returns) > 0 else 0
        
        # 夏普比率
        sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0
        
        # 索提诺比率
        downside_returns = daily_returns[daily_returns < 0]
        sortino = np.sqrt(252) * daily_returns.mean() / downside_returns.std() if len(downside_returns) > 0 and downside_returns.std() > 0 else 0
        
        # 卡尔马比率
        calmar = -annual_return / max_drawdown if max_drawdown != 0 else 0
        annual_return = ((1 + daily_returns.mean()) ** 252 - 1) * 100 if len(daily_returns) > 0 else 0
        
        return {
            'max_drawdown': max_drawdown,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'downside_deviation': downside_returns.std() * np.sqrt(252) * 100 if len(downside_returns) > 0 else 0
        }
    
    @staticmethod
    def _calc_trade_metrics(trades: List) -> Dict:
        """计算交易统计"""
        if not trades:
            return {}
        
        buy_trades = [t for t in trades if t.action == 'buy']
        sell_trades = [t for t in trades if t.action == 'sell']
        
        # 配对交易计算盈亏
        paired_trades = []
        for sell in sell_trades:
            # 找到对应的买入
            buy = next((b for b in buy_trades if b.code == sell.code), None)
            if buy:
                profit = (sell.price - buy.price) * sell.volume - sell.commission - buy.commission
                profit_pct = (sell.price / buy.price - 1) * 100
                paired_trades.append({
                    'code': sell.code,
                    'buy_price': buy.price,
                    'sell_price': sell.price,
                    'volume': sell.volume,
                    'profit': profit,
                    'profit_pct': profit_pct,
                    'hold_days': (sell.date - buy.date).days if hasattr(sell.date, 'days') else 0
                })
        
        if not paired_trades:
            return {
                'total_trades': len(trades),
                'buy_trades': len(buy_trades),
                'sell_trades': len(sell_trades)
            }
        
        profits = [t['profit'] for t in paired_trades]
        profit_pcts = [t['profit_pct'] for t in paired_trades]
        
        winning_trades = [p for p in profits if p > 0]
        losing_trades = [p for p in profits if p < 0]
        
        return {
            'total_trades': len(trades),
            'buy_trades': len(buy_trades),
            'sell_trades': len(sell_trades),
            'paired_trades': len(paired_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(paired_trades) * 100 if paired_trades else 0,
            'avg_profit': np.mean(profits) if profits else 0,
            'avg_profit_pct': np.mean(profit_pcts) if profit_pcts else 0,
            'avg_winning': np.mean(winning_trades) if winning_trades else 0,
            'avg_losing': np.mean(losing_trades) if losing_trades else 0,
            'profit_factor': abs(sum(winning_trades) / sum(losing_trades)) if losing_trades else 0,
            'total_commission': sum(t.commission for t in trades),
            'total_slippage': sum(t.slippage for t in trades)
        }
