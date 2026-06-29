"""增强回测指标"""

import pandas as pd
import numpy as np
from typing import Dict, List
from loguru import logger


class EnhancedMetrics:
    """增强回测指标体系"""
    
    @staticmethod
    def calculate_all(equity_df: pd.DataFrame, trades: List = None,
                     benchmark_df: pd.DataFrame = None) -> Dict:
        """一站式计算所有增强指标"""
        metrics = {}
        
        equity_df = equity_df.copy()
        if 'daily_return' not in equity_df.columns:
            equity_df['daily_return'] = equity_df['equity'].pct_change()
        
        # 收益指标
        metrics.update(EnhancedMetrics.return_metrics(equity_df))
        
        # 风险指标
        metrics.update(EnhancedMetrics.risk_metrics(equity_df))
        
        # 交易指标
        if trades:
            metrics.update(EnhancedMetrics.trade_metrics(trades))
        
        # 基准对比
        if benchmark_df is not None:
            metrics.update(EnhancedMetrics.benchmark_metrics(equity_df, benchmark_df))
        
        return metrics
    
    @staticmethod
    def return_metrics(equity_df: pd.DataFrame) -> Dict:
        """收益指标体系"""
        initial = equity_df.iloc[0]['equity']
        final = equity_df.iloc[-1]['equity']
        total_return = (final / initial - 1) * 100
        
        days = len(equity_df)
        annual_return = ((1 + total_return / 100) ** (252 / days) - 1) * 100
        
        # 月度收益
        equity_df = equity_df.copy()
        equity_df['date'] = pd.to_datetime(equity_df['date'])
        equity_df['month'] = equity_df['date'].dt.to_period('M')
        
        monthly_equity = equity_df.groupby('month')['equity'].last()
        monthly_returns = monthly_equity.pct_change().dropna()
        
        # 年度收益
        equity_df['year'] = equity_df['date'].dt.year
        yearly_equity = equity_df.groupby('year')['equity'].last()
        yearly_returns = yearly_equity.pct_change().dropna()
        
        # 正收益月份比例
        positive_months = (monthly_returns > 0).sum()
        total_months = len(monthly_returns)
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'monthly_return_mean': monthly_returns.mean() * 100 if total_months > 0 else 0,
            'monthly_return_std': monthly_returns.std() * 100 if total_months > 0 else 0,
            'monthly_win_rate': positive_months / total_months * 100 if total_months > 0 else 0,
            'best_month': monthly_returns.max() * 100 if total_months > 0 else 0,
            'worst_month': monthly_returns.min() * 100 if total_months > 0 else 0,
            'positive_months': int(positive_months),
            'total_months': int(total_months),
            'yearly_returns': yearly_returns.to_dict() if len(yearly_returns) > 0 else {},
            'monthly_returns_list': monthly_returns.tolist() if total_months > 0 else []
        }
    
    @staticmethod
    def risk_metrics(equity_df: pd.DataFrame) -> Dict:
        """风险指标体系"""
        daily_ret = equity_df['daily_return'].dropna()
        
        if len(daily_ret) == 0:
            return {}
        
        # 波动率
        annual_vol = daily_ret.std() * np.sqrt(252) * 100
        
        # 夏普比率
        sharpe = np.sqrt(252) * daily_ret.mean() / daily_ret.std() if daily_ret.std() > 0 else 0
        
        # 索提诺比率
        downside = daily_ret[daily_ret < 0]
        sortino = np.sqrt(252) * daily_ret.mean() / downside.std() if len(downside) > 0 and downside.std() > 0 else 0
        
        # 最大回撤
        equity_df = equity_df.copy()
        equity_df['cummax'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax']
        max_dd = equity_df['drawdown'].min() * 100
        max_dd_date = equity_df.loc[equity_df['drawdown'].idxmin(), 'date'] if len(equity_df) > 0 else None
        
        # 回撤恢复天数
        dd_end = equity_df['drawdown'].idxmin()
        equity_after_dd = equity_df.loc[dd_end:]
        recovery = equity_after_dd[equity_after_dd['drawdown'] >= -0.01]
        recovery_days = len(recovery) if len(recovery) > 0 else 0
        
        # 卡尔玛比率
        annual_return = ((1 + daily_ret.mean()) ** 252 - 1) * 100
        calmar = -annual_return / max_dd if max_dd != 0 else 0
        
        # SQN (System Quality Number)
        trades_pnl = daily_ret[daily_ret != 0]
        if len(trades_pnl) > 0:
            sqn = np.sqrt(len(trades_pnl)) * trades_pnl.mean() / trades_pnl.std() if trades_pnl.std() > 0 else 0
        else:
            sqn = 0
        
        # VaR (95%)
        var_95 = np.percentile(daily_ret, 5) * 100
        
        # CVaR (95%)
        cvar_95 = daily_ret[daily_ret <= np.percentile(daily_ret, 5)].mean() * 100
        
        # 最大连胜/连亏
        streak_pnl = np.sign(daily_ret.values)
        max_win_streak = 0
        max_lose_streak = 0
        current_win = 0
        current_lose = 0
        
        for s in streak_pnl:
            if s > 0:
                current_win += 1
                current_lose = 0
                max_win_streak = max(max_win_streak, current_win)
            elif s < 0:
                current_lose += 1
                current_win = 0
                max_lose_streak = max(max_lose_streak, current_lose)
        
        return {
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_dd,
            'max_drawdown_date': str(max_dd_date),
            'recovery_days': int(recovery_days),
            'calmar_ratio': calmar,
            'sqn': sqn,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'max_win_streak': max_win_streak,
            'max_lose_streak': max_lose_streak
        }
    
    @staticmethod
    def trade_metrics(trades: List) -> Dict:
        """交易统计指标"""
        if not trades:
            return {'total_trades': 0}
        
        buy_trades = [t for t in trades if t.action == 'buy']
        sell_trades = [t for t in trades if t.action == 'sell']
        
        # 配对交易
        paired = []
        remaining_buys = list(buy_trades)
        
        for sell in sell_trades:
            if remaining_buys:
                buy = remaining_buys.pop(0)
                profit = (sell.price - buy.price) * sell.shares
                profit_pct = (sell.price / buy.price - 1) * 100
                paired.append({
                    'code': sell.code,
                    'profit': profit,
                    'profit_pct': profit_pct,
                    'buy_date': str(buy.date),
                    'sell_date': str(sell.date)
                })
        
        if not paired:
            return {
                'total_trades': len(trades),
                'buy_count': len(buy_trades),
                'sell_count': len(sell_trades),
                'win_rate': 0
            }
        
        profits = [p['profit'] for p in paired]
        profit_pcts = [p['profit_pct'] for p in paired]
        wins = [p for p in profits if p > 0]
        losses = [p for p in profits if p < 0]
        
        return {
            'total_trades': len(trades),
            'buy_count': len(buy_trades),
            'sell_count': len(sell_trades),
            'paired_trades': len(paired),
            'win_count': len(wins),
            'lose_count': len(losses),
            'win_rate': len(wins) / len(paired) * 100 if paired else 0,
            'avg_profit': np.mean(profits) if profits else 0,
            'avg_profit_pct': np.mean(profit_pcts) if profit_pcts else 0,
            'avg_win': np.mean(wins) if wins else 0,
            'avg_loss': np.mean(losses) if losses else 0,
            'max_win': max(profits) if profits else 0,
            'max_loss': min(profits) if profits else 0,
            'profit_factor': abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else 0,
            'total_commission': sum(t.commission for t in trades)
        }
    
    @staticmethod
    def benchmark_metrics(equity_df: pd.DataFrame, benchmark_df: pd.DataFrame) -> Dict:
        """基准对比指标"""
        benchmark_df = benchmark_df.copy()
        benchmark_df['bench_return'] = benchmark_df['close'].pct_change()
        
        merged = equity_df[['date', 'daily_return']].copy()
        merged['date'] = pd.to_datetime(merged['date'])
        benchmark_df['date'] = pd.to_datetime(benchmark_df['date'])
        merged = merged.merge(benchmark_df[['date', 'bench_return']], on='date', how='inner')
        merged = merged.dropna()
        
        if len(merged) == 0:
            return {}
        
        # 超额收益
        merged['excess_return'] = merged['daily_return'] - merged['bench_return']
        excess_annual = merged['excess_return'].mean() * 252 * 100
        
        # 信息比率
        ir = merged['excess_return'].mean() / merged['excess_return'].std() * np.sqrt(252) if merged['excess_return'].std() > 0 else 0
        
        # Beta
        cov = merged[['daily_return', 'bench_return']].cov()
        beta = cov.iloc[0, 1] / cov.iloc[1, 1] if cov.iloc[1, 1] > 0 else 1
        
        # Alpha (年化)
        alpha = (merged['daily_return'].mean() - beta * merged['bench_return'].mean()) * 252 * 100
        
        # 相关性
        corr = merged['daily_return'].corr(merged['bench_return'])
        
        return {
            'excess_return': excess_annual,
            'information_ratio': ir,
            'beta': beta,
            'alpha': alpha,
            'benchmark_correlation': corr
        }
