#!/usr/bin/env python3
"""
因子潜力评估 - 对所有因子进行非 agent 回测

评估维度：
1. IC 分析：IC 均值、IC_IR、IC 稳定性
2. 分组收益：多空收益、单调性
3. 策略回测：单因子策略的收益、夏普、回撤
4. 综合评分：多维度加权评分
"""

import sys
import os
from pathlib import Path
import time
from typing import Optional
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent
backend_root = project_root / "backend"
sys.path.insert(0, str(backend_root))

import pandas as pd
import numpy as np

from app.services.market_store import ParquetMarketStore
from app.services.new_factors_fast import compute_factors_vectorized, FACTOR_REGISTRY_VECTORIZED
from app.services.factor_ic_analysis import calculate_ic_series, calculate_ic_summary
from app.services.research_backtest import run_scored_backtest, build_weekly_feature_panel
from app.services.strategies.v8_canonical import v8_candidate_filter, merge_alpha_132


def run_factor_evaluation(
    data_dir: str = "/Volumes/xhrrrrr_macmini副盘/quantlab/market",
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    max_stocks: Optional[int] = None,
    output_file: str = None
):
    """
    运行因子潜力评估
    
    Args:
        data_dir: 数据目录
        start_date: 开始日期
        end_date: 结束日期
        max_stocks: 最大股票数量（None 表示不限制）
        output_file: 输出文件路径
    """
    print("=" * 70)
    print("因子潜力评估（非 Agent 回测）")
    print("=" * 70)
    print(f"数据目录: {data_dir}")
    print(f"实验区间: {start_date} 至 {end_date}")
    print(f"股票数量: {'全部' if max_stocks is None else max_stocks}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    start_time = time.time()
    
    # 1. 加载数据
    print("步骤 1/4: 加载数据...")
    store = ParquetMarketStore(data_dir)
    start_dt = pd.to_datetime(start_date).date()
    end_dt = pd.to_datetime(end_date).date()
    
    panel = build_weekly_feature_panel(store, start_dt, end_dt)
    panel_with_alpha132 = merge_alpha_132(panel)
    print(f"  面板数据: {len(panel)} 行")
    
    # 2. 准备因子计算数据
    print("\n步骤 2/4: 准备因子计算数据...")
    all_daily = store.read(start_dt, end_dt, symbols=None, fill_suspensions=False)
    
    if all_daily.empty:
        print("错误: 无法加载日线数据")
        return
    
    print(f"  日线数据: {len(all_daily)} 行, {all_daily['symbol'].nunique()} 只股票")
    
    # 构建透视表
    close_prices = all_daily.pivot(index='date', columns='symbol', values='close')
    open_prices = all_daily.pivot(index='date', columns='symbol', values='open')
    high_prices = all_daily.pivot(index='date', columns='symbol', values='high')
    low_prices = all_daily.pivot(index='date', columns='symbol', values='low')
    volume = all_daily.pivot(index='date', columns='symbol', values='volume')
    amount = all_daily.pivot(index='date', columns='symbol', values='amount')
    
    returns = close_prices.pct_change()
    market_returns = returns.mean(axis=1)
    high_250d = close_prices.rolling(250).max()
    prev_close = close_prices.shift(1)
    
    # 生成行业收益率
    symbols = close_prices.columns.tolist()
    n_industries = 3
    industry_returns = pd.DataFrame(index=close_prices.index)
    for i in range(n_industries):
        industry_col = symbols[i::n_industries]
        industry_returns[f'INDUSTRY_{i}'] = returns[industry_col].mean(axis=1)
    
    factor_input_data = {
        'close': close_prices,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'volume': volume,
        'amount': amount,
        'returns': returns,
        'market_returns': market_returns,
        'industry_returns': industry_returns,
        'high_250d': high_250d,
        'prev_close': prev_close,
    }
    
    # 3. 计算因子
    print("\n步骤 3/4: 计算因子...")
    factor_df = compute_factors_vectorized(
        factor_input_data,
        factor_names=list(FACTOR_REGISTRY_VECTORIZED.keys()),
        max_stocks=max_stocks
    )
    
    # 将因子合并到面板
    factor_df_reset = factor_df.reset_index()
    factor_df_reset['date'] = pd.to_datetime(factor_df_reset['date'])
    panel = panel.merge(factor_df_reset, on=['date', 'symbol'], how='left')
    
    print(f"  面板现在有 {len(panel.columns)} 列")
    
    # 4. 评估每个因子
    print("\n步骤 4/4: 评估因子...")
    
    # 计算前瞻收益率
    close_prices_panel = panel.pivot(index='date', columns='symbol', values='close')
    forward_returns = close_prices_panel.pct_change(5).shift(-5)
    
    factor_results = []
    
    for factor_name in FACTOR_REGISTRY_VECTORIZED.keys():
        if factor_name not in panel.columns:
            continue
        
        print(f"\n  评估因子: {factor_name}")
        
        factor_info = FACTOR_REGISTRY_VECTORIZED[factor_name]
        factor_direction = factor_info.get('direction', 'positive')
        
        # 4.1 IC 分析
        factor_values = panel.pivot(index='date', columns='symbol', values=factor_name)
        ic_series = calculate_ic_series(factor_values, forward_returns)
        ic_summary = calculate_ic_summary(ic_series)
        
        # 4.2 分组收益分析
        group_returns = calculate_group_returns(factor_values, forward_returns, n_groups=5)
        
        # 计算多空收益和单调性
        if not group_returns.empty and 1 in group_returns.index and 5 in group_returns.index:
            long_return = group_returns.loc[5, 'avg_return']
            short_return = group_returns.loc[1, 'avg_return']
            long_short_return = long_return - short_return
            
            # 检查单调性（5组收益是否单调递增或递减）
            group_avg = group_returns.loc[1:5, 'avg_return'].values
            is_monotonic = (
                all(group_avg[i] <= group_avg[i+1] for i in range(len(group_avg)-1)) or
                all(group_avg[i] >= group_avg[i+1] for i in range(len(group_avg)-1))
            )
        else:
            long_short_return = 0
            is_monotonic = False
        
        # 4.3 单因子策略回测
        def single_factor_scorer(panel_with_factor: pd.DataFrame) -> pd.Series:
            if factor_name not in panel_with_factor.columns:
                return pd.Series(0.0, index=panel_with_factor.index)
            
            factor_val = panel_with_factor[factor_name].copy()
            factor_val = factor_val.fillna(factor_val.median())
            factor_rank = factor_val.rank(pct=True)
            
            # 根据因子方向调整
            if factor_direction == 'negative':
                factor_rank = 1 - factor_rank
            
            return factor_rank
        
        try:
            # 确保面板包含因子列
            if factor_name not in panel.columns:
                print(f"    警告: 因子 {factor_name} 不在面板中")
                total_return = 0
                sharpe = 0
                max_drawdown = 0
                win_rate = 0
                closed_trades = 0
            else:
                backtest_result = run_scored_backtest(
                    panel,  # 使用包含因子的面板
                    single_factor_scorer,
                    top_n=5,
                    initial_cash=100000,
                    market_filter=True,
                    retention_multiple=3,
                    universe_size=1000,
                    candidate_filter=v8_candidate_filter
                )
                
                bt_metrics = backtest_result['metrics']
                total_return = bt_metrics['total_return']
                sharpe = bt_metrics['sharpe']
                max_drawdown = bt_metrics['max_drawdown']
                win_rate = bt_metrics['win_rate']
                closed_trades = bt_metrics['closed_trades']
        except Exception as e:
            print(f"    回测失败: {e}")
            import traceback
            traceback.print_exc()
            total_return = 0
            sharpe = 0
            max_drawdown = 0
            win_rate = 0
            closed_trades = 0
        
        # 4.4 综合评分
        ic_score = min(ic_summary['ic_ir'] / 0.5, 1.0) * 30  # IC_IR 权重 30%
        return_score = min(max(total_return, 0) / 0.2, 1.0) * 25  # 收益权重 25%
        sharpe_score = min(max(sharpe, 0) / 1.0, 1.0) * 20  # 夏普权重 20%
        drawdown_score = min(max(-max_drawdown, 0) / 0.3, 1.0) * 15  # 回撤权重 15%
        stability_score = (1 if is_monotonic else 0) * 10  # 单调性权重 10%
        
        overall_score = ic_score + return_score + sharpe_score + drawdown_score + stability_score
        
        result = {
            '因子名称': factor_name,
            '类别': factor_info.get('category', '未知'),
            '方向': factor_direction,
            'IC 均值': ic_summary['ic_mean'],
            'IC_IR': ic_summary['ic_ir'],
            'IC>0 比例': ic_summary['ic_positive_ratio'],
            '多空收益': long_short_return,
            '单调性': '是' if is_monotonic else '否',
            '总收益': total_return,
            '夏普比率': sharpe,
            '最大回撤': max_drawdown,
            '胜率': win_rate,
            '交易次数': closed_trades,
            '综合评分': overall_score,
            'IC 评分': ic_score,
            '收益评分': return_score,
            '夏普评分': sharpe_score,
            '回撤评分': drawdown_score,
            '稳定性评分': stability_score,
        }
        
        factor_results.append(result)
        
        print(f"    IC_IR: {ic_summary['ic_ir']:.4f}")
        print(f"    多空收益: {long_short_return:.4f}")
        print(f"    总收益: {total_return:.2%}")
        print(f"    夏普: {sharpe:.4f}")
        print(f"    综合评分: {overall_score:.2f}")
    
    # 5. 生成报告
    elapsed_time = time.time() - start_time
    
    results_df = pd.DataFrame(factor_results)
    results_df = results_df.sort_values('综合评分', ascending=False)
    
    print("\n" + "=" * 70)
    print("因子潜力评估结果")
    print("=" * 70)
    print(f"总耗时: {elapsed_time:.1f} 秒")
    print()
    
    # 打印排名
    print("因子排名（按综合评分）:")
    print("-" * 70)
    for idx, row in results_df.iterrows():
        print(f"{idx+1}. {row['因子名称']}")
        print(f"   类别: {row['类别']} | 方向: {row['方向']}")
        print(f"   IC_IR: {row['IC_IR']:.4f} | 多空收益: {row['多空收益']:.4f} | 单调性: {row['单调性']}")
        print(f"   总收益: {row['总收益']:.2%} | 夏普: {row['夏普比率']:.4f} | 回撤: {row['最大回撤']:.2%}")
        print(f"   综合评分: {row['综合评分']:.2f} (IC:{row['IC 评分']:.1f} + 收益:{row['收益评分']:.1f} + 夏普:{row['夏普评分']:.1f} + 回撤:{row['回撤评分']:.1f} + 稳定:{row['稳定性评分']:.1f})")
        print()
    
    # 保存报告
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("因子潜力评估报告（非 Agent 回测）\n")
            f.write("=" * 70 + "\n")
            f.write(f"实验时间: {start_date} 至 {end_date}\n")
            f.write(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总耗时: {elapsed_time:.1f} 秒\n")
            f.write("\n")
            f.write("因子排名\n")
            f.write("-" * 70 + "\n")
            f.write(results_df.to_string(index=False))
            f.write("\n")
        
        print(f"报告已保存到: {output_file}")
    
    # 保存 CSV
    csv_file = output_file.replace('.txt', '.csv') if output_file else None
    if csv_file:
        results_df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f"CSV 已保存到: {csv_file}")
    
    return results_df


def calculate_group_returns(factor_df: pd.DataFrame, returns_df: pd.DataFrame, n_groups: int = 5) -> pd.DataFrame:
    """计算分组收益"""
    group_returns = []
    
    for date in factor_df.index:
        if date not in returns_df.index:
            continue
        
        factor_values = factor_df.loc[date].dropna()
        forward_returns = returns_df.loc[date]
        
        # 共同有效的股票
        common_stocks = factor_values.index.intersection(forward_returns.dropna().index)
        if len(common_stocks) < n_groups * 2:
            continue
        
        factor_clean = factor_values[common_stocks]
        returns_clean = forward_returns[common_stocks]
        
        # 按因子值分组
        try:
            groups = pd.qcut(factor_clean, q=n_groups, labels=False, duplicates='drop')
        except ValueError:
            continue
        
        # 计算每组平均收益
        for group in range(n_groups):
            group_stocks = groups[groups == group].index
            if len(group_stocks) == 0:
                continue
            group_return = returns_clean[group_stocks].mean()
            group_returns.append({
                'date': date,
                'group': group + 1,
                'return': group_return,
                'n_stocks': len(group_stocks)
            })
    
    group_df = pd.DataFrame(group_returns)
    
    if group_df.empty:
        return pd.DataFrame(columns=['avg_return', 'std_return', 'n_periods', 'avg_n_stocks'])
    
    # 计算每组平均收益
    group_summary = group_df.groupby('group').agg({
        'return': ['mean', 'std', 'count'],
        'n_stocks': 'mean'
    }).round(6)
    
    group_summary.columns = ['avg_return', 'std_return', 'n_periods', 'avg_n_stocks']
    
    return group_summary


if __name__ == "__main__":
    # 运行因子评估
    output_dir = project_root / "reports"
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"factor_evaluation_{timestamp}.txt"
    
    run_factor_evaluation(
        start_date="2024-01-01",
        end_date="2024-12-31",
        max_stocks=None,  # 使用全部股票
        output_file=str(output_file)
    )
