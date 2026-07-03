#!/usr/bin/env python3
"""
快速因子对比实验 - 使用向量化计算大幅提升速度

优化点：
1. 使用向量化因子计算（避免逐股票循环）
2. 限制股票数量（默认前 1000 只）
3. 减少时间范围（默认最近 1 年）
4. 添加进度显示和预计时间
"""

import sys
import os
from pathlib import Path
import time
from typing import Optional

# 添加项目根目录到路径
project_root = Path(__file__).parent
backend_root = project_root / "backend"
sys.path.insert(0, str(backend_root))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from app.services.market_store import ParquetMarketStore
from app.services.new_factors_fast import compute_factors_vectorized, FACTOR_REGISTRY_VECTORIZED
from app.services.factor_ic_analysis import batch_analyze_factors
from app.services.research_backtest import run_scored_backtest, build_weekly_feature_panel
from app.services.strategies.v8_canonical import v8_scorer, v8_candidate_filter, merge_alpha_132
from app.services.strategies.agent_gate import agent_gate


def run_fast_comparison_experiment(
    data_dir: str = "/Volumes/xhrrrrr_macmini副盘/quantlab/market",
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    max_stocks: Optional[int] = None,  # None 表示不限制，使用全部股票
    top_n_factors: int = 5,
    output_file: str = None
):
    """
    运行快速对比实验
    
    Args:
        data_dir: 数据目录
        start_date: 开始日期
        end_date: 结束日期
        max_stocks: 最大股票数量（None 表示不限制，使用全部股票）
        top_n_factors: 选择的最佳因子数量
        output_file: 输出文件路径
    """
    print("=" * 60)
    print("快速因子对比实验")
    print("=" * 60)
    print(f"数据目录: {data_dir}")
    print(f"实验区间: {start_date} 至 {end_date}")
    print(f"最大股票数: {max_stocks}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    start_time = time.time()
    
    # 1. 加载数据
    print("步骤 1/5: 加载数据...")
    store = ParquetMarketStore(data_dir)
    
    start_dt = pd.to_datetime(start_date).date()
    end_dt = pd.to_datetime(end_date).date()
    
    # 构建周频面板
    panel = build_weekly_feature_panel(store, start_dt, end_dt)
    panel_with_alpha132 = merge_alpha_132(panel)
    
    print(f"  面板数据: {len(panel)} 行")
    
    # 2. 准备因子计算数据
    print("\n步骤 2/5: 准备因子计算数据...")
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
    
    # 3. 计算因子（向量化）
    print("\n步骤 3/5: 计算因子（向量化）...")
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
    
    # 4. 分析因子 IC
    print("\n步骤 4/5: 分析因子 IC...")
    close_prices_panel = panel.pivot(index='date', columns='symbol', values='close')
    forward_returns = close_prices_panel.pct_change(5).shift(-5)
    
    factor_dict = {}
    for factor_name in FACTOR_REGISTRY_VECTORIZED.keys():
        if factor_name in panel.columns:
            factor_values = panel.pivot(index='date', columns='symbol', values=factor_name)
            factor_dict[factor_name] = factor_values
    
    ic_results = batch_analyze_factors(factor_dict, forward_returns, n_groups=5)
    
    # 选择最佳因子
    if ic_results.empty:
        print("警告: 没有有效的因子 IC 结果")
        best_factors = list(FACTOR_REGISTRY_VECTORIZED.keys())[:top_n_factors]
    else:
        best_factors = ic_results.nlargest(top_n_factors, 'ic_ir')['factor_name'].tolist()
    
    print(f"\n最佳 {top_n_factors} 个因子:")
    for i, factor in enumerate(best_factors, 1):
        if not ic_results.empty and factor in ic_results['factor_name'].values:
            ic_ir = ic_results[ic_results['factor_name'] == factor]['ic_ir'].values[0]
            print(f"  {i}. {factor} (IC_IR: {ic_ir:.4f})")
        else:
            print(f"  {i}. {factor}")
    
    # 5. 运行回测
    print("\n步骤 5/5: 运行回测...")
    
    # 创建新因子评分函数
    def new_factor_scorer(panel_with_factors: pd.DataFrame) -> pd.Series:
        scores = pd.Series(0.0, index=panel_with_factors.index)
        weights = 1.0 / len(best_factors)
        
        for factor_name in best_factors:
            if factor_name in panel_with_factors.columns:
                factor_values = panel_with_factors[factor_name].copy()
                factor_values = factor_values.fillna(factor_values.median())
                factor_rank = factor_values.rank(pct=True)
                
                # 根据因子方向调整
                direction = FACTOR_REGISTRY_VECTORIZED.get(factor_name, {}).get('direction', 'positive')
                if direction == 'negative':
                    factor_rank = 1 - factor_rank
                
                scores += factor_rank * weights
        
        return scores
    
    # 运行 4 个实验
    experiments = {
        'Baseline (Alpha191)': (v8_scorer, v8_candidate_filter, False),
        'Baseline + Agent': (v8_scorer, lambda g: agent_gate(v8_candidate_filter(g)), True),
        'New Factors': (new_factor_scorer, v8_candidate_filter, False),
        'New Factors + Agent': (new_factor_scorer, lambda g: agent_gate(v8_candidate_filter(g)), True),
    }
    
    results = {}
    for exp_name, (scorer, candidate_filter, use_agent) in experiments.items():
        print(f"\n  运行实验: {exp_name}")
        
        result = run_scored_backtest(
            panel_with_alpha132 if 'alpha_132' in panel_with_alpha132.columns else panel,
            scorer,
            top_n=5,
            initial_cash=100000,
            market_filter=True,
            retention_multiple=3,
            universe_size=1000,
            candidate_filter=candidate_filter
        )
        
        metrics = result['metrics']
        results[exp_name] = {
            'metrics': metrics,
            'use_agent': use_agent
        }
        
        print(f"    总收益: {metrics['total_return']:.2%}")
        print(f"    夏普比率: {metrics['sharpe']:.4f}")
        print(f"    最大回撤: {metrics['max_drawdown']:.2%}")
    
    # 生成报告
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("实验完成")
    print("=" * 60)
    print(f"总耗时: {elapsed_time:.1f} 秒 ({elapsed_time/60:.1f} 分钟)")
    
    # 生成对比表
    comparison_data = []
    for exp_name, result in results.items():
        metrics = result['metrics']
        comparison_data.append({
            '实验': exp_name,
            '使用 Agent': '是' if result['use_agent'] else '否',
            '总收益': f"{metrics['total_return']:.2%}",
            '年化收益': f"{metrics['annual_return']:.2%}",
            '夏普比率': f"{metrics['sharpe']:.4f}",
            '最大回撤': f"{metrics['max_drawdown']:.2%}",
            '胜率': f"{metrics['win_rate']:.2%}",
            '交易次数': metrics['closed_trades']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\n实验结果对比:")
    print(comparison_df.to_string(index=False))
    
    # 保存报告
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("快速因子对比实验报告\n")
            f.write("=" * 60 + "\n")
            f.write(f"实验时间: {start_date} 至 {end_date}\n")
            f.write(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总耗时: {elapsed_time:.1f} 秒\n")
            f.write("\n")
            f.write("实验结果对比\n")
            f.write("-" * 60 + "\n")
            f.write(comparison_df.to_string(index=False))
            f.write("\n")
        
        print(f"\n报告已保存到: {output_file}")
    
    return results, comparison_df, ic_results


if __name__ == "__main__":
    # 运行快速实验（默认：全部股票，最近 1 年）
    output_dir = project_root / "reports"
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"fast_comparison_{timestamp}.txt"
    
    run_fast_comparison_experiment(
        start_date="2024-01-01",
        end_date="2024-12-31",
        max_stocks=None,  # 使用全部股票
        top_n_factors=5,
        output_file=str(output_file)
    )
