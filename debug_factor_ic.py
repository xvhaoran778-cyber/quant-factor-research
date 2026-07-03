#!/usr/bin/env python3
"""
调试因子 IC 计算问题
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
backend_root = project_root / "backend"
sys.path.insert(0, str(backend_root))

import pandas as pd
import numpy as np
from app.services.market_store import ParquetMarketStore
from app.services.new_factors import compute_all_factors, FACTOR_REGISTRY

def test_factor_calculation():
    """测试因子计算"""
    print("=" * 60)
    print("测试因子计算")
    print("=" * 60)
    
    # 加载数据
    data_dir = "/Volumes/xhrrrrr_macmini副盘/quantlab/market"
    store = ParquetMarketStore(data_dir)
    
    start_date = pd.to_datetime("2023-01-01").date()
    end_date = pd.to_datetime("2024-03-31").date()
    
    print(f"加载 {start_date} 至 {end_date} 的日线数据...")
    all_daily = store.read(start_date, end_date, symbols=None, fill_suspensions=False)
    
    if all_daily.empty:
        print("无法加载数据")
        return
    
    print(f"加载了 {len(all_daily)} 条日线数据")
    
    # 只取前 10 只股票用于调试
    symbols = all_daily['symbol'].unique()[:10]
    all_daily = all_daily[all_daily['symbol'].isin(symbols)]
    print(f"只使用 {len(symbols)} 只股票进行调试: {symbols}")
    
    # 构建透视表
    close_prices = all_daily.pivot(index='date', columns='symbol', values='close')
    returns = close_prices.pct_change()
    market_returns = returns.mean(axis=1)
    
    # 准备数据字典
    factor_input_data = {
        'close_prices_matrix': close_prices,
        'stock_returns': returns,
        'market_returns': market_returns,
        'close': close_prices,
        'returns': returns,
        'prices': close_prices,
    }
    
    # 只测试一个简单的因子
    test_factors = ['return_skewness']
    
    print(f"\n计算因子: {test_factors}")
    factor_df = compute_all_factors(factor_input_data, test_factors)
    
    print(f"\n因子 DataFrame 形状: {factor_df.shape}")
    print(f"因子 DataFrame 索引类型: {type(factor_df.index)}")
    print(f"\n前 10 行:")
    print(factor_df.head(10))
    
    print(f"\nNaN 统计:")
    print(factor_df.isna().sum())
    
    print(f"\n非 NaN 值统计:")
    print(factor_df.notna().sum())
    
    # 检查具体值
    print(f"\n因子值分布:")
    print(factor_df['return_skewness'].describe())
    
    # 转换为透视表格式
    print("\n" + "=" * 60)
    print("转换为透视表格式")
    print("=" * 60)
    
    factor_pivot = factor_df['return_skewness'].unstack()
    print(f"\n透视表形状: {factor_pivot.shape}")
    print(f"\n前 5 行前 5 列:")
    print(factor_pivot.iloc[:5, :5])
    
    print(f"\n透视表 NaN 统计:")
    print(f"总 NaN 数量: {factor_pivot.isna().sum().sum()}")
    print(f"总非 NaN 数量: {factor_pivot.notna().sum().sum()}")
    
    # 计算前瞻收益率
    print("\n" + "=" * 60)
    print("计算前瞻收益率")
    print("=" * 60)
    
    forward_days = 5
    forward_returns = close_prices.pct_change(forward_days).shift(-forward_days)
    
    print(f"\n前瞻收益率形状: {forward_returns.shape}")
    print(f"\n前 5 行前 5 列:")
    print(forward_returns.iloc[:5, :5])
    
    print(f"\n前瞻收益率 NaN 统计:")
    print(f"总 NaN 数量: {forward_returns.isna().sum().sum()}")
    print(f"总非 NaN 数量: {forward_returns.notna().sum().sum()}")
    
    # 计算 IC
    print("\n" + "=" * 60)
    print("计算 IC")
    print("=" * 60)
    
    from app.services.factor_ic_analysis import calculate_ic_series
    
    ic_series = calculate_ic_series(factor_pivot, forward_returns)
    
    print(f"\nIC 时间序列:")
    print(ic_series.head(10))
    
    print(f"\nIC 统计:")
    print(ic_series.describe())

if __name__ == "__main__":
    test_factor_calculation()
