#!/usr/bin/env python3
"""
简单测试单个因子计算
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
from app.services.new_factors import return_skewness

def test_single_factor():
    """测试单个因子计算"""
    print("=" * 60)
    print("测试单个因子计算")
    print("=" * 60)
    
    # 创建测试数据
    dates = pd.date_range('2023-01-01', periods=100, freq='B')
    returns = pd.Series(np.random.randn(100) * 0.02, index=dates)
    
    print(f"\n测试数据:")
    print(f"  日期范围: {dates[0]} 到 {dates[-1]}")
    print(f"  数据点数: {len(returns)}")
    print(f"  非 NaN 数据点: {returns.notna().sum()}")
    
    # 测试因子计算
    print(f"\n计算 return_skewness (window=60)...")
    
    # 方法 1: 直接调用函数
    try:
        result = return_skewness(returns, window=60)
        print(f"  结果: {result}")
        print(f"  是否为 NaN: {pd.isna(result)}")
    except Exception as e:
        print(f"  错误: {e}")
    
    # 方法 2: 手动计算
    print(f"\n手动计算:")
    rolling_skew = returns.rolling(60).skew()
    print(f"  rolling_skew 形状: {rolling_skew.shape}")
    print(f"  rolling_skew 非 NaN 数量: {rolling_skew.notna().sum()}")
    print(f"  最后一个值: {rolling_skew.iloc[-1]}")
    print(f"  最后一个值是否为 NaN: {pd.isna(rolling_skew.iloc[-1])}")
    
    # 方法 3: 检查不同长度的数据
    print(f"\n测试不同长度的数据:")
    for length in [10, 30, 60, 90, 100]:
        test_returns = returns.iloc[:length]
        result = return_skewness(test_returns, window=60)
        print(f"  长度 {length}: 结果 = {result}, 是否为 NaN = {pd.isna(result)}")

if __name__ == "__main__":
    test_single_factor()
