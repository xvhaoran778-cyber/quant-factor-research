#!/usr/bin/env python3
"""
新因子测试脚本

测试 15 个新因子是否能正确计算
"""

import sys
import os
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent
backend_root = project_root / "backend"
sys.path.insert(0, str(backend_root))

from app.services.new_factors import (
    rank_velocity, alpha_momentum, correlation_breakdown,
    momentum_acceleration, bid_ask_spread, order_flow_imbalance,
    price_impact, liquidity_shock, disposition_effect,
    herding_indicator, overreaction_reversal, return_skewness,
    return_kurtosis, hurst_exponent, FACTOR_REGISTRY
)


def generate_test_data(n_stocks: int = 10, n_days: int = 100):
    """生成测试数据"""
    np.random.seed(42)
    
    dates = pd.date_range('2020-01-01', periods=n_days, freq='B')
    symbols = [f'STOCK_{i:03d}' for i in range(n_stocks)]
    
    # 生成价格数据
    close = pd.DataFrame(
        np.random.randn(n_days, n_stocks).cumsum(axis=0) + 100,
        index=dates,
        columns=symbols
    )
    
    open_price = close.shift(1).fillna(close.iloc[0])
    high = close * (1 + np.abs(np.random.randn(n_days, n_stocks) * 0.02))
    low = close * (1 - np.abs(np.random.randn(n_days, n_stocks) * 0.02))
    volume = pd.DataFrame(
        np.random.randint(1000, 10000, (n_days, n_stocks)),
        index=dates,
        columns=symbols
    )
    
    returns = close.pct_change()
    market_returns = returns.mean(axis=1)
    
    # 生成行业收益率（简单模拟：将股票分成3个行业）
    n_industries = 3
    industry_returns = pd.DataFrame(index=dates)
    for i in range(n_industries):
        industry_col = symbols[i::n_industries]  # 每隔n_industries取一个
        industry_returns[f'INDUSTRY_{i}'] = returns[industry_col].mean(axis=1)
    
    high_250d = close.rolling(250, min_periods=1).max()
    prev_close = close.shift(1)
    
    return {
        'close_prices_matrix': close,
        'stock_returns': returns,
        'market_returns': market_returns,
        'industry_returns': industry_returns,
        'close': close,
        'open': open_price,
        'open_price': open_price,  # 别名
        'high': high,
        'low': low,
        'volume': volume,
        'returns': returns,
        'prices': close,
        'high_250d': high_250d,
        'prev_close': prev_close
    }


def test_single_factor(factor_name: str, test_data: dict):
    """测试单个因子"""
    print(f"\n测试因子: {factor_name}")
    
    try:
        factor_info = FACTOR_REGISTRY[factor_name]
        func = factor_info['func']
        requires = factor_info['requires']
        
        # 准备参数
        if factor_name == 'rank_velocity':
            # 特殊处理：需要整个矩阵
            result = func(test_data['close_prices_matrix'])
            print(f"  结果类型: {type(result)}")
            print(f"  结果形状: {result.shape if hasattr(result, 'shape') else 'N/A'}")
            print(f"  结果示例: {result.iloc[:3] if hasattr(result, 'iloc') else result}")
            
        elif factor_name in ['alpha_momentum', 'correlation_breakdown', 'herding_indicator']:
            # 需要股票和市场数据，测试第一个股票
            symbol = test_data['stock_returns'].columns[0]
            result = func(
                test_data['stock_returns'][symbol],
                test_data['market_returns']
            )
            print(f"  结果: {result}")
            
        elif factor_name == 'industry_relative_strength':
            # 需要股票和行业数据，测试第一个股票和第一个行业
            symbol = test_data['stock_returns'].columns[0]
            industry = test_data['industry_returns'].columns[0]
            result = func(
                test_data['stock_returns'][symbol],
                test_data['industry_returns'][industry]
            )
            print(f"  结果: {result}")
            
        else:
            # 逐股票计算，测试第一个股票
            symbol = test_data['close'].columns[0]
            symbol_data = {}
            for req in requires:
                data = test_data[req]
                if hasattr(data, 'columns'):
                    symbol_data[req] = data[symbol]
                else:
                    symbol_data[req] = data
            
            result = func(**symbol_data)
            print(f"  结果: {result}")
        
        print(f"  ✓ 因子 {factor_name} 计算成功")
        return True
        
    except Exception as e:
        print(f"  ✗ 因子 {factor_name} 计算失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("="*60)
    print("新因子测试")
    print("="*60)
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 生成测试数据
    print("生成测试数据...")
    test_data = generate_test_data(n_stocks=10, n_days=300)
    print(f"数据形状: {test_data['close'].shape}")
    print()
    
    # 测试所有因子
    print("测试所有因子...")
    results = {}
    
    for factor_name in FACTOR_REGISTRY.keys():
        success = test_single_factor(factor_name, test_data)
        results[factor_name] = success
    
    # 汇总结果
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    
    success_count = sum(results.values())
    total_count = len(results)
    
    print(f"\n成功: {success_count}/{total_count}")
    print(f"失败: {total_count - success_count}/{total_count}")
    
    if success_count == total_count:
        print("\n✓ 所有因子测试通过！")
        return 0
    else:
        print("\n✗ 部分因子测试失败:")
        for factor_name, success in results.items():
            if not success:
                print(f"  - {factor_name}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
