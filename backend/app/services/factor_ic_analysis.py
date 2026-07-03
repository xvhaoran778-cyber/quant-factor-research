"""
因子 IC 分析框架

计算因子的 IC (Information Coefficient)、Rank IC、分组收益等指标
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


def calculate_ic(factor_values: pd.Series, forward_returns: pd.Series) -> float:
    """
    计算 IC (Information Coefficient)
    
    IC = corr(factor_t, return_{t+1})
    
    Args:
        factor_values: 因子值序列
        forward_returns: 未来收益率序列
    
    Returns:
        IC 值
    """
    # 去除 NaN
    valid_mask = factor_values.notna() & forward_returns.notna()
    if valid_mask.sum() < 10:
        return np.nan
    
    factor_clean = factor_values[valid_mask]
    returns_clean = forward_returns[valid_mask]
    
    # 计算相关系数
    ic = factor_clean.corr(returns_clean)
    return ic


def calculate_rank_ic(factor_values: pd.Series, forward_returns: pd.Series) -> float:
    """
    计算 Rank IC
    
    Rank IC = spearman_corr(factor_t, return_{t+1})
    
    Args:
        factor_values: 因子值序列
        forward_returns: 未来收益率序列
    
    Returns:
        Rank IC 值
    """
    # 去除 NaN
    valid_mask = factor_values.notna() & forward_returns.notna()
    if valid_mask.sum() < 10:
        return np.nan
    
    factor_clean = factor_values[valid_mask]
    returns_clean = forward_returns[valid_mask]
    
    # 计算排名相关系数
    factor_rank = factor_clean.rank()
    returns_rank = returns_clean.rank()
    rank_ic = factor_rank.corr(returns_rank)
    
    return rank_ic


def calculate_ic_series(factor_df: pd.DataFrame, returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    计算时间序列 IC
    
    Args:
        factor_df: 因子值矩阵 (日期 x 股票)
        returns_df: 未来收益率矩阵 (日期 x 股票)
    
    Returns:
        IC 时间序列 DataFrame
    """
    ic_list = []
    rank_ic_list = []
    
    for date in factor_df.index:
        if date not in returns_df.index:
            continue
        
        factor_values = factor_df.loc[date]
        forward_returns = returns_df.loc[date]
        
        ic = calculate_ic(factor_values, forward_returns)
        rank_ic = calculate_rank_ic(factor_values, forward_returns)
        
        ic_list.append({'date': date, 'ic': ic})
        rank_ic_list.append({'date': date, 'rank_ic': rank_ic})
    
    ic_df = pd.DataFrame(ic_list).set_index('date')
    rank_ic_df = pd.DataFrame(rank_ic_list).set_index('date')
    
    return pd.concat([ic_df, rank_ic_df], axis=1)


def calculate_ic_summary(ic_series: pd.DataFrame) -> Dict:
    """
    计算 IC 汇总统计
    
    Args:
        ic_series: IC 时间序列
    
    Returns:
        汇总统计字典
    """
    ic = ic_series['ic'].dropna()
    rank_ic = ic_series['rank_ic'].dropna()
    
    summary = {
        'ic_mean': ic.mean(),
        'ic_std': ic.std(),
        'ic_ir': ic.mean() / ic.std() if ic.std() > 0 else 0,
        'ic_positive_ratio': (ic > 0).mean(),
        'rank_ic_mean': rank_ic.mean(),
        'rank_ic_std': rank_ic.std(),
        'rank_ic_ir': rank_ic.mean() / rank_ic.std() if rank_ic.std() > 0 else 0,
        'rank_ic_positive_ratio': (rank_ic > 0).mean(),
        'ic_max': ic.max(),
        'ic_min': ic.min(),
        'sample_size': len(ic)
    }
    
    return summary


def calculate_group_returns(factor_df: pd.DataFrame, 
                           returns_df: pd.DataFrame,
                           n_groups: int = 5) -> pd.DataFrame:
    """
    计算分组收益
    
    Args:
        factor_df: 因子值矩阵 (日期 x 股票)
        returns_df: 未来收益率矩阵 (日期 x 股票)
        n_groups: 分组数量
    
    Returns:
        分组收益 DataFrame
    """
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
    
    # 检查是否有数据
    if group_df.empty:
        # 返回空的分组收益表
        return pd.DataFrame(columns=['avg_return', 'std_return', 'n_periods', 'avg_n_stocks'])
    
    # 计算每组平均收益
    group_summary = group_df.groupby('group').agg({
        'return': ['mean', 'std', 'count'],
        'n_stocks': 'mean'
    }).round(6)
    
    group_summary.columns = ['avg_return', 'std_return', 'n_periods', 'avg_n_stocks']
    
    # 计算多空收益
    if 1 in group_summary.index and n_groups in group_summary.index:
        long_short_return = (
            group_summary.loc[n_groups, 'avg_return'] - 
            group_summary.loc[1, 'avg_return']
        )
        group_summary.loc['long_short'] = [
            long_short_return,
            np.nan,
            np.nan,
            np.nan
        ]
    
    return group_summary


def calculate_yearly_ic(factor_df: pd.DataFrame, 
                       returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    计算逐年 IC
    
    Args:
        factor_df: 因子值矩阵 (日期 x 股票)
        returns_df: 未来收益率矩阵 (日期 x 股票)
    
    Returns:
        逐年 IC DataFrame
    """
    # 提取年份
    factor_df = factor_df.copy()
    factor_df['year'] = factor_df.index.year
    
    returns_df = returns_df.copy()
    returns_df['year'] = returns_df.index.year
    
    yearly_ic = []
    
    for year in sorted(factor_df['year'].unique()):
        year_factor = factor_df[factor_df['year'] == year].drop('year', axis=1)
        year_returns = returns_df[returns_df['year'] == year].drop('year', axis=1)
        
        ic_series = calculate_ic_series(year_factor, year_returns)
        ic_summary = calculate_ic_summary(ic_series)
        ic_summary['year'] = year
        
        yearly_ic.append(ic_summary)
    
    yearly_df = pd.DataFrame(yearly_ic).set_index('year')
    
    return yearly_df


def analyze_factor(factor_name: str,
                  factor_df: pd.DataFrame,
                  returns_df: pd.DataFrame,
                  n_groups: int = 5) -> Dict:
    """
    完整分析单个因子
    
    Args:
        factor_name: 因子名称
        factor_df: 因子值矩阵 (日期 x 股票)
        returns_df: 未来收益率矩阵 (日期 x 股票)
        n_groups: 分组数量
    
    Returns:
        分析结果字典
    """
    print(f"\n{'='*60}")
    print(f"分析因子: {factor_name}")
    print(f"{'='*60}")
    
    # 1. 计算 IC 时间序列
    print("计算 IC 时间序列...")
    ic_series = calculate_ic_series(factor_df, returns_df)
    
    # 2. 计算 IC 汇总统计
    print("计算 IC 汇总统计...")
    ic_summary = calculate_ic_summary(ic_series)
    
    # 3. 计算分组收益
    print(f"计算分组收益 ({n_groups} 组)...")
    group_returns = calculate_group_returns(factor_df, returns_df, n_groups)
    
    # 4. 计算逐年 IC
    print("计算逐年 IC...")
    yearly_ic = calculate_yearly_ic(factor_df, returns_df)
    
    # 5. 打印结果
    print("\n--- IC 汇总统计 ---")
    print(f"IC 均值: {ic_summary['ic_mean']:.4f}")
    print(f"IC 标准差: {ic_summary['ic_std']:.4f}")
    print(f"IC_IR: {ic_summary['ic_ir']:.4f}")
    print(f"IC > 0 比例: {ic_summary['ic_positive_ratio']:.2%}")
    print(f"\nRank IC 均值: {ic_summary['rank_ic_mean']:.4f}")
    print(f"Rank IC 标准差: {ic_summary['rank_ic_std']:.4f}")
    print(f"Rank IC_IR: {ic_summary['rank_ic_ir']:.4f}")
    print(f"Rank IC > 0 比例: {ic_summary['rank_ic_positive_ratio']:.2%}")
    
    print("\n--- 分组收益 ---")
    print(group_returns[['avg_return', 'avg_n_stocks']])
    
    print("\n--- 逐年 IC ---")
    print(yearly_ic[['ic_mean', 'ic_ir', 'ic_positive_ratio']].round(4))
    
    # 6. 判断因子质量
    print("\n--- 因子质量评估 ---")
    quality = assess_factor_quality(ic_summary, yearly_ic)
    for criterion, passed in quality.items():
        status = "✓" if passed else "✗"
        print(f"{status} {criterion}")
    
    overall_passed = sum(quality.values()) / len(quality)
    print(f"\n总体评分: {overall_passed:.0%} 通过")
    
    if overall_passed >= 0.6:
        print("结论: 因子质量良好，可以使用")
    elif overall_passed >= 0.4:
        print("结论: 因子质量一般，需要进一步优化")
    else:
        print("结论: 因子质量较差，建议放弃")
    
    return {
        'factor_name': factor_name,
        'ic_summary': ic_summary,
        'group_returns': group_returns,
        'yearly_ic': yearly_ic,
        'quality': quality,
        'overall_score': overall_passed
    }


def assess_factor_quality(ic_summary: Dict, yearly_ic: pd.DataFrame) -> Dict[str, bool]:
    """
    评估因子质量
    
    Args:
        ic_summary: IC 汇总统计
        yearly_ic: 逐年 IC
    
    Returns:
        评估结果字典
    """
    quality = {}
    
    # 1. IC 均值 > 0.03
    quality['IC 均值 > 0.03'] = abs(ic_summary['ic_mean']) > 0.03
    
    # 2. IC_IR > 0.5
    quality['IC_IR > 0.5'] = abs(ic_summary['ic_ir']) > 0.5
    
    # 3. IC > 0 比例 > 55%
    quality['IC > 0 比例 > 55%'] = ic_summary['ic_positive_ratio'] > 0.55
    
    # 4. 逐年 IC 稳定性（至少 60% 年份 IC 方向一致）
    if len(yearly_ic) > 0:
        ic_direction = (yearly_ic['ic_mean'] > 0).mean()
        quality['逐年 IC 稳定性 > 60%'] = ic_direction > 0.6
    else:
        quality['逐年 IC 稳定性 > 60%'] = False
    
    # 5. 样本量充足
    quality['样本量 > 100'] = ic_summary['sample_size'] > 100
    
    return quality


def batch_analyze_factors(factor_dict: Dict[str, pd.DataFrame],
                         returns_df: pd.DataFrame,
                         n_groups: int = 5) -> pd.DataFrame:
    """
    批量分析多个因子
    
    Args:
        factor_dict: 因子名称到因子矩阵的映射
        returns_df: 未来收益率矩阵
        n_groups: 分组数量
    
    Returns:
        因子分析汇总 DataFrame
    """
    results = []
    
    for factor_name, factor_df in factor_dict.items():
        try:
            analysis = analyze_factor(factor_name, factor_df, returns_df, n_groups)
            
            results.append({
                'factor_name': factor_name,
                'ic_mean': analysis['ic_summary']['ic_mean'],
                'ic_ir': analysis['ic_summary']['ic_ir'],
                'ic_positive_ratio': analysis['ic_summary']['ic_positive_ratio'],
                'rank_ic_mean': analysis['ic_summary']['rank_ic_mean'],
                'overall_score': analysis['overall_score'],
                'quality_passed': sum(analysis['quality'].values()),
                'quality_total': len(analysis['quality'])
            })
        except Exception as e:
            print(f"分析因子 {factor_name} 时出错: {e}")
            continue
    
    summary_df = pd.DataFrame(results)
    
    if not summary_df.empty:
        # 按 IC_IR 排序
        summary_df = summary_df.sort_values('ic_ir', ascending=False)
        
        print("\n" + "="*60)
        print("因子分析汇总")
        print("="*60)
        print(summary_df.to_string(index=False))
    
    return summary_df
