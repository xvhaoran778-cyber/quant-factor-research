"""
优化版因子计算 - 使用向量化操作大幅提升速度
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


def compute_factors_vectorized(data: Dict[str, pd.DataFrame], 
                               factor_names: Optional[List[str]] = None,
                               max_stocks: Optional[int] = None) -> pd.DataFrame:
    """
    向量化计算因子（优化版）
    
    优化点：
    1. 使用 DataFrame 的 rolling 方法批量计算，避免逐股票循环
    2. 一次性计算所有日期的因子
    3. 支持全量股票（不设上限）
    
    Args:
        data: 数据字典
        factor_names: 因子名称列表
        max_stocks: 最大股票数量（None 表示不限制）
    
    Returns:
        因子值 DataFrame，索引为 (date, symbol) 的 MultiIndex
    """
    if factor_names is None:
        factor_names = list(FACTOR_REGISTRY_VECTORIZED.keys())
    
    # 获取数据
    close_prices = data['close']
    returns = data['returns']
    volume = data['volume']
    high = data['high']
    low = data['low']
    open_price = data['open']
    
    # 可选：按流动性筛选股票
    if max_stocks is not None:
        if 'amount' in data:
            avg_amount = data['amount'].mean()
            top_stocks = avg_amount.nlargest(max_stocks).index
        else:
            # 如果没有成交额数据，按成交量排序
            avg_volume = volume.mean()
            top_stocks = avg_volume.nlargest(max_stocks).index
        
        # 筛选股票
        close_prices = close_prices[top_stocks]
        returns = returns[top_stocks]
        volume = volume[top_stocks]
        high = high[top_stocks]
        low = low[top_stocks]
        open_price = open_price[top_stocks]
        print(f"按流动性筛选前 {max_stocks} 只股票")
    
    dates = close_prices.index
    symbols = close_prices.columns
    
    print(f"开始计算 {len(factor_names)} 个因子，共 {len(symbols)} 只股票")
    
    # 创建 MultiIndex
    index = pd.MultiIndex.from_product([dates, symbols], names=['date', 'symbol'])
    results = pd.DataFrame(index=index, columns=factor_names, dtype=float)
    
    # 逐因子计算（向量化）
    for factor_name in factor_names:
        if factor_name not in FACTOR_REGISTRY_VECTORIZED:
            continue
        
        print(f"  计算因子: {factor_name}")
        
        try:
            factor_func = FACTOR_REGISTRY_VECTORIZED[factor_name]['func']
            factor_values = factor_func(close_prices, returns, volume, high, low, open_price)
            
            # 将 DataFrame 转换为 MultiIndex Series
            factor_series = factor_values.stack()
            factor_series.index.names = ['date', 'symbol']
            
            # 填充结果
            results[factor_name] = factor_series
            
        except Exception as e:
            print(f"    警告: {factor_name} 计算失败 - {e}")
    
    print(f"因子计算完成")
    return results


# ============================================================================
# 向量化因子函数
# ============================================================================

def calc_return_skewness_vectorized(close_prices, returns, volume, high, low, open_price, window=60):
    """收益偏度 - 向量化版本"""
    return returns.rolling(window).skew()


def calc_return_kurtosis_vectorized(close_prices, returns, volume, high, low, open_price, window=60):
    """收益峰度 - 向量化版本"""
    return returns.rolling(window).kurt()


def calc_bid_ask_spread_vectorized(close_prices, returns, volume, high, low, open_price):
    """买卖价差代理 - 向量化版本"""
    return (high - low) / close_prices


def calc_liquidity_shock_vectorized(close_prices, returns, volume, high, low, open_price, window=20):
    """流动性冲击 - 向量化版本"""
    volume_ma = volume.rolling(window).mean()
    volume_std = volume.rolling(window).std()
    shock = (volume - volume_ma) / volume_std.replace(0, np.nan)
    return shock


def calc_price_impact_vectorized(close_prices, returns, volume, high, low, open_price, window=20):
    """价格冲击 - 向量化版本"""
    volume_std = volume.rolling(window).std()
    impact = returns / volume_std.replace(0, np.nan)
    return impact


def calc_momentum_acceleration_vectorized(close_prices, returns, volume, high, low, open_price, window=20):
    """动量加速度 - 向量化版本"""
    momentum = close_prices.pct_change(window)
    acceleration = momentum.diff(window)
    return acceleration


def calc_disposition_effect_vectorized(close_prices, returns, volume, high, low, open_price):
    """处置效应 - 向量化版本"""
    high_250d = close_prices.rolling(250).max()
    return (close_prices - high_250d) / high_250d


def calc_order_flow_imbalance_vectorized(close_prices, returns, volume, high, low, open_price):
    """订单流不平衡 - 向量化版本"""
    prev_close = close_prices.shift(1)
    gap = open_price - prev_close
    direction = np.sign(gap)
    magnitude = abs(gap) / prev_close
    return direction * magnitude


def calc_alpha_momentum_vectorized(close_prices, returns, volume, high, low, open_price, window=60):
    """Alpha 动量 - 向量化版本（简化版）"""
    # 计算市场收益率
    market_returns = returns.mean(axis=1)
    
    # 计算 beta（简化：使用相关性）
    # 注意：这是一个近似，完整的 CAPM beta 需要更复杂的计算
    alpha = returns - market_returns
    return alpha.rolling(window).sum()


def calc_correlation_breakdown_vectorized(close_prices, returns, volume, high, low, open_price, window=20):
    """相关性突变 - 向量化版本"""
    market_returns = returns.mean(axis=1)
    
    # 计算滚动相关性
    corr_recent = returns.rolling(window).corr(market_returns)
    corr_long = returns.rolling(window * 3).corr(market_returns)
    
    return corr_recent - corr_long


def calc_hurst_exponent_vectorized(close_prices, returns, volume, high, low, open_price, max_lag=20):
    """Hurst 指数 - 向量化版本（简化版）"""
    # 使用 R/S 分析的简化版本
    # 这里使用一个近似方法：计算不同时间尺度的波动率比值
    
    # 计算对数收益率
    log_returns = np.log(close_prices / close_prices.shift(1))
    
    # 计算不同时间尺度的标准差
    std_short = log_returns.rolling(5).std()
    std_long = log_returns.rolling(20).std()
    
    # Hurst 指数近似
    hurst = np.log(std_long / std_short.replace(0, np.nan)) / np.log(4)
    
    return hurst


def calc_rank_velocity_vectorized(close_prices, returns, volume, high, low, open_price, window=20):
    """排名变化速度 - 向量化版本"""
    # 横截面排名
    ranks = close_prices.rank(axis=1, pct=True)
    # 排名变化速度
    velocity = ranks.diff(window) / window
    return velocity


def calc_industry_relative_strength_vectorized(close_prices, returns, volume, high, low, open_price, window=20):
    """行业相对强度。

    ponytail: no real point-in-time industry map is passed here; returning NaN
    prevents fake equal-split "industries" from entering factor ranks.
    """
    return pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)


def calc_overreaction_reversal_vectorized(close_prices, returns, volume, high, low, open_price, window=20, threshold=2):
    """过度反应反转 - 向量化版本"""
    cumulative_returns = returns.rolling(window).sum()
    returns_std = returns.rolling(window * 3).std()
    
    is_extreme = cumulative_returns.abs() > returns_std * threshold
    
    reversal = -returns.rolling(window).sum().shift(window)
    
    # 只在极端情况下返回反转信号
    return reversal.where(is_extreme, 0)


def calc_herding_indicator_vectorized(close_prices, returns, volume, high, low, open_price, window=20):
    """羊群效应 - 向量化版本（简化版）"""
    market_returns = returns.mean(axis=1)
    market_std = market_returns.rolling(window).std()
    
    # 识别极端市场日
    extreme_days = market_returns.abs() > market_std * 2
    
    # 计算极端日和正常日的相关性
    # 注意：这是一个近似，完整的计算需要更复杂的逻辑
    corr_extreme = returns.where(extreme_days).rolling(window).corr(market_returns.where(extreme_days))
    corr_normal = returns.where(~extreme_days).rolling(window).corr(market_returns.where(~extreme_days))
    
    return corr_extreme - corr_normal


def calc_low_volatility_20_vectorized(close_prices, returns, volume, high, low, open_price, window=20):
    """短期低波动：值越低越好。"""
    return returns.rolling(window).std()


def calc_downside_volatility_20_vectorized(close_prices, returns, volume, high, low, open_price, window=20):
    """下行波动：只统计负收益波动，值越低越好。"""
    downside_returns = returns.where(returns < 0, 0)
    return downside_returns.rolling(window).std()


def calc_risk_adjusted_momentum_20_vectorized(close_prices, returns, volume, high, low, open_price, window=20):
    """风险调整动量：20 日收益除以 20 日波动。"""
    momentum = close_prices.pct_change(window)
    volatility = returns.rolling(window).std()
    return momentum / volatility.replace(0, np.nan)


def calc_trend_consistency_60_vectorized(close_prices, returns, volume, high, low, open_price, window=60):
    """趋势一致性：过去 60 日收盘价站上 20 日均线的比例。"""
    ma20 = close_prices.rolling(20).mean()
    above_ma20 = (close_prices > ma20).astype(float)
    return above_ma20.rolling(window).mean()


def calc_liquidity_strength_20_vectorized(close_prices, returns, volume, high, low, open_price, window=20):
    """流动性强度：20 日平均成交额代理，值越高越好。"""
    amount_proxy = close_prices * volume
    return amount_proxy.rolling(window).mean()


def calc_volume_price_confirmation_vectorized(close_prices, returns, volume, high, low, open_price, window=20):
    """量价确认：20 日动量乘以成交活跃度。"""
    momentum = close_prices.pct_change(window)
    volume_base = volume.rolling(window).mean().replace(0, np.nan)
    volume_ratio = (volume.rolling(5).mean() / volume_base).clip(upper=3)
    return momentum * volume_ratio


def calc_breakout_position_60_vectorized(close_prices, returns, volume, high, low, open_price, window=60):
    """60 日突破位置：越接近区间高位越强。"""
    rolling_low = close_prices.rolling(window).min()
    rolling_high = close_prices.rolling(window).max()
    return (close_prices - rolling_low) / (rolling_high - rolling_low).replace(0, np.nan)


def calc_distance_to_high_20_vectorized(close_prices, returns, volume, high, low, open_price, window=20):
    """距离 20 日高点：越接近高点越强。"""
    rolling_high = close_prices.rolling(window).max()
    return close_prices / rolling_high.replace(0, np.nan) - 1


def calc_liquidity_acceleration_20_vectorized(close_prices, returns, volume, high, low, open_price, window=20):
    """流动性加速：5 日成交额相对 20 日成交额。"""
    amount_proxy = close_prices * volume
    return amount_proxy.rolling(5).mean() / amount_proxy.rolling(window).mean().replace(0, np.nan)


def calc_low_drawdown_momentum_60_vectorized(close_prices, returns, volume, high, low, open_price, window=60):
    """低回撤动量：60 日收益除以期间最大回撤绝对值。"""
    momentum = close_prices.pct_change(window)
    running_high = close_prices.rolling(window).max()
    drawdown = close_prices / running_high.replace(0, np.nan) - 1
    max_drawdown = drawdown.rolling(window).min().abs()
    return momentum / max_drawdown.replace(0, np.nan)


def calc_market_relative_strength_60_vectorized(close_prices, returns, volume, high, low, open_price, window=60):
    """相对大盘强度：个股 60 日收益减去全市场等权 60 日收益。"""
    stock_return = close_prices.pct_change(window)
    market_return = close_prices.mean(axis=1).pct_change(window)
    return stock_return.sub(market_return, axis=0)


def calc_volatility_squeeze_20_vectorized(close_prices, returns, volume, high, low, open_price, window=20):
    """波动收缩：20 日平均振幅相对 60 日平均振幅，越低越收缩。"""
    daily_range = (high - low) / close_prices.replace(0, np.nan)
    return daily_range.rolling(window).mean() / daily_range.rolling(window * 3).mean().replace(0, np.nan)


def calc_dry_up_breakout_60_vectorized(close_prices, returns, volume, high, low, open_price, window=60):
    """缩量突破：区间高位位置乘以低量能拥挤度。"""
    rolling_low = close_prices.rolling(window).min()
    rolling_high = close_prices.rolling(window).max()
    position = (close_prices - rolling_low) / (rolling_high - rolling_low).replace(0, np.nan)
    volume_ratio = volume.rolling(5).mean() / volume.rolling(20).mean().replace(0, np.nan)
    calm_volume = (1 / volume_ratio.replace(0, np.nan)).clip(upper=3)
    return position * calm_volume


def calc_money_flow_persistence_20_vectorized(close_prices, returns, volume, high, low, open_price, window=20):
    """资金流持续性：涨跌方向加权成交额占比。"""
    amount_proxy = close_prices * volume
    signed_amount = amount_proxy * np.sign(returns.fillna(0))
    return signed_amount.rolling(window).sum() / amount_proxy.rolling(window).sum().replace(0, np.nan)


def calc_short_reversal_5_vectorized(close_prices, returns, volume, high, low, open_price, window=5):
    """短期反转：近 5 日跌幅越大，反转分越高。"""
    return -close_prices.pct_change(window)


def calc_medium_reversal_20_vectorized(close_prices, returns, volume, high, low, open_price, window=20):
    """中期反转：近 20 日跌幅越大，反转分越高。"""
    return -close_prices.pct_change(window)


def calc_calm_pullback_20_vectorized(close_prices, returns, volume, high, low, open_price, window=20):
    """低波动回调：回调幅度除以波动，偏好温和下跌后的修复。"""
    pullback = -close_prices.pct_change(window)
    volatility = returns.rolling(window).std()
    return pullback / volatility.replace(0, np.nan)


def calc_drawdown_recovery_60_vectorized(close_prices, returns, volume, high, low, open_price, window=60):
    """回撤修复：距离 60 日高点越远但近 5 日越强，分数越高。"""
    high_60 = close_prices.rolling(window).max()
    drawdown = close_prices / high_60.replace(0, np.nan) - 1
    rebound = close_prices.pct_change(5)
    return (-drawdown) * rebound


# ============================================================================
# 向量化因子注册表
# ============================================================================

FACTOR_REGISTRY_VECTORIZED = {
    'return_skewness': {
        'func': calc_return_skewness_vectorized,
        'category': '统计分布',
        'description': '收益分布的不对称性',
        'direction': 'positive',
    },
    'return_kurtosis': {
        'func': calc_return_kurtosis_vectorized,
        'category': '统计分布',
        'description': '收益分布的尾部厚度',
        'direction': 'negative',
    },
    'bid_ask_spread': {
        'func': calc_bid_ask_spread_vectorized,
        'category': '市场微观结构',
        'description': '用日内振幅代理买卖价差',
        'direction': 'negative',
    },
    'liquidity_shock': {
        'func': calc_liquidity_shock_vectorized,
        'category': '市场微观结构',
        'description': '成交量突然放大',
        'direction': 'positive',
    },
    'price_impact': {
        'func': calc_price_impact_vectorized,
        'category': '市场微观结构',
        'description': '单位成交量的价格变化',
        'direction': 'negative',
    },
    'momentum_acceleration': {
        'func': calc_momentum_acceleration_vectorized,
        'category': '横截面关系',
        'description': '动量的变化速度（二阶导数）',
        'direction': 'positive',
    },
    'disposition_effect': {
        'func': calc_disposition_effect_vectorized,
        'category': '行为金融',
        'description': '处置效应代理：距离 250 日高点的距离，不等同于账户级处置效应定义',
        'direction': 'negative',
    },
    'order_flow_imbalance': {
        'func': calc_order_flow_imbalance_vectorized,
        'category': '市场微观结构',
        'description': '开盘跳空方向 × 幅度',
        'direction': 'positive',
    },
    'alpha_momentum': {
        'func': calc_alpha_momentum_vectorized,
        'category': '横截面关系',
        'description': 'Alpha 动量代理：个股收益减等权市场收益的滚动和，不是完整 CAPM/多因子 alpha',
        'direction': 'positive',
    },
    'correlation_breakdown': {
        'func': calc_correlation_breakdown_vectorized,
        'category': '横截面关系',
        'description': '与市场相关性突然下降',
        'direction': 'positive',
    },
    'hurst_exponent': {
        'func': calc_hurst_exponent_vectorized,
        'category': '统计分布',
        'description': 'Hurst 代理：短长窗口波动率比值，不等同于严格 R/S 或 DFA 估计',
        'direction': 'positive',
    },
    'rank_velocity': {
        'func': calc_rank_velocity_vectorized,
        'category': '横截面关系',
        'description': '股票在全市场的排名变化速度',
        'direction': 'positive',
    },
    'industry_relative_strength': {
        'func': calc_industry_relative_strength_vectorized,
        'category': '横截面关系',
        'description': '已禁用：缺少真实 PIT 行业分类时返回空值，避免等分股票伪行业',
        'direction': 'positive',
    },
    'overreaction_reversal': {
        'func': calc_overreaction_reversal_vectorized,
        'category': '行为金融',
        'description': '极端收益后的反转',
        'direction': 'positive',
    },
    'herding_indicator': {
        'func': calc_herding_indicator_vectorized,
        'category': '行为金融',
        'description': '极端市场日的相关性',
        'direction': 'negative',
    },
    'low_volatility_20': {
        'func': calc_low_volatility_20_vectorized,
        'category': '风险质量',
        'description': '20 日收益波动率',
        'direction': 'negative',
    },
    'downside_volatility_20': {
        'func': calc_downside_volatility_20_vectorized,
        'category': '风险质量',
        'description': '20 日下行波动率',
        'direction': 'negative',
    },
    'risk_adjusted_momentum_20': {
        'func': calc_risk_adjusted_momentum_20_vectorized,
        'category': '趋势质量',
        'description': '20 日收益除以 20 日波动',
        'direction': 'positive',
    },
    'trend_consistency_60': {
        'func': calc_trend_consistency_60_vectorized,
        'category': '趋势质量',
        'description': '60 日内站上 20 日均线的比例',
        'direction': 'positive',
    },
    'liquidity_strength_20': {
        'func': calc_liquidity_strength_20_vectorized,
        'category': '流动性质量',
        'description': '20 日平均成交额代理',
        'direction': 'positive',
    },
    'volume_price_confirmation': {
        'func': calc_volume_price_confirmation_vectorized,
        'category': '量价关系',
        'description': '20 日动量与成交活跃度确认',
        'direction': 'positive',
    },
    'breakout_position_60': {
        'func': calc_breakout_position_60_vectorized,
        'category': '趋势突破',
        'description': '收盘价在 60 日区间中的相对位置',
        'direction': 'positive',
    },
    'distance_to_high_20': {
        'func': calc_distance_to_high_20_vectorized,
        'category': '趋势突破',
        'description': '距离 20 日高点的比例',
        'direction': 'positive',
    },
    'liquidity_acceleration_20': {
        'func': calc_liquidity_acceleration_20_vectorized,
        'category': '流动性质量',
        'description': '5 日成交额相对 20 日成交额',
        'direction': 'positive',
    },
    'low_drawdown_momentum_60': {
        'func': calc_low_drawdown_momentum_60_vectorized,
        'category': '趋势质量',
        'description': '60 日收益除以 60 日最大回撤',
        'direction': 'positive',
    },
    'market_relative_strength_60': {
        'func': calc_market_relative_strength_60_vectorized,
        'category': '相对强度',
        'description': '个股 60 日收益减全市场等权 60 日收益',
        'direction': 'positive',
    },
    'volatility_squeeze_20': {
        'func': calc_volatility_squeeze_20_vectorized,
        'category': '波动结构',
        'description': '20 日平均振幅相对 60 日平均振幅',
        'direction': 'negative',
    },
    'dry_up_breakout_60': {
        'func': calc_dry_up_breakout_60_vectorized,
        'category': '趋势突破',
        'description': '60 日区间高位位置乘以缩量确认',
        'direction': 'positive',
    },
    'money_flow_persistence_20': {
        'func': calc_money_flow_persistence_20_vectorized,
        'category': '资金流',
        'description': '20 日涨跌方向加权成交额占比',
        'direction': 'positive',
    },
    'short_reversal_5': {
        'func': calc_short_reversal_5_vectorized,
        'category': '反转修复',
        'description': '近 5 日反向收益',
        'direction': 'positive',
    },
    'medium_reversal_20': {
        'func': calc_medium_reversal_20_vectorized,
        'category': '反转修复',
        'description': '近 20 日反向收益',
        'direction': 'positive',
    },
    'calm_pullback_20': {
        'func': calc_calm_pullback_20_vectorized,
        'category': '反转修复',
        'description': '20 日回调幅度除以波动',
        'direction': 'positive',
    },
    'drawdown_recovery_60': {
        'func': calc_drawdown_recovery_60_vectorized,
        'category': '反转修复',
        'description': '60 日回撤深度乘以近 5 日修复强度',
        'direction': 'positive',
    },
}
