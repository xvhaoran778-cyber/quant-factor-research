"""
新因子库 - 15 个与 Alpha191 完全不同的因子

包含四类因子：
1. 横截面关系因子 (5 个)
2. 市场微观结构因子 (4 个)
3. 行为金融因子 (3 个)
4. 统计分布因子 (3 个)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional


def rank_velocity(close_prices: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    因子 1: 排名变化速度 (Rank Velocity)
    
    股票在全市场的排名变化速度
    经济含义: 排名快速上升的股票可能有正向信息驱动
    
    Args:
        close_prices: 收盘价矩阵 (日期 x 股票)
        window: 计算窗口
    
    Returns:
        排名变化速度值
    """
    # 横截面排名
    ranks = close_prices.rank(pct=True, axis=1)
    # 排名变化速度
    velocity = ranks.diff(window) / window
    return velocity.iloc[-1]


def alpha_momentum(stock_returns: pd.Series, market_returns: pd.Series, 
                   window: int = 60) -> float:
    """
    因子 2: Beta 调整动量 (Alpha Momentum)
    
    CAPM 残差的动量（去除市场影响）
    经济含义: 去除市场 beta 后的真实选股能力
    
    Args:
        stock_returns: 股票收益率序列
        market_returns: 市场收益率序列
        window: 计算窗口
    
    Returns:
        累计超额收益
    """
    # 计算 beta
    covariance = stock_returns.rolling(window).cov(market_returns)
    market_variance = market_returns.rolling(window).var()
    beta = covariance / market_variance.replace(0, np.nan)
    
    # 计算 alpha (残差)
    alpha = stock_returns - beta * market_returns
    
    # 累计 alpha
    return alpha.rolling(window).sum().iloc[-1]


def correlation_breakdown(stock_returns: pd.Series, market_returns: pd.Series, 
                          window: int = 20) -> float:
    """
    因子 3: 相关性突变 (Correlation Breakdown)
    
    与市场相关性突然下降
    经济含义: 相关性下降可能意味着个股特有信息开始主导
    
    Args:
        stock_returns: 股票收益率序列
        market_returns: 市场收益率序列
        window: 计算窗口
    
    Returns:
        相关性变化值
    """
    # 近期相关性
    corr_recent = stock_returns.rolling(window).corr(market_returns)
    # 长期相关性
    corr_long = stock_returns.rolling(window * 3).corr(market_returns)
    
    # 相关性下降为正
    return (corr_recent - corr_long).iloc[-1]


def industry_relative_strength(stock_returns: pd.Series, 
                               industry_returns: pd.Series, 
                               window: int = 20) -> float:
    """
    因子 4: 行业相对强度 (Industry Relative Strength)
    
    相对行业的超额收益
    经济含义: 行业内选股能力
    
    Args:
        stock_returns: 股票收益率序列
        industry_returns: 行业收益率序列
        window: 计算窗口
    
    Returns:
        相对行业的超额收益
    """
    stock_cumret = stock_returns.rolling(window).sum()
    industry_cumret = industry_returns.rolling(window).sum()
    return (stock_cumret - industry_cumret).iloc[-1]


def momentum_acceleration(close: pd.Series, window: int = 20) -> float:
    """
    因子 5: 横截面动量加速度 (Cross-sectional Momentum Acceleration)
    
    动量的变化速度（二阶导数）
    经济含义: 动量加速的股票可能处于趋势初期
    
    Args:
        close: 收盘价序列
        window: 计算窗口
    
    Returns:
        动量加速度
    """
    # 计算动量
    momentum = close.pct_change(window)
    # 计算加速度（动量的变化）
    acceleration = momentum.diff(window)
    return acceleration.iloc[-1]


def bid_ask_spread(high: pd.Series, low: pd.Series, close: pd.Series) -> float:
    """
    因子 6: 买卖价差代理 (Bid-Ask Spread Proxy)
    
    用日内振幅代理买卖价差
    经济含义: 价差大的股票流动性差，可能有流动性溢价
    
    Args:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        买卖价差代理值
    """
    spread = (high - low) / close
    return spread.iloc[-1]


def order_flow_imbalance(open_price: pd.Series, prev_close: pd.Series) -> float:
    """
    因子 7: 订单流不平衡 (Order Flow Imbalance)
    
    开盘跳空方向 × 幅度
    经济含义: 开盘跳空反映隔夜信息，正向跳空可能有持续效应
    
    Args:
        open_price: 开盘价序列
        prev_close: 前一日收盘价序列
    
    Returns:
        订单流不平衡值
    """
    gap = open_price - prev_close
    direction = np.sign(gap)
    magnitude = abs(gap) / prev_close
    return (direction * magnitude).iloc[-1]


def price_impact(close: pd.Series, volume: pd.Series, window: int = 20) -> float:
    """
    因子 8: 价格冲击 (Price Impact)
    
    单位成交量的价格变化
    经济含义: 价格冲击小的股票流动性好，大资金更容易进出
    
    Args:
        close: 收盘价序列
        volume: 成交量序列
        window: 计算窗口
    
    Returns:
        标准化价格冲击
    """
    returns = close.pct_change()
    volume_std = volume.rolling(window).std()
    impact = returns / volume_std.replace(0, np.nan)
    return impact.iloc[-1]


def liquidity_shock(volume: pd.Series, window: int = 20) -> float:
    """
    因子 9: 流动性冲击 (Liquidity Shock)
    
    成交量突然放大
    经济含义: 流动性冲击可能意味着信息事件或大资金介入
    
    Args:
        volume: 成交量序列
        window: 计算窗口
    
    Returns:
        流动性冲击值（标准差倍数）
    """
    volume_ma = volume.rolling(window).mean()
    volume_std = volume.rolling(window).std()
    
    # 避免除以零
    std_value = volume_std.iloc[-1]
    if std_value == 0 or pd.isna(std_value):
        return 0.0
    
    shock = (volume.iloc[-1] - volume_ma.iloc[-1]) / std_value
    return shock


def disposition_effect(close: pd.Series, high_250d: pd.Series) -> float:
    """
    因子 10: 处置效应 (Disposition Effect)
    
    距离 250 日高点的距离
    经济含义: 接近高点的股票可能有"解套卖出"压力
    
    Args:
        close: 收盘价序列
        high_250d: 250 日最高价序列
    
    Returns:
        距离高点的比例
    """
    return ((close - high_250d) / high_250d).iloc[-1]


def herding_indicator(stock_returns: pd.Series, market_returns: pd.Series, 
                      window: int = 20) -> float:
    """
    因子 11: 羊群效应 (Herding Indicator)
    
    极端市场日的相关性
    经济含义: 羊群效应强的股票可能在市场反转时跌得更惨
    
    Args:
        stock_returns: 股票收益率序列
        market_returns: 市场收益率序列
        window: 计算窗口
    
    Returns:
        羊群效应指标
    """
    # 识别极端市场日（超过 2 倍标准差）
    market_std = market_returns.rolling(window).std()
    extreme_days = market_returns.abs() > market_std * 2
    
    # 极端日的相关性
    corr_extreme = stock_returns[extreme_days].corr(market_returns[extreme_days])
    # 正常日的相关性
    corr_normal = stock_returns[~extreme_days].corr(market_returns[~extreme_days])
    
    # 羊群效应 = 极端日相关性 - 正常日相关性
    return corr_extreme - corr_normal


def overreaction_reversal(returns: pd.Series, window: int = 20, 
                          threshold: float = 2) -> float:
    """
    因子 12: 过度反应反转 (Overreaction Reversal)
    
    极端收益后的反转
    经济含义: 过度反应后会有反转
    
    Args:
        returns: 收益率序列
        window: 计算窗口
        threshold: 极端值阈值（标准差倍数）
    
    Returns:
        反转信号值
    """
    # 累计收益
    cumulative_returns = returns.rolling(window).sum()
    # 识别极端收益
    returns_std = returns.rolling(window * 3).std()
    is_extreme = cumulative_returns.abs() > returns_std * threshold
    
    # 反转信号（反向）
    reversal = -returns.rolling(window).sum().shift(window)
    
    # 只在极端情况下返回反转信号
    return reversal.where(is_extreme, 0).iloc[-1]


def return_skewness(returns: pd.Series, window: int = 60) -> float:
    """
    因子 13: 收益偏度 (Return Skewness)
    
    收益分布的不对称性
    经济含义: 正偏度意味着大涨概率大于大跌，可能有正向尾部风险
    
    Args:
        returns: 收益率序列
        window: 计算窗口
    
    Returns:
        偏度值
    """
    # 检查数据是否足够
    if len(returns.dropna()) < window:
        return np.nan
    
    skew_series = returns.rolling(window).skew()
    if skew_series.empty or skew_series.dropna().empty:
        return np.nan
    
    return float(skew_series.iloc[-1])


def return_kurtosis(returns: pd.Series, window: int = 60) -> float:
    """
    因子 14: 收益峰度 (Return Kurtosis)
    
    收益分布的尾部厚度
    经济含义: 高峰度意味着极端事件概率高，风险更大
    
    Args:
        returns: 收益率序列
        window: 计算窗口
    
    Returns:
        峰度值
    """
    # 检查数据是否足够
    if len(returns.dropna()) < window:
        return np.nan
    
    kurt_series = returns.rolling(window).kurt()
    if kurt_series.empty or kurt_series.dropna().empty:
        return np.nan
    
    return float(kurt_series.iloc[-1])


def hurst_exponent(prices: pd.Series, max_lag: int = 20) -> float:
    """
    因子 15: Hurst 指数 (Hurst Exponent)
    
    衡量时间序列的长记忆性
    经济含义: H > 0.5 说明有趋势持续性，< 0.5 说明有反转
    
    Args:
        prices: 价格序列
        max_lag: 最大滞后阶数
    
    Returns:
        Hurst 指数
    """
    lags = range(2, max_lag)
    tau = []
    
    for lag in lags:
        # 计算滞后 lag 的差分标准差
        diff = prices.diff(lag).dropna()
        tau.append(np.std(diff))
    
    # 对数线性回归
    log_lags = np.log(list(lags))
    log_tau = np.log(tau)
    
    # 线性回归
    reg = np.polyfit(log_lags, log_tau, 1)
    
    return reg[0]  # Hurst 指数


# 因子注册表
FACTOR_REGISTRY = {
    'rank_velocity': {
        'func': rank_velocity,
        'category': '横截面关系',
        'description': '股票在全市场的排名变化速度',
        'direction': 'positive',  # 值越大越好
        'requires': ['close_prices_matrix']
    },
    'alpha_momentum': {
        'func': alpha_momentum,
        'category': '横截面关系',
        'description': 'CAPM 残差的动量（去除市场影响）',
        'direction': 'positive',
        'requires': ['stock_returns', 'market_returns']
    },
    'correlation_breakdown': {
        'func': correlation_breakdown,
        'category': '横截面关系',
        'description': '与市场相关性突然下降',
        'direction': 'positive',
        'requires': ['stock_returns', 'market_returns']
    },
    'industry_relative_strength': {
        'func': industry_relative_strength,
        'category': '横截面关系',
        'description': '相对行业的超额收益',
        'direction': 'positive',
        'requires': ['stock_returns', 'industry_returns']
    },
    'momentum_acceleration': {
        'func': momentum_acceleration,
        'category': '横截面关系',
        'description': '动量的变化速度（二阶导数）',
        'direction': 'positive',
        'requires': ['close']
    },
    'bid_ask_spread': {
        'func': bid_ask_spread,
        'category': '市场微观结构',
        'description': '用日内振幅代理买卖价差',
        'direction': 'negative',  # 流动性差可能有溢价，但也可能风险大
        'requires': ['high', 'low', 'close']
    },
    'order_flow_imbalance': {
        'func': order_flow_imbalance,
        'category': '市场微观结构',
        'description': '开盘跳空方向 × 幅度',
        'direction': 'positive',
        'requires': ['open_price', 'prev_close']
    },
    'price_impact': {
        'func': price_impact,
        'category': '市场微观结构',
        'description': '单位成交量的价格变化',
        'direction': 'negative',  # 价格冲击小更好
        'requires': ['close', 'volume']
    },
    'liquidity_shock': {
        'func': liquidity_shock,
        'category': '市场微观结构',
        'description': '成交量突然放大',
        'direction': 'positive',
        'requires': ['volume']
    },
    'disposition_effect': {
        'func': disposition_effect,
        'category': '行为金融',
        'description': '距离 250 日高点的距离',
        'direction': 'negative',  # 接近高点有卖出压力
        'requires': ['close', 'high_250d']
    },
    'herding_indicator': {
        'func': herding_indicator,
        'category': '行为金融',
        'description': '极端市场日的相关性',
        'direction': 'negative',  # 羊群效应强风险大
        'requires': ['stock_returns', 'market_returns']
    },
    'overreaction_reversal': {
        'func': overreaction_reversal,
        'category': '行为金融',
        'description': '极端收益后的反转',
        'direction': 'positive',  # 反转信号
        'requires': ['returns']
    },
    'return_skewness': {
        'func': return_skewness,
        'category': '统计分布',
        'description': '收益分布的不对称性',
        'direction': 'positive',  # 正偏度好
        'requires': ['returns']
    },
    'return_kurtosis': {
        'func': return_kurtosis,
        'category': '统计分布',
        'description': '收益分布的尾部厚度',
        'direction': 'negative',  # 高峰度风险大
        'requires': ['returns']
    },
    'hurst_exponent': {
        'func': hurst_exponent,
        'category': '统计分布',
        'description': '衡量时间序列的长记忆性',
        'direction': 'positive',  # H > 0.5 有趋势持续性
        'requires': ['prices']
    }
}


def compute_all_factors(data: Dict[str, pd.DataFrame], 
                        factor_names: Optional[List[str]] = None) -> pd.DataFrame:
    """
    计算所有因子
    
    Args:
        data: 数据字典，包含:
            - close_prices_matrix: 收盘价矩阵 (日期 x 股票)
            - stock_returns: 股票收益率
            - market_returns: 市场收益率
            - close: 收盘价
            - open: 开盘价
            - high: 最高价
            - low: 最低价
            - volume: 成交量
            - returns: 收益率
            - prices: 价格序列
            - high_250d: 250日最高价
            - prev_close: 前一日收盘价
            - industry_returns: 行业收益率（可选）
        factor_names: 要计算的因子名称列表，None 表示全部
    
    Returns:
        因子值 DataFrame，索引为 (date, symbol) 的 MultiIndex
    """
    if factor_names is None:
        factor_names = list(FACTOR_REGISTRY.keys())
    
    # 获取所有日期和股票
    dates = data['close'].index.tolist()
    symbols = data['close'].columns.tolist()
    
    # 创建 MultiIndex
    index = pd.MultiIndex.from_product([dates, symbols], names=['date', 'symbol'])
    
    # 初始化结果 DataFrame
    results = pd.DataFrame(index=index, columns=factor_names, dtype=float)
    
    print(f"开始计算 {len(factor_names)} 个因子，共 {len(symbols)} 只股票")
    
    # 逐股票计算因子（时间序列方式）
    for symbol_idx, symbol in enumerate(symbols):
        if symbol_idx % 500 == 0:
            print(f"计算因子: {symbol_idx+1}/{len(symbols)} 股票")
        
        # 为该股票准备历史数据
        symbol_data = {}
        for key, value in data.items():
            if isinstance(value, pd.DataFrame):
                if symbol in value.columns:
                    symbol_data[key] = value[symbol]  # 获取该股票的时间序列
            elif isinstance(value, pd.Series):
                symbol_data[key] = value  # 市场收益率等是 Series
        
            # 为每个日期计算因子
            for date_idx, date in enumerate(dates):
                # 获取截止到当前日期的历史数据
                hist_data = {}
                for key, value in symbol_data.items():
                    if isinstance(value, pd.Series):
                        # 获取截止到当前日期的数据
                        hist_data[key] = value[:date]
                    else:
                        hist_data[key] = value
                
                # 计算每个因子
                for factor_name in factor_names:
                    if factor_name not in FACTOR_REGISTRY:
                        continue
                    
                    factor_info = FACTOR_REGISTRY[factor_name]
                    func = factor_info['func']
                    requires = factor_info['requires']
                    
                    # 检查所需数据
                    missing_data = [r for r in requires if r not in hist_data]
                    if missing_data:
                        continue
                    
                    try:
                        # 准备参数（传入历史数据）
                        kwargs = {}
                        for r in requires:
                            if r == 'close_prices_matrix':
                                # rank_velocity 需要整个矩阵
                                kwargs[r] = data['close'][:date]
                            elif r in ['market_returns', 'industry_returns']:
                                # 市场/行业收益率是 Series
                                kwargs[r] = hist_data[r][:date]
                            else:
                                # 股票的时间序列数据
                                kwargs[r] = hist_data[r][:date]
                        
                        # 检查数据是否足够
                        # 对于需要 rolling 计算的因子，至少需要 window 个数据点
                        min_data_points = 60  # 默认最小数据点数
                        if 'window' in func.__code__.co_varnames:
                            # 如果函数有 window 参数，使用它
                            import inspect
                            sig = inspect.signature(func)
                            if 'window' in sig.parameters:
                                min_data_points = sig.parameters['window'].default
                        
                        # 检查主要数据序列的长度
                        main_data = None
                        for key in ['returns', 'close', 'prices']:
                            if key in kwargs:
                                main_data = kwargs[key]
                                break
                        
                        if main_data is not None:
                            non_null_count = main_data.dropna().shape[0]
                            if non_null_count < min_data_points:
                                # 数据不足，跳过这个日期
                                continue
                        
                        # 计算因子值
                        value = func(**kwargs)
                        
                        # 确保返回的是标量值
                        if isinstance(value, pd.Series):
                            # 如果是 Series，取最后一个非 NaN 值
                            value = value.dropna().iloc[-1] if not value.dropna().empty else np.nan
                        
                        # 存储结果
                        if pd.notna(value):
                            results.loc[(date, symbol), factor_name] = value
                            
                    except Exception:
                        # 计算失败时保持 NaN
                        pass
    
    print(f"因子计算完成")
    return results


def get_factor_info() -> pd.DataFrame:
    """获取所有因子的信息"""
    info = []
    for name, factor in FACTOR_REGISTRY.items():
        info.append({
            'factor_name': name,
            'category': factor['category'],
            'description': factor['description'],
            'direction': factor['direction'],
            'requires': ', '.join(factor['requires'])
        })
    return pd.DataFrame(info)
