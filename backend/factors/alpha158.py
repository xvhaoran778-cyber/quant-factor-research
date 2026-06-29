"""Alpha158因子库 - 158个量价因子"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from loguru import logger


class Alpha158:
    """Alpha158因子库
    
    包含158个量价因子，参考QLib的Alpha158实现
    适合短线和中线策略
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.factor_names = []
    
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有Alpha158因子
        
        Args:
            df: 包含 OHLCV 数据的 DataFrame
                必须包含: open, high, low, close, volume
        
        Returns:
            添加了158个因子的 DataFrame
        """
        df = df.copy()
        
        # 确保列名小写
        df.columns = [c.lower() for c in df.columns]
        
        # 基础特征
        df = self._calc_basic_features(df)
        
        # KBAR特征
        df = self._calc_kbar_features(df)
        
        # 价格变化特征
        df = self._calc_price_change_features(df)
        
        # 成交量特征
        df = self._calc_volume_features(df)
        
        # 滚动统计特征
        df = self._calc_rolling_features(df)
        
        # 技术指标特征
        df = self._calc_technical_features(df)
        
        # 获取所有因子列名
        self.factor_names = [c for c in df.columns if c.startswith('alpha_')]
        
        logger.info(f"计算完成，共{len(self.factor_names)}个因子")
        return df
    
    def _calc_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算基础特征"""
        # 收益率
        for d in [1, 2, 3, 5, 10, 20]:
            df[f'alpha_return_{d}d'] = df['close'].pct_change(d)
        
        # 对数收益率
        for d in [1, 5, 10, 20]:
            df[f'alpha_log_return_{d}d'] = np.log(df['close'] / df['close'].shift(d))
        
        # 价格位置
        df['alpha_price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
        
        return df
    
    def _calc_kbar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算KBAR特征"""
        # 实体
        df['alpha_body'] = df['close'] - df['open']
        df['alpha_body_pct'] = df['alpha_body'] / (df['open'] + 1e-8)
        
        # 上影线
        df['alpha_upper_shadow'] = df['high'] - df[['close', 'open']].max(axis=1)
        df['alpha_upper_shadow_pct'] = df['alpha_upper_shadow'] / (df['high'] - df['low'] + 1e-8)
        
        # 下影线
        df['alpha_lower_shadow'] = df[['close', 'open']].min(axis=1) - df['low']
        df['alpha_lower_shadow_pct'] = df['alpha_lower_shadow'] / (df['high'] - df['low'] + 1e-8)
        
        # 实体占比
        df['alpha_body_ratio'] = abs(df['alpha_body']) / (df['high'] - df['low'] + 1e-8)
        
        # 涨跌方向
        df['alpha_direction'] = np.sign(df['alpha_body'])
        
        # 连涨连跌天数
        df['alpha_consecutive_up'] = (df['alpha_direction'] > 0).astype(int).groupby(
            (df['alpha_direction'] != df['alpha_direction'].shift()).cumsum()
        ).cumsum()
        df['alpha_consecutive_down'] = (df['alpha_direction'] < 0).astype(int).groupby(
            (df['alpha_direction'] != df['alpha_direction'].shift()).cumsum()
        ).cumsum()
        
        return df
    
    def _calc_price_change_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算价格变化特征"""
        # 价格变化
        df['alpha_price_change'] = df['close'] - df['close'].shift(1)
        df['alpha_price_change_pct'] = df['close'].pct_change()
        
        # 最高价/最低价变化
        df['alpha_high_change'] = df['high'] - df['high'].shift(1)
        df['alpha_low_change'] = df['low'] - df['low'].shift(1)
        
        # 价格波动范围
        df['alpha_range'] = df['high'] - df['low']
        df['alpha_range_pct'] = df['alpha_range'] / (df['close'] + 1e-8)
        
        # 价格缺口
        df['alpha_gap'] = df['open'] - df['close'].shift(1)
        df['alpha_gap_pct'] = df['alpha_gap'] / (df['close'].shift(1) + 1e-8)
        
        # 价格相对位置
        for d in [5, 10, 20, 60]:
            high_d = df['high'].rolling(d).max()
            low_d = df['low'].rolling(d).min()
            df[f'alpha_price_position_{d}d'] = (df['close'] - low_d) / (high_d - low_d + 1e-8)
        
        return df
    
    def _calc_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算成交量特征"""
        # 成交量变化
        df['alpha_volume_change'] = df['volume'].pct_change()
        
        # 量比
        for d in [5, 10, 20]:
            df[f'alpha_volume_ratio_{d}d'] = df['volume'] / (df['volume'].rolling(d).mean() + 1e-8)
        
        # 成交量波动
        df['alpha_volume_std_5d'] = df['volume'].rolling(5).std() / (df['volume'].rolling(5).mean() + 1e-8)
        df['alpha_volume_std_20d'] = df['volume'].rolling(20).std() / (df['volume'].rolling(20).mean() + 1e-8)
        
        # 量价相关性
        for d in [5, 10, 20]:
            df[f'alpha_price_volume_corr_{d}d'] = df['close'].rolling(d).corr(df['volume'])
        
        # OBV
        obv = [0]
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv.append(obv[-1] + df['volume'].iloc[i])
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv.append(obv[-1] - df['volume'].iloc[i])
            else:
                obv.append(obv[-1])
        df['alpha_obv'] = obv
        df['alpha_obv_ma20'] = df['alpha_obv'].rolling(20).mean()
        
        # VWAP
        df['alpha_vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        df['alpha_price_vwap_ratio'] = df['close'] / (df['alpha_vwap'] + 1e-8)
        
        return df
    
    def _calc_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算滚动统计特征"""
        # 均值
        for d in [5, 10, 20, 60]:
            df[f'alpha_ma_{d}'] = df['close'].rolling(d).mean()
            df[f'alpha_price_ma_ratio_{d}'] = df['close'] / (df[f'alpha_ma_{d}'] + 1e-8)
        
        # 标准差
        for d in [5, 10, 20]:
            df[f'alpha_std_{d}'] = df['close'].rolling(d).std()
            df[f'alpha_std_ratio_{d}'] = df[f'alpha_std_{d}'] / (df['close'] + 1e-8)
        
        # 偏度
        for d in [20, 60]:
            df[f'alpha_skew_{d}'] = df['close'].pct_change().rolling(d).skew()
        
        # 峰度
        for d in [20, 60]:
            df[f'alpha_kurt_{d}'] = df['close'].pct_change().rolling(d).kurt()
        
        # 分位数
        for d in [5, 10, 20]:
            df[f'alpha_quantile_{d}'] = df['close'].rolling(d).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
            )
        
        # 最大值/最小值
        for d in [5, 10, 20, 60]:
            df[f'alpha_high_{d}'] = df['high'].rolling(d).max()
            df[f'alpha_low_{d}'] = df['low'].rolling(d).min()
            df[f'alpha_range_{d}'] = df[f'alpha_high_{d}'] - df[f'alpha_low_{d}']
        
        return df
    
    def _calc_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标特征"""
        # RSI
        for period in [6, 14, 24]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / (loss + 1e-8)
            df[f'alpha_rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['alpha_macd_dif'] = ema12 - ema26
        df['alpha_macd_dea'] = df['alpha_macd_dif'].ewm(span=9, adjust=False).mean()
        df['alpha_macd_hist'] = (df['alpha_macd_dif'] - df['alpha_macd_dea']) * 2
        
        # KDJ
        low_9 = df['low'].rolling(9).min()
        high_9 = df['high'].rolling(9).max()
        rsv = (df['close'] - low_9) / (high_9 - low_9 + 1e-8) * 100
        df['alpha_kdj_k'] = rsv.ewm(com=2, adjust=False).mean()
        df['alpha_kdj_d'] = df['alpha_kdj_k'].ewm(com=2, adjust=False).mean()
        df['alpha_kdj_j'] = 3 * df['alpha_kdj_k'] - 2 * df['alpha_kdj_d']
        
        # 布林带
        df['alpha_boll_mid'] = df['close'].rolling(20).mean()
        df['alpha_boll_std'] = df['close'].rolling(20).std()
        df['alpha_boll_upper'] = df['alpha_boll_mid'] + 2 * df['alpha_boll_std']
        df['alpha_boll_lower'] = df['alpha_boll_mid'] - 2 * df['alpha_boll_std']
        df['alpha_boll_width'] = (df['alpha_boll_upper'] - df['alpha_boll_lower']) / (df['alpha_boll_mid'] + 1e-8)
        df['alpha_boll_position'] = (df['close'] - df['alpha_boll_lower']) / (df['alpha_boll_upper'] - df['alpha_boll_lower'] + 1e-8)
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['alpha_atr_14'] = tr.rolling(14).mean()
        df['alpha_atr_ratio'] = df['alpha_atr_14'] / (df['close'] + 1e-8)
        
        # 威廉指标
        for period in [14, 28]:
            high_n = df['high'].rolling(period).max()
            low_n = df['low'].rolling(period).min()
            df[f'alpha_williams_{period}'] = (high_n - df['close']) / (high_n - low_n + 1e-8) * -100
        
        # CCI
        for period in [14, 20]:
            tp = (df['high'] + df['low'] + df['close']) / 3
            ma_tp = tp.rolling(period).mean()
            md_tp = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
            df[f'alpha_cci_{period}'] = (tp - ma_tp) / (0.015 * md_tp + 1e-8)
        
        return df
    
    def get_factor_names(self) -> List[str]:
        """获取所有因子名称"""
        return self.factor_names
    
    def get_factor_count(self) -> int:
        """获取因子数量"""
        return len(self.factor_names)
