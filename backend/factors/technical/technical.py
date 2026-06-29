"""Technical Factors - 技术因子计算"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from loguru import logger


class TechnicalFactors:
    """技术因子计算器"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
    
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有技术因子
        
        Args:
            df: 包含 OHLCV 数据的 DataFrame
        
        Returns:
            添加了技术因子的 DataFrame
        """
        df = df.copy()
        
        # 动量因子
        df = self._calc_momentum(df)
        
        # 波动率因子
        df = self._calc_volatility(df)
        
        # 成交量因子
        df = self._calc_volume_factors(df)
        
        # 趋势因子
        df = self._calc_trend(df)
        
        return df
    
    def _calc_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算动量因子"""
        # ROC (Rate of Change)
        for period in [5, 10, 20, 60]:
            df[f'roc_{period}'] = df['close'].pct_change(period) * 100
        
        # 动量
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['close'] - df['close'].shift(period)
        
        # 相对强弱
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 威廉指标
        high_14 = df['high'].rolling(window=14).max()
        low_14 = df['low'].rolling(window=14).min()
        df['williams_r'] = (high_14 - df['close']) / (high_14 - low_14) * -100
        
        return df
    
    def _calc_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算波动率因子"""
        # 历史波动率
        for period in [5, 10, 20]:
            df[f'volatility_{period}'] = df['close'].pct_change().rolling(window=period).std() * np.sqrt(252) * 100
        
        # ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr_14'] = tr.rolling(window=14).mean()
        df['atr_pct'] = df['atr_14'] / df['close'] * 100
        
        # 布林带宽度
        df['boll_mid'] = df['close'].rolling(window=20).mean()
        df['boll_std'] = df['close'].rolling(window=20).std()
        df['boll_upper'] = df['boll_mid'] + 2 * df['boll_std']
        df['boll_lower'] = df['boll_mid'] - 2 * df['boll_std']
        df['boll_width'] = (df['boll_upper'] - df['boll_lower']) / df['boll_mid'] * 100
        
        return df
    
    def _calc_volume_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算成交量因子"""
        # 量比
        df['volume_ma5'] = df['volume'].rolling(window=5).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma5']
        
        # 成交量变化率
        df['volume_roc'] = df['volume'].pct_change(5) * 100
        
        # OBV (On Balance Volume)
        obv = [0]
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv.append(obv[-1] + df['volume'].iloc[i])
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv.append(obv[-1] - df['volume'].iloc[i])
            else:
                obv.append(obv[-1])
        df['obv'] = obv
        df['obv_ma20'] = df['obv'].rolling(window=20).mean()
        
        # 量价相关性
        df['price_volume_corr'] = df['close'].rolling(window=20).corr(df['volume'])
        
        return df
    
    def _calc_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算趋势因子"""
        # 均线
        for period in [5, 10, 20, 60, 120]:
            df[f'ma_{period}'] = df['close'].rolling(window=period).mean()
        
        # 均线斜率
        for period in [5, 10, 20]:
            ma = df[f'ma_{period}']
            df[f'ma_{period}_slope'] = (ma - ma.shift(5)) / ma.shift(5) * 100
        
        # 均线多头排列得分
        df['ma_alignment'] = 0
        for i in range(len(df)):
            if (df['ma_5'].iloc[i] > df['ma_10'].iloc[i] > 
                df['ma_20'].iloc[i] > df['ma_60'].iloc[i]):
                df.loc[df.index[i], 'ma_alignment'] = 1  # 多头排列
            elif (df['ma_5'].iloc[i] < df['ma_10'].iloc[i] < 
                  df['ma_20'].iloc[i] < df['ma_60'].iloc[i]):
                df.loc[df.index[i], 'ma_alignment'] = -1  # 空头排列
        
        # MACD
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd_dif'] = df['ema_12'] - df['ema_26']
        df['macd_dea'] = df['macd_dif'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = (df['macd_dif'] - df['macd_dea']) * 2
        
        # ADX (Average Directional Index)
        df = self._calc_adx(df)
        
        return df
    
    def _calc_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """计算ADX"""
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift()).abs(),
            (df['low'] - df['close'].shift()).abs()
        ], axis=1).max(axis=1)
        
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        df['adx'] = dx.rolling(window=period).mean()
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di
        
        return df
    
    def get_latest_factors(self, df: pd.DataFrame) -> Dict[str, float]:
        """获取最新一行的因子值"""
        df = self.calculate_all(df)
        latest = df.iloc[-1]
        
        factors = {}
        for col in df.columns:
            if col not in ['date', 'code', 'open', 'high', 'low', 'close', 'volume', 'amount']:
                val = latest.get(col, np.nan)
                if not pd.isna(val):
                    factors[col] = float(val)
        
        return factors
