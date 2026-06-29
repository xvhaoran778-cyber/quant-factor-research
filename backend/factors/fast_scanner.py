"""高速扫描引擎 - 可配置因子+并行计算+内存缓存"""

import pandas as pd
import numpy as np
import hashlib
from typing import Dict, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from loguru import logger
from datetime import datetime

from functools import lru_cache


class FastScanner:
    """高速A股扫描器 - 支持可配置因子"""
    
    # 因子方向: ↑=正向(越大越好), ↓=反向(越小越好), ~=中性(接近50最好)
    FACTOR_DIRECTIONS = {
        'alpha_return_5d': '↑', 'alpha_return_10d': '↑', 'alpha_return_20d': '↑',
        'alpha_rsi_6': '~', 'alpha_rsi_14': '~',
        'alpha_volume_ratio_5d': '↑', 'alpha_volume_ratio_10d': '↑',
        'alpha_price_ma_ratio_5': '↑', 'alpha_price_ma_ratio_20': '↑',
        'alpha_boll_position': '~', 'alpha_boll_width': '↓',
    }
    
    # 预设因子配置
    PRESETS = {
        '趋势动量': {
            'alpha_return_5d': 0.3, 'alpha_return_10d': 0.3, 'alpha_return_20d': 0.4
        },
        '超跌反弹': {
            'alpha_rsi_6': 0.4, 'alpha_rsi_14': 0.6
        },
        '放量突破': {
            'alpha_volume_ratio_5d': 0.5, 'alpha_volume_ratio_10d': 0.5
        },
        '均线多头': {
            'alpha_price_ma_ratio_5': 0.3, 'alpha_price_ma_ratio_20': 0.7
        },
        '趋势+量价': {
            'alpha_return_5d': 0.2, 'alpha_return_20d': 0.3,
            'alpha_volume_ratio_5d': 0.3, 'alpha_price_ma_ratio_20': 0.2
        },
        '综合多因子': {
            'alpha_return_5d': 0.15, 'alpha_return_20d': 0.2,
            'alpha_rsi_14': 0.2, 'alpha_volume_ratio_5d': 0.2,
            'alpha_price_ma_ratio_20': 0.15, 'alpha_boll_position': 0.1
        }
    }
    
    def __init__(self, max_workers: int = 8):
        self.max_workers = max_workers
        self._cache = {}  # 内存缓存
    
    def scan(self, stock_codes: List[str], factor_weights: Dict[str, float],
            top_n: int = 20, min_volume: int = 50000,
            use_full_alpha: bool = False,
            progress_callback=None) -> pd.DataFrame:
        """高速扫描
        
        Args:
            stock_codes: 股票列表
            factor_weights: {因子名: 权重}
            top_n: 返回前N只
            min_volume: 最小成交量
            use_full_alpha: 是否使用完整Alpha158计算
            progress_callback: 进度回调
        """
        logger.info(f"扫描 {len(stock_codes)}只, 因子={list(factor_weights.keys())[:3]}..., 并发={self.max_workers}")
        
        # 缓存键
        today = datetime.now().strftime("%Y%m%d")
        factor_hash = hashlib.md5(str(sorted(factor_weights.items())).encode()).hexdigest()[:8]
        
        cached_scores = {}
        uncached_codes = []
        
        for code in stock_codes:
            cache_key = f"s_{code}_{today}_{factor_hash}"
            cached = self._cache.get(cache_key)
            if cached:
                cached_scores[code] = cached
            else:
                uncached_codes.append(code)
        
        logger.info(f"缓存命中 {len(cached_scores)}, 需计算 {len(uncached_codes)}")
        
        results = list(cached_scores.values())
        completed = len(cached_scores)
        
        if uncached_codes:
            batch_size = max(10, len(uncached_codes) // self.max_workers)
            batches = [uncached_codes[i:i+batch_size] 
                      for i in range(0, len(uncached_codes), batch_size)]
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(self._scan_batch, batch, factor_weights, 
                                          min_volume, use_full_alpha): i 
                          for i, batch in enumerate(batches)}
                
                for future in as_completed(futures):
                    try:
                        batch_results = future.result()
                    except Exception as e:
                        logger.error(f"批次扫描失败: {e}")
                        batch_results = []
                    
                    for r in batch_results:
                        code = r['code']
                        cache_key = f"s_{code}_{today}_{factor_hash}"
                        self._cache[cache_key] = r
                        results.append(r)
                    
                    completed += len(batch_results)
                    if progress_callback:
                        progress_callback(completed, len(stock_codes))
        
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame(results).sort_values('score', ascending=False)
        return df.head(top_n) if top_n else df
    
    def _scan_batch(self, codes: List[str], factor_weights: Dict[str, float],
                   min_volume: int, use_full_alpha: bool) -> List[Dict]:
        """扫描一批股票"""
        from data.collectors import TencentCollector
        
        tencent = TencentCollector({'base_url': 'http://qt.gtimg.cn'})
        if not tencent.connect():
            return []
        
        results = []
        
        for code in codes:
            try:
                kline = tencent.get_kline(code, period='daily', count=120)
                if kline is None or len(kline) < 30:
                    continue
                
                avg_vol = kline['volume'].tail(20).mean()
                if avg_vol < min_volume:
                    continue
                
                # 计算因子得分
                if use_full_alpha:
                    factor_values = self._calc_full_factors(kline, factor_weights)
                else:
                    factor_values = self._calc_fast_factors(kline, factor_weights)
                
                if factor_values is None:
                    continue
                
                quote = tencent.get_realtime_quote(code)
                name = quote.get('name', code) if quote else code
                price = quote.get('price', 0) if quote else 0
                change_pct = quote.get('change_pct', 0) if quote else 0
                
                if price <= 0:
                    continue
                
                factor_values['code'] = code
                factor_values['name'] = name
                factor_values['price'] = price
                factor_values['change_pct'] = change_pct
                results.append(factor_values)
                
            except Exception:
                pass
        
        return results
    
    def _normalize(self, value: float, factor_name: str) -> float:
        """方向感知归一化"""
        direction = self.FACTOR_DIRECTIONS.get(factor_name, '~')
        
        if 'return' in factor_name or 'momentum' in factor_name:
            normalized = (value + 0.5) * 100
        elif 'rsi' in factor_name:
            normalized = 100 - abs(value - 50)  # 中性指标
        elif 'ratio' in factor_name:
            normalized = min(value * 30, 100)
        elif 'volatility' in factor_name:
            normalized = (0.5 - value) * 200
        elif 'boll' in factor_name and 'position' in factor_name:
            normalized = 100 - abs(value - 0.5) * 200  # 中性：0.5最好
        else:
            normalized = 50 + value * 50
        
        # 方向反转
        if direction == '↓':
            normalized = 100 - normalized
        elif direction == '⬇':
            pass  # already handled
        
        return max(0, min(100, normalized))
    
    def _calc_fast_factors(self, kline: pd.DataFrame, weights: Dict[str, float]) -> Optional[Dict]:
        """快速计算因子（numpy向量化）"""
        close = kline['close'].values
        volume = kline['volume'].values
        n = len(close)
        
        if n < 30:
            return None
        
        score = 0
        weight_sum = 0
        detail = {}
        
        for factor_name, weight in weights.items():
            value = 0
            normalized = 0
            
            # 收益率因子
            if 'return_5d' in factor_name:
                value = close[-1] / close[-6] - 1 if n > 5 else 0
                normalized = self._normalize(value, factor_name)
            elif 'return_10d' in factor_name:
                value = close[-1] / close[-11] - 1 if n > 10 else 0
                normalized = self._normalize(value, factor_name)
            elif 'return_20d' in factor_name:
                value = close[-1] / close[-21] - 1 if n > 20 else 0
                normalized = self._normalize(value, factor_name)
            
            # RSI因子
            elif 'rsi' in factor_name:
                period = 14 if '14' in factor_name else 6
                if n > period + 1:
                    deltas = np.diff(close[-(period+1):])
                    gains = np.sum(deltas[deltas > 0])
                    losses = -np.sum(deltas[deltas < 0])
                    value = 100 - 100/(1 + gains/(losses + 1e-8)) if losses > 0 else (80 if gains > 0 else 50)
                    normalized = self._normalize(value, factor_name)
                else:
                    normalized = 50
            
            # 成交量因子
            elif 'volume_ratio_5d' in factor_name:
                if n > 15:
                    vol_recent = np.mean(volume[-5:])
                    vol_prev = np.mean(volume[-15:-5])
                    value = vol_recent / (vol_prev + 1)
                    normalized = self._normalize(value, factor_name)
                else:
                    normalized = 50
            elif 'volume_ratio_10d' in factor_name:
                if n > 20:
                    value = np.mean(volume[-10:]) / (np.mean(volume[-20:-10]) + 1)
                    normalized = self._normalize(value, factor_name)
                else:
                    normalized = 50
            
            # 均线因子
            elif 'price_ma_ratio_5' in factor_name:
                if n > 5:
                    ma5 = np.mean(close[-5:])
                    value = close[-1] / (ma5 + 1e-8)
                    normalized = self._normalize(value, factor_name)
                else:
                    normalized = 50
            elif 'price_ma_ratio_20' in factor_name:
                if n > 20:
                    ma20 = np.mean(close[-20:])
                    value = close[-1] / (ma20 + 1e-8)
                    normalized = self._normalize(value, factor_name)
                else:
                    normalized = 50
            
            # 布林带因子
            elif 'boll_position' in factor_name:
                if n > 20:
                    ma20 = np.mean(close[-20:])
                    std20 = np.std(close[-20:])
                    lower = ma20 - 2 * std20
                    upper = ma20 + 2 * std20
                    value = (close[-1] - lower) / (upper - lower + 1e-8)
                    normalized = self._normalize(value, factor_name)
                else:
                    normalized = 50
            
            else:
                normalized = 50  # 未知因子默认中性
            
            score += normalized * weight
            weight_sum += weight
            detail[factor_name] = value
        
        final_score = score / weight_sum if weight_sum > 0 else 50
        
        # 始终包含基础展示字段
        return {
            'score': final_score,
            'momentum': (close[-1]/close[-21]-1)*100 if n>20 else 0,
            'rsi': self._quick_rsi(close),
            'vol_ratio': (np.mean(volume[-5:])/(np.mean(volume[-15:-5])+1)) if n>15 else 1,
            **detail
        }
    
    def _quick_rsi(self, close: np.ndarray, period: int = 14) -> float:
        """快速计算RSI"""
        if len(close) < period + 1:
            return 50
        deltas = np.diff(close[-(period+1):])
        gains = np.sum(deltas[deltas > 0])
        losses = -np.sum(deltas[deltas < 0])
        if losses > 0:
            return float(100 - 100/(1 + gains/losses))
        return 80.0 if gains > 0 else 50.0
    
    def _calc_full_factors(self, kline: pd.DataFrame, weights: Dict[str, float]) -> Optional[Dict]:
        """完整Alpha158因子计算"""
        from factors.alpha158 import Alpha158
        
        alpha158 = Alpha158()
        factors_df = alpha158.calculate_all(kline)
        latest = factors_df.iloc[-1]
        
        score = 0
        weight_sum = 0
        detail = {}
        
        for factor_name, weight in weights.items():
            value = latest.get(factor_name, 0)
            if value is None or pd.isna(value):
                value = 0
            
            # 归一化
            if 'return' in factor_name or 'momentum' in factor_name:
                normalized = (value + 0.5) * 100
            elif 'rsi' in factor_name:
                normalized = 100 - abs(value - 50)
            elif 'ratio' in factor_name:
                normalized = min(value * 30, 100)
            elif 'volatility' in factor_name:
                normalized = (0.5 - value) * 200
            else:
                normalized = 50 + value * 50
            
            normalized = max(0, min(100, normalized))
            score += normalized * weight
            weight_sum += weight
            detail[factor_name] = value
        
        final_score = score / weight_sum if weight_sum > 0 else 50
        
        # 始终包含基础展示字段
        close_vals = kline['close'].values
        vol_vals = kline['volume'].values
        n = len(close_vals)
        return {
            'score': final_score,
            'momentum': (close_vals[-1]/close_vals[-21]-1)*100 if n>20 else 0,
            'rsi': self._quick_rsi(close_vals),
            'vol_ratio': (np.mean(vol_vals[-5:])/(np.mean(vol_vals[-15:-5])+1)) if n>15 else 1,
            **detail
        }
    
    def clear_cache(self):
        self._cache = {}
        logger.info("扫描缓存已清除")
