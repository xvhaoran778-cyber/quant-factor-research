"""Macro Factors - 宏观因子计算"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from loguru import logger


class MacroFactors:
    """宏观因子计算器"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
    
    def calculate_gdp_factors(self, gdp_data: pd.DataFrame) -> Dict[str, float]:
        """计算GDP因子"""
        if gdp_data is None or gdp_data.empty:
            return {'gdp_growth': 0, 'gdp_score': 50}
        
        try:
            latest = gdp_data.iloc[-1]
            gdp_growth = float(latest.get('同比增长', latest.get('GDP同比增长', 0)))
            
            if pd.isna(gdp_growth):
                return {'gdp_growth': 0, 'gdp_score': 50}
            
            # GDP评分
            if gdp_growth >= 6:
                gdp_score = 85
            elif gdp_growth >= 5:
                gdp_score = 70
            elif gdp_growth >= 3:
                gdp_score = 50
            elif gdp_growth >= 0:
                gdp_score = 35
            else:
                gdp_score = 20
            
            return {
                'gdp_growth': gdp_growth,
                'gdp_score': gdp_score
            }
        except Exception as e:
            logger.error(f"Error calculating GDP factors: {e}")
            return {'gdp_growth': 0, 'gdp_score': 50}
    
    def calculate_pmi_factors(self, pmi_data: pd.DataFrame) -> Dict[str, float]:
        """计算PMI因子"""
        if pmi_data is None or pmi_data.empty:
            return {'pmi': 50, 'pmi_score': 50}
        
        try:
            latest = pmi_data.iloc[-1]
            pmi_value = None
            
            for col in ['制造业PMI', 'PMI', '制造业采购经理指数']:
                if col in latest.index:
                    pmi_value = float(latest[col])
                    break
            
            if pmi_value is None or pd.isna(pmi_value):
                return {'pmi': 50, 'pmi_score': 50}
            
            # PMI评分
            if pmi_value >= 55:
                pmi_score = 85
            elif pmi_value >= 50:
                pmi_score = 65
            elif pmi_value >= 48:
                pmi_score = 45
            else:
                pmi_score = 25
            
            return {
                'pmi': pmi_value,
                'pmi_score': pmi_score,
                'pmi_expansion': 1 if pmi_value >= 50 else 0
            }
        except Exception as e:
            logger.error(f"Error calculating PMI factors: {e}")
            return {'pmi': 50, 'pmi_score': 50}
    
    def calculate_cpi_factors(self, cpi_data: pd.DataFrame) -> Dict[str, float]:
        """计算CPI因子"""
        if cpi_data is None or cpi_data.empty:
            return {'cpi': 0, 'cpi_score': 50}
        
        try:
            latest = cpi_data.iloc[-1]
            cpi_value = None
            
            for col in ['同比增长', 'CPI同比', '全国']:
                if col in latest.index:
                    cpi_value = float(latest[col])
                    break
            
            if cpi_value is None or pd.isna(cpi_value):
                return {'cpi': 0, 'cpi_score': 50}
            
            # CPI评分（温和通胀最佳）
            if 1.5 <= cpi_value <= 3:
                cpi_score = 75
            elif 0 <= cpi_value < 1.5:
                cpi_score = 55
            elif 3 < cpi_value <= 5:
                cpi_score = 45
            elif cpi_value > 5:
                cpi_score = 25
            else:
                cpi_score = 35
            
            return {
                'cpi': cpi_value,
                'cpi_score': cpi_score,
                'inflation_type': self._get_inflation_type(cpi_value)
            }
        except Exception as e:
            logger.error(f"Error calculating CPI factors: {e}")
            return {'cpi': 0, 'cpi_score': 50}
    
    def calculate_money_supply_factors(self, money_data: pd.DataFrame) -> Dict[str, float]:
        """计算货币供应量因子"""
        if money_data is None or money_data.empty:
            return {'m2_growth': 0, 'money_score': 50}
        
        try:
            latest = money_data.iloc[-1]
            m2_growth = None
            
            for col in ['M2-同比增长', 'M2同比', '同比增长']:
                if col in latest.index:
                    m2_growth = float(latest[col])
                    break
            
            if m2_growth is None or pd.isna(m2_growth):
                return {'m2_growth': 0, 'money_score': 50}
            
            # M2评分
            if 8 <= m2_growth <= 12:
                money_score = 70
            elif 12 < m2_growth <= 15:
                money_score = 60
            elif m2_growth > 15:
                money_score = 45
            elif 5 <= m2_growth < 8:
                money_score = 50
            else:
                money_score = 35
            
            return {
                'm2_growth': m2_growth,
                'money_score': money_score,
                'liquidity_type': self._get_liquidity_type(m2_growth)
            }
        except Exception as e:
            logger.error(f"Error calculating money supply factors: {e}")
            return {'m2_growth': 0, 'money_score': 50}
    
    def _get_inflation_type(self, cpi: float) -> str:
        """获取通胀类型"""
        if cpi < 0:
            return "通缩"
        elif cpi < 1.5:
            return "低通胀"
        elif cpi <= 3:
            return "温和通胀"
        elif cpi <= 5:
            return "通胀偏高"
        else:
            return "高通胀"
    
    def _get_liquidity_type(self, m2_growth: float) -> str:
        """获取流动性类型"""
        if m2_growth < 5:
            return "流动性紧缩"
        elif m2_growth < 8:
            return "流动性偏紧"
        elif m2_growth <= 12:
            return "流动性适度"
        elif m2_growth <= 15:
            return "流动性偏松"
        else:
            return "流动性过剩"
    
    def calculate_composite_score(self, factors: Dict[str, float]) -> float:
        """计算宏观综合评分"""
        scores = []
        
        gdp_score = factors.get('gdp_score', 50)
        pmi_score = factors.get('pmi_score', 50)
        cpi_score = factors.get('cpi_score', 50)
        money_score = factors.get('money_score', 50)
        
        scores = [gdp_score, pmi_score, cpi_score, money_score]
        return np.mean(scores) if scores else 50
