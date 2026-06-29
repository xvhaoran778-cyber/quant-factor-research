"""Fundamental Factors - 基本面因子计算"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from loguru import logger


class FundamentalFactors:
    """基本面因子计算器"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
    
    def calculate_valuation_factors(self, data: Dict) -> Dict[str, float]:
        """计算估值因子"""
        factors = {}
        
        # PE因子
        pe = data.get('pe_ratio', 0)
        if pe > 0:
            factors['pe'] = pe
            factors['ep'] = 1 / pe  # 盈利收益率
        
        # PB因子
        pb = data.get('pb_ratio', 0)
        if pb > 0:
            factors['pb'] = pb
            factors['bp'] = 1 / pb  # 账面价值比
        
        # PS因子
        ps = data.get('ps_ratio', 0)
        if ps > 0:
            factors['ps'] = ps
            factors['sp'] = 1 / ps
        
        # 市值因子
        market_cap = data.get('market_cap', 0)
        if market_cap > 0:
            factors['ln_market_cap'] = np.log(market_cap)
        
        return factors
    
    def calculate_quality_factors(self, data: Dict) -> Dict[str, float]:
        """计算质量因子"""
        factors = {}
        
        # ROE (净资产收益率)
        roe = data.get('roe', 0)
        if roe != 0:
            factors['roe'] = roe
        
        # ROA (总资产收益率)
        roa = data.get('roa', 0)
        if roa != 0:
            factors['roa'] = roa
        
        # 毛利率
        gross_margin = data.get('gross_margin', 0)
        if gross_margin != 0:
            factors['gross_margin'] = gross_margin
        
        # 净利率
        net_margin = data.get('net_margin', 0)
        if net_margin != 0:
            factors['net_margin'] = net_margin
        
        # 资产负债率
        debt_ratio = data.get('debt_ratio', 0)
        if debt_ratio != 0:
            factors['debt_ratio'] = debt_ratio
        
        return factors
    
    def calculate_growth_factors(self, data: Dict) -> Dict[str, float]:
        """计算成长因子"""
        factors = {}
        
        # 营收增长率
        revenue_growth = data.get('revenue_growth', 0)
        if revenue_growth != 0:
            factors['revenue_growth'] = revenue_growth
        
        # 净利润增长率
        profit_growth = data.get('profit_growth', 0)
        if profit_growth != 0:
            factors['profit_growth'] = profit_growth
        
        # 每股收益增长率
        eps_growth = data.get('eps_growth', 0)
        if eps_growth != 0:
            factors['eps_growth'] = eps_growth
        
        return factors
    
    def calculate_composite_score(self, factors: Dict[str, float]) -> float:
        """计算基本面综合评分"""
        scores = []
        
        # 估值评分
        pe = factors.get('pe', 0)
        if pe > 0:
            if pe < 15:
                scores.append(90)
            elif pe < 25:
                scores.append(70)
            elif pe < 40:
                scores.append(50)
            else:
                scores.append(30)
        
        # 质量评分
        roe = factors.get('roe', 0)
        if roe > 0:
            if roe > 20:
                scores.append(90)
            elif roe > 15:
                scores.append(75)
            elif roe > 10:
                scores.append(55)
            else:
                scores.append(35)
        
        # 成长评分
        growth = factors.get('revenue_growth', 0)
        if growth != 0:
            if growth > 30:
                scores.append(90)
            elif growth > 20:
                scores.append(75)
            elif growth > 10:
                scores.append(55)
            elif growth > 0:
                scores.append(40)
            else:
                scores.append(20)
        
        return np.mean(scores) if scores else 50
