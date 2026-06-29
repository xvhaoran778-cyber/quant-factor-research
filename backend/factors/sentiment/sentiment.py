"""Sentiment Factors - 情绪因子计算"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from loguru import logger


class SentimentFactors:
    """情绪因子计算器"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # 情绪关键词
        self.positive_keywords = [
            '利好', '上涨', '突破', '新高', '增长', '盈利', '回购', '增持',
            '涨停', '强势', '龙头', '景气', '扩张', '超预期', '创新'
        ]
        self.negative_keywords = [
            '利空', '下跌', '破位', '新低', '下滑', '亏损', '减持', '质押',
            '跌停', '弱势', '风险', '衰退', '萎缩', '不及预期', '监管'
        ]
    
    def calculate_news_sentiment(self, news_df: pd.DataFrame) -> Dict[str, float]:
        """计算新闻情绪因子"""
        if news_df is None or news_df.empty:
            return {'news_sentiment': 50, 'news_positive_ratio': 0, 'news_negative_ratio': 0}
        
        positive_count = 0
        negative_count = 0
        total_count = min(len(news_df), 30)
        
        for i in range(total_count):
            title = str(news_df.iloc[i].get('新闻标题', ''))
            content = str(news_df.iloc[i].get('新闻内容', ''))
            text = title + content
            
            for keyword in self.positive_keywords:
                if keyword in text:
                    positive_count += 1
                    break
            
            for keyword in self.negative_keywords:
                if keyword in text:
                    negative_count += 1
                    break
        
        if total_count == 0:
            return {'news_sentiment': 50, 'news_positive_ratio': 0, 'news_negative_ratio': 0}
        
        positive_ratio = positive_count / total_count
        negative_ratio = negative_count / total_count
        sentiment_score = 50 + (positive_ratio - negative_ratio) * 50
        sentiment_score = max(0, min(100, sentiment_score))
        
        return {
            'news_sentiment': sentiment_score,
            'news_positive_ratio': positive_ratio,
            'news_negative_ratio': negative_ratio,
            'news_total_count': total_count
        }
    
    def calculate_fund_flow_factors(self, fund_flow: Dict) -> Dict[str, float]:
        """计算资金流向因子"""
        factors = {}
        
        # 主力净流入占比
        main_pct = fund_flow.get('main_net_pct', 0)
        factors['main_net_pct'] = main_pct
        
        # 超大单净流入
        super_large = fund_flow.get('super_large_net', 0)
        factors['super_large_net'] = super_large
        
        # 大单净流入
        large_net = fund_flow.get('large_net', 0)
        factors['large_net'] = large_net
        
        # 资金流向综合得分
        if main_pct > 10:
            factors['fund_flow_score'] = 90
        elif main_pct > 5:
            factors['fund_flow_score'] = 75
        elif main_pct > 0:
            factors['fund_flow_score'] = 60
        elif main_pct > -5:
            factors['fund_flow_score'] = 40
        else:
            factors['fund_flow_score'] = 20
        
        return factors
    
    def calculate_analyst_sentiment(self, ratings_df: pd.DataFrame) -> Dict[str, float]:
        """计算分析师情绪因子"""
        if ratings_df is None or ratings_df.empty:
            return {'analyst_score': 50, 'buy_ratio': 0}
        
        buy_count = 0
        hold_count = 0
        sell_count = 0
        
        for _, row in ratings_df.iterrows():
            rating = str(row.get('最新评级', '')).lower()
            if '买入' in rating or '增持' in rating or '推荐' in rating:
                buy_count += 1
            elif '卖出' in rating or '减持' in rating:
                sell_count += 1
            else:
                hold_count += 1
        
        total = buy_count + hold_count + sell_count
        if total == 0:
            return {'analyst_score': 50, 'buy_ratio': 0}
        
        buy_ratio = buy_count / total
        analyst_score = 30 + buy_ratio * 60
        
        return {
            'analyst_score': analyst_score,
            'buy_ratio': buy_ratio,
            'buy_count': buy_count,
            'total_ratings': total
        }
    
    def calculate_market_sentiment(self, telegraph_df: pd.DataFrame) -> Dict[str, float]:
        """计算市场整体情绪"""
        if telegraph_df is None or telegraph_df.empty:
            return {'market_sentiment': 50}
        
        positive_count = 0
        negative_count = 0
        total_count = min(len(telegraph_df), 50)
        
        for i in range(total_count):
            content = str(telegraph_df.iloc[i].get('内容', ''))
            
            for keyword in self.positive_keywords:
                if keyword in content:
                    positive_count += 1
                    break
            
            for keyword in self.negative_keywords:
                if keyword in content:
                    negative_count += 1
                    break
        
        if total_count == 0:
            return {'market_sentiment': 50}
        
        sentiment_ratio = (positive_count - negative_count) / total_count
        market_sentiment = 50 + sentiment_ratio * 50
        market_sentiment = max(0, min(100, market_sentiment))
        
        return {
            'market_sentiment': market_sentiment,
            'market_positive_count': positive_count,
            'market_negative_count': negative_count
        }
    
    def calculate_composite_score(self, factors: Dict[str, float]) -> float:
        """计算情绪综合评分"""
        scores = []
        
        # 新闻情绪
        news_score = factors.get('news_sentiment', 50)
        scores.append(news_score)
        
        # 资金流向
        fund_score = factors.get('fund_flow_score', 50)
        scores.append(fund_score)
        
        # 分析师评级
        analyst_score = factors.get('analyst_score', 50)
        scores.append(analyst_score)
        
        # 市场情绪
        market_score = factors.get('market_sentiment', 50)
        scores.append(market_score)
        
        return np.mean(scores) if scores else 50
