"""自动交易风控"""

from typing import Dict, List, Optional
from datetime import datetime
from loguru import logger
from .rules import filter_by_price_limit


class RiskManager:
    """自动交易风控器"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.stop_loss_pct = self.config.get('stop_loss', 0.05)
        self.take_profit_pct = self.config.get('take_profit', 0.10)
        self.max_daily_trades = self.config.get('max_daily_trades', 20)
        self.min_price = self.config.get('min_price', 3)
        self.max_price = self.config.get('max_price', 200)
        
        self._today = datetime.now().strftime("%Y%m%d")
        self._daily_trades = 0
    
    def check_daily_limit(self) -> bool:
        """检查每日交易次数限制"""
        today = datetime.now().strftime("%Y%m%d")
        if today != self._today:
            self._today = today
            self._daily_trades = 0
        return self._daily_trades < self.max_daily_trades
    
    def record_trade(self):
        self._daily_trades += 1
    
    def should_stop_loss(self, buy_price: float, current_price: float) -> bool:
        """止损检查"""
        return current_price < buy_price * (1 - self.stop_loss_pct)
    
    def should_take_profit(self, buy_price: float, current_price: float) -> bool:
        """止盈检查"""
        return current_price > buy_price * (1 + self.take_profit_pct)
    
    def filter_stock(self, code: str, price: float, prev_close: float = None) -> bool:
        """过滤股票"""
        if price < self.min_price or price > self.max_price:
            return False
        if prev_close and not filter_by_price_limit(code, price, prev_close):
            return False
        return True
    
    def check_buy_votes(self, agent_results: Dict) -> bool:
        """检查Agent投票"""
        buy = agent_results.get('强烈买入', 0) + agent_results.get('买入', 0)
        return buy >= 3
    
    def check_sell_votes(self, agent_results: Dict) -> bool:
        """检查Agent卖出投票"""
        sell = agent_results.get('强烈卖出', 0) + agent_results.get('卖出', 0)
        return sell >= 3
