"""模拟交易账户 - 纸交易/模拟资金"""

import json
import os
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class PaperPosition:
    """模拟持仓"""
    code: str
    name: str
    shares: int
    avg_cost: float
    current_price: float = 0
    market_value: float = 0
    profit: float = 0
    profit_pct: float = 0
    buy_date: str = ""
    buy_price: float = 0
    
    def to_dict(self):
        return {
            'code': self.code, 'name': self.name, 'shares': self.shares,
            'avg_cost': self.avg_cost, 'current_price': self.current_price,
            'market_value': self.market_value, 'profit': self.profit,
            'profit_pct': self.profit_pct, 'buy_date': self.buy_date
        }


@dataclass
class PaperTrade:
    """模拟交易记录"""
    date: str
    code: str
    name: str
    action: str  # buy/sell
    price: float
    shares: int
    amount: float
    commission: float = 0
    reason: str = ""
    
    def to_dict(self):
        return {
            'date': self.date, 'code': self.code, 'name': self.name,
            'action': self.action, 'price': self.price, 'shares': self.shares,
            'amount': self.amount, 'commission': self.commission, 'reason': self.reason
        }


class PaperAccount:
    """模拟交易账户"""
    
    def __init__(self, initial_capital: float = 10000, save_path: str = None):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions: Dict[str, PaperPosition] = {}
        self.trade_history: List[PaperTrade] = []
        self.daily_equity: List[Dict] = []
        
        if save_path is None:
            save_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                    'data', 'paper_account.json')
        self.save_path = save_path
        
        self._load()
    
    def _load(self):
        """从文件加载状态"""
        try:
            if os.path.exists(self.save_path):
                with open(self.save_path, 'r') as f:
                    data = json.load(f)
                self.capital = data.get('capital', self.initial_capital)
                self.trade_history = [PaperTrade(**t) for t in data.get('trades', [])]
                for p in data.get('positions', []):
                    self.positions[p['code']] = PaperPosition(**p)
                logger.info(f"加载模拟账户: 资金={self.capital:.0f}, 持仓={len(self.positions)}")
        except:
            pass
    
    def save(self):
        """保存状态到文件"""
        try:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            data = {
                'initial_capital': self.initial_capital,
                'capital': self.capital,
                'positions': [p.to_dict() for p in self.positions.values()],
                'trades': [t.to_dict() for t in self.trade_history[-200:]]
            }
            with open(self.save_path, 'w') as f:
                json.dump(data, f, ensure_ascii=False, default=str)
        except Exception as e:
            logger.error(f"保存模拟账户失败: {e}")
    
    def buy(self, code: str, name: str, price: float, shares: int = None,
            amount: float = None, reason: str = "") -> bool:
        """买入"""
        if shares is None and amount is not None:
            commission = max(amount * 0.0003, 5)
            available = amount - commission
            shares = int(available / price / 100) * 100
        
        if shares is None or shares <= 0:
            return False
        
        total_cost = shares * price
        commission = max(total_cost * 0.0003, 5)
        cost = total_cost + commission
        
        if cost > self.capital:
            return False
        
        self.capital -= cost
        
        if code in self.positions:
            pos = self.positions[code]
            total_shares = pos.shares + shares
            pos.avg_cost = (pos.avg_cost * pos.shares + price * shares) / total_shares
            pos.shares = total_shares
            pos.market_value = pos.shares * price
            pos.current_price = price
        else:
            self.positions[code] = PaperPosition(
                code=code, name=name, shares=shares,
                avg_cost=price, current_price=price,
                market_value=shares * price,
                buy_date=datetime.now().strftime("%Y-%m-%d"),
                buy_price=price
            )
        
        trade = PaperTrade(
            date=datetime.now().strftime("%Y-%m-%d %H:%M"),
            code=code, name=name, action='buy',
            price=price, shares=shares, amount=total_cost,
            commission=commission, reason=reason
        )
        self.trade_history.append(trade)
        self.save()
        logger.info(f"模拟买入: {name}({code}) {shares}股 @ {price:.2f}")
        return True
    
    def sell(self, code: str, price: float, shares: int = None,
             pct: float = None, reason: str = "") -> bool:
        """卖出"""
        if code not in self.positions:
            return False
        
        pos = self.positions[code]
        
        if shares is None and pct is not None:
            shares = int(pos.shares * pct / 100) * 100
        if shares is None or shares <= 0:
            shares = pos.shares
        
        if shares > pos.shares:
            shares = pos.shares
        
        total_revenue = shares * price
        commission = max(total_revenue * 0.0003, 5)
        stamp_tax = total_revenue * 0.001
        revenue = total_revenue - commission - stamp_tax
        
        self.capital += revenue
        
        if shares >= pos.shares:
            del self.positions[code]
        else:
            pos.shares -= shares
            pos.market_value = pos.shares * price
        
        trade = PaperTrade(
            date=datetime.now().strftime("%Y-%m-%d %H:%M"),
            code=code, name=pos.name, action='sell',
            price=price, shares=shares, amount=total_revenue,
            commission=commission + stamp_tax, reason=reason
        )
        self.trade_history.append(trade)
        self.save()
        logger.info(f"模拟卖出: {pos.name}({code}) {shares}股 @ {price:.2f}")
        return True
    
    def update_prices(self, quotes: Dict[str, Dict]):
        """更新持仓价格"""
        for code, pos in self.positions.items():
            if code in quotes:
                pos.current_price = quotes[code].get('price', pos.current_price)
                pos.market_value = pos.shares * pos.current_price
                pos.profit = (pos.current_price - pos.avg_cost) * pos.shares
                pos.profit_pct = (pos.current_price / pos.avg_cost - 1) * 100 if pos.avg_cost > 0 else 0
        self.save()
    
    def get_total_equity(self) -> float:
        """总权益"""
        position_value = sum(p.market_value for p in self.positions.values())
        return self.capital + position_value
    
    def get_total_return(self) -> float:
        """总收益率"""
        return (self.get_total_equity() / self.initial_capital - 1) * 100
    
    def get_summary(self) -> Dict:
        """获取账户摘要"""
        equity = self.get_total_equity()
        position_count = len(self.positions)
        position_value = sum(p.market_value for p in self.positions.values())
        
        return {
            'initial_capital': self.initial_capital,
            'capital': self.capital,
            'total_equity': equity,
            'total_return': self.get_total_return(),
            'position_count': position_count,
            'position_value': position_value,
            'position_pct': position_value / equity * 100 if equity > 0 else 0,
            'cash_pct': self.capital / equity * 100 if equity > 0 else 100,
            'trade_count': len(self.trade_history),
            'positions': [p.to_dict() for p in self.positions.values()]
        }
    
    def reset(self):
        """重置账户"""
        self.capital = self.initial_capital
        self.positions = {}
        self.trade_history = []
        self.save()
