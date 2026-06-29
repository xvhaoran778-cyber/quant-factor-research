"""PushPlus 微信推送通知"""

import requests
from typing import Dict, Optional
from loguru import logger


class PushPlusNotifier:
    """PushPlus微信推送"""
    
    def __init__(self, token: str = None):
        self.token = token
        self.api_url = "http://www.pushplus.plus/send"
    
    def send(self, title: str, content: str, template: str = "txt") -> bool:
        """发送推送消息
        
        Args:
            title: 标题
            content: 内容
            template: 模板类型 (txt/html/json/markdown)
        
        Returns:
            是否发送成功
        """
        if not self.token:
            logger.warning("PushPlus token not configured")
            return False
        
        data = {
            "token": self.token,
            "title": title,
            "content": content,
            "template": template
        }
        
        try:
            response = requests.post(self.api_url, json=data, timeout=10)
            result = response.json()
            
            if result.get('code') == 200:
                logger.info(f"PushPlus notification sent: {title}")
                return True
            else:
                logger.error(f"PushPlus error: {result.get('msg')}")
                return False
                
        except Exception as e:
            logger.error(f"PushPlus request error: {e}")
            return False
    
    def send_trade_signal(self, stock_code: str, stock_name: str, 
                         signal: str, score: float, reason: str) -> bool:
        """发送交易信号通知"""
        signal_emoji = {
            'strong_buy': '🟢🟢',
            'buy': '🟢',
            'hold': '🟡',
            'sell': '🔴',
            'strong_sell': '🔴🔴'
        }
        
        signal_cn = {
            'strong_buy': '强烈买入',
            'buy': '买入',
            'hold': '持有',
            'sell': '卖出',
            'strong_sell': '强烈卖出'
        }
        
        emoji = signal_emoji.get(signal, '❓')
        signal_text = signal_cn.get(signal, signal)
        
        title = f"{emoji} {stock_name}({stock_code}) - {signal_text}"
        
        content = f"""
## 交易信号通知

**股票**: {stock_name} ({stock_code})

**信号**: {emoji} {signal_text}

**评分**: {score:.1f}/100

**理由**: {reason}

---
*QuantAgent 暗黑量化交易系统*
        """
        
        return self.send(title, content, template="markdown")
    
    def send_system_alert(self, alert_type: str, message: str) -> bool:
        """发送系统告警"""
        title = f"⚠️ 系统告警 - {alert_type}"
        content = f"""
## 系统告警

**类型**: {alert_type}

**详情**: {message}

---
*QuantAgent 暗黑量化交易系统*
        """
        
        return self.send(title, content, template="markdown")
    
    def send_daily_report(self, report: Dict) -> bool:
        """发送每日报告"""
        title = f"📊 每日报告 - {report.get('date', '')}"
        
        content = f"""
## 每日交易报告

**日期**: {report.get('date', '')}

### 📈 收益情况
- 总资金: ¥{report.get('total_equity', 0):,.2f}
- 今日收益: {report.get('daily_return', 0):+.2f}%
- 累计收益: {report.get('total_return', 0):+.2f}%

### 🎯 今日信号
- 买入信号: {report.get('buy_signals', 0)} 个
- 卖出信号: {report.get('sell_signals', 0)} 个
- 持有信号: {report.get('hold_signals', 0)} 个

### 🤖 Agent状态
- 基本面分析师: {report.get('fundamental_score', 0):.1f}
- 技术面分析师: {report.get('technical_score', 0):.1f}
- 情绪分析师: {report.get('sentiment_score', 0):.1f}
- 宏观分析师: {report.get('macro_score', 0):.1f}

---
*QuantAgent 暗黑量化交易系统*
        """
        
        return self.send(title, content, template="markdown")
