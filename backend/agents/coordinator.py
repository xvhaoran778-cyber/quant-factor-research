"""决策协调员Agent - 综合机构分析框架+BRK风格最终决策"""

import json
from typing import Dict, Any, List
from loguru import logger
from .base_agent import BaseAgent, AgentResult, AgentSignal, SignalType
from .fundamental_agent import FundamentalAgent
from .technical_agent import TechnicalAgent
from .sentiment_agent import SentimentAgent
from .macro_agent import MacroAgent
from .news_agent import NewsAgent


class CoordinatorAgent(BaseAgent):
    """决策协调员Agent - 融合多维度机构分析的综合决策"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("决策协调员", config)
        self.fundamental_agent = FundamentalAgent(config)
        self.technical_agent = TechnicalAgent(config)
        self.sentiment_agent = SentimentAgent(config)
        self.news_agent = NewsAgent(config)
        self.macro_agent = MacroAgent(config)
        self._agent_results: Dict[str, AgentResult] = {}
    
    def get_system_prompt(self) -> str:
        return """你是首席投资官(CIO)，需要综合以下五个专业分析师的意见做出最终决策：

- 基本面分析师(Graham/Dodd价值投资+McKinsey战略分析)
- 技术面分析师(Wyckoff量价分析+Market Profile)
- 情绪分析师(Kahneman行为金融学+逆向投资)
- 消息面分析师(Event-Driven+Reflexivity)
- 宏观分析师(Dalio经济机器+美林投资时钟)

## 决策原则

### 1. 多维度验证 (Confluence)
- **3/5法则**: 至少3个分析师指向同一方向才可操作
- **一致性强**: 5/5一致看多/看空=极高确定性
- **分歧大**: 2:3或2:2:1=不确定性高，应减仓或观望

### 2. 矛盾信号的加权处理
- **宏观>基本面>技术面>消息面>情绪面**: 框架越宏观权重越高
- **宏观定仓位**: 宏观经济决定总体持仓比例
- **基本面选个股**: 在公司层面做多/空决策
- **技术面择时**: 精确入场/出场时机
- **消息面催化剂**: 加速器或触发器
- **情绪面止盈止损**: 过度乐观时减仓，过度悲观时加仓

### 3. 风险管理 (Risk-First)
- **凯利公式(Kelly Criterion)**: 仓位 = (胜率×盈亏比 - 败率) / 盈亏比
- **下行风险优先**: 最大回撤控制远比收益最大化重要
- **黑天鹅准备**: 永远假设你不知道的事情可能发生

## 输出格式
{
    "final_score": 0-100,
    "final_signal": "强烈买入/买入/持有/卖出/强烈卖出",
    "confidence": 0-1,
    "action_plan": "具体的操作建议（含仓位比例、止损止盈价位）",
    "risk_warning": "最重要的一条风险",
    "consensus": {"agent_count": N, "direction": "一致看多/偏多/分歧/偏空/一致看空"},
    "key_factors": ["最重要的2-3个因子"],
    "stop_loss": number,
    "take_profit": number
}"""
    
    def format_user_prompt(self, stock_code: str, data: Dict[str, Any]) -> str:
        stock_name = ''
        prompt = f"""## 最终决策任务
**股票代码**: {stock_code}
**五位分析师意见如下**:
"""
        for key in ['fundamental', 'technical', 'sentiment', 'news', 'macro']:
            result = self._agent_results.get(key)
            if result:
                stock_name = stock_name or result.stock_name
                prompt += f"""
### {result.agent_name}
- 信号: {result.signal.signal_type.value}
- 评分: {result.signal.score:.0f}
- 理由: {result.signal.reason}
- 关键因子: {', '.join(result.signal.key_factors) if result.signal.key_factors else '无'}
- 风险点: {', '.join(result.signal.risk_points) if result.signal.risk_points else '无'}
"""
        prompt += f"""
## 请综合决策
1. 各分析师的分歧点是什么？谁的观点更有说服力？
2. 按照宏观>基本面>技术面>消息面>情绪面的权重给出最终建议
3. 仓位建议（凯利公式估算）
4. 止损止盈价位

直接返回JSON。"""
        return prompt
    
    def parse_llm_response(self, response: str, stock_code: str, stock_name: str) -> AgentResult:
        json_data = self._parse_json_from_response(response)
        if not json_data:
            return self._get_error_result(stock_code, "无法解析分析结果")
        signal_str = json_data.get('final_signal', '持有')
        signal_map = {'强烈买入': SignalType.STRONG_BUY, '买入': SignalType.BUY,
                      '持有': SignalType.HOLD, '卖出': SignalType.SELL, '强烈卖出': SignalType.STRONG_SELL}
        signal = AgentSignal(
            signal_type=signal_map.get(signal_str, SignalType.HOLD),
            confidence=float(json_data.get('confidence', 0.5)),
            score=float(json_data.get('final_score', 50)),
            reason=json_data.get('action_plan', ''),
            key_factors=json_data.get('key_factors', []),
            risk_points=[json_data.get('risk_warning', '')]
        )
        return AgentResult(agent_name=self.name, stock_code=stock_code,
                          stock_name=stock_name, signal=signal, raw_data=json_data)
    
    def analyze(self, stock_code: str, data: Dict[str, Any] = None) -> AgentResult:
        logger.info(f"[{self.name}] 综合分析 {stock_code}")
        self._collect_agent_results(stock_code)
        
        stock_name = ''
        for result in self._agent_results.values():
            if result.stock_name:
                stock_name = result.stock_name
                break
        
        system_prompt = self.get_system_prompt()
        user_prompt = self.format_user_prompt(stock_code, {'stock_name': stock_name})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system_prompt},
                         {"role": "user", "content": user_prompt}],
                temperature=self.temperature, max_tokens=self.max_tokens
            )
            llm_response = response.choices[0].message.content
            result = self.parse_llm_response(llm_response, stock_code, stock_name)
            result.llm_response = llm_response
            self._last_result = result
            return result
        except Exception as e:
            logger.error(f"[{self.name}] LLM调用失败: {e}")
            return self._get_error_result(stock_code, str(e))
    
    def _collect_agent_results(self, stock_code: str):
        self._agent_results = {}
        agents = {'fundamental': self.fundamental_agent, 'technical': self.technical_agent,
                  'sentiment': self.sentiment_agent, 'news': self.news_agent, 'macro': self.macro_agent}
        for name, agent in agents.items():
            try:
                result = agent.analyze(stock_code)
                self._agent_results[name] = result
            except Exception as e:
                logger.error(f"[{self.name}] {name} Agent失败: {e}")
    
    def get_agent_details(self) -> Dict[str, Dict]:
        details = {}
        for name, result in self._agent_results.items():
            details[name] = {
                'agent_name': result.agent_name, 'score': result.signal.score,
                'signal': result.signal.signal_type.value, 'confidence': result.signal.confidence,
                'reason': result.signal.reason, 'key_factors': result.signal.key_factors,
                'risk_points': result.signal.risk_points
            }
        return details
