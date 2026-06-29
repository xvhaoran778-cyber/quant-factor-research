"""情绪分析师Agent - 行为金融学 + 机构情绪指标框架"""

import json
from typing import Dict, Any
from loguru import logger
from .base_agent import BaseAgent, AgentResult, AgentSignal, SignalType
from data.collectors import TencentCollector, AKShareCollector


class SentimentAgent(BaseAgent):
    """情绪分析师Agent
    
    学习对象：Kahneman/Tversky行为金融学(Behavioral Finance)、
            AAII情绪调查、CNN恐惧贪婪指数、VIX波动率框架、
            高盛/摩根大通资金流向分析
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("情绪分析师", config)
        self.tencent = TencentCollector(config.get('tencent', {}))
        self.akshare = AKShareCollector(config.get('akshare', {}))
        self._ensure_connection()
    
    def _ensure_connection(self):
        if not self.tencent.is_connected():
            self.tencent.connect()
        if not self.akshare.is_connected():
            self.akshare.connect()
    
    def _get_quote(self, stock_code: str) -> Dict:
        quote = self.tencent.get_realtime_quote(stock_code)
        return quote or self.akshare.get_realtime_quote(stock_code) or {}
    
    def get_system_prompt(self) -> str:
        return """你是行为金融学专家+市场情绪分析师，融合了诺贝尔奖得主Kahneman的前景理论、以及华尔街情绪指标框架。

## 核心分析框架

### 1. 行为金融学基础 (Behavioral Finance)
- **前景理论(Prospect Theory)**: 投资者在盈利区间风险厌恶(过早止盈)，亏损区间风险偏好(不愿止损)
- **过度反应与反应不足**: 连续大涨后过度乐观→回调风险，连续大跌后过度悲观→反弹机会
- **羊群效应(Herding)**: 追涨杀跌的散户行为放大趋势，终极信号是极端一致预期
- **锚定效应(Anchoring)**: 投资者锚定某个历史价格(如成本价、历史高点)，影响决策
- **确认偏误(Confirmation Bias)**: 选择性接收支持自己持仓方向的信息

### 2. 市场情绪指标 (Sentiment Indicators)
- **恐惧贪婪指数(Fear & Greed Index)**: 0-25极度恐惧(买入机会)，75-100极度贪婪(卖出信号)
- **资金流向(Money Flow)**: 主力净流入/流出是关键，大单/超大单比中单/小单更有信息含量
- **北向资金(沪港深通)**: 外资行为往往领先内资，连续流入/流出信号强
- **融资融券余额**: 融资余额激增=散户加杠杆看多(反向指标)，融券余额增=机构对冲增加
- **涨停/跌停家数比**: 市场情绪温度计，涨停>100家=亢奋，跌停>50家=恐慌

### 3. 逆向投资框架 (Contrarian Framework)
- **巴菲特名言**: "在别人贪婪时恐惧，在别人恐惧时贪婪"
- **情绪极端时的反转**: 情绪极度一致时(99%看多/看空)，反向操作胜率极高
- **交易拥挤度**: 当一个赛道所有人都在聊的时候，离顶不远了
- **散户持仓比例**: 散户持仓占比激增=筹码分散=上涨动力减弱

### 4. 机构行为解读
- **龙虎榜分析**: 游资/机构席位动向，机构买入=中期看好，游资接力=短期博弈
- **大宗交易**: 折价大宗=老股东减持，溢价大宗=战略投资者入场
- **股东增减持**: 大股东增持(信号强)>高管增持>员工持股计划，高管集中减持=危险信号

## 输出格式
{
    "score": 0-100,
    "signal": "强烈买入/买入/持有/卖出/强烈卖出",
    "confidence": 0-1,
    "reason": "行为金融学语言描述（150-250字）",
    "key_factors": ["列出3-5个关键情绪因子"],
    "risk_points": ["列出行为偏差风险点"],
    "sentiment": {"level": "极度恐惧/恐惧/中性/贪婪/极度贪婪", "trend": "转暖/稳定/转冷"},
    "crowd_behavior": "恐慌出逃/犹豫观望/理性参与/追涨情绪/极度亢奋",
    "contrarian_signal": "逆向买入机会/中性/泡沫卖出信号"
}"""
    
    def format_user_prompt(self, stock_code: str, data: Dict[str, Any]) -> str:
        quote = self._get_quote(stock_code)
        stock_name = quote.get('name', stock_code)
        fund_flow = self.akshare.get_fund_flow(stock_code)
        news_df = self.akshare.get_stock_news(stock_code)
        telegraph = self.akshare.get_cls_telegraph()
        
        prompt = f"""## 情绪分析任务
**股票**: {stock_name}({stock_code})

### 行情数据
- 现价: {quote.get('price', 'N/A')}
- 涨跌幅: {quote.get('change_pct', 'N/A')}%
- 换手率: {quote.get('turnover_rate', 'N/A')}%
- 量比: {quote.get('volume_ratio', 'N/A')}
"""
        if float(quote.get('turnover_rate', 0) or 0) > 10:
            prompt += "- ⚠️ 换手率>10% → 关注筹码交换是否充分\n"
        if abs(float(quote.get('change_pct', 0) or 0)) > 7:
            prompt += "- ⚠️ 涨跌幅>7% → 情绪极端，存在回归压力\n"
        
        prompt += "\n### 资金流向\n"
        if fund_flow:
            prompt += f"""- 主力净流入: {fund_flow.get('main_net_inflow', 'N/A')}
- 主力净占比: {fund_flow.get('main_net_pct', 'N/A')}%
- 超大单: {fund_flow.get('super_large_net', 'N/A')}
- 大单: {fund_flow.get('large_net', 'N/A')}
- 中单: {fund_flow.get('medium_net', 'N/A')}
- 小单: {fund_flow.get('small_net', 'N/A')}
"""
            main_pct = fund_flow.get('main_net_pct', 0) or 0
            if main_pct > 5:
                prompt += "→ 主力大幅流入，机构建仓信号\n"
            elif main_pct < -5:
                prompt += "→ 主力大幅流出，机构减仓信号\n"
        else:
            prompt += "- 资金流数据获取失败\n"
        
        prompt += "\n### 个股新闻（最近）\n"
        if news_df is not None and not news_df.empty:
            for _, row in news_df.head(8).iterrows():
                title = row.get('新闻标题', row.get('标题', ''))
                if title:
                    prompt += f"- {title}\n"
        else:
            prompt += "- 暂无个股新闻\n"
        
        prompt += "\n### 市场快讯（财联社）\n"
        if telegraph is not None and not telegraph.empty:
            for _, row in telegraph.head(5).iterrows():
                content = row.get('内容', row.get('标题', ''))
                if content:
                    prompt += f"- {content}\n"
        else:
            prompt += "- 暂无快讯\n"
        
        prompt += """
## 请分析

1. **情绪温度**: 当前处于恐惧/贪婪的哪个区间？换手率和涨跌幅暗示什么？
2. **资金博弈**: 主力vs散户的资金流向如何？谁在买谁在卖？
3. **逆向机会**: 当前是否有情绪极端带来的逆向操作机会？
4. **行为偏差**: 投资者可能陷入哪些心理陷阱？

用行为金融学语言，直接返回JSON。"""
        return prompt
    
    def parse_llm_response(self, response: str, stock_code: str, stock_name: str) -> AgentResult:
        json_data = self._parse_json_from_response(response)
        if not json_data:
            return self._get_error_result(stock_code, "无法解析分析结果")
        signal_str = json_data.get('signal', '持有')
        signal_map = {'强烈买入': SignalType.STRONG_BUY, '买入': SignalType.BUY,
                      '持有': SignalType.HOLD, '卖出': SignalType.SELL, '强烈卖出': SignalType.STRONG_SELL}
        signal_type = signal_map.get(signal_str, SignalType.HOLD)
        signal = AgentSignal(
            signal_type=signal_type,
            confidence=float(json_data.get('confidence', 0.5)),
            score=float(json_data.get('score', 50)),
            reason=json_data.get('reason', ''),
            key_factors=json_data.get('key_factors', []),
            risk_points=json_data.get('risk_points', [])
        )
        return AgentResult(agent_name=self.name, stock_code=stock_code,
                          stock_name=stock_name, signal=signal, raw_data=json_data)
