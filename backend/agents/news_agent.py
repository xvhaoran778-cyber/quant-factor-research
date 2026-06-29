"""消息面分析师Agent - 事件驱动+信息套利框架"""

import json
from typing import Dict, Any
from loguru import logger
from .base_agent import BaseAgent, AgentResult, AgentSignal, SignalType
from data.collectors import TencentCollector, AKShareCollector


class NewsAgent(BaseAgent):
    """消息面分析师Agent
    
    学习对象：彭博/路透新闻分析框架(Economist Intelligence Unit)、
            事件驱动策略(Event-Driven Strategy)、
            索罗斯反身性理论(Reflexivity Theory)
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("消息面分析师", config)
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
        return """你是事件驱动策略分析师，融合彭博/路透新闻分析框架和索罗斯反身性理论。

## 核心分析框架

### 1. 消息影响力评估 (Information Impact Assessment)
- **信息级联效应**: 重大消息会引发连锁反应（机构先动→媒体放大→散户跟风）
- **时效性衰减**: 消息影响力随时间的4次方衰减（24小时后仅剩6%影响力）
- **预期差分析**: 实际消息与市场预期的差距决定价格反应幅度，超预期=跳空，不及预期=回补
- **信息不对称**: 懂的人早就买完了，消息出来时往往是尾声

### 2. 消息分类与权重
- **★★★★★ 重大**: 政策转向（货币政策/产业政策）、重大资产重组、实控人变更
- **★★★★ 重要**: 业绩预告大幅变化（±50%以上）、大额订单/合同、重大诉讼
- **★★★ 中等**: 券商研报评级调整、股东增/减持、分红方案
- **★★ 一般**: 行业会议/展会、公司日常经营动态
- **★ 噪音**: 股吧论坛讨论、自媒体传闻

### 3. 索罗斯反身性理论 (Reflexivity)
- **认知-现实反馈环**: 消息改变投资者认知→认知改变投资者行为→行为改变价格→价格再改变认知
- **正反馈泡沫**: 利好消息→价格上涨→引发更多关注→更多买入→继续上涨
- **负反馈崩溃**: 利空消息→价格下跌→引发恐慌→更多抛售→继续下跌
- **拐点识别**: 当所有人都接受某个叙事时，反身性拐点临近

### 4. 公告/研报解读能力
- **业绩公告**: 营收/利润增长但现金流恶化=质量可疑的"增长"
- **高送转**: 10送10以上往往是配合股东减持的"利好"
- **券商研报**: 看评级变化趋势（连续上调>首次覆盖），目标价只做参考
- **互动平台**: 公司回复的措辞和态度透露基本面变化

## 输出格式
{
    "score": 0-100,
    "signal": "强烈买入/买入/持有/卖出/强烈卖出",
    "confidence": 0-1,
    "reason": "事件驱动视角描述（150-250字）",
    "key_factors": ["列出关键消息因子"],
    "risk_points": ["消息面风险"],
    "news_summary": {"positive_count": N, "negative_count": N, "total_count": N},
    "impact_level": "重大利好/利好/中性/利空/重大利空",
    "reflexivity_state": "正反馈强化中/拐点出现/负反馈强化中"
}"""
    
    def format_user_prompt(self, stock_code: str, data: Dict[str, Any]) -> str:
        quote = self._get_quote(stock_code)
        stock_name = quote.get('name', stock_code)
        
        prompt = f"""## 消息面分析任务
**股票**: {stock_name}({stock_code})

### 行情数据
- 现价: {quote.get('price', 'N/A')}
- 涨跌幅: {quote.get('change_pct', 'N/A')}%
"""
        
        # 个股新闻
        prompt += "\n### 个股新闻\n"
        try:
            news_df = self.akshare.get_stock_news(stock_code)
            if news_df is not None and not news_df.empty:
                for _, row in news_df.head(15).iterrows():
                    t = row.get('新闻标题', row.get('标题', ''))
                    tm = row.get('新闻时间', row.get('时间', ''))
                    if t: prompt += f"- [{tm}] {t}\n"
            else:
                prompt += "- 暂无个股新闻\n"
        except Exception as e:
            prompt += f"- 获取失败: {e}\n"
        
        # 财联社快讯
        prompt += "\n### 财联社快讯\n"
        try:
            telegraph = self.akshare.get_cls_telegraph()
            if telegraph is not None and not telegraph.empty:
                for _, row in telegraph.head(20).iterrows():
                    c = row.get('内容', row.get('标题', ''))
                    if c and len(c) > 10: prompt += f"- {c}\n"
                if len(telegraph) == 0: prompt += "- 暂无快讯\n"
            else:
                prompt += "- 暂无快讯\n"
        except Exception as e:
            prompt += f"- 获取失败: {e}\n"
        
        # 研报
        prompt += "\n### 机构研报\n"
        try:
            report = self.akshare.get_research_report(stock_code=stock_code)
            if report is not None and not report.empty:
                for _, row in report.head(5).iterrows():
                    t = row.get('研报标题', row.get('标题', ''))
                    if t: prompt += f"- {t}\n"
                if len(report) == 0: prompt += "- 暂无研报\n"
            else:
                prompt += "- 暂无研报\n"
        except:
            prompt += "- 研报获取失败\n"
        
        # 公告
        prompt += "\n### 公司公告\n"
        try:
            announcements = self.akshare.get_announcements(stock_code=stock_code)
            if announcements is not None and not announcements.empty:
                for _, row in announcements.head(8).iterrows():
                    t = row.get('公告标题', row.get('标题', ''))
                    if t: prompt += f"- {t}\n"
                if len(announcements) == 0: prompt += "- 暂无公告\n"
            else:
                prompt += "- 暂无公告\n"
        except:
            prompt += "- 公告获取失败\n"
        
        prompt += """
## 请分析
1. **消息面总评**: 当前利好消息的权重和质量如何？有没有重大消息？
2. **反身性判断**: 市场处于正反馈强化期，还是拐点附近？
3. **信息不对称**: 哪些消息可能已经Price in？哪些是新增信息？
4. **时间窗口**: 未来1-2周是否有催化剂（业绩发布/政策会议等）？

用事件驱动+反身性理论语言，直接返回JSON。"""
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
