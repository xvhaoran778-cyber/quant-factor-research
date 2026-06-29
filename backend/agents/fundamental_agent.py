"""基本面分析师Agent - 融合顶级机构估值分析框架"""

import json
from typing import Dict, Any, List
from loguru import logger
from .base_agent import BaseAgent, AgentResult, AgentSignal, SignalType
from data.collectors import AKShareCollector, TencentCollector


class FundamentalAgent(BaseAgent):
    """基本面分析师Agent
    
    学习对象：巴菲特/格雷厄姆价值投资(Graham & Dodd)、
           麦肯锡战略分析(McKinsey 7S/Porter五力)、
           高盛/摩根士丹利卖方研究估值模型
    
    分析框架：DCF绝对估值 + 相对估值 + 杜邦分析 + 波特五力
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("基本面分析师", config)
        self.akshare = AKShareCollector(config.get('akshare', {}))
        self.tencent = TencentCollector(config.get('tencent', {}))
        self._ensure_connection()
    
    def _ensure_connection(self):
        if not self.akshare.is_connected():
            self.akshare.connect()
        if not self.tencent.is_connected():
            self.tencent.connect()
    
    def get_system_prompt(self) -> str:
        return """你是一位资深基本面分析师，融合了巴菲特/格雷厄姆价值投资学派、麦肯锡战略咨询、以及高盛/摩根士丹利卖方研究的方法论。

## 你的知识体系

### 1. 估值方法论 (Valuation Framework)
- **DCF绝对估值**: 自由现金流折现，关注永续增长率(g)和WACC的合理性
- **相对估值**: PE/PE/Growth(PEG)、PB-ROE体系（PB=ROE×PE的逻辑链）、EV/EBITDA排除资本结构干扰
- **格雷厄姆安全边际**: 内在价值×0.7以下才是买入区，市盈率×市净率<22.5作为快速筛选
- **股息折现模型(DDM)**: 适用于成熟稳定企业

### 2. 盈利质量分析 (Quality of Earnings)
- **杜邦三分法**: ROE = 净利率×资产周转率×权益乘数，区分高ROE来源（高利润率 vs 高杠杆）
- **应计利润分析**: 经营现金流/净利润>1 表示盈利质量好
- **收入确认风险**: 大量应收账款、关联交易、收入确认时点异常
- **费用资本化**: 研发费用资本化率过高是危险信号

### 3. 行业竞争优势 (Porter's Five Forces)
- **进入壁垒**: 牌照、专利、品牌、规模经济、用户迁移成本
- **供应商议价能力**: 上游集中度、原材料依赖度
- **客户议价能力**: 下游集中度、产品差异化程度
- **替代品威胁**: 技术颠覆风险、跨界竞争
- **现有竞争格局**: 行业集中度(CR5/CR10)、价格战可能性

### 4. 财务健康度 (Financial Health)
- **Altman Z-Score**: 制造业: 1.2A+1.4B+3.3C+0.6D+1.0E，>3安全，<1.8危险
- **负债结构**: 短期/长期债务比例，利息保障倍数(EBIT/利息费用>3为安全)
- **现金流三表勾稽**: 经营现金流应能覆盖资本开支+分红

## 评分逻辑
- PE<15 且 ROE>15% → 加分（格雷厄姆捡烟蒂）
- PEG<1 且 revenue_growth>20% → 加分（费雪成长股）
- PB<1 且 ROE>10% → 破净但盈利，潜在低估
- 负债率>70% 且 现金流/负债<0.2 → 减分（财务风险）
- 毛利率连续3年下滑 → 减分（竞争恶化）

## 输出格式
{
    "score": 0-100,
    "signal": "强烈买入/买入/持有/卖出/强烈卖出",
    "confidence": 0-1,
    "reason": "用投资大师的语言分析（150-250字），引用具体指标",
    "key_factors": ["列出3-5个关键支撑因子"],
    "risk_points": ["列出2-4个核心风险点"],
    "valuation": {"pe": number, "pb": number, "status": "低估/合理/高估", "method": "DCF/PB-ROE/Graham"},
    "profitability": {"roe": number, "quality": "优秀/良好/一般/差"},
    "growth_outlook": "高增长/稳定/放缓/衰退",
    "margin_of_safety": "充足/适中/不足"
}"""
    
    def format_user_prompt(self, stock_code: str, data: Dict[str, Any]) -> str:
        valuation = self.tencent.get_valuation_metrics(stock_code)
        quote = self.akshare.get_realtime_quote(stock_code)
        stock_name = quote.get('name', stock_code) if quote else stock_code
        
        prompt = f"""## 基本面分析任务
**股票**: {stock_name}({stock_code})

### 当前估值数据
"""
        if valuation:
            pe = valuation.get('pe_ratio', 'N/A')
            pb = valuation.get('pb_ratio', 'N/A')
            mc = valuation.get('market_cap', 'N/A')
            turnover = valuation.get('turnover_rate', 'N/A')
            prompt += f"""- 市盈率(PE): {pe}
- 市净率(PB): {pb}
- 总市值: {mc}亿
- 换手率: {turnover}%
"""
            # 格雷厄姆快速检查
            try:
                pe_val = float(pe) if pe != 'N/A' else 0
                pb_val = float(pb) if pb != 'N/A' else 0
                if pe_val > 0 and pb_val > 0:
                    graham = pe_val * pb_val
                    graham_status = "✓ 通过" if graham < 22.5 else "✗ 不通过"
                    prompt += f"- **格雷厄姆指标(PE×PB)**: {graham:.1f} {graham_status}(<22.5为安全边际区)\n"
            except:
                pass

        if quote:
            prompt += f"""
### 实时行情
- 现价: {quote.get('price', 'N/A')}
- 涨跌幅: {quote.get('change_pct', 'N/A')}%
"""
        
        prompt += """
## 请按以下框架分析:

1. **估值分析**: 当前PE/PB在行业中处于什么位置？用PEG和格雷厄姆指标判断是否有安全边际
2. **盈利质量**: ROE的来源是什么（高利润率/高周转/高杠杆）？现金流是否健康？
3. **竞争优势**: 公司是否具备护城河（品牌/技术/渠道/规模）？波特五力中的位置？
4. **财务风险**: Altman Z-Score逻辑定性判断，负债结构和偿债能力

请用投资大师的语言风格输出分析，直接返回JSON。"""
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
