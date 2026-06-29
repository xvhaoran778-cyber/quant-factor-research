"""宏观分析师Agent - 全球宏观/货币政策分析框架"""

import json
from typing import Dict, Any
from loguru import logger
from .base_agent import BaseAgent, AgentResult, AgentSignal, SignalType
from data.collectors import TencentCollector, AKShareCollector


class MacroAgent(BaseAgent):
    """宏观分析师Agent
    
    学习对象：美联储/人民银行货币政策框架、
            桥水全天候策略(Ray Dalio's All-Weather)、
            美林投资时钟(Merrill Lynch Investment Clock)、
            高盛/大摩全球宏观研究
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("宏观分析师", config)
        self.tencent = TencentCollector(config.get('tencent', {}))
        self.akshare = AKShareCollector(config.get('akshare', {}))
        self._ensure_connection()
    
    def _ensure_connection(self):
        if not self.tencent.is_connected():
            self.tencent.connect()
        if not self.akshare.is_connected():
            self.akshare.connect()
    
    def get_system_prompt(self) -> str:
        return """你是全球宏观策略师，融合Ray Dalio的经济机器模型、美林投资时钟理论和全球央行的货币政策框架。

## 核心分析框架

### 1. 经济周期定位 (Dalio's Economic Machine)
- **三大驱动力**: 生产率增长(长期)、短期债务周期(5-8年)、长期债务周期(50-75年)
- **交易等式**: 一个人的支出=另一个人的收入，信贷是经济的加速器也是减速器
- **去杠杆的四种方式**: 1)债务削减 2)紧缩支出 3)财富转移 4)央行印钱
- **美丽的去杠杆**: 印钱速度=通缩压力时，可以实现不衰退的去杠杆

### 2. 美林投资时钟 (Investment Clock)
- **复苏期(Recovery)**: GDP↑ CPI↓ → 股票最佳，周期性行业领涨
- **过热期(Overheat)**: GDP↑ CPI↑ → 商品最佳，通胀受益行业
- **滞胀期(Stagflation)**: GDP↓ CPI↑ → 现金为王，防御性行业
- **衰退期(Recession)**: GDP↓ CPI↓ → 债券最佳，必需消费/医药
- **时钟转速**: 当前周期切换速度，央行政策可以加速/减慢周期

### 3. 流动性框架 (Liquidity Analysis)
- **三层次流动性**: 央行→银行间(基础货币)→实体经济(信用派生)→金融市场(风险偏好)
- **社融-M2剪刀差**: 社融增速>M2增速=资金需求旺(经济向好)，反之为需求不足
- **剩余流动性**: M1增速-工业增加值增速，正值=资金流入金融市场
- **信用脉冲**: 新增社融的加速度，领先股市6-9个月

### 4. 政策信号解读
- **货币政策**: 降准=放水5000亿一级，降息=降低资金成本，公开市场操作=短期调节
- **财政政策**: 赤字率、专项债额度、基建投资增速
- **产业政策**: 十四五规划、碳中和、数字经济、半导体自主等
- **监管态度**: 窗口指导、问询函、反垄断等

### 5. 全球联动
- **中美利差**: 利差扩大=资金流出A股压力，利差缩小=外资流入
- **美元指数**: DXY↑→新兴市场承压，DXY↓→利好A股外资流入
- **VIX恐慌指数**: >30=全球避险模式，<15=风险偏好高
- **CRB商品指数**: 反映全球需求，领先PPI

## 输出格式
{
    "score": 0-100,
    "signal": "强烈买入/买入/持有/卖出/强烈卖出",
    "confidence": 0-1,
    "reason": "宏观经济视角（150-250字）",
    "key_factors": ["列出关键宏观因子"],
    "risk_points": ["宏观风险点"],
    "cycle_position": "复苏/过热/滞胀/衰退",
    "liquidity": {"m2_growth": N, "stance": "宽松/中性/收紧"},
    "policy_outlook": "积极/中性/收紧",
    "clock_recommendation": "适合配置的资产方向"
}"""
    
    def format_user_prompt(self, stock_code: str, data: Dict[str, Any]) -> str:
        quote = self.tencent.get_realtime_quote('000001')
        stock_name = data.get('stock_name', stock_code) if data else stock_code
        
        prompt = f"""## 宏观经济分析任务
**参考标的**: {stock_name}({stock_code})

### 经济增长数据
"""
        try:
            gdp = self.akshare.get_macro_gdp()
            if gdp is not None and not gdp.empty:
                prompt += f"- GDP: {gdp.iloc[-1].to_dict()}\n"
            else: prompt += "- GDP数据获取失败\n"
        except: prompt += "- GDP获取失败\n"

        try:
            pmi = self.akshare.get_macro_pmi()
            if pmi is not None and not pmi.empty:
                prompt += f"- PMI: {pmi.iloc[-1].to_dict()}\n"
            else: prompt += "- PMI数据获取失败\n"
        except: prompt += "- PMI获取失败\n"

        try:
            cpi = self.akshare.get_macro_cpi()
            if cpi is not None and not cpi.empty:
                prompt += f"- CPI: {cpi.iloc[-1].to_dict()}\n"
            else: prompt += "- CPI数据获取失败\n"
        except: prompt += "- CPI获取失败\n"

        try:
            money = self.akshare.get_money_supply()
            if money is not None and not money.empty:
                prompt += f"- M2/货币: {money.iloc[-1].to_dict()}\n"
            else: prompt += "- M2数据获取失败\n"
        except: prompt += "- M2获取失败\n"
        
        if quote:
            prompt += f"\n### 上证指数\n- 点位: {quote.get('price', 'N/A')}\n- 涨跌: {quote.get('change_pct', 'N/A')}%\n"
        
        prompt += """
## 请分析
1. **经济周期定位**: 当前处于美林时钟哪个象限？用什么证据支持？
2. **流动性环境**: M2增速、社融、利率环境如何？股市增量资金是否充裕？
3. **政策方向**: 货币政策/财政政策/产业政策的立场是什么？
4. **全球环境**: 中美关系、美联储政策、全球风险偏好的影响？

用Dalio经济机器模型+美林时钟语言，直接返回JSON。"""
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
