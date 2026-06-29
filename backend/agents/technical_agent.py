"""技术面分析师Agent - 融合顶级交易机构量价分析方法论"""

import json
from typing import Dict, Any
import pandas as pd
from loguru import logger
from .base_agent import BaseAgent, AgentResult, AgentSignal, SignalType
from data.collectors import TencentCollector, AKShareCollector


class TechnicalAgent(BaseAgent):
    """技术面分析师Agent
    
    学习对象：Wyckoff量价分析法(Wyckoff Method)、
            Market Profile/Volume Profile(CME芝加哥商品交易所)、
            Elliott Wave Theory、Ichimoku Kinko Hyo(一目均衡表)、
            缠论、Darvas Box Method
    
    核心框架：量价关系 > 趋势跟踪 > 形态识别 > 指标辅助
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("技术面分析师", config)
        self.tencent = TencentCollector(config.get('tencent', {}))
        self.akshare = AKShareCollector(config.get('akshare', {}))
        self._ensure_connection()
    
    def _ensure_connection(self):
        if not self.tencent.is_connected():
            self.tencent.connect()
        if not self.akshare.is_connected():
            self.akshare.connect()
    
    def _get_kline(self, stock_code: str, count: int = 60) -> pd.DataFrame:
        kline = self.tencent.get_kline(stock_code, period='daily', count=count)
        if kline is not None and len(kline) > 20:
            return kline
        kline = self.akshare.get_kline(stock_code, period='daily', count=count)
        return kline
    
    def _get_quote(self, stock_code: str) -> Dict:
        quote = self.tencent.get_realtime_quote(stock_code)
        return quote or self.akshare.get_realtime_quote(stock_code) or {}
    
    def get_system_prompt(self) -> str:
        return """你是一位资深技术分析师，融合了Richard Wyckoff的量价分析体系、CME的市场轮廓理论、以及华尔街顶级交易员的方法论。

## 你的核心分析框架

### 1. Wyckoff量价分析 (Volume Price Analysis)
- **量是价的先行指标**: 量在价先，成交量验证价格方向
- **Effort vs Result**: 放量不涨=供应方占优(派发信号)，缩量上涨=供应枯竭(吸筹信号)
- **Wyckoff三段论**: 
  - 吸筹区(Accumulation): 底部放量止跌，spring测试后反弹
  - 趋势推进(Markup): 缩量回调+放量上涨，上升趋势确认
  - 派发区(Distribution): 顶部放量滞涨，UTAD(上冲回落)
- **关键价位**: 支撑/阻力位的量价验证，突破需要放量确认

### 2. 市场轮廓理论 (Market Profile)
- **Value Area**: 70%成交量集中的价格区域，价格在VA内=平衡市，突破VA=趋势
- **POC(Point of Control)**: 最大成交量价位，是核心支撑/阻力
- **单边/双边分布**: 单边分布=趋势延续，正态分布=区间震荡
- **Initial Balance**: 开盘第一个30分钟的高低点区间，突破方向预示日内趋势

### 3. 技术指标的正确使用 (Indicator Science)
- **MACD**: DIF/DEA的金叉死叉，更关注背离(Divergence)——价格新高但MACD未新高=顶背离
- **RSI(14)**: 40-60为中性区，<30超卖但趋势中可继续跌，>70超买但强势可继续涨
  - RSI背离比超买超卖更重要
- **布林带(20,2)**: 带宽收缩=波动率降低=即将突破，价格沿带宽外侧运行=强势趋势
- **一目均衡表**: 转换线/基准线金叉死叉，云层厚度=支撑阻力强度

### 4. 形态学 (Pattern Recognition)
- **VCP(Volatility Contraction Pattern)**: Mark Minervini的波动收缩形态，收缩后放量突破
- **杯柄形态**: William O'Neil的CANSLIM选股法核心形态
- **头肩顶/底**: 经典反转形态，需右肩量缩确认
- **旗形/三角整理**: 趋势中的中继形态，突破方向大概率延续原趋势

## 评分逻辑
- 缩量回调+放量上涨(Wyckoff买入信号) → 85-95分
- MACD+RSI同时底背离 → 80-90分
- 布林带极度收缩(带宽<5%) → 关注突破方向
- 放量滞涨(Wyckoff派发信号) → 10-25分
- 连续放量跌破关键支撑 → 15-30分

## 输出格式
{
    "score": 0-100,
    "signal": "强烈买入/买入/持有/卖出/强烈卖出",
    "confidence": 0-1,
    "reason": "用量价分析语言描述（150-250字），引用具体指标和形态",
    "key_factors": ["列出3-5个关键技术因子"],
    "risk_points": ["列出2-4个技术风险点"],
    "trend": {"direction": "上升/震荡/下降", "strength": "强/中/弱"},
    "volume_analysis": "放量上涨健康/缩量上涨警惕/放量滞涨派发/缩量下跌吸筹",
    "wyckoff_phase": "吸筹/推进/派发/下跌",
    "support": number,
    "resistance": number,
    "indicators": {"macd": "金叉/死叉/顶背离/底背离", "rsi": "超买/超卖/中性/背离"}
}"""
    
    def format_user_prompt(self, stock_code: str, data: Dict[str, Any]) -> str:
        quote = self._get_quote(stock_code)
        kline = self._get_kline(stock_code, count=60)
        stock_name = quote.get('name', stock_code)
        
        prompt = f"""## 技术面分析任务
**股票**: {stock_name}({stock_code})

### 实时行情
"""
        if quote:
            prompt += f"""- 现价: {quote.get('price', 'N/A')}
- 涨跌幅: {quote.get('change_pct', 'N/A')}%
- 换手率: {quote.get('turnover_rate', 'N/A')}%
- 量比: {quote.get('volume_ratio', 'N/A')}
"""
        
        if kline is not None and len(kline) > 20:
            recent = kline.tail(20)
            close = kline['close'].values
            
            ma5 = recent['close'].rolling(5).mean().iloc[-1] if len(recent) >= 5 else None
            ma10 = recent['close'].rolling(10).mean().iloc[-1] if len(recent) >= 10 else None
            ma20 = recent['close'].rolling(20).mean().iloc[-1] if len(recent) >= 20 else None
            
            # RSI
            n = len(close)
            rsi = 50
            if n > 14:
                deltas = __import__('numpy').diff(close[-15:])
                gains = deltas[deltas > 0].sum()
                losses = -deltas[deltas < 0].sum()
                rsi = 100 - 100/(1 + gains/(losses + 1e-8)) if losses > 0 else (80 if gains > 0 else 50)
            
            # MACD
            ema12 = __import__('pandas').Series(close).ewm(span=12).mean()
            ema26 = __import__('pandas').Series(close).ewm(span=26).mean()
            dif = ema12.iloc[-1] - ema26.iloc[-1]
            dea = (ema12 - ema26).ewm(span=9).mean().iloc[-1]
            prev_dif = ema12.iloc[-2] - ema26.iloc[-2]
            prev_dea = (ema12 - ema26).ewm(span=9).mean().iloc[-2]
            
            macd_state = "金叉" if dif > dea and prev_dif <= prev_dea else \
                        "死叉" if dif < dea and prev_dif >= prev_dea else \
                        "多头" if dif > dea else "空头"
            
            # 量比
            vol_recent = recent['volume'].tail(5).mean()
            vol_prev = kline['volume'].iloc[-15:-5].mean() if len(kline) > 15 else vol_recent
            vol_ratio = vol_recent / vol_prev if vol_prev > 0 else 1
            
            # 5/20日涨跌
            chg5 = (close[-1]/close[-6]-1)*100 if n>5 else 0
            chg20 = (close[-1]/close[-21]-1)*100 if n>20 else 0
            
            # Wyckoff初步判断
            price_up = close[-1] > close[-2] if n > 1 else False
            vol_up = kline['volume'].iloc[-1] > kline['volume'].iloc[-2] if n > 1 else False
            
            wyckoff_hint = ""
            if price_up and vol_up:
                wyckoff_hint = "放量上涨 → 需求方主导，趋势推进阶段"
            elif price_up and not vol_up:
                wyckoff_hint = "缩量上涨 → 供应枯竭信号，需关注后续量能"
            elif not price_up and vol_up:
                wyckoff_hint = "放量下跌 → 供应方主导，派发/下跌阶段"
            else:
                wyckoff_hint = "缩量下跌 → 供需两弱，底部吸筹可能"
            
            prompt += f"""
### 技术指标计算
- MA5: {ma5:.2f if ma5 else 'N/A'} | MA10: {ma10:.2f if ma10 else 'N/A'} | MA20: {ma20:.2f if ma20 else 'N/A'}
- 均线排列: {'多头' if ma5 and ma20 and ma5>ma20 else '空头' if ma5 and ma20 and ma5<ma20 else '交叉'}
- RSI(14): {rsi:.1f} {'超买' if rsi>70 else '超卖' if rsi<30 else '中性'}
- MACD: {macd_state} (DIF={dif:.3f}, DEA={dea:.3f})
- 5日涨跌: {chg5:.2f}% | 20日涨跌: {chg20:.2f}%
- 量比(5/10): {vol_ratio:.2f}
- **Wyckoff初步判断**: {wyckoff_hint}

### 近10日K线
"""
            for _, row in kline.tail(10).iterrows():
                vol_mark = "🔥" if row['volume'] > kline['volume'].tail(20).mean() * 1.5 else ""
                prompt += f"- {row['date']}: O{row['open']:.2f} H{row['high']:.2f} L{row['low']:.2f} C{row['close']:.2f} V{row['volume']} {vol_mark}\n"
        
        prompt += """
## 请按以下框架分析

1. **Wyckoff阶段判断**: 当前处于吸筹/推进/派发/下跌哪个阶段？量价配合是否健康？
2. **指标综合分析**: MACD/RSI是否有背离？均线排列方向？布林带位置？
3. **关键价位**: 最近的强支撑和强阻力在哪里（用量密集区判断）？
4. **形态识别**: 是否有VCP/杯柄/旗形等形态？突破方向预判？

用Wyckoff和华尔街交易员的分析语言，直接返回JSON。"""
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
