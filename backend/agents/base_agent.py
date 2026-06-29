"""Agent基类 - 支持LLM调用和结果缓存"""

import json
import hashlib
from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from loguru import logger

from openai import OpenAI


class SignalType(Enum):
    """交易信号类型"""
    STRONG_BUY = "强烈买入"
    BUY = "买入"
    HOLD = "持有"
    SELL = "卖出"
    STRONG_SELL = "强烈卖出"


@dataclass
class AgentSignal:
    """Agent分析信号"""
    signal_type: SignalType
    confidence: float  # 0-1 信心度
    score: float  # 0-100 评分
    reason: str  # 判断理由
    key_factors: List[str] = field(default_factory=list)  # 关键因子
    risk_points: List[str] = field(default_factory=list)  # 风险点
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            'signal_type': self.signal_type.value,
            'confidence': self.confidence,
            'score': self.score,
            'reason': self.reason,
            'key_factors': self.key_factors,
            'risk_points': self.risk_points,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class AgentResult:
    """Agent分析结果"""
    agent_name: str
    stock_code: str
    stock_name: str
    signal: AgentSignal
    raw_data: Dict[str, Any] = field(default_factory=dict)
    llm_response: str = ""  # LLM原始回复
    analysis_time: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            'agent_name': self.agent_name,
            'stock_code': self.stock_code,
            'stock_name': self.stock_name,
            'signal': self.signal.to_dict(),
            'llm_response': self.llm_response,
            'analysis_time': self.analysis_time.isoformat()
        }


class BaseAgent(ABC):
    """Agent基类 - LLM驱动"""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        
        # 初始化LLM客户端
        llm_config = self.config.get('llm', {})
        self.client = OpenAI(
            api_key=llm_config.get('api_key', ''),
            base_url=llm_config.get('base_url', 'https://api.deepseek.com')
        )
        self.model = llm_config.get('model', 'deepseek-chat')
        self.temperature = llm_config.get('temperature', 0.7)
        self.max_tokens = llm_config.get('max_tokens', 2000)
        
        # 初始化内存缓存
        cache_config = llm_config.get('cache', {})
        if cache_config.get('enabled', True):
            self.cache = {}  # 内存缓存
            self.cache_ttl = cache_config.get('ttl', 3600)
        else:
            self.cache = None
            self.cache_ttl = 0
        
        self._last_result: Optional[AgentResult] = None
        logger.info(f"初始化Agent: {name}")
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """获取Agent角色的系统提示词"""
        pass
    
    @abstractmethod
    def format_user_prompt(self, stock_code: str, data: Dict[str, Any]) -> str:
        """格式化用户提示词（包含股票数据）"""
        pass
    
    @abstractmethod
    def parse_llm_response(self, response: str, stock_code: str, stock_name: str) -> AgentResult:
        """解析LLM返回结果"""
        pass
    
    def analyze(self, stock_code: str, data: Dict[str, Any] = None) -> AgentResult:
        """调用LLM分析股票"""
        logger.info(f"[{self.name}] 开始分析 {stock_code}")
        
        # 检查缓存
        cache_key = self._get_cache_key(stock_code, data)
        if self.cache and cache_key in self.cache:
            logger.info(f"[{self.name}] 使用缓存结果: {stock_code}")
            return self.cache[cache_key]
        
        # 构建提示词
        system_prompt = self.get_system_prompt()
        user_prompt = self.format_user_prompt(stock_code, data or {})
        
        # 调用LLM
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            llm_response = response.choices[0].message.content
            logger.info(f"[{self.name}] LLM分析完成: {stock_code}")
            
            # 解析结果
            stock_name = data.get('stock_name', '') if data else ''
            result = self.parse_llm_response(llm_response, stock_code, stock_name)
            result.llm_response = llm_response
            
            # 缓存结果（内存）
            if self.cache is not None:
                self.cache[cache_key] = result
            
            self._last_result = result
            return result
            
        except Exception as e:
            logger.error(f"[{self.name}] LLM调用失败: {e}")
            return self._get_error_result(stock_code, str(e))
    
    def _get_cache_key(self, stock_code: str, data: Dict = None) -> str:
        """生成缓存键"""
        # 使用日期+股票代码+数据hash作为缓存键
        today = datetime.now().strftime('%Y%m%d')
        data_str = json.dumps(data or {}, sort_keys=True, default=str)
        data_hash = hashlib.md5(data_str.encode()).hexdigest()[:8]
        return f"{self.name}_{stock_code}_{today}_{data_hash}"
    
    def _parse_json_from_response(self, response: str) -> Dict:
        """从LLM回复中提取JSON"""
        # 尝试直接解析
        try:
            return json.loads(response)
        except:
            pass
        
        # 尝试提取```json ... ```中的内容
        import re
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except:
                pass
        
        # 尝试提取{ ... }中的内容
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except:
                pass
        
        return {}
    
    def _score_to_signal(self, score: float) -> SignalType:
        """将评分转换为信号类型"""
        if score >= 80:
            return SignalType.STRONG_BUY
        elif score >= 60:
            return SignalType.BUY
        elif score >= 40:
            return SignalType.HOLD
        elif score >= 20:
            return SignalType.SELL
        else:
            return SignalType.STRONG_SELL
    
    def _get_error_result(self, stock_code: str, error_msg: str) -> AgentResult:
        """生成错误结果"""
        signal = AgentSignal(
            signal_type=SignalType.HOLD,
            confidence=0.0,
            score=50,
            reason=f"分析失败: {error_msg}",
            key_factors=[],
            risk_points=["LLM调用异常"]
        )
        return AgentResult(
            agent_name=self.name,
            stock_code=stock_code,
            stock_name='',
            signal=signal
        )
    
    def get_last_result(self) -> Optional[AgentResult]:
        """获取上次分析结果"""
        return self._last_result
    
    def clear_cache(self, stock_code: str = None):
        """清除缓存"""
        if self.cache is not None:
            if stock_code:
                keys_to_delete = [k for k in list(self.cache.keys()) if stock_code in k]
                for key in keys_to_delete:
                    del self.cache[key]
            else:
                self.cache.clear()
            logger.info(f"[{self.name}] 缓存已清除")
