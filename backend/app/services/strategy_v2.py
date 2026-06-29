"""strategy_v2.py — V2 策略体系：回撤归因 + 多策略方向 + 版本管理

本文件是策略版本管理的统一入口，包含:
  - 所有策略的版本注册
  - 回撤归因分析
  - 优化后的风控逻辑
  - 多策略方向并行

版本命名: V{major}.{minor}.{patch}
  主版本: 策略方向更换
  次版本: 权重/参数调整  
  补丁: bug修复/微调
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Callable
from enum import Enum

# ═══════════════════════════════════════════════════════════════
# 策略版本注册表
# ═══════════════════════════════════════════════════════════════

class StrategyVersion:
    """策略版本元数据"""
    def __init__(self, vid: str, name: str, category: str, weights: dict,
                 params: dict | None = None, description: str = ""):
        self.vid = vid          # 版本ID: V1.0.0
        self.name = name        # 策略名称
        self.category = category # 策略类别: reversal/momentum/lowvol/hybrid
        self.weights = weights  # 因子权重
        self.params = params or {}  # 其他参数 (止损/持仓数/过滤条件)
        self.description = description

    def __repr__(self):
        return f"[{self.vid}] {self.name} ({self.category})"


# ── 版本历史 ──────────────────────────────────────────────────
VERSIONS: dict[str, StrategyVersion] = {}

def register(v: StrategyVersion):
    VERSIONS[v.vid] = v
    return v

# V1.0.0 — 原始小盘超跌反转 (夏普优化后)
register(StrategyVersion(
    vid="V1.0.0", name="小盘超跌反转-基础版", category="reversal",
    weights={"small": 0.45, "reversal": 0.25, "trend": 0.20, "lowvol": 0.10},
    params={"top_n": 5, "retention": 12, "min_price": 3.0},
    description="小盘+反转+趋势+低波，夏普优化后的基础权重"
))

# V1.1.0 — 加波动率风控
register(StrategyVersion(
    vid="V1.1.0", name="小盘超跌反转-波动风控版", category="reversal",
    weights={"small": 0.45, "reversal": 0.25, "trend": 0.20, "lowvol": 0.10},
    params={"top_n": 5, "retention": 12, "min_price": 3.0,
            "vol_shrink": 0.5},  # 市场波动率高时减仓
    description="V1.0.0 + 市场波动率过高时主动减仓"
))

# V1.4.0 — 加入动量强度因子 alpha030 (夏普提升最大)
register(StrategyVersion(
    vid="V1.4.0", name="小盘超跌反转-alpha030增强版", category="reversal",
    weights={"small": 0.40, "reversal": 0.18, "trend": 0.18, "lowvol": 0.14, "alpha030": 0.05},
    params={"top_n": 5, "retention": 12, "min_price": 3.0},
    description="V1.3.0 + alpha030(动量强度)5%, 夏普提升+0.05"
))
register(StrategyVersion(
    vid="V1.3.0", name="小盘超跌反转-夏普优化版", category="reversal",
    weights={"small": 0.45, "reversal": 0.20, "trend": 0.20, "lowvol": 0.15},
    params={"top_n": 5, "retention": 12, "min_price": 3.0},
    description="夏普优化版: 低波从10%→15%, 夏普0.548→0.562"
))
register(StrategyVersion(
    vid="V1.2.0", name="小盘超跌反转-趋势加速出场版", category="reversal",
    weights={"small": 0.45, "reversal": 0.25, "trend": 0.20, "lowvol": 0.10},
    params={"top_n": 5, "retention": 12, "min_price": 3.0,
            "trend_exit": 0.05},  # 趋势加速向下时提前出场
    description="V1.0.0 + 趋势加速下跌时提前空仓"
))

# V2.0.0 — 低波动稳健
register(StrategyVersion(
    vid="V2.0.0", name="低波动稳健", category="lowvol",
    weights={"lowvol": 1.0},
    params={"top_n": 5, "retention": 12, "min_price": 3.0},
    description="纯低波动因子，IC研究显示稳定正向IC"
))

# V2.1.0 — 低波动+反转
register(StrategyVersion(
    vid="V2.1.0", name="低波动反转复合", category="hybrid",
    weights={"lowvol": 0.5, "reversal": 0.3, "small": 0.2},
    params={"top_n": 5, "retention": 12, "min_price": 3.0},
    description="低波动为主+反转辅助+小盘过滤"
))

# V3.0.0 — 量价背离(IC研究发现alpha172最强但单独用亏钱)
register(StrategyVersion(
    vid="V3.0.0", name="量价背离-防守型", category="hybrid",
    weights={"alpha172": 0.3, "lowvol": 0.3, "reversal": 0.25, "trend": 0.15},
    params={"top_n": 5, "retention": 12, "min_price": 3.0},
    description="量价相关性(IC最强) + 低波 + 反转 + 趋势，综合防守"
))

# V3.1.0 — 量价增强(用IC Top因子的加权组合)
register(StrategyVersion(
    vid="V3.1.0", name="IC多因子综合", category="hybrid",
    weights={"alpha172": 0.20, "alpha149": 0.15, "alpha144": 0.15,
             "lowvol": 0.20, "reversal": 0.15, "trend": 0.15},
    params={"top_n": 5, "retention": 12, "min_price": 3.0},
    description="综合IC Top因子: 量价+动量+低波+反转"
))

# V4.0.0 — 市场择时增强版(只在趋势明确时入场)
register(StrategyVersion(
    vid="V4.0.0", name="增强择时-小盘反转", category="reversal",
    weights={"small": 0.45, "reversal": 0.25, "trend": 0.20, "lowvol": 0.10},
    params={"top_n": 5, "retention": 12, "min_price": 3.0,
            "timing_ma_short": 10, "timing_ma_long": 30,
            "timing_vol_mult": 1.5},
    description="V1.0.0 + 改进的择时(短期MA+波动率阈值)"
))


def get_version(vid: str) -> StrategyVersion | None:
    return VERSIONS.get(vid)

def list_versions(category: str | None = None) -> list[StrategyVersion]:
    vs = list(VERSIONS.values())
    if category:
        vs = [v for v in vs if v.category == category]
    return sorted(vs, key=lambda v: v.vid)


# ═══════════════════════════════════════════════════════════════
# 回撤归因分析
# ═══════════════════════════════════════════════════════════════

def analyze_drawdowns(equity_curve: list[float], bench_close: list[float],
                      dates: list[str], top_n: int = 5) -> list[dict]:
    """分析最大回撤时期的市场状态。"""
    eq = np.array(equity_curve)
    bm = np.array(bench_close)
    
    # 计算回撤
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    
    # 找 Top N 回撤区间
    drawdowns = []
    in_dd = False
    dd_start, dd_peak_val = 0, 0
    
    for i in range(len(dd)):
        if not in_dd and dd[i] < -0.02:  # 进入回撤
            in_dd = True
            dd_start = i
            dd_peak_val = peak[i]
        elif in_dd and (dd[i] >= -0.01 or i == len(dd)-1):  # 恢复
            if i > dd_start:
                dd_depth = (eq[i] - dd_peak_val) / dd_peak_val * 100
                if dd_depth < -5:  # 只记录 >5% 的回撤
                    # 市场状态分析
                    bm_start = bm[dd_start] if dd_start < len(bm) else 1
                    bm_end = bm[i] if i < len(bm) else 1
                    bm_change = (bm_end / bm_start - 1) * 100
                    vol_period = max(0, dd_start - 20)
                    bm_vol = np.std(np.diff(bm[vol_period:dd_start+1]) / bm[vol_period:dd_start]) * 100 if dd_start > vol_period else 0
                    
                    drawdowns.append({
                        'start_date': dates[dd_start],
                        'end_date': dates[i] if i < len(dates) else dates[-1],
                        'duration_days': i - dd_start,
                        'depth': round(dd_depth, 2),
                        'benchmark_change': round(bm_change, 2),
                        'benchmark_vol': round(bm_vol, 4),
                        'type': '同步下跌' if bm_change < -3 else '独立回撤',
                    })
            in_dd = False
    
    drawdowns.sort(key=lambda x: x['depth'])
    return drawdowns[:top_n]


def classify_market_regime(bench_close: np.ndarray, lookback: int = 60) -> str:
    """分类市场状态。"""
    if len(bench_close) < lookback:
        return "unknown"
    ret = bench_close[-1] / bench_close[-lookback] - 1
    vol = np.std(np.diff(bench_close[-lookback:]) / bench_close[-lookback:-1])
    ma20 = np.mean(bench_close[-20:])
    ma60 = np.mean(bench_close[-60:])
    
    if ret > 0.05 and vol < 0.015:
        return "平稳上涨"
    elif ret > 0.05 and vol >= 0.015:
        return "急涨"
    elif ret < -0.05 and vol < 0.015:
        return "阴跌"
    elif ret < -0.05 and vol >= 0.015:
        return "急跌"
    elif bench_close[-1] < ma20 < ma60:
        return "空头排列"
    elif bench_close[-1] > ma20 > ma60:
        return "多头排列"
    else:
        return "震荡"


# ═══════════════════════════════════════════════════════════════
# 增强风控模块
# ═══════════════════════════════════════════════════════════════

def enhanced_risk_check(
    bench_series: pd.Series,
    current_idx: int,
    params: dict,
) -> tuple[bool, str]:
    """增强版风控检查，返回 (是否可交易, 原因)。

    规则:
      1. 基础: close < MA20 < MA60 → 空仓 (原规则)
      2. 波动率: 20日波动率 > 阈值 → 减仓
      3. 趋势加速: 5日跌幅 > 阈值 → 空仓
      4. VIX代理: 近期波动率剧增 → 减仓
    """
    if current_idx < 60:
        return True, "warmup"
    
    close = bench_series.iloc[:current_idx + 1].astype(float)
    ma20 = close.rolling(20).mean().iloc[-1]
    ma60 = close.rolling(60).mean().iloc[-1]
    
    # 规则1: 原MA空仓规则
    if pd.notna(ma20) and pd.notna(ma60):
        if close.iloc[-1] < ma20 < ma60:
            return False, f"MA空仓: close({close.iloc[-1]:.0f})<MA20({ma20:.0f})<MA60({ma60:.0f})"
    
    # 规则2: 波动率检查
    vol_shrink = params.get("vol_shrink", 0)
    if vol_shrink > 0 and current_idx > 20:
        recent_vol = close.pct_change().iloc[-20:].std()
        hist_vol = close.pct_change().iloc[-60:].std() if current_idx >= 60 else recent_vol
        if hist_vol > 0:
            vol_ratio = recent_vol / hist_vol
            if vol_ratio > 1.5 + vol_shrink:
                return False, f"波动率激增: 近期/历史={vol_ratio:.2f}"
    
    # 规则3: 趋势加速
    trend_exit = params.get("trend_exit", 0)
    if trend_exit > 0 and current_idx >= 5:
        ret5 = close.iloc[-1] / close.iloc[-6] - 1
        if ret5 < -trend_exit:
            return False, f"趋势加速下跌: 5日={ret5:.2%}"
    
    # 规则4: 短期均线加速
    timing_ma_short = params.get("timing_ma_short", 0)
    timing_ma_long = params.get("timing_ma_long", 0)
    if timing_ma_short > 0 and timing_ma_long > 0 and current_idx >= timing_ma_long:
        ma_short = close.rolling(timing_ma_short).mean().iloc[-1]
        ma_long = close.rolling(timing_ma_long).mean().iloc[-1]
        if ma_short < ma_long:
            return False, f"短期均线下穿: MA{timing_ma_short}<MA{timing_ma_long}"
    
    return True, ""


# ═══════════════════════════════════════════════════════════════
# 多策略评分函数注册表
# ═══════════════════════════════════════════════════════════════

def score_reversal_v1(features: pd.DataFrame) -> pd.DataFrame:
    """V1.0.0: 小盘超跌反转"""
    df = features.copy()
    df['score'] = ((1 - df['liquidity'].rank(pct=True)) * 0.45
                   + (1 - df['ret5'].rank(pct=True)) * 0.25
                   + df['trend60'].rank(pct=True) * 0.20
                   + (1 - df['vol20'].rank(pct=True)) * 0.10)
    return df.sort_values('score', ascending=False)


def score_lowvol_v1(features: pd.DataFrame) -> pd.DataFrame:
    """V2.0.0: 纯低波动"""
    df = features.copy()
    df['score'] = (1 - df['vol20'].rank(pct=True))
    return df.sort_values('score', ascending=False)


def score_lowvol_reversal(features: pd.DataFrame) -> pd.DataFrame:
    """V2.1.0: 低波动+反转"""
    df = features.copy()
    df['score'] = ((1 - df['vol20'].rank(pct=True)) * 0.50
                   + (1 - df['ret5'].rank(pct=True)) * 0.30
                   + (1 - df['liquidity'].rank(pct=True)) * 0.20)
    return df.sort_values('score', ascending=False)


def score_hybrid_v1(features: pd.DataFrame) -> pd.DataFrame:
    """V3.1.0: IC多因子综合"""
    df = features.copy()
    # alpha172 = rank(corr(c,v,10)) + rank(corr(c,v,5))
    # alpha144 = 20d return
    # 这些需要在 compute_features 中计算，这里用代理因子
    df['score'] = ((1 - df['vol20'].rank(pct=True)) * 0.20
                   + (1 - df['ret5'].rank(pct=True)) * 0.15
                   + df['trend60'].rank(pct=True) * 0.15
                   + (1 - df['liquidity'].rank(pct=True)) * 0.20
                   + df['ret20'].rank(pct=True) * 0.15
                   + df['liquidity'].rank(pct=True) * 0.15)
    return df.sort_values('score', ascending=False)


# ── 评分函数注册表 ──
SCORING_REGISTRY: dict[str, Callable] = {
    "V1.0.0": score_reversal_v1,
    "V2.0.0": score_lowvol_v1,
    "V2.1.0": score_lowvol_reversal,
    "V3.1.0": score_hybrid_v1,
}


def get_scoring_fn(vid: str) -> Callable | None:
    return SCORING_REGISTRY.get(vid)
