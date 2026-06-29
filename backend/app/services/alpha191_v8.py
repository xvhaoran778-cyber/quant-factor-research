"""alpha191_v8.py — V8 动量趋势主导策略 (Alpha191 因子版)

基于 GTJA Alpha191 全量因子，使用核心动量/趋势因子的加权评分。
评分因子组合（来自旧策略分析报告的 momentum/trend 策略族）：
  35% Alpha030 (动量) + 25% Alpha175 (趋势) + 20% Alpha076 (低波动) + 20% Alpha034 (流动性代理)
"""

from __future__ import annotations

import pandas as pd
import numpy as np


def v8_scoring_fn(df: pd.DataFrame) -> pd.DataFrame:
    """V8 评分函数：基于 Alpha191 核心因子的动量趋势主导。

    使用 Alpha191 因子的子集作为评分基础:
      - 动量: alpha030 (动量强度), alpha046 (短期趋势)
      - 趋势: alpha095 (价格位置), alpha175 (均线差)
      - 低波动: alpha076 (波动稳定性), alpha070 (低波动)
      - 流动性: alpha100 (成交活跃度)
    """
    df = df.copy()

    # 基础过滤
    df = df[df["close"] >= 3.0].copy()
    if df.empty:
        return df

    # 可用 Alpha191 因子列表
    alpha_cols = [c for c in df.columns if c.startswith("alpha")]

    # 如果没有 Alpha191 因子（回退到基础因子）
    if not alpha_cols:
        df["score"] = (
            df.get("trend60", df.get("ret20", 0)).rank(pct=True) * 0.40
            + df.get("ret20", 0).rank(pct=True) * 0.30
            + (1 - df.get("vol20", pd.Series(0)).rank(pct=True)) * 0.20
            + df.get("liquidity", pd.Series(0)).rank(pct=True) * 0.10
        )
        return df.sort_values("score", ascending=False)

    # Alpha191 核心因子评分
    # 动量组: alpha030(动量强度), alpha046(短期趋势强度), alpha144(累积收益)
    momentum_factors = [c for c in alpha_cols if c in ["alpha030", "alpha046", "alpha144", "alpha149"]]
    # 趋势组: alpha095(价格位置), alpha175(均线差), alpha176(趋势Z-score)
    trend_factors = [c for c in alpha_cols if c in ["alpha095", "alpha175", "alpha176", "alpha184"]]
    # 低波动组: alpha076(波动稳定性), alpha070(低波动), alpha173(低波动)
    lowvol_factors = [c for c in alpha_cols if c in ["alpha070", "alpha076", "alpha097", "alpha100", "alpha173"]]
    # 流动性代理: alpha034(量比), alpha100(成交额)
    liquidity_factors = [c for c in alpha_cols if c in ["alpha034", "alpha190"]]

    # 计算各因子组得分（百分位排名）
    weights = {"momentum": 0.35, "trend": 0.25, "lowvol": 0.20, "liquidity": 0.20}

    momentum_score = _composite_rank(df, momentum_factors, ascending=True) if momentum_factors else 0
    trend_score = _composite_rank(df, trend_factors, ascending=True) if trend_factors else 0
    lowvol_score = _composite_rank(df, lowvol_factors, ascending=False) if lowvol_factors else 0
    liquidity_score = _composite_rank(df, liquidity_factors, ascending=True) if liquidity_factors else 0

    df["score"] = (
        momentum_score * weights["momentum"]
        + trend_score * weights["trend"]
        + lowvol_score * weights["lowvol"]
        + liquidity_score * weights["liquidity"]
    )

    return df.sort_values("score", ascending=False)


def _composite_rank(df: pd.DataFrame, factors: list[str], ascending: bool = True) -> pd.Series:
    """多因子复合百分位排名。"""
    ranks = []
    for f in factors:
        if f in df.columns and df[f].notna().sum() > 10:
            r = df[f].rank(pct=True, method="average")
            if not ascending:
                r = 1 - r
            ranks.append(r)
    if not ranks:
        return pd.Series(0, index=df.index)
    return pd.concat(ranks, axis=1).mean(axis=1).fillna(0)


STRATEGY_META = {
    "name": "V8 动量趋势主导 (Alpha191)",
    "version": "alpha191_v8",
    "factor_set": "GTJA Alpha191 (191个量价因子子集)",
    "scoring": "35% Alpha191动量 + 25% Alpha191趋势 + 20% Alpha191低波动 + 20% 流动性",
    "top_n": 5,
    "retention": 12,
    "status": "可研究，禁止上线",
    "known_issues": ["Alpha191 因子计算慢（191个因子）", "部分因子可能在未来数据"],
}
