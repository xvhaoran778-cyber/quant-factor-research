"""v8_canonical.py — 标准 V8 策略 (α132 衰减版)

评分函数:
  50% α132 (逐年衰减 -1.5%/yr from 2018)
  + 30% ret60 (60日动量)
  + 20% -liquidity (小盘代理, 成交额反向)
参数:
  top_n=5, retention_multiple=3, universe_size=1000, 大盘过滤
"""

from __future__ import annotations

import os
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

MARKET_DIR = os.environ.get("MARKET_DATA_DIR", "/Volumes/xhrrrrr_macmini副盘/quantlab/market")
ALPHA_132_CACHE = os.path.join(MARKET_DIR, "alpha191_v4_annual", "factor_cache", "alpha_132.parquet")


def load_alpha_132() -> pd.DataFrame:
    df = pd.read_parquet(ALPHA_132_CACHE)
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "symbol", "factor"]].rename(columns={"factor": "alpha_132"})


def alpha_132_decay(df: pd.DataFrame) -> pd.Series:
    """对 α132 施加逐年衰减: -1.5%/yr from 2018."""
    year = pd.to_datetime(df["date"]).dt.year
    decay = np.maximum(1 - 0.015 * np.maximum(year - 2018, 0), 0.1)
    return df["alpha_132"].fillna(0) * decay


def v8_scorer(panel_with_132: pd.DataFrame) -> pd.Series:
    """V8 评分: α132(衰减) + ret60 + size_small"""
    rank = lambda s, asc=True: s.rank(pct=True, ascending=asc, method="average")
    a132_score = rank(alpha_132_decay(panel_with_132))
    mom_score = rank(panel_with_132.get("ret60", 0))
    size_score = rank(panel_with_132.get("liquidity", 0), asc=False)
    return a132_score * 0.50 + mom_score * 0.30 + size_score * 0.20


def v8_candidate_filter(group: pd.DataFrame) -> pd.DataFrame:
    """V8 候选过滤: 价格≥3, 上市≥120天, 成交额>0"""
    f = group.copy()
    f = f[(f["close"] >= 3) & (f["listed_days"] >= 120) & (f["liquidity"] > 0)]
    return f


def merge_alpha_132(panel: pd.DataFrame) -> pd.DataFrame:
    """把 α132 因子合并入 weekly panel."""
    a132 = load_alpha_132()
    panel["date"] = pd.to_datetime(panel["date"])
    merged = panel.merge(a132, on=["date", "symbol"], how="left")
    return merged


V8_CONFIG: dict[str, Any] = {
    "name": "V8 标准版 (α132 衰减)",
    "summary": "α132 动量因子(逐年衰减) + ret60 中期动量 + 小盘代理(成交额反向)；+7350% 历史收益",
    "formula": "50% α132(decay) + 30% ret60 + 20% -liquidity",
    "top_n": 5,
    "retention_multiple": 3,
    "universe_size": 1000,
    "status": "研究基准",
}
