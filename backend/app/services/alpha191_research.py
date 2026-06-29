"""alpha191_research.py — Alpha191 因子研究与 IC 分析

基于 Alpha158 的 100 个量价因子，提供：
  1. IC / RankIC 计算（逐日/逐月）
  2. IC 衰减分析
  3. 因子分组收益测试
  4. 因子相关性分析
"""

from __future__ import annotations

import os
import sys
import warnings
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# 添加 quant-agent-system 路径
QAS_DIR = "/Users/xuhaoran/quant-agent-system"
if QAS_DIR not in sys.path:
    sys.path.insert(0, QAS_DIR)

from factors.alpha158 import Alpha158


# ═══════════════════════════════════════════════════════════════
# 1. IC 计算
# ═══════════════════════════════════════════════════════════════

def compute_ic(
    factor_series: pd.Series,
    forward_return: pd.Series,
    method: str = "spearman",
) -> float:
    """计算单期 IC。

    Args:
        factor_series: 截面因子值
        forward_return: 未来 N 日收益
        method: 'pearson' 或 'spearman' (RankIC)

    Returns:
        IC 值
    """
    valid = factor_series.notna() & forward_return.notna()
    if valid.sum() < 10:
        return 0.0
    if method == "spearman":
        return factor_series[valid].corr(forward_return[valid], method="spearman")
    return factor_series[valid].corr(forward_return[valid], method="pearson")


def compute_factor_ic_series(
    panel: pd.DataFrame,
    factor_col: str,
    forward_col: str = "ret_1d",
    groupby: str = "date",
    method: str = "spearman",
) -> pd.Series:
    """逐日计算因子 IC 序列。

    Args:
        panel: 面板数据，必须包含 date, factor_col, forward_col
        factor_col: 因子列名
        forward_col: 未来收益列名
        groupby: 分组列（通常是 date）
        method: 'spearman' (RankIC) 或 'pearson' (IC)

    Returns:
        IC 时间序列
    """
    def _ic(g):
        return compute_ic(g[factor_col], g[forward_col], method)

    return panel.groupby(groupby).apply(_ic)


def ic_summary(ic_series: pd.Series) -> dict[str, Any]:
    """IC 序列统计摘要。"""
    if len(ic_series) == 0:
        return {"mean_ic": 0, "std_ic": 0, "ir": 0, "win_rate": 0, "positive_pct": 0}
    return {
        "mean_ic": float(ic_series.mean()),
        "std_ic": float(ic_series.std()),
        "ir": float(ic_series.mean() / ic_series.std()) if ic_series.std() > 0 else 0,
        "win_rate": int((ic_series > 0).sum()),
        "positive_pct": float((ic_series > 0).mean() * 100),
        "t_stat": float(ic_series.mean() / ic_series.std() * np.sqrt(len(ic_series))),
    }


# ═══════════════════════════════════════════════════════════════
# 2. 因子分组收益测试
# ═══════════════════════════════════════════════════════════════

def factor_group_returns(
    panel: pd.DataFrame,
    factor_col: str,
    forward_col: str = "ret_1d",
    n_groups: int = 5,
    groupby: str = "date",
) -> pd.DataFrame:
    """截面因子分组收益。

    Args:
        panel: 面板数据
        factor_col: 因子列
        forward_col: 未来收益列
        n_groups: 分组数
        groupby: 时间列

    Returns:
        每期的分组收益
    """
    def _group_ret(g):
        g = g.dropna(subset=[factor_col, forward_col])
        if len(g) < n_groups:
            return pd.Series({i: np.nan for i in range(n_groups)})
        g["group"] = pd.qcut(g[factor_col], n_groups, labels=list(range(n_groups)), duplicates="drop")
        return g.groupby("group")[forward_col].mean()

    return panel.groupby(groupby).apply(_group_ret).unstack()


# ═══════════════════════════════════════════════════════════════
# 3. 全因子扫描（基于 Alpha158）
# ═══════════════════════════════════════════════════════════════

def scan_all_alpha158_factors(
    panel: pd.DataFrame,
    forward_col: str = "ret_1d",
) -> pd.DataFrame:
    """扫描 Alpha158 所有因子，计算平均 IC/RankIC。

    Args:
        panel: 必须包含 Alpha158 因子列和 forward_col
        forward_col: 未来收益列

    Returns:
        因子 IC 排名表
    """
    factor_cols = [c for c in panel.columns if c.startswith("alpha_") and c != forward_col]
    results = []
    for fc in factor_cols:
        ic_series = compute_factor_ic_series(panel, fc, forward_col)
        summary = ic_summary(ic_series)
        summary["factor"] = fc
        results.append(summary)

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values("ir", ascending=False).reset_index(drop=True)
    return result_df


# ═══════════════════════════════════════════════════════════════
# 4. 因子预处理：计算面板数据的未来收益
# ═══════════════════════════════════════════════════════════════

def prepare_panel(
    stock_data: dict[str, pd.DataFrame],
    start_date: str = "2024-01-01",
    end_date: str = "2026-06-19",
) -> pd.DataFrame:
    """将多只股票的 DataFrame 合并为面板数据，并计算 Alpha158 因子和未来收益。"""
    alpha = Alpha158()
    all_dfs = []

    for sym, df in stock_data.items():
        df = df.copy()
        df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
        if len(df) < 100:
            continue

        # 计算未来 1 日收益
        df["ret_1d"] = df["close"].pct_change(-1).shift(-1)
        df["ret_5d"] = df["close"].pct_change(-5).shift(-5)

        # 计算 Alpha158 因子
        try:
            df = alpha.calculate_all(df)
        except Exception:
            continue

        df["symbol"] = sym
        all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    panel = pd.concat(all_dfs, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    return panel
