"""agent_gate.py — 确定性 Agent 过滤规则（用于回测验证 Agent 是否起作用）

4 个 Agent 各自有独立否决规则（确定性，非 mock）:
  1. 风险 Agent:    vol20 > 0.60 → 否决（高波动）
  2. 趋势 Agent:    ret20 < -0.15 → 否决（强下跌）
  3. 情绪 Agent:    ret5 < -0.08 → 否决（恐慌）
  4. 质量 Agent:    liquidity < 流动性后 20% → 否决（流动性差）

注: 宏观 Agent 已在 V8 的 market_filter 中实现（大盘转弱空仓），
    这里不重复添加。

每个候选股票会接受 4 个 Agent 的独立检查。
任一 Agent 否决即被剔除。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


AGENT_RULES: dict[str, dict[str, Any]] = {
    "risk": {
        "name": "风险 Agent",
        "threshold": "vol20 > 0.60",
        "reason": "年化波动率过高",
    },
    "trend": {
        "name": "趋势 Agent",
        "threshold": "ret20 < -0.15",
        "reason": "20日动量强下跌",
    },
    "sentiment": {
        "name": "情绪 Agent",
        "threshold": "ret5 < -0.08",
        "reason": "5日恐慌性下跌",
    },
    "quality": {
        "name": "质量 Agent",
        "threshold": "liquidity < 流动性后 20%",
        "reason": "流动性不足",
    },
}


def _check_risk(group: pd.DataFrame) -> pd.Series:
    return group["vol20"] <= 0.60


def _check_trend(group: pd.DataFrame) -> pd.Series:
    return group["ret20"] >= -0.15


def _check_sentiment(group: pd.DataFrame) -> pd.Series:
    return group["ret5"] >= -0.08


def _check_quality(group: pd.DataFrame) -> pd.Series:
    if len(group) == 0:
        return pd.Series(dtype=bool, index=group.index)
    threshold = group["liquidity"].quantile(0.20)
    return group["liquidity"] >= threshold


CHECKS = {
    "risk": _check_risk,
    "trend": _check_trend,
    "sentiment": _check_sentiment,
    "quality": _check_quality,
}


def agent_gate(group: pd.DataFrame) -> pd.DataFrame:
    """5 个 Agent 联合否决；返回通过所有 Agent 的候选股。

    返回的 DataFrame 增加一列 'agent_decisions'，记录每个 Agent 的判定。
    """
    if group.empty:
        return group

    decisions: dict[str, pd.Series] = {}
    for agent_id, check_fn in CHECKS.items():
        decisions[agent_id] = check_fn(group)

    decisions_df = pd.DataFrame(decisions, index=group.index)
    passed = decisions_df.all(axis=1)

    result = group[passed].copy()
    result.attrs["agent_decisions"] = decisions_df
    result.attrs["rejected_count"] = int((~passed).sum())
    return result


def get_agent_rejection_log(panel: pd.DataFrame) -> pd.DataFrame:
    """对整个 panel 跑一遍 Agent 门控，返回每个候选被哪个 Agent 否决的统计。

    返回: DataFrame with columns [agent_id, rejected_count, sample_symbols]
    """
    log_rows = []
    for week, group in panel.groupby("week", sort=True):
        if group.empty:
            continue
        group = group.dropna(subset=["next_open", "ret5", "ret20", "vol20", "trend60", "liquidity"]).copy()
        group = group[(group["listed_days"] >= 120) & (group["liquidity"] > 0)]
        if group.empty:
            continue

        decisions: dict[str, pd.Series] = {}
        for agent_id, check_fn in CHECKS.items():
            decisions[agent_id] = check_fn(group)
        decisions_df = pd.DataFrame(decisions, index=group.index)

        for agent_id in decisions:
            rejected = ~decisions_df[agent_id]
            n_rejected = int(rejected.sum())
            if n_rejected > 0:
                log_rows.append({
                    "week": str(week),
                    "agent_id": agent_id,
                    "agent_name": AGENT_RULES[agent_id]["name"],
                    "rejected_count": n_rejected,
                    "sample_symbols": ",".join(group.loc[rejected, "symbol"].head(3).tolist()),
                })

    return pd.DataFrame(log_rows)
