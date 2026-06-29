"""alpha191_v9.py — V9 防守改造策略

核心逻辑 (V8 的防守改造版):
  评分 = 25% trend60 + 15% ret20 + 35% low_vol + 25% liquidity
  额外过滤: 排除波动率前40% + 排除负趋势
  频率：周频评分 → 次周一开盘执行

特征:
  - 牺牲收益换取更低回撤
  - 防守属性强，但完整回测风险调整收益不如 V8
  - 震荡市中相对 V8 有优势
"""

import pandas as pd


def v9_scoring_fn(df: pd.DataFrame) -> pd.DataFrame:
    """V9 评分函数：防守改造。

    Args:
        df: 包含 ret20, trend60, vol20, liquidity 列的 DataFrame

    Returns:
        添加了 score 列并按得分降序排列的 DataFrame
    """
    df = df.copy()

    # 严格过滤
    df = df[df["close"] >= 3.0].copy()
    df = df[df["trend60"] > 0].copy()  # 排除负趋势

    if df.empty:
        return df

    # 排除波动率最高的 40%
    vol_limit = df["vol20"].quantile(0.60)
    df = df[df["vol20"] < vol_limit].copy()

    if df.empty:
        return df

    # 评分 (25/15/35/25)
    df["score"] = (
        df["trend60"].rank(pct=True, method="average") * 0.25
        + df["ret20"].rank(pct=True, method="average") * 0.15
        + (1 - df["vol20"].rank(pct=True, method="average")) * 0.35
        + df["liquidity"].rank(pct=True, method="average") * 0.25
    )

    return df.sort_values("score", ascending=False)


STRATEGY_META = {
    "name": "V9 防守改造",
    "version": "alpha191_v9_defensive",
    "frequency": "周频",
    "execution": "信号日(周五)下周一开盘",
    "scoring": "25% trend60 + 15% ret20 + 35% low_vol + 25% liquidity",
    "top_n": 5,
    "retention": 12,
    "filters": "price >= 3, trend60 > 0, vol20 < p60",
    "risk_control": "低波动过滤 + 负趋势排除",
    "known_issues": [
        "风险调整收益不如 V8（夏普更低）",
        "过滤条件过于严格，候选池经常不足",
        "牛市弹性不足，容易踏空",
    ],
    "status": "可研究，禁止上线",
}
