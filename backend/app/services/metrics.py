"""metrics.py — 全量绩效指标计算

包含 CAGR、逐年收益、夏普、索提诺、卡尔玛、最大回撤、逐年 IC 等。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Any


def compute_full_metrics(
    equity_curve: list[dict] | pd.DataFrame,
    trade_log: list[dict] | None = None,
    benchmark_curve: list[dict] | pd.DataFrame | None = None,
    factor_ic_series: pd.Series | None = None,
) -> dict[str, Any]:
    if isinstance(equity_curve, list):
        eq_df = pd.DataFrame(equity_curve)
    else:
        eq_df = equity_curve.copy()
    if eq_df.empty or "value" not in eq_df.columns:
        return {"error": "empty equity curve"}

    eq_df["date"] = pd.to_datetime(eq_df["date"])
    eq_df = eq_df.sort_values("date").reset_index(drop=True)

    initial = float(eq_df["value"].iloc[0])
    final = float(eq_df["value"].iloc[-1])
    total_days = len(eq_df)
    first_date = eq_df["date"].iloc[0]
    last_date = eq_df["date"].iloc[-1]
    years = (last_date - first_date).days / 365.25

    # 1. 总收益 / CAGR
    total_return = (final / initial - 1) * 100
    cagr = ((final / initial) ** (1 / years) - 1) * 100 if years > 0 else 0.0

    # 日收益率
    eq_df["daily_return"] = eq_df["value"].pct_change()
    daily_ret = eq_df["daily_return"].dropna()
    ann_vol = float(daily_ret.std() * np.sqrt(252) * 100) if len(daily_ret) > 0 else 0.0

    # 夏普
    rf = 0.02 / 252
    excess = daily_ret - rf
    sharpe = float(np.sqrt(252) * excess.mean() / daily_ret.std()) if daily_ret.std() > 0 else 0.0

    # 索提诺
    downside = daily_ret[daily_ret < 0]
    downside_std = downside.std() if len(downside) > 0 else 1e-8
    sortino = float(np.sqrt(252) * (daily_ret.mean() - rf) / downside_std)

    # 最大回撤
    eq_df["cummax"] = eq_df["value"].cummax()
    eq_df["drawdown"] = (eq_df["value"] - eq_df["cummax"]) / eq_df["cummax"]
    max_drawdown = float(eq_df["drawdown"].min() * 100)
    dd_idx = eq_df["drawdown"].idxmin()
    max_dd_date = str(eq_df.loc[dd_idx, "date"].date()) if pd.notna(dd_idx) else ""
    dd_start = int(dd_idx) if pd.notna(dd_idx) else 0
    eq_after = eq_df.loc[dd_start:]
    recovery = eq_after[eq_after["drawdown"] >= -0.01]
    recovery_days = len(recovery) if len(recovery) > 0 else 0

    # 卡尔玛
    calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0.0

    # 逐年收益
    eq_df["year"] = eq_df["date"].dt.year
    yearly_returns = {}
    yearly_sharpe = {}
    yearly_dd = {}
    for year, group in eq_df.groupby("year"):
        if len(group) > 1:
            yr_ret = (group["value"].iloc[-1] / group["value"].iloc[0] - 1) * 100
            yearly_returns[int(year)] = round(float(yr_ret), 2)
        yr_daily = group["daily_return"].dropna()
        if len(yr_daily) > 20:
            yr_sharpe = np.sqrt(252) * yr_daily.mean() / yr_daily.std() if yr_daily.std() > 0 else 0
            yearly_sharpe[int(year)] = round(float(yr_sharpe), 4)
        yr_dd = group["drawdown"].min() * 100
        yearly_dd[int(year)] = round(float(yr_dd), 2)

    # 交易统计
    trade_stats = {}
    if trade_log is not None:
        td = pd.DataFrame(trade_log) if isinstance(trade_log, list) else trade_log.copy()
        total_trades = len(td)
        if "pnl" in td.columns:
            wins = td[td["pnl"] > 0]
            losses = td[td["pnl"] < 0]
            wc = len(wins)
            lc = len(losses)
            wr = wc / (wc + lc) * 100 if (wc + lc) > 0 else 0
            aw = float(wins["pnl"].mean()) if wc > 0 else 0
            al = float(losses["pnl"].mean()) if lc > 0 else 0
            pf = abs(float(wins["pnl"].sum() / losses["pnl"].sum())) if lc > 0 and losses["pnl"].sum() != 0 else float("inf")
            trade_stats = {
                "total_trades": total_trades, "win_count": wc, "lose_count": lc,
                "win_rate": round(wr, 2), "avg_win": round(aw, 2), "avg_loss": round(al, 2),
                "profit_factor": round(pf, 4) if pf != float("inf") else "inf",
            }

    # 逐年 IC
    yearly_ic = {}
    if factor_ic_series is not None and len(factor_ic_series) > 0:
        ic_df = factor_ic_series.reset_index()
        ic_df.columns = ["date", "ic"]
        ic_df["date"] = pd.to_datetime(ic_df["date"])
        ic_df["year"] = ic_df["date"].dt.year
        for year, group in ic_df.groupby("year"):
            yearly_ic[int(year)] = {
                "mean_ic": round(float(group["ic"].mean()), 4),
                "std_ic": round(float(group["ic"].std()), 4),
                "ir": round(float(group["ic"].mean() / group["ic"].std()), 4) if group["ic"].std() > 0 else 0,
                "win_rate": round(float((group["ic"] > 0).mean() * 100), 2),
            }

    # 基准对比
    benchmark_metrics = {}
    if benchmark_curve is not None:
        bm = pd.DataFrame(benchmark_curve) if isinstance(benchmark_curve, list) else benchmark_curve.copy()
        if "value" in bm.columns:
            bm["date"] = pd.to_datetime(bm["date"])
            m = eq_df[["date", "value"]].merge(bm[["date", "value"]], on="date", suffixes=("_s", "_b"))
            if len(m) > 0:
                m["sr"] = m["value_s"].pct_change()
                m["br"] = m["value_b"].pct_change()
                m["ex"] = m["sr"] - m["br"]
                ea = m["ex"].mean() * 252 * 100
                ir = m["ex"].mean() / m["ex"].std() * np.sqrt(252) if m["ex"].std() > 0 else 0
                cv = m[["sr", "br"]].cov()
                bv = cv.iloc[0, 1] / cv.iloc[1, 1] if cv.iloc[1, 1] > 0 else 1
                al = (m["sr"].mean() - bv * m["br"].mean()) * 252 * 100
                benchmark_metrics = {
                    "excess_return": round(float(ea), 2),
                    "information_ratio": round(float(ir), 4),
                    "beta": round(float(bv), 4),
                    "alpha": round(float(al), 4),
                }

    return {
        "start_date": str(first_date.date()),
        "end_date": str(last_date.date()),
        "total_days": total_days,
        "years": round(years, 2),
        "total_return": round(total_return, 2),
        "cagr": round(cagr, 2),
        "annual_volatility": round(ann_vol, 2),
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "calmar": round(calmar, 4),
        "max_drawdown": round(max_drawdown, 2),
        "max_drawdown_date": max_dd_date,
        "recovery_days": recovery_days,
        "final_value": round(final, 2),
        "yearly_returns": yearly_returns,
        "yearly_sharpe": yearly_sharpe,
        "yearly_max_drawdown": yearly_dd,
        "yearly_ic": yearly_ic,
        "trade_stats": trade_stats,
        "benchmark": benchmark_metrics,
    }
