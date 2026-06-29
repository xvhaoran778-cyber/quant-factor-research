"""tasks.py — QuantLab 统一回测入口

整合原始 quant-agent-system 的策略 + 新建的 V8/V9/MR/Liquidity 截面回测。
运行方式:
  python -m app.tasks                         # 全量回测
  python -m app.tasks --ic                     # IC 分析
  python -m app.tasks --start 2010-01-01       # 自定义起始日期
"""

from __future__ import annotations

import os
import sys
import argparse
from datetime import datetime

# 项目根目录
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in [PROJECT_DIR, BACKEND_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from app.services.research_backtest import (
    MarketDataLoader,
    CrossSectionalBacktest,
    generate_report,
)
from app.services.alpha191_v8 import v8_scoring_fn
from app.services.alpha191_v9 import v9_scoring_fn
from app.services.alpha191_research import prepare_panel, scan_all_alpha158_factors

REPORT_DIR = os.path.join(PROJECT_DIR, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)


# ── 策略评分函数 ──────────────────────────────────────────────

def mr_scoring_fn(df):
    """MR 均值回归：反转 + 低波动 + 低流动性。"""
    df = df.copy()
    df = df[df["close"] >= 3.0].copy()
    df = df[df["ret20"] < -0.03].copy()
    if df.empty:
        return df
    df["score"] = (
        (1 - df["ret20"].rank(pct=True)) * 0.45
        + (1 - df["trend60"].rank(pct=True)) * 0.25
        + (1 - df["vol20"].rank(pct=True)) * 0.20
        + (1 - df["liquidity"].rank(pct=True)) * 0.10
    )
    return df.sort_values("score", ascending=False)


def liquidity_scoring_fn(df):
    """Liquidity 原始评分：45% trend60 + 25% ret20 + 20% low_vol + 10% liquidity。"""
    df = df.copy()
    df = df[df["close"] >= 3.0].copy()
    df = df[df["ret20"] > 0.02].copy()
    df = df[df["trend60"] > 0].copy()
    if df.empty:
        return df
    vol_limit = df["vol20"].quantile(0.80)
    df = df[df["vol20"] < vol_limit].copy()
    if df.empty:
        return df
    df["score"] = (
        df["trend60"].rank(pct=True) * 0.45
        + df["ret20"].rank(pct=True) * 0.25
        + (1 - df["vol20"].rank(pct=True)) * 0.20
        + df["liquidity"].rank(pct=True) * 0.10
    )
    return df.sort_values("score", ascending=False)


# ── 原始 quant-agent-system 策略 ─────────────────────────────

from strategies.momentum_strategy import MomentumStrategy
from strategies.mean_reversion_strategy import MeanReversionStrategy
from strategies.multi_factor_strategy import MultiFactorStrategy


def run_original_sequential_backtest(loader: MarketDataLoader, start: str, end: str):
    """用原始 quant-agent-system 的逐股时序回测引擎跑所有策略（抽样 50 只）。"""
    import pandas as pd
    import numpy as np
    print("\n[原始引擎] 逐股时序回测（抽样 50 只）...")
    from backtest.engine import BacktestEngine
    from backtest.enhanced_metrics import EnhancedMetrics

    stock_data = loader.stock_data
    bench = loader.benchmark

    orig_strategies = {
        "动量策略(20日)": MomentumStrategy({"lookback_period": 20, "holding_period": 5, "top_n": 3}),
        "均值回归(20日)": MeanReversionStrategy({"ma_period": 20, "std_multiplier": 2.0}),
        "多因子策略": MultiFactorStrategy({
            "momentum_weight": 0.3, "value_weight": 0.3,
            "quality_weight": 0.2, "volatility_weight": 0.2
        }),
    }

    eval_start = pd.Timestamp(start)
    eval_end = pd.Timestamp(end)

    results = []
    for sname, strategy in orig_strategies.items():
        print(f"\n  运行 {sname}...")
        all_returns = []
        total_trades = 0
        valid = 0

        for sym, sdf in list(stock_data.items())[:50]:
            sdf = sdf[(sdf["date"] >= eval_start) & (sdf["date"] <= eval_end)].copy()
            if len(sdf) < 100:
                continue
            try:
                result = strategy.backtest(sdf)
                if result["total_trades"] > 0:
                    all_returns.append(result["total_return"])
                    total_trades += result["total_trades"]
                    valid += 1
            except:
                continue

        if all_returns:
            avg_ret = np.mean(all_returns)
            med_ret = np.median(all_returns)
            win_pct = sum(1 for r in all_returns if r > 0) / len(all_returns) * 100
            results.append({
                "name": f"[原始] {sname}",
                "total_return": round(avg_ret, 2),
                "annual_return": round(0, 2),
                "max_drawdown": round(0, 2),
                "sharpe": 0,
                "benchmark_return": round(0, 2),
                "excess_return": round(0, 2),
                "total_trades": total_trades,
                "note": f"逐股平均收益 {avg_ret:.2f}%, 中位 {med_ret:.2f}%, 胜率 {win_pct:.1f}%, 有效 {valid}只",
            })
            print(f"    平均收益: {avg_ret:.2f}% | 中位: {med_ret:.2f}% | 胜率: {win_pct:.1f}% | 有效: {valid}只")

    return results


# ── 策略注册表 ────────────────────────────────────────────────

STRATEGIES = {
    "v8":        ("V8 动量趋势主导 (40/30/20/10)", v8_scoring_fn, 5),
    "v9":        ("V9 防守改造 (25/15/35/25)", v9_scoring_fn, 5),
    "mr":        ("MR 均值回归超跌反弹", mr_scoring_fn, 5),
    "liquidity": ("Liquidity 强弱轮动 (45/25/20/10)", liquidity_scoring_fn, 4),
}


def run_cross_sectional_backtests(loader: MarketDataLoader, start: str, end: str):
    """运行截面回测。"""
    results = []
    for key, (name, score_fn, top_n) in STRATEGIES.items():
        print(f"\n  {name} (Top {top_n})...")
        bt = CrossSectionalBacktest(
            data_loader=loader,
            scoring_fn=score_fn,
            top_n=top_n,
            name=name,
        )
        result = bt.run(start_date=start, end_date=end)
        results.append(result)
        print(f"    收益: {result['total_return']:.2f}% | 年化: {result['annual_return']:.2f}% | "
              f"回撤: {result['max_drawdown']:.2f}% | 夏普: {result['sharpe']:.4f} | "
              f"交易: {result['total_trades']}")
    return results


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

def run_all(start: str = "2010-01-01", end: str = "2026-06-19"):
    """统一回测入口。"""
    import pandas as pd
    import numpy as np

    print("=" * 70)
    print(f"QuantLab 统一回测 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"区间: {start} ~ {end}")
    print("引擎: 截面回测(open执行+大盘择时) + 原始逐股时序回测")
    print("=" * 70)

    # 1. 加载数据
    print("\n[1/4] 加载数据...")
    loader = MarketDataLoader()
    loader.load_benchmark()
    loader.load_stocks(top_n=300)
    print(f"  股票: {len(loader.stock_data)} 只")
    print(f"  交易日: {len(loader.all_dates)} 天")

    # 2. 截面回测（新引擎）
    print("\n[2/4] 截面回测...")
    results = run_cross_sectional_backtests(loader, start, end)

    # 3. 原始策略回测（逐股时序）
    print("\n[3/4] 原始策略回测...")
    orig_results = run_original_sequential_backtest(loader, start, end)

    # 4. 合并报告
    print("\n[4/4] 生成报告...")
    all_results = results + orig_results

    report_path = os.path.join(
        REPORT_DIR, f"全局回测报告_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
    )

    # 基准收益
    bench = loader.benchmark
    bm_ret = 0
    if bench is not None:
        bm_slice = bench[(bench["date"] >= pd.Timestamp(start)) & (bench["date"] <= pd.Timestamp(end))]
        if len(bm_slice) > 0:
            bm_ret = (bm_slice["adj_close"].iloc[-1] / bm_slice["adj_close"].iloc[0] - 1) * 100

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# QuantLab 全局回测报告\n\n")
        f.write(f"**生成时间**: {now}\n")
        f.write(f"**区间**: {start} ~ {end}\n")
        f.write(f"**数据**: 本地 parquet 日线 (Top 300 按流动性)\n")
        f.write(f"**基准**: 上证指数 | **基准收益**: {bm_ret:.2f}%\n")
        f.write(f"**截面引擎**: 开盘执行 + 周频调仓 + 大盘择时\n")
        f.write(f"**原始引擎**: 逐股时序, close执行\n\n")
        f.write("---\n\n")
        f.write("## 截面回测结果\n\n")
        f.write("| 策略 | 总收益 | 年化 | 回撤 | 夏普 | 基准 | 超额 | 交易 |\n")
        f.write("|------|--------|------|------|------|------|------|------|\n")
        for r in results:
            f.write(f"| {r['name']} | {r['total_return']:.2f}% | {r['annual_return']:.2f}% | "
                    f"{r['max_drawdown']:.2f}% | {r['sharpe']:.4f} | {r['benchmark_return']:.2f}% | "
                    f"{r['excess_return']:.2f}% | {r['total_trades']} |\n")

        f.write("\n## 原始策略回测（逐股时序）\n\n")
        f.write("| 策略 | 平均收益 | 中位收益 | 胜率 | 有效股票 |\n")
        f.write("|------|---------|---------|------|---------|\n")
        for r in orig_results:
            f.write(f"| {r['name']} | {r['total_return']:.2f}% | {r['note']} |\n")

        for r in results:
            f.write(f"\n---\n## {r['name']}\n\n")
            f.write("| 指标 | 数值 |\n|------|------|\n")
            for k, v in [
                ("总收益", f"{r['total_return']:.2f}%"),
                ("年化收益", f"{r['annual_return']:.2f}%"),
                ("最大回撤", f"{r['max_drawdown']:.2f}%"),
                ("夏普比率", f"{r['sharpe']:.4f}"),
                ("超额收益", f"{r['excess_return']:.2f}%"),
                ("总交易次数", str(r['total_trades'])),
            ]:
                f.write(f"| {k} | {v} |\n")

        f.write("\n---\n## 安全提示\n")
        f.write("- 本报告基于本地回测数据，**禁止直接上线实盘**\n")
        f.write("- 所有策略定位为研究/模拟，不构成投资建议\n")

    print(f"\n✅ 全局报告已生成: {report_path}")
    return all_results


def run_ic_analysis():
    """因子 IC 分析。"""
    import pandas as pd
    import numpy as np
    import warnings as wrn
    wrn.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

    print("=" * 60)
    print("Alpha158 因子 IC 分析")
    print("=" * 60)

    print("\n[1/3] 加载数据...")
    loader = MarketDataLoader()
    loader.load_stocks(top_n=100)

    print(f"\n[2/3] 准备面板...")
    from factors.alpha158 import Alpha158
    alpha = Alpha158()
    all_rows = []
    for sym, sdf in loader.stock_data.items():
        sdf = sdf[(sdf["date"] >= "2020-01-01")].copy()
        if len(sdf) < 200:
            continue
        sdf.columns = [c.lower() for c in sdf.columns]
        sdf["ret_1d"] = sdf["close"].pct_change(-1).shift(-1)
        try:
            sdf = alpha.calculate_all(sdf)
        except:
            continue
        sdf["symbol"] = sym
        all_rows.append(sdf)
    panel = pd.concat(all_rows, ignore_index=True).dropna(subset=["ret_1d"])
    print(f"  面板: {panel.shape[0]} 行 x {panel.shape[1]} 列")

    print(f"\n[3/3] 扫描因子 IC...")
    ic_df = scan_all_alpha158_factors(panel)
    print(f"\n  Top 10 因子:")
    print(f"  {'因子':28s} {'平均IC':>8s} {'IR':>8s} {'胜率':>7s}")
    print(f"  {'-'*55}")
    for _, row in ic_df.head(10).iterrows():
        print(f"  {row['factor']:28s} {row['mean_ic']:>8.4f} {row['ir']:>8.4f} {row['positive_pct']:>6.1f}%")

    report_path = os.path.join(REPORT_DIR, f"IC分析_{datetime.now().strftime('%Y%m%d_%H%M')}.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Alpha158 因子 IC 分析\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**面板**: {panel.shape[0]} 行, {len(ic_df)} 因子\n\n")
        f.write("## Top 10 因子\n\n| 因子 | 平均IC | IC标准差 | IR | 胜率 | t统计 |\n|------|--------|---------|----|------|-------|\n")
        for _, row in ic_df.head(10).iterrows():
            f.write(f"| {row['factor']} | {row['mean_ic']:.4f} | {row['std_ic']:.4f} | {row['ir']:.4f} | {row['positive_pct']:.1f}% | {row['t_stat']:.2f} |\n")
        f.write("\n---\n_数据定位为研究用途_\n")
    print(f"\n✅ IC 分析报告: {report_path}")
    return ic_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QuantLab 统一回测")
    parser.add_argument("--ic", action="store_true", help="因子 IC 分析")
    parser.add_argument("--start", default="2010-01-01", help="开始日期")
    parser.add_argument("--end", default="2026-06-19", help="结束日期")
    args = parser.parse_args()

    if args.ic:
        run_ic_analysis()
    else:
        run_all(start=args.start, end=args.end)
