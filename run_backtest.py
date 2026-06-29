"""QuantLab 综合回测脚本 v2
使用市场数据 + 现有策略跑回测，输出报告数据
"""
import sys, os, json, glob
from datetime import datetime

import pandas as pd
import numpy as np

# 添加系统路径
sys.path.insert(0, '/Users/xuhaoran/quant-agent-system')

MARKET_DIR = '/Volumes/xhrrrrr_macmini副盘/quantlab/market'
DAILY_DIR = f'{MARKET_DIR}/daily'
BENCHMARK_FILE = f'{MARKET_DIR}/benchmarks/000001.SH.parquet'
REPORT_DIR = '/Users/xuhaoran/Documents/agent/reports'
os.makedirs(REPORT_DIR, exist_ok=True)

print("=" * 60)
print("QuantLab 综合回测")
print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

# 1. 基准指数
print("\n[1/5] 加载基准指数...")
bench_df = pd.read_parquet(BENCHMARK_FILE)
bench_df['date'] = pd.to_datetime(bench_df['date'])
bench_df = bench_df.sort_values('date')
bench_return = (bench_df['adj_close'].iloc[-1] / bench_df['adj_close'].iloc[0] - 1) * 100
print(f"  沪深300: {bench_df['date'].min().date()} -> {bench_df['date'].max().date()}, "
      f"{len(bench_df)} 天, 累计收益 {bench_return:.2f}%")

# 2. 股票数据
print("\n[2/5] 加载股票数据（按成交额筛选 top 100）...")
all_files = sorted(glob.glob(f'{DAILY_DIR}/*.parquet'))
stocks = []
for f in all_files:
    df = pd.read_parquet(f)
    if len(df) > 2000:
        stocks.append((f, df['amount'].mean(), len(df)))
stocks.sort(key=lambda x: x[1], reverse=True)
stocks = stocks[:100]
print(f"  共加载 {len(stocks)} 只股票")

# 3. 因子计算
print("\n[3/5] 计算 Alpha158 因子...")
from factors.alpha158 import Alpha158

all_factors = {}
alpha_calc = Alpha158()
import warnings
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

for fpath, avg_vol, ndays in stocks:
    symbol = os.path.basename(fpath).replace('.parquet', '')
    df = pd.read_parquet(fpath)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df = alpha_calc.calculate_all(df)
    all_factors[symbol] = df

print(f"  因子数: {len(alpha_calc.get_factor_names())}")
print(f"  总样本: {sum(len(v) for v in all_factors.values())} 行")

# 4. 策略回测
print("\n[4/5] 运行策略回测...")

from strategies.momentum_strategy import MomentumStrategy
from strategies.mean_reversion_strategy import MeanReversionStrategy
from strategies.multi_factor_strategy import MultiFactorStrategy

STRATEGIES = {
    'momentum_20': MomentumStrategy({'lookback_period': 20, 'holding_period': 5, 'top_n': 3}),
    'mean_reversion': MeanReversionStrategy({'ma_period': 20, 'std_multiplier': 2.0}),
    'multi_factor': MultiFactorStrategy({
        'momentum_weight': 0.3, 'value_weight': 0.3,
        'quality_weight': 0.2, 'volatility_weight': 0.2
    }),
}

results = {}
for sname, strategy in STRATEGIES.items():
    print(f"\n  --- {strategy.name} ---")
    stock_rets = {}
    total_signals = {'buy': 0, 'sell': 0, 'hold': 0}

    for symbol, df in list(all_factors.items()):
        try:
            result = strategy.backtest(df)
            if result['total_trades'] > 1:
                ret = result['total_return']
                stock_rets[symbol] = ret
                for t in result['trades']:
                    if t['type'] == 'buy':
                        total_signals['buy'] += 1
                    else:
                        total_signals['sell'] += 1
        except Exception as e:
            pass

    if stock_rets:
        returns = list(stock_rets.values())
        results[sname] = {
            'name': strategy.name,
            'description': strategy.get_description(),
            'valid_stocks': len(stock_rets),
            'total_stocks': len(all_factors),
            'avg_return': float(np.mean(returns)),
            'median_return': float(np.median(returns)),
            'positive_pct': sum(1 for r in returns if r > 0) / len(returns) * 100,
            'best_return': float(max(returns)),
            'worst_return': float(min(returns)),
            'std_return': float(np.std(returns)) if len(returns) > 1 else 0,
            'total_trades': total_signals['buy'] + total_signals['sell'],
            'stock_returns': {k: round(v, 2) for k, v in
                              sorted(stock_rets.items(), key=lambda x: x[1], reverse=True)},
        }
        print(f"    有效: {len(stock_rets)}/{len(all_factors)}, "
              f"平均收益: {results[sname]['avg_return']:.2f}%, "
              f"胜率: {results[sname]['positive_pct']:.1f}%")
    else:
        results[sname] = {
            'name': strategy.name,
            'description': strategy.get_description(),
            'valid_stocks': 0, 'total_stocks': len(all_factors),
            'avg_return': 0, 'median_return': 0, 'positive_pct': 0,
            'best_return': 0, 'worst_return': 0, 'std_return': 0,
            'total_trades': 0, 'stock_returns': {},
        }
        print(f"    无有效回测结果")

# 5. 生成报告
print("\n[5/5] 生成报告...")

report_path = f'{REPORT_DIR}/回测报告_{datetime.now().strftime("%Y%m%d_%H%M")}.md'

with open(report_path, 'w', encoding='utf-8') as f:
    f.write(f"""# QuantLab 策略回测报告

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**数据范围**: {bench_df['date'].min().date()} ~ {bench_df['date'].max().date()}
**基准指数**: 沪深300 (000001.SH) — 累计收益 **{bench_return:.2f}%**
**股票样本**: top 100（按日均成交额）
**因子类型**: Alpha158（{len(alpha_calc.get_factor_names())} 个量价因子）
**回测引擎**: quant-agent-system 内置策略 backtest()

---

## 一、策略绩效总览

| 策略 | 有效/总数 | 平均收益 | 中位收益 | 胜率 | 最佳 | 最差 | 收益标准差 | 总交易 |
|------|----------|---------|---------|------|------|------|-----------|-------|""")
    for r in results.values():
        f.write(f"\n| {r['name']} | {r['valid_stocks']}/{r['total_stocks']} | "
                f"{r['avg_return']:.2f}% | {r['median_return']:.2f}% | "
                f"{r['positive_pct']:.1f}% | {r['best_return']:.2f}% | "
                f"{r['worst_return']:.2f}% | {r['std_return']:.2f}% | "
                f"{r['total_trades']} |")

    f.write(f"""

### 与基准对比

| 指标 | 沪深300 | 
|------|---------|
| 累计收益 | {bench_return:.2f}% |
| 年化收益（按{len(bench_df)}个交易日） | {((1+bench_return/100)**(252/len(bench_df))-1)*100:.2f}% |

---

## 二、各策略详细分析

""")

    for sname, r in results.items():
        f.write(f"""### {r['name']}

#### 策略描述
{r['description']}

#### 回测统计

| 指标 | 数值 |
|------|------|
| 参与股票数 | {r['valid_stocks']}/{r['total_stocks']} |
| 平均单股收益 | {r['avg_return']:.2f}% |
| 中位单股收益 | {r['median_return']:.2f}% |
| 收益标准差 | {r['std_return']:.2f}% |
| 正收益占比 | {r['positive_pct']:.1f}% |
| 最佳单股收益 | {r['best_return']:.2f}% |
| 最差单股收益 | {r['worst_return']:.2f}% |
| 总交易次数 | {r['total_trades']} |

""")

    # 收益分布
    f.write("## 三、收益分布亮点\n\n")
    for sname, r in results.items():
        stock_rets = r['stock_returns']
        items = list(stock_rets.items())
        top5 = items[:5]
        bot5 = items[-5:] if len(items) >= 5 else items

        f.write(f"""### {r['name']}

**最佳 5 只:**
| 股票 | 收益 |
|------|------|""")
        for sym, ret in top5:
            f.write(f"\n| {sym} | {ret:.2f}% |")

        f.write(f"""

**最差 5 只:**
| 股票 | 收益 |
|------|------|""")
        for sym, ret in bot5:
            f.write(f"\n| {sym} | {ret:.2f}% |")
        f.write("\n\n")

    # 对比分析
    f.write("## 四、策略对比分析\n\n")
    if results:
        by_avg = sorted(results.values(), key=lambda x: x['avg_return'], reverse=True)
        by_win = sorted(results.values(), key=lambda x: x['positive_pct'], reverse=True)

        f.write(f"""### 按平均收益排序
1. **{by_avg[0]['name']}** — {by_avg[0]['avg_return']:.2f}%
""")
        if len(by_avg) > 1:
            f.write(f"""2. **{by_avg[1]['name']}** — {by_avg[1]['avg_return']:.2f}%
""")
        if len(by_avg) > 2:
            f.write(f"""3. **{by_avg[2]['name']}** — {by_avg[2]['avg_return']:.2f}%
""")

        f.write(f"""\n### 按胜率排序
1. **{by_win[0]['name']}** — {by_win[0]['positive_pct']:.1f}%
""")
        if len(by_win) > 1:
            f.write(f"2. **{by_win[1]['name']}** — {by_win[1]['positive_pct']:.1f}%\n")
        if len(by_win) > 2:
            f.write(f"3. **{by_win[2]['name']}** — {by_win[2]['positive_pct']:.1f}%\n")

    # 问题与改进建议
    f.write(f"""
## 五、现有问题与改进建议

### 发现的问题

1. **策略仅适用于单股票时序** — 现有 MomentumStrategy / MeanReversionStrategy / MultiFactorStrategy 的 backtest() 方法只对单只股票的日线做选时。**这是最大的结构问题**：实际量化研究需要的是截面选股——在每天从全市场选 N 只股票，而不是每只股票独立跑时序。
   
2. **信号太稀疏** — 动量策略的买入条件是"动量转正且排名前 20%"，在单只股票上信号很少；均值回归需要价格触及 2 倍布林带，在长期趋势股上几乎不会触发。

3. **缺乏组合管理** — 单只股票独立回测没法做组合级别的风控（行业、个股、换手率约束）。

4. **Alpha158 计算性能差** — 每只股票独立调用 calculate_all 导致 PerformanceWarning，应该在面板数据上批量计算。

5. **缺少 IC/RankIC 分析** — 没有对 100 个 Alpha158 因子做截面有效性检验。

### 改进建议

| 问题 | 改进方向 | 优先级 |
|------|---------|--------|
| 单股票时序→截面选股 | 重构为面板数据回测：每天计算所有股票的因子得分，选 Top N 买入 | **高** |
| 因子有效性检查 | 添加 IC / RankIC 计算，淘汰无效因子 | **高** |
| 组合风控 | 加行业约束、个股仓位上限、换手率限制 | 中 |
| 交易成本 | 现在的0.03%佣金+0.1%滑点偏保守，加冲击成本模型 | 中 |
| 多周期测试 | 在 2023-2024、2025、2026 分段回测 | 低 |

### 安全提示（参照 HERMES_AGENT.md）

- 本次所有策略均为本地研究，**禁止上线实盘**
- V8/V9 相关策略需按规范单独验证
- 本报告结论仅用于研究参考，不构成投资建议

---
""")

print(f"\n✅ 报告已生成: {report_path}")
print(f"   报告大小: {os.path.getsize(report_path)} bytes")
