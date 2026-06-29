"""
Alpha191 × 截面回测 — 完整集成

流程:
  1. 加载股票日线数据 → 批量计算 Alpha191 因子
  2. 周频信号日：提取每只股票的最新 Alpha191 因子值 → 评分选股
  3. 下周一开盘执行调仓
  4. 输出完整指标 (CAGR/逐年/夏普/索提诺/IC)
"""

import sys, time, os, glob, warnings
import pandas as pd
import numpy as np
from datetime import datetime

warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
sys.path.insert(0, '/Users/xuhaoran/Documents/agent/backend')
sys.path.insert(0, '/Users/xuhaoran/quant-agent-system')

from app.services.alpha191_factors import Alpha191
from app.services.metrics import compute_full_metrics

DAILY_DIR = '/Volumes/xhrrrrr_macmini副盘/quantlab/market/daily'
BENCH_FILE = '/Volumes/xhrrrrr_macmini副盘/quantlab/market/benchmarks/000001.SH.parquet'
REPORT_DIR = '/Users/xuhaoran/Documents/agent/reports'

INITIAL_CASH = 1_000_000
BUY_COMM = 0.0003
SELL_COMM = 0.0003
SELL_TAX = 0.0005
LOT_SIZE = 100
SLIPPAGE = 0.001
TOP_N = 5


def get_date_index(df, date_str):
    m = df[df['date'] == pd.Timestamp(date_str)]
    return m.index[0] if len(m) > 0 else None


def v8_alpha191_scoring(factor_df):
    """基于预计算 Alpha191 因子的 V8 评分。"""
    df = factor_df.copy()
    if df.empty:
        return df

    alpha_cols = [c for c in df.columns if c.startswith('alpha')]
    if not alpha_cols:
        return df

    # 动量组
    momentum = [c for c in alpha_cols if c in ['alpha030','alpha046','alpha144','alpha149','alpha095']]
    trend = [c for c in alpha_cols if c in ['alpha095','alpha175','alpha176','alpha184','alpha046']]
    lowvol = [c for c in alpha_cols if c in ['alpha070','alpha076','alpha097','alpha100','alpha173']]
    liquid = [c for c in alpha_cols if c in ['alpha034','alpha190','alpha100']]

    def composite(factors, asc=True):
        ranks = [df[f].rank(pct=True) if asc else (1 - df[f].rank(pct=True))
                 for f in factors if f in df.columns and df[f].notna().sum() > 10]
        return pd.concat(ranks, axis=1).mean(axis=1).fillna(0) if ranks else pd.Series(0, index=df.index)

    ms = composite(momentum) if momentum else 0
    ts = composite(trend) if trend else 0
    vs = composite(lowvol, asc=False) if lowvol else 0
    ls = composite(liquid) if liquid else 0

    df['score'] = ms * 0.35 + ts * 0.25 + vs * 0.20 + ls * 0.20
    return df.sort_values('score', ascending=False)


def run_backtest(stock_count=100, start='2010-01-01', end='2026-06-19'):
    print(f'[{datetime.now().strftime("%H:%M:%S")}] Alpha191 × 截面回测')
    print(f'   股票: {stock_count} | 区间: {start} ~ {end}')
    t_start = time.time()

    # 1. 加载基准
    bench = pd.read_parquet(BENCH_FILE)
    bench['date'] = pd.to_datetime(bench['date'])
    bench = bench.sort_values('date').reset_index(drop=True)

    # 2. 选股
    print(f'   选股...', end=' ', flush=True)
    all_files = sorted(glob.glob(f'{DAILY_DIR}/*.parquet'))
    all_files = [f for f in all_files if not os.path.basename(f).startswith('688')]
    metas = []
    for f in all_files[:500]:
        try:
            df = pd.read_parquet(f, columns=['date','amount'])
            metas.append((os.path.basename(f).replace('.parquet',''), float(df['amount'].mean())))
        except: pass
    metas.sort(key=lambda x: x[1], reverse=True)
    symbols = [m[0] for m in metas[:stock_count]]
    print(f'{len(symbols)} 只')

    # 3. 批量计算 Alpha191 因子
    print(f'   计算 Alpha191 因子...', end=' ', flush=True)
    alpha = Alpha191()
    stock_data = {}
    for i, sym in enumerate(symbols):
        try:
            df = pd.read_parquet(f'{DAILY_DIR}/{sym}.parquet')
            df.columns = [c.lower() for c in df.columns]
            df['date'] = pd.to_datetime(df['date'])
            df = df[df['date'] >= '2010-01-01'].copy()
            if len(df) < 300: continue
            df = alpha.calculate_all(df)
            stock_data[sym] = df
        except: pass
    print(f'{len(stock_data)} 只有效')

    # 4. 构建交易日历
    all_dates = sorted(set(
        d for s in stock_data for d in stock_data[s]['date'].dt.strftime('%Y-%m-%d').values
    ))
    eval_start, eval_end = pd.Timestamp(start), pd.Timestamp(end)

    # 5. 回测主循环
    print(f'   回测中...', flush=True)
    cash = float(INITIAL_CASH)
    positions = {}
    equity_curve = []
    trade_log = []
    pending_rebalance = False
    pending_targets = []
    risk_off = False
    bt_start = time.time()

    for i, date_str in enumerate(all_dates):
        dt = pd.Timestamp(date_str)
        if dt < eval_start or dt > eval_end:
            continue

        # 信号日检测
        is_signal = False
        if i < len(all_dates) - 1:
            nxt = pd.Timestamp(all_dates[i+1])
            if dt.isocalendar()[1] != nxt.isocalendar()[1] or (nxt - dt).days > 3:
                is_signal = True
        else:
            is_signal = True

        # 大盘择时
        risk_on = True
        bm_idx = bench[bench['date'] == dt].index
        if len(bm_idx) > 0:
            bi = bm_idx[0]
            if bi >= 60:
                bc = bench['close'].iloc[:bi+1].astype(float)
                ma20 = bc.rolling(20).mean().iloc[-1]
                ma60 = bc.rolling(60).mean().iloc[-1]
                if pd.notna(ma20) and pd.notna(ma60):
                    risk_on = not (bc.iloc[-1] < ma20 < ma60)

        # 执行调仓
        if pending_rebalance:
            to_sell = list(positions.keys()) if risk_off else [s for s in positions if s not in pending_targets]
            for sym in to_sell:
                pos = positions.pop(sym)
                sp = None
                if sym in stock_data:
                    r = stock_data[sym][stock_data[sym]['date'] == dt]
                    if len(r) > 0:
                        sp = float(r.iloc[0]['open']) * (1 - SLIPPAGE)
                sp = sp or (pos.get('last_close', pos['entry_price']) * 0.95)
                proceeds = pos['shares'] * sp * (1 - SELL_COMM - SELL_TAX)
                cash += proceeds
                trade_log.append({
                    'date': date_str, 'symbol': sym, 'action': 'sell',
                    'price': round(sp, 3), 'shares': pos['shares'],
                    'pnl': round(proceeds - pos['cost'], 2),
                    'entry_date': pos['entry_date'],
                })

            if not risk_off:
                to_buy = [s for s in pending_targets if s not in positions]
                if to_buy:
                    per_cash = cash / len(to_buy)
                    for sym in to_buy:
                        bp = None
                        if sym in stock_data:
                            r = stock_data[sym][stock_data[sym]['date'] == dt]
                            if len(r) > 0:
                                bp = float(r.iloc[0]['open']) * (1 + SLIPPAGE)
                        if bp is None or bp <= 0: continue
                        size = (int(per_cash / (bp * (1 + BUY_COMM)) // LOT_SIZE)) * LOT_SIZE
                        if size <= 0: continue
                        cost = size * bp * (1 + BUY_COMM)
                        if cost > cash: continue
                        cash -= cost
                        positions[sym] = {'shares': size, 'cost': cost, 'entry_price': bp, 'entry_date': date_str}
            pending_rebalance = False
            pending_targets = []

        # 信号日评分
        if is_signal:
            risk_off = not risk_on
            features = []
            for sym, sdf in stock_data.items():
                idx = get_date_index(sdf, date_str)
                if idx is None or idx < 60: continue
                row = sdf.iloc[idx]
                feat = {c: row[c] for c in sdf.columns if c.startswith('alpha') or c == 'close'}
                feat['symbol'] = sym
                features.append(feat)

            if features:
                fdf = pd.DataFrame(features)
                fdf = v8_alpha191_scoring(fdf)
                if not risk_off and not fdf.empty:
                    cur = set(positions.keys())
                    targets = fdf.head(TOP_N)['symbol'].tolist()
                    top12 = set(fdf.head(12)['symbol']) if len(fdf) >= 12 else set(fdf['symbol'])
                    targets = [s for s in cur if s in top12] + [s for s in targets if s not in cur]
                    pending_targets = targets[:TOP_N]
                else:
                    pending_targets = []
                pending_rebalance = True

        # 净值
        pv = 0
        for sym, pos in list(positions.items()):
            if sym in stock_data:
                r = stock_data[sym][stock_data[sym]['date'] == dt]
                if len(r) > 0:
                    px = float(r.iloc[0]['close'])
                    pv += pos['shares'] * px
                    pos['last_close'] = px
        equity_curve.append({'date': date_str, 'value': round(cash + pv, 2)})

        if (i+1) % 1000 == 0:
            print(f'     {(i+1)}/{len(all_dates)} days ({time.time()-bt_start:.0f}s)', flush=True)

    # 6. 指标计算
    print(f'   计算指标...', flush=True)
    bench_curve = [{'date': str(bench['date'].iloc[i].date()), 'value': float(bench['adj_close'].iloc[i])}
                   for i in range(len(bench))]
    metrics = compute_full_metrics(equity_curve, trade_log=trade_log, benchmark_curve=bench_curve)

    ret = metrics.get('total_return', 0)
    cagr = metrics.get('cagr', 0)
    sharpe = metrics.get('sharpe', 0)
    sortino = metrics.get('sortino', 0)
    calmar = metrics.get('calmar', 0)
    mdd = metrics.get('max_drawdown', 0)
    trades = metrics.get('trade_stats', {}).get('total_trades', 0)
    wr = metrics.get('trade_stats', {}).get('win_rate', 0)

    print(f'\n   === 结果 ===')
    print(f'   总收益: {ret:.2f}% | CAGR: {cagr:.2f}%')
    print(f'   夏普: {sharpe:.4f} | 索提诺: {sortino:.4f} | 卡尔玛: {calmar:.4f}')
    print(f'   最大回撤: {mdd:.2f}% | 交易: {trades} | 胜率: {wr:.1f}%')
    print(f'   耗时: {time.time()-t_start:.0f}s')
    print(f'\n   逐年:')
    for y in sorted(metrics.get('yearly_returns', {}).keys()):
        print(f'    {y}: {metrics["yearly_returns"][y]:.2f}% | 夏普: {metrics["yearly_sharpe"].get(y,0):.4f}')

    # 7. 报告
    now_str = datetime.now().strftime('%Y%m%d_%H%M')
    with open(f'{REPORT_DIR}/V8_Alpha191_xs_{now_str}.md', 'w', encoding='utf-8') as f:
        f.write(f'# V8 Alpha191 截面回测报告\n\n')
        f.write(f'**生成**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'**区间**: {metrics.get("start_date","")} ~ {metrics.get("end_date","")}\n')
        f.write(f'**股票**: {len(stock_data)} 只 | **因子**: Alpha191 ({alpha.get_factor_count()}个)\n\n')
        f.write('| 指标 | 数值 |\n|------|------|\n')
        for k, v in [('总收益', f'{ret:.2f}%'), ('CAGR', f'{cagr:.2f}%'),
            ('夏普', f'{sharpe:.4f}'), ('索提诺', f'{sortino:.4f}'),
            ('卡尔玛', f'{calmar:.4f}'), ('最大回撤', f'{mdd:.2f}%'),
            ('交易次数', str(trades)), ('胜率', f'{wr:.1f}%')]:
            f.write(f'| {k} | {v} |\n')
        f.write('\n## 逐年\n\n| 年份 | 收益 | 夏普 | 回撤 |\n|------|------|------|------|\n')
        for y in sorted(metrics.get('yearly_returns', {}).keys()):
            f.write(f'| {y} | {metrics["yearly_returns"][y]:.2f}% | {metrics["yearly_sharpe"].get(y,0):.4f} | {metrics["yearly_max_drawdown"].get(y,0):.2f}% |\n')
        f.write('\n---\n**风险提示**: 本报告基于历史回测，不构成投资建议。\n')

    print(f'\n   报告: {REPORT_DIR}/V8_Alpha191_xs_{now_str}.md')


if __name__ == '__main__':
    run_backtest(stock_count=100)
