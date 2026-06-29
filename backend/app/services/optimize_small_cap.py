"""optimize_small_cap.py — 小盘超跌反转权重优化 + 止损测试"""

import sys, time, os, glob, warnings
import pandas as pd
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/xuhaoran/Documents/agent/backend')

DAILY_DIR = '/Volumes/xhrrrrr_macmini副盘/quantlab/market/daily'
BENCH_FILE = '/Volumes/xhrrrrr_macmini副盘/quantlab/market/benchmarks/000001.SH.parquet'

INITIAL_CASH = 1_000_000
TOP_N = 5
BUY_COMM = 0.0003
SELL_COMM = 0.0003
SELL_TAX = 0.0005
LOT_SIZE = 100
SLIPPAGE = 0.001


def load_data():
    bench = pd.read_parquet(BENCH_FILE)
    bench['date'] = pd.to_datetime(bench['date'])
    bench = bench.sort_values('date').reset_index(drop=True)
    all_dates = [str(d.date()) for d in bench['date'] if d >= pd.Timestamp('2010-01-01')]

    all_files = sorted(glob.glob(f'{DAILY_DIR}/*.parquet'))
    all_files = [f for f in all_files if not os.path.basename(f).startswith('688')]
    metas = []
    for f in all_files[:500]:
        try:
            df = pd.read_parquet(f, columns=['date','amount'])
            metas.append((os.path.basename(f).replace('.parquet',''), float(df['amount'].mean())))
        except: pass
    metas.sort(key=lambda x: x[1], reverse=True)
    symbols = [m[0] for m in metas[:100]]

    stock_data = {}
    for sym in symbols:
        try:
            df = pd.read_parquet(f'{DAILY_DIR}/{sym}.parquet')
            df.columns = [c.lower() for c in df.columns]
            df['date'] = pd.to_datetime(df['date'])
            df = df[df['date'] >= '2009-06-01'].copy()
            if len(df) < 300: continue
            stock_data[sym] = df
        except: pass

    risk_on_cache = {}
    for i, ds in enumerate(all_dates):
        if i < 60: risk_on_cache[ds] = True; continue
        bc = bench['close'].iloc[:i+1].astype(float)
        ma20 = bc.rolling(20).mean().iloc[-1]
        ma60 = bc.rolling(60).mean().iloc[-1]
        risk_on_cache[ds] = True
        if pd.notna(ma20) and pd.notna(ma60):
            risk_on_cache[ds] = not (bc.iloc[-1] < ma20 < ma60)

    return stock_data, bench, all_dates, risk_on_cache


def run_bt(stock_data, all_dates, risk_on_cache, w_small, w_rev, w_trend, w_lv, stop_loss_pct=0):
    """回测，支持可选止损。"""
    cash = float(INITIAL_CASH)
    positions = {}
    equity_curve = []
    pending_rebalance = False
    pending_targets = []
    risk_off = False

    for i, date_str in enumerate(all_dates):
        dt = pd.Timestamp(date_str)
        is_signal = False
        if i < len(all_dates) - 1:
            nxt = pd.Timestamp(all_dates[i+1])
            if dt.isocalendar()[1] != nxt.isocalendar()[1] or (nxt-dt).days > 3:
                is_signal = True
        else: is_signal = True

        if pending_rebalance:
            to_sell = list(positions.keys()) if risk_off else [s for s in positions if s not in pending_targets]
            for sym in to_sell:
                pos = positions.pop(sym)
                sp = None
                if sym in stock_data:
                    r = stock_data[sym][stock_data[sym]['date'] == dt]
                    if len(r) > 0: sp = float(r.iloc[0]['open']) * (1 - SLIPPAGE)
                sp = sp or (pos.get('last_close', pos['entry_price']) * 0.95)
                cash += pos['shares'] * sp * (1 - SELL_COMM - SELL_TAX)
            if not risk_off:
                to_buy = [s for s in pending_targets if s not in positions]
                if to_buy:
                    pc = cash / len(to_buy)
                    for sym in to_buy:
                        bp = None
                        if sym in stock_data:
                            r = stock_data[sym][stock_data[sym]['date'] == dt]
                            if len(r) > 0: bp = float(r.iloc[0]['open']) * (1 + SLIPPAGE)
                        if bp is None or bp <= 0: continue
                        size = (int(pc / (bp * (1 + BUY_COMM)) // LOT_SIZE)) * LOT_SIZE
                        if size <= 0: continue
                        cost = size * bp * (1 + BUY_COMM)
                        if cost > cash: continue
                        cash -= cost
                        positions[sym] = {'shares':size,'cost':cost,'entry_price':bp,'entry_date':date_str}
            pending_rebalance = False; pending_targets = []

        # 止损检查
        if stop_loss_pct > 0:
            for sym in list(positions.keys()):
                pos = positions[sym]
                px = None
                if sym in stock_data:
                    r = stock_data[sym][stock_data[sym]['date'] == dt]
                    if len(r) > 0: px = float(r.iloc[0]['close'])
                if px and px < pos['entry_price'] * (1 - stop_loss_pct):
                    p = positions.pop(sym)
                    sp = px * (1 - SLIPPAGE)
                    cash += p['shares'] * sp * (1 - SELL_COMM - SELL_TAX)

        # 信号
        if is_signal:
            risk_off = not risk_on_cache.get(date_str, True)
            features = []
            for sym, sdf in stock_data.items():
                m = sdf[sdf['date'] == dt]
                if len(m) == 0: continue
                idx = m.index[0]
                if idx < 60: continue
                r = sdf.iloc[idx]
                c = float(r['close'])
                c5 = float(sdf.iloc[idx-5]['close']) if idx>=5 else c
                c20 = float(sdf.iloc[idx-20]['close']) if idx>=20 else c
                ma60 = float(sdf['close'].iloc[:idx+1].rolling(60).mean().iloc[-1])
                vol20 = float(sdf['close'].iloc[:idx+1].pct_change().iloc[-20:].std())
                liq = float(sdf['amount'].iloc[:idx+1].tail(20).mean()) if idx>=20 else 0
                features.append({
                    'symbol': sym, 'close': c, 'ret5': c/c5-1, 'ret20': c/c20-1,
                    'trend60': c/ma60-1 if ma60>0 else 0, 'vol20': vol20, 'liquidity': liq,
                })
            if features:
                fdf = pd.DataFrame(features)
                fdf['score'] = (
                    (1 - fdf['liquidity'].rank(pct=True)) * w_small
                    + (1 - fdf['ret5'].rank(pct=True)) * w_rev
                    + fdf['trend60'].rank(pct=True) * w_trend
                    + (1 - fdf['vol20'].rank(pct=True)) * w_lv
                )
                fdf = fdf.sort_values('score', ascending=False)
                if not risk_off and not fdf.empty:
                    cur = set(positions.keys())
                    t = fdf.head(TOP_N)['symbol'].tolist()
                    t12 = set(fdf.head(12)['symbol']) if len(fdf)>=12 else set(fdf['symbol'])
                    t = [s for s in cur if s in t12] + [s for s in t if s not in cur]
                    pending_targets = t[:TOP_N]; pending_rebalance = True

        pv = 0
        for sym, pos in list(positions.items()):
            if sym in stock_data:
                r = stock_data[sym][stock_data[sym]['date'] == dt]
                if len(r) > 0:
                    pv += pos['shares'] * float(r.iloc[0]['close'])
                    pos['last_close'] = float(r.iloc[0]['close'])
        equity_curve.append(cash + pv)

    eq = np.array(equity_curve)
    total_ret = (eq[-1]/eq[0]-1)*100
    years = len(eq)/252
    cagr = ((eq[-1]/eq[0])**(1/years)-1)*100 if years>0 else 0
    daily_ret = pd.Series(eq).pct_change().dropna()
    sharpe = np.sqrt(252)*daily_ret.mean()/daily_ret.std() if daily_ret.std()>0 else 0
    dd = (eq - np.maximum.accumulate(eq))/np.maximum.accumulate(eq)
    max_dd = dd.min()*100
    calmar = cagr/abs(max_dd) if max_dd!=0 else 0
    return total_ret, cagr, sharpe, max_dd, calmar


# ── 权重组合 ──
WEIGHTS = [
    ('当前(35/30/20/15)', 0.35, 0.30, 0.20, 0.15),
    ('小盘加重(45/25/20/10)', 0.45, 0.25, 0.20, 0.10),
    ('小盘极重(55/20/15/10)', 0.55, 0.20, 0.15, 0.10),
    ('小盘+反转(40/35/10/15)', 0.40, 0.35, 0.10, 0.15),
    ('防守(20/20/30/30)', 0.20, 0.20, 0.30, 0.30),
    ('反转加重(25/45/15/15)', 0.25, 0.45, 0.15, 0.15),
]

STOP_LOSSES = [0, 0.15, 0.25]


if __name__ == '__main__':
    print('Loading...', flush=True)
    stock_data, bench, all_dates, risk_cache = load_data()
    print(f'Stocks: {len(stock_data)}, Days: {len(all_dates)}', flush=True)

    all_results = []
    for name, ws, wr, wt, wl in WEIGHTS:
        for sl in STOP_LOSSES:
            t0 = time.time()
            ret, cagr, sh, mdd, cal = run_bt(stock_data, all_dates, risk_cache, ws, wr, wt, wl, sl)
            score = sh*0.3 + max(cagr/10, 0)*0.3 + cal*0.2 + (1-abs(mdd)/100)*0.2
            all_results.append((name, ws, wr, wt, wl, sl, ret, cagr, sh, mdd, cal, score))
            sl_label = f'止损{sl*100:.0f}%' if sl > 0 else '无止损'
            print(f'  {name:15s} {sl_label:8s} | 收益:{ret:>7.2f}% CAGR:{cagr:>5.2f}% 夏普:{sh:>6.4f} 回撤:{mdd:>6.2f}% 评分:{score:.4f} ({time.time()-t0:.0f}s)', flush=True)

    # 排名
    all_results.sort(key=lambda x: x[11], reverse=True)
    print(f'\n{"="*70}')
    print(f'Top 10 组合')
    print(f'{"="*70}')
    for i, (name, ws, wr, wt, wl, sl, ret, cagr, sh, mdd, cal, score) in enumerate(all_results[:10]):
        sl_txt = f'止损{sl*100:.0f}%' if sl > 0 else '无止损'
        print(f'  #{i+1}: {name:15s} {sl_txt:8s} → 收益{ret:>7.2f}% CAGR{cagr:>5.2f}% 夏普{sh:>6.4f} 回撤{mdd:>6.2f}% 卡尔玛{cal:.4f} (评分{score:.4f})')

    # 保存报告
    report_path = f'/Users/xuhaoran/Documents/agent/reports/小盘超跌优化_{pd.Timestamp.now().strftime("%Y%m%d_%H%M")}.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f'# 小盘超跌反转 权重×止损优化报告\n\n')
        f.write(f'**生成**: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'**股票**: {len(stock_data)} 只 | **区间**: 2010-2026\n\n')
        f.write('| 排名 | 权重 | 止损 | 总收益 | CAGR | 夏普 | 回撤 | 卡尔玛 |\n')
        f.write('|------|------|------|--------|------|------|------|--------|\n')
        for i, (name, ws, wr, wt, wl, sl, ret, cagr, sh, mdd, cal, score) in enumerate(all_results[:10]):
            sl_txt = f'{sl*100:.0f}%' if sl > 0 else '无'
            f.write(f'| #{i+1} | {name} | {sl_txt} | {ret:.2f}% | {cagr:.2f}% | {sh:.4f} | {mdd:.2f}% | {cal:.4f} |\n')
    print(f'\n报告: {report_path}')
