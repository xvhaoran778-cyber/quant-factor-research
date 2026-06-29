"""alpha191_optimizer.py — V8 Alpha191 权重优化

尝试多组权重组合，用截面回测对比效果。
缓存因子避免重复计算。
"""

import sys, time, os, glob, warnings
import pandas as pd, numpy as np
from datetime import datetime
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
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
    """加载数据并预计算 Alpha191。"""
    bench = pd.read_parquet(BENCH_FILE)
    bench['date'] = pd.to_datetime(bench['date'])
    bench = bench.sort_values('date').reset_index(drop=True)
    all_dates = [str(d.date()) for d in bench['date'] if d >= pd.Timestamp('2010-01-01')]

    # 选股
    all_files = sorted(glob.glob(f'{DAILY_DIR}/*.parquet'))
    all_files = [f for f in all_files if not os.path.basename(f).startswith('688')]
    metas = []
    for f in all_files[:300]:
        try:
            df = pd.read_parquet(f, columns=['date','amount'])
            metas.append((os.path.basename(f).replace('.parquet',''), float(df['amount'].mean())))
        except: pass
    metas.sort(key=lambda x: x[1], reverse=True)
    symbols = [m[0] for m in metas[:80]]

    # Alpha191 因子
    from app.services.alpha191_factors import Alpha191
    alpha = Alpha191()
    stock_data = {}
    for sym in symbols:
        try:
            df = pd.read_parquet(f'{DAILY_DIR}/{sym}.parquet')
            df.columns = [c.lower() for c in df.columns]
            df['date'] = pd.to_datetime(df['date'])
            df = df[df['date'] >= '2009-06-01'].copy()
            if len(df) < 300: continue
            df = alpha.calculate_all(df)
            stock_data[sym] = df
        except: pass

    # 大盘择时缓存
    risk_on_cache = {}
    for i, ds in enumerate(all_dates):
        if i < 60:
            risk_on_cache[ds] = True
            continue
        bc = bench['close'].iloc[:i+1].astype(float)
        ma20 = bc.rolling(20).mean().iloc[-1]
        ma60 = bc.rolling(60).mean().iloc[-1]
        risk_on_cache[ds] = True
        if pd.notna(ma20) and pd.notna(ma60):
            risk_on_cache[ds] = not (bc.iloc[-1] < ma20 < ma60)

    return stock_data, bench, all_dates, risk_on_cache, alpha.get_factor_count()


def run_backtest(stock_data, bench, all_dates, risk_on_cache, weights):
    """用给定权重跑一次截面回测。"""
    w_m, w_t, w_v, w_l = weights  # momentum, trend, lowvol, liquidity
    cash = float(INITIAL_CASH)
    positions = {}
    equity_curve = []
    pending_rebalance = False
    pending_targets = []
    risk_off = False

    for i, date_str in enumerate(all_dates):
        dt = pd.Timestamp(date_str)
        # 信号日
        is_signal = False
        if i < len(all_dates) - 1:
            nxt = pd.Timestamp(all_dates[i+1])
            if dt.isocalendar()[1] != nxt.isocalendar()[1] or (nxt-dt).days > 3:
                is_signal = True
        else:
            is_signal = True

        # 执行
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
                cash += pos['shares'] * sp * (1 - SELL_COMM - SELL_TAX)

            if not risk_off:
                to_buy = [s for s in pending_targets if s not in positions]
                if to_buy:
                    pc = cash / len(to_buy)
                    for sym in to_buy:
                        bp = None
                        if sym in stock_data:
                            r = stock_data[sym][stock_data[sym]['date'] == dt]
                            if len(r) > 0:
                                bp = float(r.iloc[0]['open']) * (1 + SLIPPAGE)
                        if bp is None or bp <= 0: continue
                        size = (int(pc / (bp * (1 + BUY_COMM)) // LOT_SIZE)) * LOT_SIZE
                        if size <= 0: continue
                        cost = size * bp * (1 + BUY_COMM)
                        if cost > cash: continue
                        cash -= cost
                        positions[sym] = {'shares':size,'cost':cost,'entry_price':bp,'entry_date':date_str}
            pending_rebalance = False
            pending_targets = []

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
                # 用 Alpha191 因子评分
                alpha_cols = [c for c in sdf.columns if c.startswith('alpha')]
                momentum = [c for c in alpha_cols if c in ['alpha030','alpha046','alpha144','alpha149','alpha095']]
                trend = [c for c in alpha_cols if c in ['alpha095','alpha175','alpha176','alpha184','alpha046']]
                lowvol = [c for c in alpha_cols if c in ['alpha070','alpha076','alpha097','alpha173']]
                liquid = [c for c in alpha_cols if c in ['alpha034','alpha190','alpha100']]
                def comp(fs, asc=True):
                    vals = [r[f] for f in fs if f in sdf.columns and not pd.isna(r.get(f))]
                    vals = [v for v in vals if not pd.isna(v)]
                    return np.mean(vals) if vals else 0
                features.append({
                    'symbol': sym,
                    'ms': comp(momentum), 'ts': comp(trend),
                    'vs': -comp(lowvol) if lowvol else 0,  # 低波动→反向
                    'ls': comp(liquid),
                })

            if features:
                fdf = pd.DataFrame(features)
                fdf['score'] = (fdf['ms'].rank(pct=True)*w_m + fdf['ts'].rank(pct=True)*w_t
                              + fdf['vs'].rank(pct=True)*w_v + fdf['ls'].rank(pct=True)*w_l)
                fdf = fdf.sort_values('score', ascending=False)
                cur = set(positions.keys())
                t = fdf.head(TOP_N)['symbol'].tolist()
                t12 = set(fdf.head(12)['symbol']) if len(fdf) >= 12 else set(fdf['symbol'])
                t = [s for s in cur if s in t12] + [s for s in t if s not in cur]
                pending_targets = t[:TOP_N]
                pending_rebalance = True

        # 净值
        pv = sum(pos['shares'] * float(stock_data[sym][stock_data[sym]['date']==dt].iloc[0]['close'])
                 for sym, pos in list(positions.items())
                 if sym in stock_data and len(stock_data[sym][stock_data[sym]['date']==dt]) > 0)
        equity_curve.append(cash + pv)

    # 计算指标
    eq = np.array(equity_curve)
    total_ret = (eq[-1]/eq[0]-1)*100
    days = len(eq)
    years = days/252
    cagr = ((eq[-1]/eq[0])**(1/years)-1)*100 if years > 0 else 0
    daily_ret = pd.Series(eq).pct_change().dropna()
    sharpe = np.sqrt(252)*daily_ret.mean()/daily_ret.std() if daily_ret.std() > 0 else 0
    dd = (eq - np.maximum.accumulate(eq))/np.maximum.accumulate(eq)
    max_dd = dd.min()*100
    calmar = cagr/abs(max_dd) if max_dd != 0 else 0

    return total_ret, cagr, sharpe, max_dd, calmar


# ── 权重组合 ──
WEIGHT_COMBOS = [
    ('当前配置(35/25/20/20)', (0.35, 0.25, 0.20, 0.20)),
    ('动量偏重(45/20/20/15)', (0.45, 0.20, 0.20, 0.15)),
    ('动量极重(55/15/15/15)', (0.55, 0.15, 0.15, 0.15)),
    ('趋势偏重(25/40/20/15)', (0.25, 0.40, 0.20, 0.15)),
    ('低波偏重(20/20/40/20)', (0.20, 0.20, 0.40, 0.20)),
    ('均衡(25/25/25/25)', (0.25, 0.25, 0.25, 0.25)),
    ('V8原始(40/30/20/10)', (0.40, 0.30, 0.20, 0.10)),
    ('防守型(20/15/45/20)', (0.20, 0.15, 0.45, 0.20)),
    ('流动性偏重(20/20/20/40)', (0.20, 0.20, 0.20, 0.40)),
    ('动量+趋势(40/30/15/15)', (0.40, 0.30, 0.15, 0.15)),
]

if __name__ == '__main__':
    print(f'[{datetime.now().strftime("%H:%M:%S")}] Loading data...', flush=True)
    stock_data, bench, all_dates, risk_on_cache, n_factors = load_data()
    print(f'Stocks: {len(stock_data)}, Factors: {n_factors}, Days: {len(all_dates)}', flush=True)

    results = []
    for name, weights in WEIGHT_COMBOS:
        t0 = time.time()
        ret, cagr, sharpe, mdd, calmar = run_backtest(stock_data, bench, all_dates, risk_on_cache, weights)
        elapsed = time.time() - t0
        results.append((name, weights, ret, cagr, sharpe, mdd, calmar))
        score = sharpe * 0.4 + (cagr/10) * 0.3 + calmar * 0.2 + (1 - abs(mdd)/100) * 0.1
        print(f'  {name:20s} | 收益:{ret:>7.2f}% | CAGR:{cagr:>6.2f}% | 夏普:{sharpe:>7.4f} | 回撤:{mdd:>6.2f}% | 卡尔玛:{calmar:>6.4f} | 评分:{score:.4f} | {elapsed:.0f}s', flush=True)

    # 排序
    results.sort(key=lambda x: x[4]*0.4 + (x[3]/10)*0.3 + x[6]*0.2 + (1-abs(x[5])/100)*0.1, reverse=True)
    print(f'\n{"="*60}')
    print(f'最优权重排名 (综合评分=夏普×0.4 + CAGR/10×0.3 + 卡尔玛×0.2 + 回撤分×0.1)')
    print(f'{"="*60}')
    for i, (name, w, ret, cagr, sharpe, mdd, calmar) in enumerate(results):
        score = sharpe * 0.4 + (cagr/10) * 0.3 + calmar * 0.2 + (1 - abs(mdd)/100) * 0.1
        print(f'  #{i+1}: {name:20s} M{w[0]:.0f}/T{w[1]:.0f}/V{w[2]:.0f}/L{w[3]:.0f} → 收益{ret:.2f}% CAGR{cagr:.2f}% 夏普{sharpe:.4f} 回撤{mdd:.2f}% (评分{score:.4f})')