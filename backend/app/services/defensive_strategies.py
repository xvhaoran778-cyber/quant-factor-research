"""defensive_strategies.py — 防守型多因子策略

基于旧策略分析报告的稳健型策略设计:
  - reversal_5d: 短期反转75% + 低波动25%
  - low_volatility: 纯低波动
  - weekly_defensive: 反转40% + 低波30% + 趋势20% + 流动性10%
  - weekly_multi_factor: 反转25% + 趋势25% + 低波25% + 量能15% + 流动性10%
"""

import pandas as pd
import numpy as np


def reversal_5d_scoring(df):
    """短期反转增强: -5日收益75% + 低波动25%"""
    df = df.copy()
    df = df[df["close"] >= 3.0].copy()
    if df.empty: return df
    df["score"] = (
        (1 - df["ret5"].rank(pct=True)) * 0.75
        + (1 - df["vol20"].rank(pct=True)) * 0.25
    )
    return df.sort_values("score", ascending=False)


def low_vol_scoring(df):
    """低波动稳健: 纯低波动"""
    df = df.copy()
    df = df[df["close"] >= 3.0].copy()
    if df.empty: return df
    df["score"] = (1 - df["vol20"].rank(pct=True))
    return df.sort_values("score", ascending=False)


def weekly_defensive_scoring(df):
    """周频低波动反转: 反转40% + 低波30% + 趋势20% + 流动性10%
    附加大盘过滤 (close < MA20 < MA60 时空仓)
    """
    df = df.copy()
    df = df[df["close"] >= 3.0].copy()
    if df.empty: return df
    df["score"] = (
        (1 - df["ret5"].rank(pct=True)) * 0.40
        + (1 - df["vol20"].rank(pct=True)) * 0.30
        + df["trend60"].rank(pct=True) * 0.20
        + df["liquidity"].rank(pct=True) * 0.10
    )
    return df.sort_values("score", ascending=False)


def weekly_multi_factor_scoring(df):
    """周频防守多因子: 反转25% + 趋势25% + 低波25% + 量能15% + 流动性10%"""
    df = df.copy()
    df = df[df["close"] >= 3.0].copy()
    if df.empty: return df
    df["score"] = (
        (1 - df["ret5"].rank(pct=True)) * 0.25
        + df["trend60"].rank(pct=True) * 0.25
        + (1 - df["vol20"].rank(pct=True)) * 0.25
        + df.get("volume_ratio", df["ret20"]).rank(pct=True) * 0.15
        + df["liquidity"].rank(pct=True) * 0.10
    )
    return df.sort_values("score", ascending=False)


def small_cap_reversal_scoring(df):
    """小盘超跌反转: 小盘35% + 超跌30% + 趋势20% + 低波15%
    使用流动性反向作为小盘代理 (低流动性 = 小盘)
    """
    df = df.copy()
    df = df[df["close"] >= 3.0].copy()
    if df.empty: return df
    df["score"] = (
        (1 - df["liquidity"].rank(pct=True)) * 0.35
        + (1 - df["ret5"].rank(pct=True)) * 0.30
        + df["trend60"].rank(pct=True) * 0.20
        + (1 - df["vol20"].rank(pct=True)) * 0.15
    )
    return df.sort_values("score", ascending=False)


def mean_reversion_cross_scoring(df):
    """均值回归截面版: ret20反向 + trend60反向 + 低波动
    限 ret20 < -3% 的股票
    """
    df = df.copy()
    df = df[df["close"] >= 3.0].copy()
    df = df[df["ret20"] < -0.03].copy()
    if df.empty: return df
    df["score"] = (
        (1 - df["ret20"].rank(pct=True)) * 0.45
        + (1 - df["trend60"].rank(pct=True)) * 0.25
        + (1 - df["vol20"].rank(pct=True)) * 0.20
        + (1 - df["liquidity"].rank(pct=True)) * 0.10
    )
    return df.sort_values("score", ascending=False)


# ── 测试用简单回测器 ──

INITIAL_CASH = 1_000_000
TOP_N = 5
BUY_COMM = 0.0003
SELL_COMM = 0.0003
SELL_TAX = 0.0005
LOT_SIZE = 100
SLIPPAGE = 0.001


def quick_test(scoring_fn, stock_data, all_dates, risk_on_cache, name="策略"):
    """快速回测给定评分函数。"""
    import numpy as np
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
        else:
            is_signal = True

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
            pending_rebalance = False
            pending_targets = []

        if is_signal:
            risk_off = not risk_on_cache.get(date_str, True)
            features = []
            for sym, sdf in stock_data.items():
                m = sdf[sdf['date'] == dt]
                if len(m) == 0: continue
                idx = m.index[0]
                if idx < 60: continue
                r = sdf.iloc[idx]
                features.append({
                    'symbol': sym, 'close': float(r['close']),
                    'ret5': float(r['close'])/float(sdf.iloc[idx-5]['close'])-1 if idx>=5 else 0,
                    'ret20': float(r['close'])/float(sdf.iloc[idx-20]['close'])-1 if idx>=20 else 0,
                    'trend60': float(r['close'])/float(sdf['close'].iloc[:idx+1].rolling(60).mean().iloc[-1])-1,
                    'vol20': float(sdf['close'].iloc[:idx+1].pct_change().iloc[-20:].std()),
                    'liquidity': float(sdf['amount'].iloc[:idx+1].tail(20).mean()) if idx>=20 else 0,
                })
            if features:
                fdf = pd.DataFrame(features)
                fdf = scoring_fn(fdf)
                if not risk_off and not fdf.empty:
                    cur = set(positions.keys())
                    t = fdf.head(TOP_N)['symbol'].tolist()
                    t12 = set(fdf.head(12)['symbol']) if len(fdf)>=12 else set(fdf['symbol'])
                    t = [s for s in cur if s in t12] + [s for s in t if s not in cur]
                    pending_targets = t[:TOP_N]
                    pending_rebalance = True

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
    days = len(eq)
    years = days/252
    cagr = ((eq[-1]/eq[0])**(1/years)-1)*100 if years>0 else 0
    daily_ret = pd.Series(eq).pct_change().dropna()
    sharpe = np.sqrt(252)*daily_ret.mean()/daily_ret.std() if daily_ret.std()>0 else 0
    dd = (eq - np.maximum.accumulate(eq))/np.maximum.accumulate(eq)
    max_dd = dd.min()*100
    calmar = cagr/abs(max_dd) if max_dd != 0 else 0
    return total_ret, cagr, sharpe, max_dd, calmar


if __name__ == '__main__':
    import sys, os, glob, time, warnings
    warnings.filterwarnings('ignore')
    
    DAILY_DIR = '/Volumes/xhrrrrr_macmini副盘/quantlab/market/daily'
    BENCH_FILE = '/Volumes/xhrrrrr_macmini副盘/quantlab/market/benchmarks/000001.SH.parquet'
    
    bench = pd.read_parquet(BENCH_FILE)
    bench['date'] = pd.to_datetime(bench['date'])
    bench = bench.sort_values('date').reset_index(drop=True)
    all_dates = [str(d.date()) for d in bench['date'] if d >= pd.Timestamp('2010-01-01')]
    
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
    
    print('Loading data...', flush=True)
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
        if i < 60:
            risk_on_cache[ds] = True
            continue
        bc = bench['close'].iloc[:i+1].astype(float)
        ma20 = bc.rolling(20).mean().iloc[-1]
        ma60 = bc.rolling(60).mean().iloc[-1]
        risk_on_cache[ds] = True
        if pd.notna(ma20) and pd.notna(ma60):
            risk_on_cache[ds] = not (bc.iloc[-1] < ma20 < ma60)
    
    strategies = [
        ('短期反转(75/25)', reversal_5d_scoring),
        ('低波动稳健(100)', low_vol_scoring),
        ('周频防守反转(40/30/20/10)', weekly_defensive_scoring),
        ('周频多因子(25/25/25/15/10)', weekly_multi_factor_scoring),
        ('小盘超跌(35/30/20/15)', small_cap_reversal_scoring),
        ('均值回归截面版(45/25/20/10)', mean_reversion_cross_scoring),
    ]
    
    print(f'Stocks: {len(stock_data)}, Days: {len(all_dates)}', flush=True)
    results = []
    for name, fn in strategies:
        t0 = time.time()
        ret, cagr, sharpe, mdd, calmar = quick_test(fn, stock_data, all_dates, risk_on_cache, name)
        score = sharpe*0.4 + max(cagr/20,0)*0.3 + calmar*0.2 + (1-abs(mdd)/100)*0.1
        results.append((name, ret, cagr, sharpe, mdd, calmar, score))
        print(f'  {name:25s} | 收益:{ret:>7.2f}% | CAGR:{cagr:>6.2f}% | 夏普:{sharpe:>7.4f} | 回撤:{mdd:>6.2f}% | 评分:{score:.4f} ({time.time()-t0:.0f}s)', flush=True)
    
    results.sort(key=lambda x: x[6], reverse=True)
    print(f'\n{"="*60}')
    print(f'最优策略排名')
    print(f'{"="*60}')
    for i, (name, ret, cagr, sharpe, mdd, calmar, score) in enumerate(results):
        print(f'  #{i+1}: {name:25s} → 收益{ret:.2f}% CAGR{cagr:.2f}% 夏普{sharpe:.4f} 回撤{mdd:.2f}% (评分{score:.4f})')
