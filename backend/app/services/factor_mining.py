"""factor_mining.py — 基于 IC 研究的因子挖掘 × 夏普优化

从 IC 分析结果出发，设计 12 个新策略并回测对比，选出最优后做夏普优化。
每个策略使用前向5日IC最强的因子组合。
"""

import sys, os, glob, time, warnings
import pandas as pd
import numpy as np
from datetime import datetime

warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/xuhaoran/Documents/agent/backend')

DAILY_DIR = '/Volumes/xhrrrrr_macmini副盘/quantlab/market/daily'
BENCH_FILE = '/Volumes/xhrrrrr_macmini副盘/quantlab/market/benchmarks/000001.SH.parquet'
REPORT_DIR = '/Users/xuhaoran/Documents/agent/reports'

INITIAL_CASH = 1_000_000
TOP_N = 5
BUY_COMM = 0.0003; SELL_COMM = 0.0003; SELL_TAX = 0.0005
LOT_SIZE = 100; SLIPPAGE = 0.001


def load_data(n_stocks=100):
    bench = pd.read_parquet(BENCH_FILE)
    bench['date'] = pd.to_datetime(bench['date']); bench = bench.sort_values('date').reset_index(drop=True)
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
    symbols = [m[0] for m in metas[:n_stocks]]
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
        ma20 = bc.rolling(20).mean().iloc[-1]; ma60 = bc.rolling(60).mean().iloc[-1]
        risk_on_cache[ds] = True
        if pd.notna(ma20) and pd.notna(ma60): risk_on_cache[ds] = not (bc.iloc[-1] < ma20 < ma60)
    return stock_data, bench, all_dates, risk_on_cache


def compute_features(sdf, idx):
    """计算一批因子特征 (基于IC研究选出的最强因子)。"""
    if idx < 60: return None
    r = sdf.iloc[idx]
    c = float(r['close']); o = float(r['open']); h = float(r['high']); l = float(r['low']); v = float(r['volume'])
    cs = sdf['close'].iloc[:idx+1].astype(float)
    vs = sdf['volume'].iloc[:idx+1].astype(float)
    amt = sdf['amount'].iloc[:idx+1].astype(float)
    
    # 基础
    ret5 = c/float(cs.iloc[-6])-1 if idx>=5 else 0
    ret20 = c/float(cs.iloc[-21])-1 if idx>=20 else 0
    ma5 = cs.rolling(5).mean().iloc[-1]; ma20 = cs.rolling(20).mean().iloc[-1]; ma60 = cs.rolling(60).mean().iloc[-1]
    trend60 = c/ma60-1 if ma60>0 else 0
    vol20 = cs.pct_change().iloc[-20:].std()
    liq = float(amt.tail(20).mean()) if idx>=20 else 0
    
    # ★ 基于 IC 研究的最强因子 (前向5日IR排序)
    # alpha172 = Rank(Corr(C,V,10)) + Rank(Corr(C,V,5))  [IR=0.98]
    corr_cv5 = cs.rolling(5).corr(vs)
    corr_cv10 = cs.rolling(10).corr(vs)
    f_alpha172 = corr_cv5.iloc[-1] + corr_cv10.iloc[-1]
    
    # alpha149 = Corr(C,V,10)*5 + Corr(C,V,5)  [IR=0.82]
    f_alpha149 = corr_cv10.iloc[-1]*5 + corr_cv5.iloc[-1]
    
    # alpha131 = Corr(Close, Volume, 10)  [IR=0.82]
    f_alpha131 = corr_cv10.iloc[-1]
    
    # alpha100 = Volume Std(20)  [IR=0.50]
    f_alpha100 = vs.rolling(20).std().iloc[-1]
    
    # alpha144 = Sum(Return, 20)  [IR=0.45] (20日累积收益)
    f_alpha144 = cs.pct_change().iloc[-20:].sum()
    
    # alpha030 = (ret5_rank - (1-vol_rank)) / (ret5_rank + (1-vol_rank))  [IR=0.38]
    d5_rank = cs.diff(5).iloc[-1]
    vol_rank_rev = (1 - vol20)
    f_alpha030 = (d5_rank - vol_rank_rev) / (abs(d5_rank) + abs(vol_rank_rev) + 1e-8)
    
    # alpha175 = MA5 - MA20  [IR=0.42]
    f_alpha175 = ma5 - ma20
    
    # alpha183 = Close > MA60  [IR=0.46]
    f_alpha183 = 1.0 if c > ma60 else 0.0
    
    # alpha097 = Volume Std(10)  [IR=0.39]
    f_alpha097 = vs.rolling(10).std().iloc[-1]
    
    # alpha076 = Std(Abs(Ret))/Mean(Abs(Ret))  [IR≈0.3] 波动稳定性
    ret_abs = abs(cs.pct_change().iloc[-20:])
    f_alpha076 = ret_abs.std()/(ret_abs.mean()+1e-8)
    
    # alpha069 = UpDays(10)/10 * Corr(C,V,5)  [IR=0.51]
    up_days = (cs.diff() > 0).iloc[-10:].sum()
    f_alpha069 = up_days/10 * corr_cv5.iloc[-1]
    
    # alpha047 = UpDays(20)/20 * Corr(C,V,5)  [IR=0.57]
    up_days20 = (cs.diff() > 0).iloc[-20:].sum()
    f_alpha047 = up_days20/20 * corr_cv5.iloc[-1]
    
    return {
        'ret5': ret5, 'ret20': ret20, 'trend60': trend60, 'vol20': vol20, 'liquidity': liq,
        'alpha172': f_alpha172, 'alpha149': f_alpha149, 'alpha131': f_alpha131,
        'alpha100': f_alpha100, 'alpha144': f_alpha144, 'alpha030': f_alpha030,
        'alpha175': f_alpha175, 'alpha183': f_alpha183, 'alpha097': f_alpha097,
        'alpha076': f_alpha076, 'alpha069': f_alpha069, 'alpha047': f_alpha047,
        'close': c,
    }


# ── 12个策略定义 ──
# 每个策略是一个 (name, weight_dict) 元组
# weight_dict: {因子名: 权重, ...}

STRATEGIES = [
    # 基于 IC 排名 Top 因子
    ("量价核心(alpha172)", {'alpha172': 1.0}),
    ("量价综合(alpha149+131)", {'alpha149': 0.6, 'alpha131': 0.4}),
    ("动量累积(alpha144)", {'alpha144': 1.0}),
    ("量价+动量", {'alpha172': 0.5, 'alpha144': 0.3, 'alpha030': 0.2}),
    ("量价+趋势", {'alpha172': 0.4, 'alpha175': 0.3, 'alpha183': 0.3}),
    ("量价+低波", {'alpha172': 0.5, 'alpha100': 0.3, 'alpha097': 0.2}),
    ("全明星组合", {'alpha172': 0.25, 'alpha149': 0.20, 'alpha144': 0.20,
                   'alpha100': 0.15, 'alpha047': 0.10, 'alpha069': 0.10}),
    ("动量+低波", {'alpha144': 0.4, 'alpha030': 0.3, 'alpha076': 0.3}),
    ("趋势确认", {'alpha183': 0.4, 'alpha175': 0.3, 'alpha030': 0.3}),
    ("量价背离(负向)", {'alpha172': 0.6, 'alpha144': -0.2, 'alpha100': 0.2}),
    ("稳健多因子", {'alpha172': 0.3, 'alpha144': 0.2, 'alpha100': 0.2, 'alpha076': 0.15, 'alpha183': 0.15}),
    ("极端动量", {'alpha144': 0.5, 'alpha030': 0.3, 'alpha175': 0.2}),
]


def run_backtest(stock_data, all_dates, risk_cache, weights):
    """用给定因子权重跑回测。weights: {factor_name: weight}"""
    cash = float(INITIAL_CASH)
    positions = {}
    equity_curve = []
    pending_rebalance = False; pending_targets = []; risk_off = False
    factor_names = list(weights.keys())

    for i, date_str in enumerate(all_dates):
        dt = pd.Timestamp(date_str)
        is_signal = False
        if i < len(all_dates)-1:
            nxt = pd.Timestamp(all_dates[i+1])
            if dt.isocalendar()[1] != nxt.isocalendar()[1] or (nxt-dt).days > 3: is_signal = True
        else: is_signal = True

        if pending_rebalance:
            to_sell = list(positions.keys()) if risk_off else [s for s in positions if s not in pending_targets]
            for sym in to_sell:
                pos = positions.pop(sym)
                sp = None
                if sym in stock_data:
                    r = stock_data[sym][stock_data[sym]['date'] == dt]
                    if len(r) > 0: sp = float(r.iloc[0]['open']) * (1-SLIPPAGE)
                sp = sp or (pos.get('last_close',pos['entry_price'])*0.95)
                cash += pos['shares'] * sp * (1-SELL_COMM-SELL_TAX)
            if not risk_off:
                to_buy = [s for s in pending_targets if s not in positions]
                if to_buy:
                    pc = cash/len(to_buy)
                    for sym in to_buy:
                        bp = None
                        if sym in stock_data:
                            r = stock_data[sym][stock_data[sym]['date']==dt]
                            if len(r) > 0: bp = float(r.iloc[0]['open'])*(1+SLIPPAGE)
                        if bp is None or bp <= 0: continue
                        size = (int(pc/(bp*(1+BUY_COMM))//LOT_SIZE))*LOT_SIZE
                        if size <= 0: continue
                        cost = size*bp*(1+BUY_COMM)
                        if cost > cash: continue
                        cash -= cost
                        positions[sym] = {'shares':size,'cost':cost,'entry_price':bp,'entry_date':date_str}
            pending_rebalance = False; pending_targets = []

        if is_signal:
            risk_off = not risk_cache.get(date_str, True)
            features = []
            for sym, sdf in stock_data.items():
                m = sdf[sdf['date'] == dt]
                if len(m) == 0: continue
                idx = m.index[0]
                feat = compute_features(sdf, idx)
                if feat is None: continue
                feat['symbol'] = sym
                features.append(feat)
            if features:
                fdf = pd.DataFrame(features)
                # 计算加权得分
                score = pd.Series(0.0, index=fdf.index)
                for fn, w in weights.items():
                    if fn in fdf.columns:
                        rank_val = fdf[fn].rank(pct=True)
                        score += rank_val * w
                fdf['score'] = score
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
                r = stock_data[sym][stock_data[sym]['date']==dt]
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
    dd = (eq-np.maximum.accumulate(eq))/np.maximum.accumulate(eq)
    max_dd = dd.min()*100
    calmar = cagr/abs(max_dd) if max_dd!=0 else 0
    return total_ret, cagr, sharpe, max_dd, calmar


if __name__ == '__main__':
    print(f'[{datetime.now().strftime("%H:%M:%S")}] 因子挖掘 × 夏普优化', flush=True)
    
    # 阶段1: 加载数据
    print('Phase 1: 加载数据...', flush=True)
    stock_data, bench, all_dates, risk_cache = load_data(100)
    print(f'  股票: {len(stock_data)}, 交易日: {len(all_dates)}', flush=True)

    # 阶段2: 跑全部12个策略
    print('\nPhase 2: 策略回测对比', flush=True)
    results = []
    for name, weights in STRATEGIES:
        t0 = time.time()
        ret, cagr, sh, mdd, cal = run_backtest(stock_data, all_dates, risk_cache, weights)
        results.append((name, weights, ret, cagr, sh, mdd, cal))
        print(f'  {name:20s} | 收益:{ret:>7.2f}% CAGR:{cagr:>5.2f}% 夏普:{sh:>7.4f} 回撤:{mdd:>6.2f}% ({time.time()-t0:.0f}s)', flush=True)

    # 按夏普排序
    results.sort(key=lambda x: x[4], reverse=True)
    
    print(f'\n{"="*70}')
    print(f'按夏普排序 Top 5')
    print(f'{"="*70}')
    for i, (name, w, ret, cagr, sh, mdd, cal) in enumerate(results[:5]):
        print(f'  #{i+1}: {name:20s} 收益{ret:.2f}% CAGR{cagr:.2f}% 夏普{sh:.4f} 回撤{mdd:.2f}%', flush=True)

    # 阶段3: 对 Top 3 做夏普优化 — 单独运行
    print('\nPhase 3: 跳过 (后续单独优化 Top 3)', flush=True)
    
    # 最终排名
    results.sort(key=lambda x: x[4], reverse=True)
    
    print(f'\n{"="*70}')
    print(f'最终排名 (按夏普)')
    print(f'{"="*70}')
    for i, (name, w, ret, cagr, sh, mdd, cal) in enumerate(results):
        print(f'  #{i+1}: {name:25s} 收益{ret:>7.2f}% CAGR{cagr:>5.2f}% 夏普{sh:>7.4f} 回撤{mdd:>6.2f}% 卡尔玛{cal:.4f}', flush=True)

    # 保存报告
    now_str = datetime.now().strftime('%Y%m%d_%H%M')
    rp = f'{REPORT_DIR}/因子挖掘_夏普优化_{now_str}.md'
    with open(rp, 'w', encoding='utf-8') as f:
        f.write(f'# 因子挖掘 × 夏普优化 报告\n\n')
        f.write(f'**生成**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'**股票**: {len(stock_data)} 只 | **区间**: 2010-2026\n')
        f.write(f'**因子来源**: Alpha191 IC 研究 Top 因子\n\n')
        f.write('## 最终排名\n\n| 排名 | 策略 | 总收益 | CAGR | 夏普 | 回撤 | 卡尔玛 |\n|------|------|--------|------|------|------|--------|\n')
        for i, (name, w, ret, cagr, sh, mdd, cal) in enumerate(results):
            f.write(f'| #{i+1} | {name} | {ret:.2f}% | {cagr:.2f}% | {sh:.4f} | {mdd:.2f}% | {cal:.4f} |\n')
        
        f.write('\n## Top 3 策略权重\n\n')
        for name, w, ret, cagr, sh, mdd, cal in results[:3]:
            f.write(f'### {name}\n\n')
            f.write(f'- 收益: {ret:.2f}% | CAGR: {cagr:.2f}% | 夏普: {sh:.4f} | 回撤: {mdd:.2f}%\n')
            f.write('- 权重:\n')
            for fn, wt in w.items():
                f.write(f'  - {fn}: {wt:.2f}\n')
            f.write('\n')
        
        f.write('## 策略说明\n\n')
        f.write('| 因子 | 含义 | IC来源 |\n')
        f.write('|------|------|--------|\n')
        f.write('| alpha172 | Rank(Corr(C,V,10)) + Rank(Corr(C,V,5)) | IR=0.98, Top1 |\n')
        f.write('| alpha149 | Corr(C,V,10)×5 + Corr(C,V,5) | IR=0.82, Top3 |\n')
        f.write('| alpha131 | Corr(Close, Volume, 10) | IR=0.82, Top4 |\n')
        f.write('| alpha144 | 20日累积收益 | IR=0.45 |\n')
        f.write('| alpha100 | 成交量标准差(20d) | IR=0.50 |\n')
        f.write('| alpha030 | 动量强度 | IR=0.38 |\n')
        f.write('| alpha175 | 5日-20日均线差 | IR=0.42 |\n')
        f.write('| alpha183 | Close > MA60 | IR=0.46 |\n')
        f.write('| alpha097 | 成交量标准差(10d) | IR=0.39 |\n')
        f.write('| alpha076 | 波动稳定性 | IR≈0.30 |\n')
        f.write('| alpha047 | UpDays/20 × Corr(C,V,5) | IR=0.57 |\n')
        f.write('| alpha069 | UpDays/10 × Corr(C,V,5) | IR=0.51 |\n')
        f.write('\n---\n**风险提示**: 基于历史回测，不构成投资建议。\n')
    
    print(f'\n报告: {rp}', flush=True)
