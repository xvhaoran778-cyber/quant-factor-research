"""factor_enhance.py — 向最优策略逐个加入Alpha191因子，找收益提升最大的因子"""

import sys, os, glob, time, warnings
import pandas as pd, numpy as np
from datetime import datetime
warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/xuhaoran/Documents/agent/backend')

DAILY_DIR = '/Volumes/xhrrrrr_macmini副盘/quantlab/market/daily'
BENCH_FILE = '/Volumes/xhrrrrr_macmini副盘/quantlab/market/benchmarks/000001.SH.parquet'
REPORT_DIR = '/Users/xuhaoran/Documents/agent/reports'
IC = 1000000; TN = 5; LS = 100; SL = 0.001; BC = 0.0003; SC = 0.0003; ST = 0.0005


def load(n=60):
    b = pd.read_parquet(BENCH_FILE); b['date'] = pd.to_datetime(b['date']); b = b.sort_values('date').reset_index(drop=True)
    ad = [str(d.date()) for d in b['date'] if d >= pd.Timestamp('2010-01-01')]
    af = sorted(glob.glob(f'{DAILY_DIR}/*.parquet')); af = [f for f in af if not os.path.basename(f).startswith('688')]
    m = []
    for f in af[:400]:
        try:
            d = pd.read_parquet(f, columns=['date', 'amount']); m.append((os.path.basename(f).replace('.parquet', ''), float(d['amount'].mean())))
        except: pass
    m.sort(key=lambda x: x[1], reverse=True); syms = [x[0] for x in m[:n]]
    
    # 预计算Alpha191因子
    from app.services.alpha191_factors import Alpha191
    alpha = Alpha191()
    sd = {}
    t0 = time.time()
    for i, s in enumerate(syms):
        try:
            d = pd.read_parquet(f'{DAILY_DIR}/{s}.parquet'); d.columns = [c.lower() for c in d.columns]
            d['date'] = pd.to_datetime(d['date']); d = d[d['date'] >= '2009-06-01'].copy()
            if len(d) < 300: continue
            d = alpha.calculate_all(d)
            sd[s] = d
        except: pass
        if (i+1) % 20 == 0: print(f'  Alpha191: {i+1}/{len(syms)} ({time.time()-t0:.0f}s)', flush=True)
    print(f'  Done: {len(sd)} stocks, {alpha.get_factor_count()} factors ({time.time()-t0:.0f}s)', flush=True)
    return sd, b, ad, alpha


def run_bt_score(sd, ad, score_fn):
    """回测评分函数。"""
    c = float(IC); p = {}; eq = []; pr = False; pt = []; ro = False
    for i, ds in enumerate(ad):
        dt = pd.Timestamp(ds)
        sig = False
        if i < len(ad) - 1:
            nx = pd.Timestamp(ad[i+1])
            if dt.isocalendar()[1] != nx.isocalendar()[1] or (nx-dt).days > 3: sig = True
        else: sig = True
        if pr:
            ts = list(p.keys()) if ro else [s for s in p if s not in pt]
            for s in ts:
                pos = p.pop(s); sp = None
                if s in sd:
                    r = sd[s][sd[s]['date'] == dt]
                    if len(r) > 0: sp = float(r.iloc[0]['open']) * (1 - SL)
                sp = sp or (pos.get('last_close', pos['entry_price']) * 0.95)
                c += pos['shares'] * sp * (1 - SC - ST)
            if not ro:
                tb = [s for s in pt if s not in p]
                if tb:
                    pc = c / len(tb)
                    for s in tb:
                        bp = None
                        if s in sd:
                            r = sd[s][sd[s]['date'] == dt]
                            if len(r) > 0: bp = float(r.iloc[0]['open']) * (1 + SL)
                        if bp is None or bp <= 0: continue
                        sz = (int(pc / (bp * (1 + BC)) // LS)) * LS
                        if sz <= 0: continue
                        co = sz * bp * (1 + BC)
                        if co > c: continue
                        c -= co; p[s] = {'shares': sz, 'cost': co, 'entry_price': bp, 'entry_date': ds}
            pr = False; pt = []
        if sig:
            ro = False
            feat = []
            for s, df in sd.items():
                m = df[df['date'] == dt]
                if len(m) == 0: continue
                ix = m.index[0]
                if ix < 60: continue
                r = df.iloc[ix]
                cv = float(r['close'])
                r5 = cv / float(df.iloc[ix-5]['close']) - 1 if ix >= 5 else 0
                c20 = float(df.iloc[ix-20]['close']) if ix >= 20 else cv
                ma60 = float(df['close'].iloc[:ix+1].rolling(60).mean().iloc[-1])
                v20 = float(df['close'].iloc[:ix+1].pct_change().iloc[-20:].std())
                lq = float(df['amount'].iloc[:ix+1].tail(20).mean()) if ix >= 20 else 0
                feat_row = {'symbol': s, 'close': cv, 'ret5': r5, 'ret20': cv / c20 - 1,
                            'trend60': cv / ma60 - 1 if ma60 > 0 else 0, 'vol20': v20, 'liquidity': lq}
                # 加入所有Alpha191因子
                for ac in [c for c in df.columns if c.startswith('alpha')]:
                    val = float(r[ac]) if ac in df.columns and not pd.isna(r.get(ac)) else 0
                    feat_row[ac] = val
                feat.append(feat_row)
            if feat:
                fdf = pd.DataFrame(feat)
                fdf = fdf.replace([np.inf, -np.inf], 0).fillna(0)
                fdf = score_fn(fdf)
                if not fdf.empty:
                    cu = set(p.keys()); t = fdf.head(TN)['symbol'].tolist()
                    t12 = set(fdf.head(12)['symbol']) if len(fdf) >= 12 else set(fdf['symbol'])
                    t = [s for s in cu if s in t12] + [s for s in t if s not in cu]
                    pt = t[:TN]; pr = True
        pv = 0
        for sym, pos in list(p.items()):
            if sym in sd:
                r = sd[sym][sd[sym]['date'] == dt]
                if len(r) > 0: pv += pos['shares'] * float(r.iloc[0]['close'])
        eq.append(c + pv)
    e = np.array(eq); tr = (e[-1]/e[0]-1)*100; yr = len(e)/252
    cagr = ((e[-1]/e[0])**(1/yr)-1)*100 if yr > 0 else 0
    dr = pd.Series(e).pct_change().dropna(); sh = np.sqrt(252)*dr.mean()/dr.std() if dr.std() > 0 else 0
    dd = (e-np.maximum.accumulate(e))/np.maximum.accumulate(e); mdd = dd.min()*100
    return tr, cagr, sh, mdd


# 最优基础权重: 45/20/20/15
def base_score(features):
    df = features.copy()
    df['score'] = ((1-df['liquidity'].rank(pct=True))*0.45 + (1-df['ret5'].rank(pct=True))*0.20
                   + df['trend60'].rank(pct=True)*0.20 + (1-df['vol20'].rank(pct=True))*0.15)
    return df.sort_values('score', ascending=False)


def make_enhanced_score(extra_factor, extra_weight=0.05, base_weights=(0.40, 0.18, 0.18, 0.14)):
    """基础4因子 + 1个额外Alpha191因子。"""
    ws, wr, wt, wl = base_weights
    total = ws + wr + wt + wl + extra_weight
    ws, wr, wt, wl = ws/total, wr/total, wt/total, wl/total
    ew = extra_weight / total

    def fn(features):
        df = features.copy()
        score = ((1-df['liquidity'].rank(pct=True))*ws + (1-df['ret5'].rank(pct=True))*wr
                 + df['trend60'].rank(pct=True)*wt + (1-df['vol20'].rank(pct=True))*wl)
        if extra_factor in df.columns:
            score += df[extra_factor].rank(pct=True) * ew
        df['score'] = score
        return df.sort_values('score', ascending=False)
    fn.__name__ = f'enhanced_{extra_factor}'
    return fn


if __name__ == '__main__':
    print(f'[{datetime.now().strftime("%H:%M:%S")}] 因子增强实验', flush=True)
    print('Loading with Alpha191...', flush=True)
    sd, b, ad, alpha = load(50)
    print(f'{len(sd)} stocks, {len(ad)} days\n', flush=True)

    # 1. 基础回测
    t0 = time.time()
    tr, cg, sh, mdd = run_bt_score(sd, ad, base_score)
    print(f'基础 V1.3.0 (45/20/20/15): ret={tr:.1f}% CAGR={cg:.1f}% Sharpe={sh:.4f} DD={mdd:.1f}% ({time.time()-t0:.0f}s)\n', flush=True)
    BASE_SH = sh

    # 2. 提取排名靠前的Alpha191因子
    alpha_cols = [c for c in sd[list(sd.keys())[0]].columns if c.startswith('alpha')]
    print(f'Total Alpha191 factors available: {len(alpha_cols)}', flush=True)

    # 用IC分析Top因子
    priority_factors = [
        'alpha172', 'alpha149', 'alpha131', 'alpha151', 'alpha144',
        'alpha100', 'alpha030', 'alpha047', 'alpha069', 'alpha175',
        'alpha183', 'alpha097', 'alpha076', 'alpha095', 'alpha188',
        'alpha152', 'alpha153', 'alpha186', 'alpha034', 'alpha040',
        'alpha116', 'alpha082', 'alpha020', 'alpha071', 'alpha084',
        'alpha163', 'alpha075', 'alpha113', 'alpha087', 'alpha166',
    ]
    available_priority = [f for f in priority_factors if f in alpha_cols]
    available_priority = available_priority[:12]  # 只测Top 12
    print(f'Priority factors: {len(available_priority)}\n', flush=True)

    # 3. 逐个测试额外因子
    results = []
    for ef in available_priority:
        fn = make_enhanced_score(ef, extra_weight=0.05)
        t0 = time.time()
        tr2, cg2, sh2, mdd2 = run_bt_score(sd, ad, fn)
        delta_sh = sh2 - BASE_SH
        results.append((ef, tr2, cg2, sh2, mdd2, delta_sh))
        marker = '★' if delta_sh > 0 else ' '
        print(f'  {marker} +{ef:10s}: ret={tr2:>7.1f}% CAGR={cg2:>4.1f}% Sharpe={sh2:>6.4f} (Δ={delta_sh:+.5f}) DD={mdd2:>5.1f}% ({time.time()-t0:.0f}s)', flush=True)

    # 按夏普排序
    results.sort(key=lambda x: x[3], reverse=True)
    improved = [r for r in results if r[5] > 0]
    worsened = [r for r in results if r[5] <= 0]

    print(f'\n{"="*65}')
    print(f'提升夏普的因子 ({len(improved)}/{len(results)})')
    print(f'{"="*65}')
    print(f'{"因子":15s} {"收益":>8s} {"CAGR":>6s} {"夏普":>8s} {"Δ夏普":>9s} {"回撤":>7s}')
    print('-'*60)
    for ef, tr, cg, sh, mdd, ds in improved[:10]:
        print(f'  {ef:15s} {tr:>7.1f}% {cg:>5.1f}% {sh:>7.4f} {ds:>+8.5f} {mdd:>6.1f}%', flush=True)

    if worsened:
        print(f'\n降低夏普的因子 ({len(worsened)}):')
        for ef, tr, cg, sh, mdd, ds in worsened[:5]:
            print(f'  {ef:15s} Sharpe={sh:.4f} (Δ={ds:+.5f})', flush=True)

    # 4. 最佳因子组合 (尝试Top 3提升因子的组合)
    top3 = improved[:3]
    print(f'\n{"="*65}')
    print(f'最佳3因子组合测试')
    print(f'{"="*65}')

    if len(top3) >= 2:
        f1, f2 = top3[0][0], top3[1][0]
        for f3 in [top3[2][0]] if len(top3) >= 3 else ['']:
            combo_factors = [f1, f2]
            if f3: combo_factors.append(f3)
            
            def combo_score(features):
                df = features.copy()
                ws, wr, wt, wl = 0.35, 0.15, 0.15, 0.10
                ew = 0.05
                total = ws + wr + wt + wl + ew * len(combo_factors)
                score = ((1-df['liquidity'].rank(pct=True))*ws/total + (1-df['ret5'].rank(pct=True))*wr/total
                         + df['trend60'].rank(pct=True)*wt/total + (1-df['vol20'].rank(pct=True))*wl/total)
                for f in combo_factors:
                    if f in df.columns:
                        score += df[f].rank(pct=True) * ew / total
                df['score'] = score
                return df.sort_values('score', ascending=False)

            t0 = time.time()
            tr3, cg3, sh3, mdd3 = run_bt_score(sd, ad, combo_score)
            print(f'  +{f1}+{f2}+{f3}: ret={tr3:.1f}% CAGR={cg3:.1f}% Sharpe={sh3:.4f} (Δ={sh3-BASE_SH:+.5f}) DD={mdd3:.1f}% ({time.time()-t0:.0f}s)', flush=True)

    # 保存报告
    ns = datetime.now().strftime('%Y%m%d_%H%M')
    with open(f'{REPORT_DIR}/因子增强_{ns}.md', 'w', encoding='utf-8') as f:
        f.write(f'# 因子增强实验报告\n\n**{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}**\n**基础**: V1.3.0 (45/20/20/15) Sharpe={BASE_SH:.4f}\n\n')
        f.write('## 单因子增强效果\n\n| 因子 | 总收益 | CAGR | 夏普 | Δ夏普 | 回撤 |\n|------|--------|------|------|-------|------|\n')
        for ef, tr, cg, sh, mdd, ds in results:
            f.write(f'| {ef} | {tr:.1f}% | {cg:.1f}% | {sh:.4f} | {ds:+.5f} | {mdd:.1f}% |\n')
        f.write('\n## 最佳组合\n\n')
        if len(top3) >= 2:
            combo_str = '+'.join([r[0] for r in top3[:3]])
            f.write(f'{combo_str}: ret={tr3:.1f}% CAGR={cg3:.1f}% Sharpe={sh3:.4f}\n')
        f.write('\n---\n')
    print(f'\n报告: {REPORT_DIR}/因子增强_{ns}.md', flush=True)
