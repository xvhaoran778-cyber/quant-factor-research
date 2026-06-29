"""test_strategies_v2.py — V2 策略体系全量测试"""

import sys, os, glob, time, warnings
import pandas as pd, numpy as np
from datetime import datetime
warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/xuhaoran/Documents/agent/backend')

from app.services.strategy_v2 import (
    VERSIONS, SCORING_REGISTRY, list_versions, get_scoring_fn,
    analyze_drawdowns, classify_market_regime, enhanced_risk_check
)

DAILY_DIR = '/Volumes/xhrrrrr_macmini副盘/quantlab/market/daily'
BENCH_FILE = '/Volumes/xhrrrrr_macmini副盘/quantlab/market/benchmarks/000001.SH.parquet'
REPORT_DIR = '/Users/xuhaoran/Documents/agent/reports'
IC = 1000000; TN=5; LS=100; SL=0.001
BC=0.0003; SC=0.0003; ST=0.0005

def load(n=80):
    bench = pd.read_parquet(BENCH_FILE); bench['date']=pd.to_datetime(bench['date']); bench=bench.sort_values('date').reset_index(drop=True)
    ad = [str(d.date()) for d in bench['date'] if d>=pd.Timestamp('2010-01-01')]
    af = sorted(glob.glob(f'{DAILY_DIR}/*.parquet')); af=[f for f in af if not os.path.basename(f).startswith('688')]
    m = []
    for f in af[:500]:
        try:
            d = pd.read_parquet(f, columns=['date','amount']); m.append((os.path.basename(f).replace('.parquet',''), float(d['amount'].mean())))
        except: pass
    m.sort(key=lambda x: x[1], reverse=True); syms=[x[0] for x in m[:n]]
    sd = {}
    for s in syms:
        try:
            d = pd.read_parquet(f'{DAILY_DIR}/{s}.parquet'); d.columns=[c.lower() for c in d.columns]
            d['date']=pd.to_datetime(d['date']); d=d[d['date']>='2009-06-01'].copy()
            if len(d)>=300: sd[s]=d
        except: pass
    return sd, bench, ad

def bt_with_curve(sd, ad, bench, scoring_fn, params=None, use_enhanced_risk=False):
    """带净值曲线输出的回测。"""
    c=float(IC); p={}; eq=[]; tl=[]; pr=False; pt=[]; ro=False
    p = p or {}
    for i,ds in enumerate(ad):
        dt=pd.Timestamp(ds)
        sig=False
        if i<len(ad)-1:
            nx=pd.Timestamp(ad[i+1])
            if dt.isocalendar()[1]!=nx.isocalendar()[1] or (nx-dt).days>3: sig=True
        else: sig=True
        
        # 增强风控
        risk_on = True; risk_reason = ""
        if use_enhanced_risk and params and i >= 60:
            bi = bench[bench['date']==dt].index
            if len(bi) > 0:
                risk_on, risk_reason = enhanced_risk_check(bench['close'], bi[0], params)
        
        if pr:
            ts=list(p.keys()) if ro else [s for s in p if s not in pt]
            for s in ts:
                pos=p.pop(s); sp=None
                if s in sd:
                    r=sd[s][sd[s]['date']==dt]
                    if len(r)>0: sp=float(r.iloc[0]['open'])*(1-SL)
                sp=sp or (pos.get('last_close',pos['entry_price'])*0.95)
                c+=pos['shares']*sp*(1-SC-ST)
                tl.append({'date':ds,'symbol':s,'action':'sell','shares':pos['shares'],'pnl':pos['shares']*sp-pos['cost']})
            if not ro and risk_on:
                tb=[s for s in pt if s not in p]
                if tb:
                    pc=c/len(tb)
                    for s in tb:
                        bp=None
                        if s in sd:
                            r=sd[s][sd[s]['date']==dt]
                            if len(r)>0: bp=float(r.iloc[0]['open'])*(1+SL)
                        if bp is None or bp<=0: continue
                        sz=(int(pc/(bp*(1+BC))//LS))*LS
                        if sz<=0: continue
                        co=sz*bp*(1+BC)
                        if co>c: continue
                        c-=co; p[s]={'shares':sz,'cost':co,'entry_price':bp,'entry_date':ds}
            pr=False; pt=[]
        if sig:
            ro=not risk_on
            feat=[]
            for s,df in sd.items():
                m=df[df['date']==dt]
                if len(m)==0: continue
                ix=m.index[0]
                if ix<60: continue
                r=df.iloc[ix]; cv=float(r['close'])
                r5=cv/float(df.iloc[ix-5]['close'])-1 if ix>=5 else 0
                c20=float(df.iloc[ix-20]['close']) if ix>=20 else cv
                ma60=float(df['close'].iloc[:ix+1].rolling(60).mean().iloc[-1])
                v20=float(df['close'].iloc[:ix+1].pct_change().iloc[-20:].std())
                lq=float(df['amount'].iloc[:ix+1].tail(20).mean()) if ix>=20 else 0
                feat.append({'symbol':s,'close':cv,'ret5':r5,'ret20':cv/c20-1,'trend60':cv/ma60-1 if ma60>0 else 0,'vol20':v20,'liquidity':lq})
            if feat:
                fdf=pd.DataFrame(feat); fdf=scoring_fn(fdf)
                if not ro and risk_on and not fdf.empty:
                    cu=set(p.keys()); t=fdf.head(TN)['symbol'].tolist()
                    t12=set(fdf.head(12)['symbol']) if len(fdf)>=12 else set(fdf['symbol'])
                    t=[s for s in cu if s in t12]+[s for s in t if s not in cu]; pt=t[:TN]; pr=True
        pv=0
        for sym,pos in list(p.items()):
            if sym in sd:
                r=sd[sym][sd[sym]['date']==dt]
                if len(r)>0: pv+=pos['shares']*float(r.iloc[0]['close'])
        eq.append(c+pv)
    return np.array(eq), tl

def compute_metrics(eq):
    tr=(eq[-1]/eq[0]-1)*100; yr=len(eq)/252
    cagr=((eq[-1]/eq[0])**(1/yr)-1)*100 if yr>0 else 0
    dr=pd.Series(eq).pct_change().dropna()
    sh=np.sqrt(252)*dr.mean()/dr.std() if dr.std()>0 else 0
    dd=(eq-np.maximum.accumulate(eq))/np.maximum.accumulate(eq); mdd=dd.min()*100
    return tr, cagr, sh, mdd

if __name__=='__main__':
    print(f'[{datetime.now().strftime("%H:%M:%S")}] V2策略体系测试', flush=True)
    print('Loading...', flush=True); sd, bench, ad = load(80)
    bm_px = bench['close'].values
    print(f'{len(sd)} stocks, {len(ad)} days', flush=True)
    
    # ── 阶段1: 回撤归因分析 ──
    print('\n=== Phase 1: 回撤归因分析 (V1.0.0) ===', flush=True)
    from app.services.strategy_v2 import score_reversal_v1
    eq, trades = bt_with_curve(sd, ad, bench, score_reversal_v1)
    tr, cagr, sh, mdd = compute_metrics(eq)
    print(f'V1.0.0: ret={tr:.2f}% CAGR={cagr:.2f}% Sharpe={sh:.4f} DD={mdd:.2f}%', flush=True)
    
    dd_analysis = analyze_drawdowns(eq.tolist(), bm_px.tolist(), ad, top_n=5)
    print(f'\nTop 5 回撤分析:')
    print(f'{"区间":25s} {"深度":>8s} {"天数":>6s} {"大盘变化":>10s} {"类型":>10s}')
    print('-'*65)
    for dd in dd_analysis:
        print(f'{dd["start_date"]}~{dd["end_date"]:15s} {dd["depth"]:>7.2f}% {dd["duration_days"]:>5d}d {dd["benchmark_change"]:>9.2f}% {dd["type"]}', flush=True)
    
    # 分析每个回撤的市场状态
    for dd in dd_analysis:
        st = ad.index(dd['start_date']) if dd['start_date'] in ad else 0
        en = ad.index(dd['end_date']) if dd['end_date'] in ad else len(ad)-1
        if en >= len(bm_px): en = len(bm_px)-1
        start_px = bm_px[st] if st < len(bm_px) else 1
        end_px = bm_px[en]
        bm_ret = (end_px/start_px-1)*100
        regime = classify_market_regime(bm_px[st:en+1]) if en > st else 'unknown'
        
        # 回撤前市场状态
        pre_st = max(0, st-60)
        pre_regime = classify_market_regime(bm_px[pre_st:st]) if st > pre_st else 'unknown'
        print(f'  [{dd["start_date"]}] 回撤前:{pre_regime} → 回撤中:{regime} 大盘{bm_ret:.1f}%', flush=True)
    
    # ── 阶段2: 全版本对比测试 ──
    print('\n=== Phase 2: 全版本对比 ===', flush=True)
    all_results = []
    for vid, version in sorted(VERSIONS.items()):
        fn = get_scoring_fn(vid)
        if fn is None: continue
        t0 = time.time()
        params = version.params or {}
        use_er = any(k in params for k in ['vol_shrink','trend_exit','timing_ma_short'])
        eq2, _ = bt_with_curve(sd, ad, bench, fn, params, use_enhanced_risk=use_er)
        tr2, cagr2, sh2, mdd2 = compute_metrics(eq2)
        all_results.append((vid, version.name, version.category, tr2, cagr2, sh2, mdd2))
        print(f'  {vid:8s} {version.name:20s} | ret={tr2:>7.2f}% CAGR={cagr2:>5.2f}% Sharpe={sh2:>7.4f} DD={mdd2:>6.2f}% ({time.time()-t0:.0f}s)', flush=True)
    
    all_results.sort(key=lambda x: x[5], reverse=True)
    print(f'\n{"="*65}')
    print(f'最终排名 by Sharpe')
    print(f'{"="*65}')
    print(f'{"版本":8s} {"名称":22s} {"类别":10s} {"收益":>8s} {"CAGR":>6s} {"夏普":>8s} {"回撤":>7s}')
    print('-'*70)
    for vid, name, cat, tr, cagr, sh, mdd in all_results:
        print(f'{vid:8s} {name:22s} {cat:10s} {tr:>7.2f}% {cagr:>5.2f}% {sh:>7.4f} {mdd:>6.2f}%', flush=True)
    
    # ── 保存报告 ──
    ns = datetime.now().strftime('%Y%m%d_%H%M')
    with open(f'{REPORT_DIR}/V2策略体系_{ns}.md', 'w', encoding='utf-8') as f:
        f.write(f'# V2 策略体系测试报告\n\n')
        f.write(f'**生成**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'**股票**: {len(sd)} 只 | **区间**: 2010-2026\n\n')
        f.write('---\n\n## 回撤归因分析\n\n')
        f.write(f'V1.0.0 基础绩效: ret={tr:.2f}% CAGR={cagr:.2f}% Sharpe={sh:.4f} DD={mdd:.2f}%\n\n')
        f.write('| 区间 | 深度 | 天数 | 大盘变化 | 类型 | 回撤前状态 | 回撤中状态 |\n|------|------|------|---------|------|-----------|-----------|\n')
        for dd in dd_analysis:
            st = ad.index(dd['start_date']) if dd['start_date'] in ad else 0
            en = ad.index(dd['end_date']) if dd['end_date'] in ad else len(ad)-1
            pre_st = max(0, st-60)
            pre_r = classify_market_regime(bm_px[pre_st:st]) if st>pre_st else '?'
            r = classify_market_regime(bm_px[st:en+1]) if en>st else '?'
            f.write(f'| {dd["start_date"]}~{dd["end_date"][:10]} | {dd["depth"]:.2f}% | {dd["duration_days"]} | {dd["benchmark_change"]:.2f}% | {dd["type"]} | {pre_r} | {r} |\n')
        
        f.write('\n## 全版本对比\n\n')
        f.write('| 版本 | 名称 | 类别 | 总收益 | CAGR | 夏普 | 回撤 |\n|------|------|------|--------|------|------|------|\n')
        for vid, name, cat, tr, cagr, sh, mdd in all_results:
            f.write(f'| {vid} | {name} | {cat} | {tr:.2f}% | {cagr:.2f}% | {sh:.4f} | {mdd:.2f}% |\n')
        
        f.write('\n## 版本详情\n\n')
        for vid, version in sorted(VERSIONS.items()):
            f.write(f'### {vid}: {version.name}\n\n')
            f.write(f'- **类别**: {version.category}\n')
            f.write(f'- **权重**: {version.weights}\n')
            f.write(f'- **参数**: {version.params}\n')
            f.write(f'- **描述**: {version.description}\n\n')
        
        f.write('\n---\n**风险提示**: 基于历史回测，不构成投资建议。\n')
    
    print(f'\n报告: {REPORT_DIR}/V2策略体系_{ns}.md', flush=True)
