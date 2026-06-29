"""sharpe_optimizer_v2.py — V2 夏普优化 (精选组合版)

限制测试组合数到 ~20/方向，确保在 600s 内完成。
"""

import sys, os, glob, time, warnings
import pandas as pd, numpy as np
from datetime import datetime
warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/xuhaoran/Documents/agent/backend')

from app.services.strategy_v2 import enhanced_risk_check

DAILY_DIR='/Volumes/xhrrrrr_macmini副盘/quantlab/market/daily'
BENCH_FILE='/Volumes/xhrrrrr_macmini副盘/quantlab/market/benchmarks/000001.SH.parquet'
REPORT_DIR='/Users/xuhaoran/Documents/agent/reports'
IC=1000000; TN=5; LS=100; SL=0.001; BC=0.0003; SC=0.0003; ST=0.0005

def load(n=80):
    b=pd.read_parquet(BENCH_FILE); b['date']=pd.to_datetime(b['date']); b=b.sort_values('date').reset_index(drop=True)
    ad=[str(d.date()) for d in b['date'] if d>=pd.Timestamp('2010-01-01')]
    af=sorted(glob.glob(f'{DAILY_DIR}/*.parquet')); af=[f for f in af if not os.path.basename(f).startswith('688')]
    m=[]
    for f in af[:500]:
        try:
            d=pd.read_parquet(f,columns=['date','amount']); m.append((os.path.basename(f).replace('.parquet',''),float(d['amount'].mean())))
        except: pass
    m.sort(key=lambda x:x[1],reverse=True); syms=[x[0] for x in m[:n]]
    sd={}
    for s in syms:
        try:
            d=pd.read_parquet(f'{DAILY_DIR}/{s}.parquet'); d.columns=[c.lower() for c in d.columns]
            d['date']=pd.to_datetime(d['date']); d=d[d['date']>='2009-06-01'].copy()
            if len(d)>=300: sd[s]=d
        except: pass
    return sd, b, ad

def run_bt(sd, ad, score_fn):
    c=float(IC); p={}; eq=[]; pr=False; pt=[]; ro=False
    for i,ds in enumerate(ad):
        dt=pd.Timestamp(ds)
        sig=False
        if i<len(ad)-1:
            nx=pd.Timestamp(ad[i+1])
            if dt.isocalendar()[1]!=nx.isocalendar()[1] or (nx-dt).days>3: sig=True
        else: sig=True
        if pr:
            ts=list(p.keys()) if ro else [s for s in p if s not in pt]
            for s in ts:
                pos=p.pop(s); sp=None
                if s in sd:
                    r=sd[s][sd[s]['date']==dt]
                    if len(r)>0: sp=float(r.iloc[0]['open'])*(1-SL)
                sp=sp or (pos.get('last_close',pos['entry_price'])*0.95)
                c+=pos['shares']*sp*(1-SC-ST)
            if not ro:
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
            ro=False
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
                fdf=pd.DataFrame(feat); fdf=score_fn(fdf)
                if not fdf.empty:
                    cu=set(p.keys()); t=fdf.head(TN)['symbol'].tolist()
                    t12=set(fdf.head(12)['symbol']) if len(fdf)>=12 else set(fdf['symbol'])
                    t=[s for s in cu if s in t12]+[s for s in t if s not in cu]; pt=t[:TN]; pr=True
        pv=0
        for sym,pos in list(p.items()):
            if sym in sd:
                r=sd[sym][sd[sym]['date']==dt]
                if len(r)>0: pv+=pos['shares']*float(r.iloc[0]['close'])
        eq.append(c+pv)
    e=np.array(eq); tr=(e[-1]/e[0]-1)*100; yr=len(e)/252
    cagr=((e[-1]/e[0])**(1/yr)-1)*100 if yr>0 else 0
    dr=pd.Series(e).pct_change().dropna(); sh=np.sqrt(252)*dr.mean()/dr.std() if dr.std()>0 else 0
    dd=(e-np.maximum.accumulate(e))/np.maximum.accumulate(e); mdd=dd.min()*100
    cal=cagr/abs(mdd) if mdd!=0 else 0
    return tr, cagr, sh, mdd, cal

def make_rev(ws,wr,wt,wl):
    def fn(f):
        f=f.copy(); f['score']=((1-f['liquidity'].rank(pct=True))*ws+(1-f['ret5'].rank(pct=True))*wr+f['trend60'].rank(pct=True)*wt+(1-f['vol20'].rank(pct=True))*wl)
        return f.sort_values('score',ascending=False)
    return fn

def make_lv(w_lv, w_r=0, w_s=0):
    def fn(f):
        f=f.copy(); s=(1-f['vol20'].rank(pct=True))*w_lv
        if w_r: s+=(1-f['ret5'].rank(pct=True))*w_r
        if w_s: s+=(1-f['liquidity'].rank(pct=True))*w_s
        f['score']=s; return f.sort_values('score',ascending=False)
    return fn

if __name__=='__main__':
    print(f'[{datetime.now().strftime("%H:%M:%S")}] 夏普优化V2', flush=True); print('Loading...', flush=True)
    sd,b,ad=load(80); print(f'{len(sd)} stocks, {len(ad)} days\n', flush=True)
    
    # ── Reversal: 精选20组合 ──
    rev_combos = [
        (0.45,0.25,0.20,0.10), (0.40,0.30,0.20,0.10), (0.50,0.20,0.20,0.10),
        (0.45,0.20,0.25,0.10), (0.40,0.25,0.25,0.10), (0.50,0.25,0.15,0.10),
        (0.45,0.30,0.15,0.10), (0.40,0.30,0.15,0.15), (0.50,0.20,0.15,0.15),
        (0.35,0.30,0.25,0.10), (0.55,0.20,0.15,0.10), (0.40,0.25,0.20,0.15),
        (0.45,0.20,0.20,0.15), (0.50,0.15,0.20,0.15), (0.35,0.25,0.25,0.15),
        (0.50,0.25,0.10,0.15), (0.40,0.20,0.20,0.20), (0.35,0.30,0.15,0.20),
        (0.45,0.15,0.25,0.15), (0.50,0.20,0.20,0.10),  # 重复1个无所谓
    ]
    print('Reversal (20 combos)...', flush=True)
    rev_res=[]
    for ws,wr,wt,wl in rev_combos:
        t0=time.time(); tr,cg,sh,mdd,cal=run_bt(sd,ad,make_rev(ws,wr,wt,wl))
        rev_res.append((ws,wr,wt,wl,tr,cg,sh,mdd,cal))
        print(f'  S{ws:.0f}R{wr:.0f}T{wt:.0f}L{wl:.0f} | ret={tr:.1f}% CAGR={cg:.1f}% Sharpe={sh:.4f} DD={mdd:.1f}% ({time.time()-t0:.0f}s)', flush=True)
    rev_res.sort(key=lambda x:x[6],reverse=True)
    print(f'  ★ Best: S{rev_res[0][0]:.0f}R{rev_res[0][1]:.0f}T{rev_res[0][2]:.0f}L{rev_res[0][3]:.0f} Sharpe={rev_res[0][6]:.4f}\n', flush=True)
    
    # ── LowVol: 精选15组合 ──
    lv_combos = [
        (1.0,0,0), (0.8,0.2,0), (0.7,0.3,0), (0.6,0.4,0), (0.5,0.5,0),
        (0.7,0.2,0.1), (0.6,0.3,0.1), (0.5,0.3,0.2), (0.6,0.2,0.2),
        (0.5,0.4,0.1), (0.4,0.4,0.2), (0.8,0.1,0.1), (0.7,0.15,0.15),
        (0.5,0.25,0.25), (0.4,0.3,0.3),
    ]
    print('LowVol (15 combos)...', flush=True)
    lv_res=[]
    for wl,wr,ws in lv_combos:
        t0=time.time(); tr,cg,sh,mdd,cal=run_bt(sd,ad,make_lv(wl,wr,ws))
        lv_res.append((wl,wr,ws,tr,cg,sh,mdd,cal))
        print(f'  LV{wl:.0f}R{wr:.0f}S{ws:.0f} | ret={tr:.1f}% CAGR={cg:.1f}% Sharpe={sh:.4f} DD={mdd:.1f}% ({time.time()-t0:.0f}s)', flush=True)
    lv_res.sort(key=lambda x:x[5],reverse=True)
    print(f'  ★ Best: LV{lv_res[0][0]:.0f}R{lv_res[0][1]:.0f}S{lv_res[0][2]:.0f} Sharpe={lv_res[0][5]:.4f}\n', flush=True)
    
    # 最终排名
    all_r = [('Reversal', rev_res[0]), ('LowVol', lv_res[0])]
    all_r.sort(key=lambda x:x[1][5], reverse=True)
    
    print('='*60)
    print('最终排名')
    print('='*60)
    print(f'{"方向":10s} {"权重":30s} {"收益":>8s} {"CAGR":>6s} {"夏普":>8s} {"回撤":>7s}')
    print('-'*70)
    for cat, best in all_r:
        ws,wr,wt,wl,tr,cg,sh,mdd,cal = best
        print(f'{cat:10s} S{ws:.0f}R{wr:.0f}T{wt:.0f}L{wl:.0f} {"":14s} {tr:>7.1f}% {cg:>5.1f}% {sh:>7.4f} {mdd:>6.1f}%', flush=True)
    
    ns=datetime.now().strftime('%Y%m%d_%H%M')
    with open(f'{REPORT_DIR}/夏普优化V2_{ns}.md','w',encoding='utf-8') as f:
        f.write(f'# 夏普优化V2 报告\n**{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}**\n\n')
        f.write('## Reversal\n\n| S | R | T | L | Ret | CAGR | Sharpe | DD |\n|---|---|---|---|-----|------|--------|-----|\n')
        for ws,wr,wt,wl,tr,cg,sh,mdd,cal in rev_res[:5]:
            f.write(f'| {ws:.0%} | {wr:.0%} | {wt:.0%} | {wl:.0%} | {tr:.1f}% | {cg:.1f}% | {sh:.4f} | {mdd:.1f}% |\n')
        f.write('\n## LowVol\n\n| LV | R | S | Ret | CAGR | Sharpe | DD |\n|----|---|---|-----|------|--------|-----|\n')
        for wl,wr,ws,tr,cg,sh,mdd,cal in lv_res[:5]:
            f.write(f'| {wl:.0%} | {wr:.0%} | {ws:.0%} | {tr:.1f}% | {cg:.1f}% | {sh:.4f} | {mdd:.1f}% |\n')
    print(f'\n报告: {REPORT_DIR}/夏普优化V2_{ns}.md', flush=True)
