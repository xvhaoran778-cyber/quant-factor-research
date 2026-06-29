"""sharpe_optimize.py — 小盘超跌反转 夏普优化"""

import sys, os, glob, time, warnings
import pandas as pd, numpy as np
from datetime import datetime
warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/xuhaoran/Documents/agent/backend')
DAILY_DIR = '/Volumes/xhrrrrr_macmini副盘/quantlab/market/daily'
BENCH_FILE = '/Volumes/xhrrrrr_macmini副盘/quantlab/market/benchmarks/000001.SH.parquet'
REPORT_DIR = '/Users/xuhaoran/Documents/agent/reports'
IC = 1000000; TN=5; BC=0.0003; SC=0.0003; ST=0.0005; LS=100; SL=0.001

def load_data(n=80):
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
    rc = {}
    for i,ds in enumerate(ad):
        if i<60: rc[ds]=True; continue
        bc=bench['close'].iloc[:i+1].astype(float); m20=bc.rolling(20).mean().iloc[-1]; m60=bc.rolling(60).mean().iloc[-1]
        rc[ds]=True
        if pd.notna(m20) and pd.notna(m60): rc[ds]=not(bc.iloc[-1]<m20<m60)
    return sd, bench, ad, rc

def bt(sd, ad, rc, ws, wr, wt, wl, tn=5):
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
            ro=not rc.get(ds,True); feat=[]
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
                fdf=pd.DataFrame(feat)
                fdf['score']=((1-fdf['liquidity'].rank(pct=True))*ws+(1-fdf['ret5'].rank(pct=True))*wr+fdf['trend60'].rank(pct=True)*wt+(1-fdf['vol20'].rank(pct=True))*wl)
                fdf=fdf.sort_values('score',ascending=False)
                if not ro and not fdf.empty:
                    cu=set(p.keys()); t=fdf.head(tn)['symbol'].tolist(); t12=set(fdf.head(12)['symbol']) if len(fdf)>=12 else set(fdf['symbol'])
                    t=[s for s in cu if s in t12]+[s for s in t if s not in cu]; pt=t[:tn]; pr=True
        pv=0
        for s,pos in list(p.items()):
            if s in sd:
                r=sd[s][sd[s]['date']==dt]
                if len(r)>0: pv+=pos['shares']*float(r.iloc[0]['close'])
        eq.append(c+pv)
    e=np.array(eq); tr=(e[-1]/e[0]-1)*100; yr=len(e)/252
    cagr=((e[-1]/e[0])**(1/yr)-1)*100 if yr>0 else 0
    dr=pd.Series(e).pct_change().dropna(); sh=np.sqrt(252)*dr.mean()/dr.std() if dr.std()>0 else 0
    dd=(e-np.maximum.accumulate(e))/np.maximum.accumulate(e); mdd=dd.min()*100
    cal=cagr/abs(mdd) if mdd!=0 else 0
    return tr, cagr, sh, mdd, cal

if __name__=='__main__':
    print(f'[{datetime.now().strftime("%H:%M:%S")}] 夏普优化', flush=True)
    print('Loading...', flush=True); sd, b, ad, rc=load_data(80)
    print(f'{len(sd)} stocks, {len(ad)} days', flush=True)
    res=[]; tc=0; ts=time.time()
    # 粗网格: 保证 ws+wr+wt+wl=1.0
    for ws in np.arange(0.2, 0.81, 0.2):
        for wr in np.arange(0.1, 0.61, 0.2):
            rem=round(1.0-ws-wr,2)
            if rem<0: continue
            for wt in np.arange(0.0, min(rem+0.001, 0.41), 0.2):
                wl=round(rem-wt,2)
                if wl<0 or wl>0.4: continue
                tc+=1
                t0=time.time(); tr,cg,sh,mdd,cal=bt(sd,ad,rc,ws,wr,wt,wl)
                res.append((ws,wr,wt,wl,tr,cg,sh,mdd,cal))
                print(f'  S{ws:.0f}R{wr:.0f}T{wt:.0f}L{wl:.0f} | {tr:>7.2f}% CAGR{cg:>5.2f}% Sharpe{sh:>7.4f} DD{mdd:>6.2f}% ({time.time()-t0:.0f}s)', flush=True)
    res.sort(key=lambda x: x[6], reverse=True)
    print(f'\nCoarse grid: {tc} combos', flush=True)
    for ws,wr,wt,wl,tr,cg,sh,mdd,cal in res[:3]:
        print(f'  S{ws:.0f}R{wr:.0f}T{wt:.0f}L{wl:.0f} ret={tr:.2f}% sharpe={sh:.4f}', flush=True)
    # Fine grid around top 3
    fine=list(res)
    for ws_b,wr_b,wt_b,wl_b,_,_,_,_,_ in res[:3]:
        for dws in [-0.05,0,0.05]:
            for dwr in [-0.05,0,0.05]:
                for dwt in [-0.05,0,0.05]:
                    for dwl in [-0.05,0,0.05]:
                        ws=round(ws_b+dws,2); wr=round(wr_b+dwr,2); wt=round(wt_b+dwt,2); wl=round(wl_b+dwl,2)
                        if ws<0.1 or wr<0.1 or wt<0 or wl<0: continue
                        if abs(ws+wr+wt+wl-1.0)>0.01: continue
                        if any(abs(fws-ws)<0.01 and abs(fwr-wr)<0.01 for fws,fwr,_,_,_,_,_,_,_ in fine): continue
                        tc+=1
                        t0=time.time(); tr,cg,sh,mdd,cal=bt(sd,ad,rc,ws,wr,wt,wl)
                        fine.append((ws,wr,wt,wl,tr,cg,sh,mdd,cal))
                        print(f'  S{ws:.0f}R{wr:.0f}T{wt:.0f}L{wl:.0f} | {tr:>7.2f}% CAGR{cg:>5.2f}% Sharpe{sh:>7.4f} DD{mdd:>6.2f}% ({time.time()-t0:.0f}s)', flush=True)
    fine.sort(key=lambda x: x[6], reverse=True)
    print(f'\nTop 10 by Sharpe ({tc} total):')
    for i,(ws,wr,wt,wl,tr,cg,sh,mdd,cal) in enumerate(fine[:10]):
        print(f'  #{i+1}: S{ws:.0f}R{wr:.0f}T{wt:.0f}L{wl:.0f} ret={tr:.2f}% CAGR={cg:.2f}% Sharpe={sh:.4f} DD={mdd:.2f}%')
    ns=datetime.now().strftime('%Y%m%d_%H%M')
    with open(f'{REPORT_DIR}/夏普优化_{ns}.md','w',encoding='utf-8') as f:
        f.write(f'# 夏普优化报告\n**{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}**\n**{len(sd)} stocks, {tc} combos**\n\n| S | R | T | L | Ret | CAGR | Sharpe | DD |\n|---|---|---|---|-----|------|--------|-----|\n')
        for ws,wr,wt,wl,tr,cg,sh,mdd,cal in fine[:10]:
            f.write(f'| {ws:.0%} | {wr:.0%} | {wt:.0%} | {wl:.0%} | {tr:.2f}% | {cg:.2f}% | {sh:.4f} | {mdd:.2f}% |\n')
    print(f'Report saved', flush=True)
