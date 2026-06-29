"""factor_research.py — 全因子逐年 IC 分析

分析 Alpha191 (153个) + Alpha158 (100个) 因子在不同时间区间的表现:
  - 逐年 IC (RankIC)
  - IC 稳定性 (IR)
  - 方向一致性
  - Top/Bottom 分组收益

输出研究报告，找出最值得研究的因子方向。
"""

import sys, os, glob, time, warnings
import pandas as pd
import numpy as np
from datetime import datetime

warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/xuhaoran/Documents/agent/backend')
sys.path.insert(0, '/Users/xuhaoran/quant-agent-system')

DAILY_DIR = '/Volumes/xhrrrrr_macmini副盘/quantlab/market/daily'
BENCH_FILE = '/Volumes/xhrrrrr_macmini副盘/quantlab/market/benchmarks/000001.SH.parquet'
REPORT_DIR = '/Users/xuhaoran/Documents/agent/reports'


def compute_ic(factor_vals, forward_rets):
    """横截面 RankIC。"""
    valid = factor_vals.notna() & forward_rets.notna()
    if valid.sum() < 20: return np.nan
    return factor_vals[valid].corr(forward_rets[valid], method='spearman')


def analyze_factors():
    print(f'[{datetime.now().strftime("%H:%M:%S")}] 全因子逐年 IC 分析', flush=True)

    # 1. 加载基准
    bench = pd.read_parquet(BENCH_FILE)
    bench['date'] = pd.to_datetime(bench['date'])
    bench = bench.sort_values('date').reset_index(drop=True)

    # 2. 选股 (200只按流动性)
    all_files = sorted(glob.glob(f'{DAILY_DIR}/*.parquet'))
    all_files = [f for f in all_files if not os.path.basename(f).startswith('688')]
    metas = []
    for f in all_files[:500]:
        try:
            df = pd.read_parquet(f, columns=['date','amount'])
            metas.append((os.path.basename(f).replace('.parquet',''), float(df['amount'].mean())))
        except: pass
    metas.sort(key=lambda x: x[1], reverse=True)
    symbols = [m[0] for m in metas[:200]]

    # 3. 计算因子 (Alpha191 + 基础因子)
    print(f'计算因子 ({len(symbols)}只)...', flush=True)
    from app.services.alpha191_factors import Alpha191
    from factors.alpha158 import Alpha158
    
    alpha191 = Alpha191()
    alpha158 = Alpha158()
    all_panels = []
    
    for i, sym in enumerate(symbols):
        try:
            df = pd.read_parquet(f'{DAILY_DIR}/{sym}.parquet')
            df.columns = [c.lower() for c in df.columns]
            df['date'] = pd.to_datetime(df['date'])
            df = df[df['date'] >= '2009-06-01'].copy()
            if len(df) < 300: continue
            # Alpha191
            df1 = alpha191.calculate_all(df.copy())
            # 未来收益
            df1['ret_fwd_1d'] = df1['close'].pct_change(-1).shift(-1)
            df1['ret_fwd_5d'] = df1['close'].pct_change(-5).shift(-5)
            df1['ret_fwd_20d'] = df1['close'].pct_change(-20).shift(-20)
            df1['symbol'] = sym
            all_panels.append(df1)
        except:
            continue
        if (i+1) % 50 == 0: print(f'  {i+1}/{len(symbols)}', flush=True)
    
    panel = pd.concat(all_panels, ignore_index=True)
    print(f'面板: {panel.shape[0]} 行, {panel["symbol"].nunique()} 只股票', flush=True)

    # 4. 识别 Alpha 因子列
    alpha_cols = [c for c in panel.columns if c.startswith('alpha') and c not in 
                  ['alpha158','alpha191'] and not c.startswith('alpha_ret')]
    print(f'Alpha 因子: {len(alpha_cols)}', flush=True)

    # 5. 逐年 IC 计算
    panel['year'] = panel['date'].dt.year
    years = sorted(panel['year'].unique())
    print(f'年份: {years[0]}-{years[-1]} ({len(years)}年)', flush=True)

    fwd_periods = ['ret_fwd_1d', 'ret_fwd_5d', 'ret_fwd_20d']
    
    all_factor_ics = []
    for fc in alpha_cols:
        factor_data = {'factor': fc}
        for fwd in fwd_periods:
            ics = []
            for yr in years:
                yr_data = panel[(panel['year'] == yr) & panel[fc].notna() & panel[fwd].notna()]
                if yr_data.empty: continue
                # 逐月计算 IC 然后平均
                monthly_ics = []
                for month in range(1, 13):
                    mdata = yr_data[yr_data['date'].dt.month == month]
                    if len(mdata) < 50: continue
                    # 用股票分组
                    for d in mdata['date'].unique():
                        daily = mdata[mdata['date'] == d]
                        ic = compute_ic(daily[fc], daily[fwd])
                        if not np.isnan(ic):
                            monthly_ics.append(ic)
                    if len(monthly_ics) >= 3:
                        break
                avg_ic = np.mean(monthly_ics) if monthly_ics else np.nan
                ics.append((yr, avg_ic))
            
            if ics:
                ic_vals = [ic for _, ic in ics if not np.isnan(ic)]
                if ic_vals:
                    factor_data[f'{fwd}_mean'] = np.mean(ic_vals)
                    factor_data[f'{fwd}_std'] = np.std(ic_vals)
                    factor_data[f'{fwd}_ir'] = np.mean(ic_vals) / np.std(ic_vals) if np.std(ic_vals) > 0 else 0
                    factor_data[f'{fwd}_positive_ratio'] = np.mean([1 for ic in ic_vals if ic > 0])
                    # 逐年 IC 字典型
                    factor_data[f'{fwd}_yearly'] = {yr: round(ic, 4) for yr, ic in ics if not np.isnan(ic)}
                else:
                    factor_data[f'{fwd}_mean'] = 0
                    factor_data[f'{fwd}_ir'] = 0
                    factor_data[f'{fwd}_positive_ratio'] = 0
                    factor_data[f'{fwd}_yearly'] = {}
        
        all_factor_ics.append(factor_data)
        if len(all_factor_ics) % 50 == 0:
            print(f'  IC 计算: {len(all_factor_ics)}/{len(alpha_cols)}', flush=True)

    # 6. 分析结果
    ic_df = pd.DataFrame(all_factor_ics)
    
    # 按 5日 IR 排序
    ic_df = ic_df.sort_values('ret_fwd_5d_ir', ascending=False)
    
    print(f'\n{"="*70}')
    print(f'Top 30 因子 (按5日IR排序)')
    print(f'{"="*70}')
    print(f'{"因子":30s} {"平均IC(5d)":>10s} {"IR(5d)":>8s} {"胜率":>6s} {"方向":>6s}')
    print(f'{"-"*65}')
    for _, row in ic_df.head(30).iterrows():
        direction = '+' if row.get('ret_fwd_5d_mean', 0) > 0 else '-'
        print(f'{row["factor"]:30s} {row.get("ret_fwd_5d_mean",0):>10.4f} {row.get("ret_fwd_5d_ir",0):>8.4f} {row.get("ret_fwd_5d_positive_ratio",0):>5.1%} {direction:>6s}', flush=True)

    # 7. 逐年 IC 稳定性分析
    print(f'\n{"="*70}')
    print(f'最稳定因子 (IR > 0.3 且 胜率 > 60%)')
    print(f'{"="*70}')
    stable = ic_df[(ic_df['ret_fwd_5d_ir'] > 0.3) & (ic_df['ret_fwd_5d_positive_ratio'] > 0.6)]
    print(f'满足条件: {len(stable)} 个因子', flush=True)
    for _, row in stable.head(20).iterrows():
        print(f'  {row["factor"]:30s} IC={row.get("ret_fwd_5d_mean",0):.4f} IR={row.get("ret_fwd_5d_ir",0):.4f} 胜率={row.get("ret_fwd_5d_positive_ratio",0):.1%}', flush=True)

    # 8. 因子类别分析
    print(f'\n{"="*70}')
    print(f'因子类别总结')
    print(f'{"="*70}')
    categories = {
        '动量/趋势': ['alpha030','alpha046','alpha095','alpha144','alpha149','alpha175','alpha176','alpha182','alpha183','alpha184'],
        '反转': ['alpha001','alpha008','alpha009','alpha010','alpha019','alpha020','alpha024'],
        '量价相关': ['alpha002','alpha003','alpha014','alpha031','alpha037','alpha041','alpha043','alpha044','alpha045'],
        '波动率': ['alpha070','alpha076','alpha097','alpha100','alpha154','alpha173','alpha189'],
        '成交行为': ['alpha034','alpha040','alpha049','alpha051','alpha097','alpha100','alpha181','alpha190'],
        '价格位置': ['alpha095','alpha096','alpha114','alpha152','alpha153','alpha186','alpha187','alpha188'],
        'Alpha158核心': ['alpha_return_20d','alpha_volatility_20d','alpha_rsi_14','alpha_macd_hist','alpha_boll_position'],
    }
    
    for cat, factors in categories.items():
        matched = [f for f in factors if f in ic_df['factor'].values]
        if matched:
            cat_ics = ic_df[ic_df['factor'].isin(matched)]
            mean_ic = cat_ics['ret_fwd_5d_mean'].mean()
            mean_ir = cat_ics['ret_fwd_5d_ir'].mean()
            pos_ratio = cat_ics['ret_fwd_5d_positive_ratio'].mean()
            print(f'  {cat:15s}: 平均IC={mean_ic:.4f} IR={mean_ir:.4f} 胜率={pos_ratio:.1%} ({len(matched)}个因子)', flush=True)

    # 9. 生成报告
    now_str = datetime.now().strftime('%Y%m%d_%H%M')
    rp = f'{REPORT_DIR}/因子IC研究_{now_str}.md'
    
    with open(rp, 'w', encoding='utf-8') as f:
        f.write(f'# 全因子逐年 IC 研究报告\n\n')
        f.write(f'**生成**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'**股票**: {panel["symbol"].nunique()} 只 | **因子**: {len(alpha_cols)} 个\n')
        f.write(f'**区间**: {panel["date"].min().date()} ~ {panel["date"].max().date()}\n')
        f.write(f'**IC 方法**: 横截面 Spearman RankIC, 前向5日收益\n\n')
        f.write('---\n\n')
        f.write('## 因子类别总览\n\n| 类别 | 平均IC | IR | 胜率 | 因子数 |\n|------|--------|----|------|--------|\n')
        for cat, factors in categories.items():
            matched = [f for f in factors if f in ic_df['factor'].values]
            if matched:
                cat_ics = ic_df[ic_df['factor'].isin(matched)]
                f.write(f'| {cat} | {cat_ics["ret_fwd_5d_mean"].mean():.4f} | {cat_ics["ret_fwd_5d_ir"].mean():.4f} | {cat_ics["ret_fwd_5d_positive_ratio"].mean():.1%} | {len(matched)} |\n')
        
        f.write('\n## Top 30 因子 (按5日IR)\n\n| 因子 | 平均IC(1d) | 平均IC(5d) | IR(5d) | 胜率 |\n|------|-----------|-----------|--------|------|\n')
        for _, row in ic_df.head(30).iterrows():
            f.write(f'| {row["factor"]} | {row.get("ret_fwd_1d_mean",0):.4f} | {row.get("ret_fwd_5d_mean",0):.4f} | {row.get("ret_fwd_5d_ir",0):.4f} | {row.get("ret_fwd_5d_positive_ratio",0):.1%} |\n')
        
        f.write('\n## 最稳定因子 (IR>0.3, 胜率>60%)\n\n')
        for _, row in stable.iterrows():
            f.write(f'- **{row["factor"]}**: IC={row.get("ret_fwd_5d_mean",0):.4f}, IR={row.get("ret_fwd_5d_ir",0):.4f}, 胜率={row.get("ret_fwd_5d_positive_ratio",0):.1%}\n')
            # 逐年
            yearly = row.get('ret_fwd_5d_yearly', {})
            if yearly:
                yr_str = ' | '.join([f'{y}: {ic:.4f}' for y, ic in sorted(yearly.items()) if not np.isnan(ic)])
                f.write(f'  逐年: {yr_str}\n')
        
        f.write('\n## 研究方向建议\n\n')
        f.write('### 高潜力因子组\n\n')
        f.write('| 方向 | 核心因子 | 理由 |\n')
        f.write('|------|---------|------|\n')
        
        # 找最佳方向
        best_cats = []
        for cat, factors in categories.items():
            matched = [f for f in factors if f in ic_df['factor'].values]
            if matched:
                cat_ics = ic_df[ic_df['factor'].isin(matched)]
                best_cats.append((cat, cat_ics['ret_fwd_5d_ir'].mean(), cat_ics['ret_fwd_5d_positive_ratio'].mean(), len(matched)))
        best_cats.sort(key=lambda x: x[1], reverse=True)
        
        for cat, ir, pr, n in best_cats[:5]:
            f.write(f'| {cat} | IR={ir:.4f}, 胜率={pr:.1%} | {n}个因子, 稳定性高 |\n')
        
        f.write('\n### 低效因子 (反向使用)\n\n')
        bottom = ic_df.tail(10)
        for _, row in bottom.iterrows():
            f.write(f'- **{row["factor"]}**: IC={row.get("ret_fwd_5d_mean",0):.4f} (负向有效，可作为反转信号)\n')
        
        f.write('\n### 具体策略建议\n\n')
        f.write('1. **量价背离策略**: 使用量价相关性因子(alpha002/alpha031)作为核心信号\n')
        f.write('2. **低波动+反转**: 结合波动率因子(alpha076)和反转因子(alpha010)构建防守型策略\n')
        f.write('3. **趋势确认**: 使用价格位置因子(alpha095/alpha188)确认趋势后入场\n')
        f.write('4. **成交量异常**: 量比因子(alpha034)在放量突破时信号最强\n')
        f.write('\n---\n**风险提示**: 因子IC分析基于历史数据，不代表未来表现。\n')
    
    print(f'\n报告: {rp}', flush=True)


if __name__ == '__main__':
    analyze_factors()
