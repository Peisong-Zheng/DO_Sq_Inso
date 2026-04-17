#!/usr/bin/env python3
from __future__ import annotations
import os, gc
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

BASE = Path('data/')
OUT = Path('results/mcv_coldduration_ccm_lagtest_outputs_v5')
FIG = OUT / 'figures'; TAB = OUT / 'tables'
for p in [OUT, FIG, TAB]: p.mkdir(parents=True, exist_ok=True)
CH4_STATE_FILE = BASE /'ch4_mcv_square_wave_selected_full_50yr.csv'
PRE_FILE = BASE / 'pre_800_inter100.txt'; OBL_FILE = BASE / 'obl_800_inter100.txt'; ECC_FILE = BASE / 'ecc_1000_inter100.txt'

DOWNSAMPLE_YR=1000; 
E=4; 
TAU=-2; 
N_SURROGATES=30; 
RNG_SEED=42; 
LIB_POINTS=6; 
N_NEIGH=E+1; 
LAG_RANGE_KYR=np.arange(-8,9,1); 
WINDOW_YR=20000
COLD_DURATION_SMOOTH_POINTS=0

def centered_smooth_with_edge_handling(x, win_pts):
    x=np.asarray(x,float)
    if win_pts<=1: return x.copy()
    return pd.Series(x).rolling(window=win_pts,center=True,min_periods=1).mean().to_numpy()

def moving_sum_centered(x,win_pts):
    y=np.convolve(x,np.ones(win_pts,float),mode='valid'); h=win_pts//2; return y,h

def moving_mean_centered(x,win_pts):
    y,h=moving_sum_centered(x,win_pts); return y/win_pts,h

def extract_intervals_from_binary(age,cold):
    age=np.asarray(age,float); cold=np.asarray(cold,bool)
    trans=np.flatnonzero(np.diff(cold.astype(int))!=0)+1; starts=np.r_[0,trans]; ends=np.r_[trans,len(age)]
    rows=[]
    dt=np.median(np.diff(age))
    for s,e in zip(starts,ends):
        if not cold[s]: continue
        rows.append({'young_age_yrBP':age[s],'old_age_yrBP':age[e-1],'duration_yr':age[e-1]-age[s]+dt,'start_truncated':bool(s==0),'end_truncated':bool(e==len(age))})
    return pd.DataFrame(rows)

def extract_warm_intervals_from_binary(age,cold):
    return extract_intervals_from_binary(age, ~np.asarray(cold,bool))

def local_mean_durations(centers, intervals_df, window_yr):
    vals=np.full_like(centers,np.nan,dtype=float); half=window_yr/2
    for i,c in enumerate(centers):
        mask=(intervals_df['young_age_yrBP']>=c-half)&(intervals_df['old_age_yrBP']<=c+half)
        if mask.any(): vals[i]=intervals_df.loc[mask,'duration_yr'].mean()
    return vals

def compute_window_stats(age,cold,warmings,cold_intervals,warm_intervals,window_yr):
    dt=np.median(np.diff(age)); win_pts=int(round(window_yr/dt)); win_pts += (win_pts%2==0)
    Eseries,h=moving_sum_centered(warmings.astype(float),win_pts); Pseries,_=moving_mean_centered(cold.astype(float),win_pts)
    centers=age[h:len(age)-h]
    warm_full=warm_intervals.loc[~warm_intervals['start_truncated'] & ~warm_intervals['end_truncated']].copy()
    return pd.DataFrame({'age_yrBP':centers,'warming_count_20k':Eseries,'cold_fraction_20k':Pseries,'mean_cold_duration_yr':local_mean_durations(centers,cold_intervals,window_yr),'mean_warm_duration_yr':local_mean_durations(centers,warm_full,window_yr)})

def prepare_ch4_sequence(path):
    df=pd.read_csv(path).sort_values('age_yrBP'); age=df['age_yrBP'].to_numpy(float); cold=(df['mcv_state'].to_numpy()==0)
    cold_intervals=extract_intervals_from_binary(age,cold); warm_intervals=extract_warm_intervals_from_binary(age,cold)
    warmings=np.zeros_like(age,int)
    for a in cold_intervals['young_age_yrBP'].values: warmings[np.argmin(np.abs(age-a))]=1
    if cold[0]: warmings[0]=0
    return {'window_stats': compute_window_stats(age,cold,warmings,cold_intervals,warm_intervals,WINDOW_YR)}

def load_orbital_series(path,name):
    df=pd.read_csv(path,sep=r'\s+',header=None,engine='python'); out=df.iloc[:,:2].copy(); out.columns=['age_raw',name]; out['age_yrBP']=out['age_raw'].abs()*1000.0; out[name]=pd.to_numeric(out[name],errors='coerce'); return out[['age_yrBP',name]].dropna().sort_values('age_yrBP').reset_index(drop=True)

def interp_to_grid(age,src_age,src_val):
    order=np.argsort(src_age); return np.interp(age,np.asarray(src_age,float)[order],np.asarray(src_val,float)[order])

def zscore(x):
    x=np.asarray(x,float); sd=np.nanstd(x); return (x-np.nanmean(x))/sd if sd>0 else np.zeros_like(x)

def build_dataset():
    stats=prepare_ch4_sequence(CH4_STATE_FILE)['window_stats'].copy(); step=max(1,int(round(DOWNSAMPLE_YR/50))); stats=stats.iloc[::step].reset_index(drop=True)
    for name,path in [('pre',PRE_FILE),('obl',OBL_FILE),('ecc',ECC_FILE)]:
        df=load_orbital_series(path,name); stats[name]=interp_to_grid(stats['age_yrBP'].values,df['age_yrBP'].values,df[name].values)
    stats['mean_cold_duration_kyr_raw']=stats['mean_cold_duration_yr']/1000.0
    stats['mean_cold_duration_kyr']=centered_smooth_with_edge_handling(stats['mean_cold_duration_kyr_raw'].to_numpy(), COLD_DURATION_SMOOTH_POINTS if COLD_DURATION_SMOOTH_POINTS>0 else 1)
    stats=stats.sort_values('age_yrBP',ascending=False).reset_index(drop=True); stats['Time']=np.arange(1,len(stats)+1)
    return stats

def build_embedding(x,E,tau):
    lag=abs(int(tau)); start=(E-1)*lag; valid_idx=np.arange(start,len(x)); emb=np.column_stack([x[valid_idx-j*lag] for j in range(E)]).astype(np.float32); return valid_idx,emb

def pearson_r(a,b):
    m=np.isfinite(a)&np.isfinite(b)
    if m.sum()<5: return np.nan
    aa=a[m]; bb=b[m]
    if np.std(aa)==0 or np.std(bb)==0: return np.nan
    return float(np.corrcoef(aa,bb)[0,1])

def libsizes_for_n(n_valid):
    vals=np.unique(np.round(np.linspace(max(E+5,int(0.1*n_valid)), int(0.9*n_valid), LIB_POINTS)).astype(int)); return vals[vals>=E+2]

def simplex_predict(dist_sub, lib_target):
    finite_counts=np.isfinite(dist_sub).sum(axis=1)
    if np.nanmax(finite_counts)<2: return None
    k=min(N_NEIGH, dist_sub.shape[1]-1)
    if k<2: return None
    idx=np.argpartition(dist_sub,kth=k-1,axis=1)[:,:k]
    d_nei=np.take_along_axis(dist_sub,idx,axis=1)
    order=np.argsort(d_nei,axis=1)
    idx=np.take_along_axis(idx,order,axis=1); d_nei=np.take_along_axis(d_nei,order,axis=1)
    tar_nei=lib_target[idx]; d1=np.maximum(d_nei[:,[0]],1e-12); w=np.exp(-d_nei/d1); return np.sum(w*tar_nei,axis=1)/np.sum(w,axis=1)

def ccm_curve_and_nulls_fast(source_series,target_series,E,tau,n_surrogates,rng_seed):
    valid_idx,manifold=build_embedding(source_series,E,tau); libs=libsizes_for_n(len(valid_idx)); rng=np.random.default_rng(rng_seed); subsets={int(L): np.sort(rng.choice(valid_idx,size=int(L),replace=False)) for L in libs}; distmat=cdist(manifold,manifold,metric='euclidean').astype(np.float32)
    obs=[]
    for L in libs:
        subset=subsets[int(L)]; lib_pos=np.searchsorted(valid_idx,subset); dsub=distmat[:,lib_pos].copy(); dsub[lib_pos,np.arange(len(subset))]=np.inf; preds=simplex_predict(dsub,target_series[subset]); obs.append(pearson_r(preds,target_series[valid_idx]) if preds is not None else np.nan)
    obs=np.array(obs,float); null_curves=np.zeros((n_surrogates,len(libs)),float)
    for s in range(n_surrogates):
        perm=np.random.default_rng(rng_seed+1000+s).permutation(target_series)
        for j,L in enumerate(libs):
            subset=subsets[int(L)]; lib_pos=np.searchsorted(valid_idx,subset); dsub=distmat[:,lib_pos].copy(); dsub[lib_pos,np.arange(len(subset))]=np.inf; preds=simplex_predict(dsub,perm[subset]); null_curves[s,j]=pearson_r(preds,perm[valid_idx]) if preds is not None else np.nan
    curve_df=pd.DataFrame({'LibSize':libs.astype(int),'rho_obs':obs,'rho_null_median':np.nanpercentile(null_curves,50,axis=0),'rho_null_lo95':np.nanpercentile(null_curves,2.5,axis=0),'rho_null_hi95':np.nanpercentile(null_curves,97.5,axis=0)})
    p=float((np.sum(null_curves[:,-1]>=obs[-1])+1)/(n_surrogates+1)); del distmat, null_curves; gc.collect(); return curve_df,float(obs[-1]),p

def ccm_lag_curve_fullsample(source_series,target_series,E,tau,lag_values):
    valid_idx,manifold=build_embedding(source_series,E,tau); distmat=cdist(manifold,manifold,metric='euclidean').astype(np.float32); obs=[]
    for lag in lag_values.astype(int):
        row_idx=valid_idx[(valid_idx+lag>=0)&(valid_idx+lag<len(target_series))]
        if len(row_idx)<E+5: obs.append(np.nan); continue
        row_pos=np.searchsorted(valid_idx,row_idx); dsub=distmat[np.ix_(row_pos,row_pos)].copy(); np.fill_diagonal(dsub,np.inf); preds=simplex_predict(dsub,target_series[row_idx+lag]); obs.append(pearson_r(preds,target_series[row_idx+lag]) if preds is not None else np.nan)
    obs=np.array(obs,float); best_idx=int(np.nanargmax(obs)); del distmat; gc.collect(); return pd.DataFrame({'lag_kyr':lag_values.astype(float),'rho_obs':obs}), {'best_lag_kyr':int(lag_values[best_idx]),'best_rho':float(obs[best_idx])}

def main():
    stats=build_dataset(); stats.to_csv(TAB/'aligned_coldduration_orbitals_for_ccm_lagtest.csv', index=False)
    metric_series=zscore(stats['mean_cold_duration_kyr'].to_numpy())
    drivers=[('pre','precession'),('obl','obliquity'),('ecc','eccentricity')]
    col_fwd='#1f77b4'; col_rev='#d62728'
    fig,axes=plt.subplots(5,1,figsize=(12,8.2),dpi=180,sharex=True)
    axes[0].plot(stats['age_yrBP']/1000,stats['mean_cold_duration_kyr_raw'],color='0.6',lw=1,label='raw'); axes[0].plot(stats['age_yrBP']/1000,stats['mean_cold_duration_kyr'],color='black',lw=1.2,label='used in CCM'); axes[0].set_ylabel('cold dur.\n(kyr)'); axes[0].legend(frameon=True,fontsize=8)
    for ax,col,lab,cc in zip(axes[1:],['pre','obl','ecc','warming_count_20k'],['pre','obl','ecc','warmings/20k'],['tab:blue','tab:green','tab:red','tab:purple']): ax.plot(stats['age_yrBP']/1000,stats[col],color=cc,lw=1); ax.set_ylabel(lab)
    axes[0].set_title(f'Cold-duration inputs (Δt={DOWNSAMPLE_YR} yr, E={E}, tau={TAU}, null={N_SURROGATES}, smooth_pts={COLD_DURATION_SMOOTH_POINTS})'); axes[-1].set_xlabel('Age (kyr BP)'); axes[-1].invert_xaxis(); fig.tight_layout(); fig.savefig(FIG/'fig01_inputs_cold_duration_and_orbitals.png',bbox_inches='tight'); plt.close(fig)
    rows=[]; std_curves={}; lag_curves={}
    for j,(driver,label) in enumerate(drivers):
        print('driver',driver,flush=True); dseries=zscore(stats[driver].to_numpy())
        f_curve,f_obs,f_p=ccm_curve_and_nulls_fast(metric_series,dseries,E,TAU,N_SURROGATES,RNG_SEED+j*10+1)
        r_curve,r_obs,r_p=ccm_curve_and_nulls_fast(dseries,metric_series,E,TAU,N_SURROGATES,RNG_SEED+j*10+2)
        lag_f,sum_f=ccm_lag_curve_fullsample(metric_series,dseries,E,TAU,LAG_RANGE_KYR)
        lag_r,sum_r=ccm_lag_curve_fullsample(dseries,metric_series,E,TAU,LAG_RANGE_KYR)
        std_curves[(driver,'fwd')]=f_curve; std_curves[(driver,'rev')]=r_curve; lag_curves[(driver,'fwd')]=lag_f; lag_curves[(driver,'rev')]=lag_r
        rows.append({'driver':driver,'driver_label':label,'rho_metric_xmap_orbital':f_obs,'p_metric_xmap_orbital':f_p,'rho_orbital_xmap_metric':r_obs,'p_orbital_xmap_metric':r_p,'best_lag_metric_xmap_orbital_kyr':sum_f['best_lag_kyr'],'best_lag_metric_xmap_orbital_rho':sum_f['best_rho'],'best_lag_orbital_xmap_metric_kyr':sum_r['best_lag_kyr'],'best_lag_orbital_xmap_metric_rho':sum_r['best_rho']}); gc.collect()
    res=pd.DataFrame(rows); res.to_csv(TAB/'coldduration_ccm_lagtest_results.csv', index=False)
    fig,axes=plt.subplots(2,3,figsize=(15,8.8),dpi=180)
    for j,(driver,label) in enumerate(drivers):
        rr=res.loc[res.driver==driver].iloc[0]; ax=axes[0,j]; df_f=std_curves[(driver,'fwd')]; df_r=std_curves[(driver,'rev')]
        ax.fill_between(df_f['LibSize'],df_f['rho_null_lo95'],df_f['rho_null_hi95'],color=col_fwd,alpha=0.15); ax.plot(df_f['LibSize'],df_f['rho_null_median'],'--',color=col_fwd,lw=1); ax.plot(df_f['LibSize'],df_f['rho_obs'],'o-',color=col_fwd,lw=1.6,label='cold duration xmap orbital')
        ax.fill_between(df_r['LibSize'],df_r['rho_null_lo95'],df_r['rho_null_hi95'],color=col_rev,alpha=0.15); ax.plot(df_r['LibSize'],df_r['rho_null_median'],'--',color=col_rev,lw=1); ax.plot(df_r['LibSize'],df_r['rho_obs'],'s-',color=col_rev,lw=1.6,label='orbital xmap cold duration')
        ax.set_title(f'{label}\nCCM: p_fwd={rr.p_metric_xmap_orbital:.3f}, p_rev={rr.p_orbital_xmap_metric:.3f}')
        if j==0: ax.set_ylabel('CCM skill ρ'); ax.legend(frameon=True,fontsize=8)
        ax.set_xlabel('Library size'); ax.grid(alpha=0.2,ls='--')
        ax=axes[1,j]; df_f=lag_curves[(driver,'fwd')]; df_r=lag_curves[(driver,'rev')]
        ax.plot(df_f['lag_kyr'],df_f['rho_obs'],'o-',color=col_fwd,lw=1.6,label='cold duration xmap orbital'); ax.plot(df_r['lag_kyr'],df_r['rho_obs'],'s-',color=col_rev,lw=1.6,label='orbital xmap cold duration'); ax.axvline(0,color='0.4',ls='--',lw=1)
        ax.set_title(f'lag-test: best_fwd={int(rr.best_lag_metric_xmap_orbital_kyr)} kyr, best_rev={int(rr.best_lag_orbital_xmap_metric_kyr)} kyr')
        if j==0: ax.set_ylabel('Cross-map skill ρ'); ax.legend(frameon=True,fontsize=8)
        ax.set_xlabel('Cross-map lag (kyr)'); ax.grid(alpha=0.2,ls='--')
    fig.suptitle('Cold-duration focused CCM and lag-test (bidirectional)\nLag-test uses full sample at each lag and no null test',y=1.02); fig.tight_layout(); fig.savefig(FIG/'fig02_coldduration_ccm_and_lagtest_2x3.png',bbox_inches='tight'); plt.close(fig)
    with open(OUT/'summary.txt','w',encoding='utf-8') as f:
        f.write('Cold-duration focused CCM and lag-test\n'); f.write(f'Downsample={DOWNSAMPLE_YR} yr, E={E}, tau={TAU}, surrogates={N_SURROGATES}\n'); f.write(f'Lag range={LAG_RANGE_KYR[0]}..{LAG_RANGE_KYR[-1]} kyr\n'); f.write(f'Cold-duration smoothing points={COLD_DURATION_SMOOTH_POINTS} (0 means none)\n'); f.write('Lag-test uses full sample at each lag and no null test.\n\n')
        for _,r in res.iterrows():
            f.write(f"Driver: {r.driver}\n"); f.write(f"  CCM   fwd (cold duration xmap orbital): rho={r.rho_metric_xmap_orbital:.3f}, p={r.p_metric_xmap_orbital:.3f}\n"); f.write(f"  CCM   rev (orbital xmap cold duration): rho={r.rho_orbital_xmap_metric:.3f}, p={r.p_orbital_xmap_metric:.3f}\n"); f.write(f"  LAG   fwd: best lag={int(r.best_lag_metric_xmap_orbital_kyr)} kyr, rho={r.best_lag_metric_xmap_orbital_rho:.3f}\n"); f.write(f"  LAG   rev: best lag={int(r.best_lag_orbital_xmap_metric_kyr)} kyr, rho={r.best_lag_orbital_xmap_metric_rho:.3f}\n\n")
if __name__=='__main__': main()
