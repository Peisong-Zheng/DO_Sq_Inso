from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from functools import lru_cache


BASE = Path('data/')
OUT = Path('results/mcv_detector')
FIG_DIR = OUT / 'figures'
TAB_DIR = OUT / 'tables'
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

GS_START = np.array([11703,14692,23020,23340,27780,28900,30840,32500,33740,35480,38220,40160,41460,43340,46860,49280,54220,55000,55800,58040,58280,59080,59440,64100,69620,72340,76440,84760,85060,90040,104040,104520,106750,108280,115370])
GS_END = np.array([12896,22900,23220,27540,28600,30600,32040,33360,34740,36580,39900,40800,42240,44280,48340,49600,54900,55400,56500,58160,58560,59300,63840,69400,70380,74100,77760,84960,87600,90140,104380,105440,106900,110640,119140])
AGE_MIN, AGE_MAX = 11703, 119140
REF_AGES = np.arange(AGE_MIN, AGE_MAX + 1, 1)
REF_STATE = np.ones_like(REF_AGES, dtype=int)
for s, e in zip(GS_START, GS_END):
    REF_STATE[(REF_AGES >= s) & (REF_AGES <= e)] = 0
REF_TRANSITIONS = REF_AGES[np.diff(REF_STATE, prepend=REF_STATE[0]) != 0]

DT_DETECT = 50
AMP_GRID = sorted(set([0.0, 0.1, 0.25, 0.5, 0.75] + [round(x, 2) for x in np.arange(0.20, 0.301, 0.01)]))
PERSIST_GRID = sorted(set([0, 100, 200, 300, 400, 500] + list(range(80, 151, 10))))
TOLS = [100, 200, 300]
SELECTED_TOL = 300

PROXIES = [
    {'name':'ch4','display':'CH$_4$','selected_csv':'ch4_mcv_square_wave_selected_full_50yr.csv','line_label':'Filtered CH$_4$','state_label':'CH$_4$ MCV','state_color':'tab:red'},
    {'name':'monsoon','display':r'$\delta^{18}$O','selected_csv':'monsoon_mcv_square_wave_selected_full_50yr.csv','line_label':r'Filtered monsoon $\delta^{18}$O','state_label':'Monsoon MCV','state_color':'tab:purple'},
]


def setup_style():
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':False,'pdf.fonttype':42,'ps.fonttype':42})

def add_panel_label(ax, label, x=-0.18, y=1.04):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=12, fontweight='bold', va='bottom', ha='left')

def save_fig(fig, name):
    fig.savefig(FIG_DIR / name, format='pdf', bbox_inches='tight')
    plt.close(fig)

def load_selected_series(filename):
    df = pd.read_csv(BASE / filename).sort_values('age_yrBP').reset_index(drop=True)
    return df

def crop_to_ref_window(age, values):
    mask = (age > AGE_MIN) & (age < AGE_MAX)
    return age[mask], values[mask]

def make_segments(age, value, med, dt_yr):
    state = (value >= med).astype(int)
    idx = np.flatnonzero(np.diff(state) != 0) + 1
    starts = np.r_[0, idx]
    ends = np.r_[idx, len(state)]
    segs = []
    for s, e in zip(starts, ends):
        segs.append({'start':int(s),'end':int(e),'state':int(state[s]),'dur_yr':float(age[e-1]-age[s]+dt_yr),'amp':float(np.max(np.abs(value[s:e]-med)))})
    return segs

def merge_segment_triplet(left, mid, right):
    return {'start':left['start'],'end':right['end'],'state':left['state'],'dur_yr':left['dur_yr']+mid['dur_yr']+right['dur_yr'],'amp':max(left['amp'],mid['amp'],right['amp'])}

def apply_detector(age, value, med, amp_sigma=0.0, min_persistence_yr=0.0, dt_yr=50):
    segs = make_segments(age, value, med, dt_yr)
    amp_thr = amp_sigma * np.std(value)
    changed = True
    while changed and len(segs) > 1:
        changed = False
        for i, seg in enumerate(segs):
            if not (seg['amp'] < amp_thr or seg['dur_yr'] < min_persistence_yr):
                continue
            changed = True
            if i == 0:
                nxt = segs[1]
                nxt['start'] = seg['start']
                nxt['dur_yr'] += seg['dur_yr']
                nxt['amp'] = max(nxt['amp'], seg['amp'])
                segs.pop(0)
            elif i == len(segs) - 1:
                prv = segs[-2]
                prv['end'] = seg['end']
                prv['dur_yr'] += seg['dur_yr']
                prv['amp'] = max(prv['amp'], seg['amp'])
                segs.pop()
            else:
                segs = segs[:i-1] + [merge_segment_triplet(segs[i-1], seg, segs[i+1])] + segs[i+2:]
            break
    flips = np.array([age[s['start']] for s in segs[1:]], dtype=float)
    states = np.zeros_like(age, dtype=int)
    segment_id = np.zeros_like(age, dtype=int)
    for k, seg in enumerate(segs):
        states[seg['start']:seg['end']] = seg['state']
        segment_id[seg['start']:seg['end']] = k
    return flips, segs, states, segment_id

def ordered_match(ref, det, tol):
    """按原脚本思路，使用动态规划做有序全局最优匹配。

    目标函数分两层：
    1. 先最大化匹配对数；
    2. 若匹配对数相同，再最小化总绝对年龄误差。

    这比简单的顺序贪心匹配更接近原始 workflow 的行为。
    """
    ref = np.asarray(ref, float)
    det = np.asarray(det, float)

    @lru_cache(None)
    def dp(i: int, j: int):
        if i == len(ref) or j == len(det):
            return (0, 0.0, ())

        best = dp(i + 1, j)
        best = (best[0], best[1], ('skip_ref',) + best[2])

        cand = dp(i, j + 1)
        cand = (cand[0], cand[1], ('skip_det',) + cand[2])
        if cand[0] > best[0] or (cand[0] == best[0] and cand[1] < best[1]):
            best = cand

        d = abs(ref[i] - det[j])
        if d <= tol:
            cand = dp(i + 1, j + 1)
            cand = (cand[0] + 1, cand[1] + d, ('match',) + cand[2])
            if cand[0] > best[0] or (cand[0] == best[0] and cand[1] < best[1]):
                best = cand

        return best

    _, _, ops = dp(0, 0)
    i = j = 0
    matches = []
    for op in ops:
        if op == 'match':
            matches.append((i, j, abs(ref[i] - det[j])))
            i += 1
            j += 1
        elif op == 'skip_ref':
            i += 1
        else:
            j += 1
    return matches

def metric_dict(ref, det, tol):
    matches = ordered_match(ref, det, tol)
    n_match = len(matches)
    recall = n_match / len(ref) if len(ref) else np.nan
    precision = n_match / len(det) if len(det) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    med_abs_err = float(np.median([m[2] for m in matches])) if matches else np.nan
    mean_abs_err = float(np.mean([m[2] for m in matches])) if matches else np.nan
    return {'n_match':int(n_match),'recall':float(recall),'precision':float(precision),'f1':float(f1),'median_abs_age_error_yr':med_abs_err,'mean_abs_age_error_yr':mean_abs_err,'matches':matches}

def build_match_tables(ref_transitions, det_transitions, matches):
    ref_table = pd.DataFrame({'ref_idx':np.arange(len(ref_transitions)),'ref_age_yrBP':ref_transitions})
    det_table = pd.DataFrame({'det_idx':np.arange(len(det_transitions)),'det_age_yrBP':det_transitions})
    rows = []
    for i, j, err in matches:
        rows.append({'ref_idx':i,'ref_age_yrBP':ref_transitions[i],'det_idx':j,'det_age_yrBP':det_transitions[j],'signed_age_error_yr':det_transitions[j]-ref_transitions[i],'abs_age_error_yr':err})
    match_df = pd.DataFrame(rows)
    unmatched_ref = ref_table[~ref_table['ref_idx'].isin(match_df['ref_idx'])].copy() if not match_df.empty else ref_table.copy()
    unmatched_det = det_table[~det_table['det_idx'].isin(match_df['det_idx'])].copy() if not match_df.empty else det_table.copy()
    return ref_table, det_table, match_df, unmatched_ref, unmatched_det

def cumulative_count_curve(all_ages, transitions):
    return np.searchsorted(np.sort(np.asarray(transitions)), all_ages, side='right')

def save_table(df, name):
    df.to_csv(TAB_DIR / name, index=False)

def compute_grid(cal_age, cal_resid, cal_median):
    rows = []
    for amp_sigma in AMP_GRID:
        for min_persistence in PERSIST_GRID:
            flips, _, _, _ = apply_detector(cal_age, cal_resid, cal_median, float(amp_sigma), float(min_persistence), DT_DETECT)
            for tol in TOLS:
                metrics = metric_dict(REF_TRANSITIONS, flips, tol)
                rows.append({'dt_yr':DT_DETECT,'amp_sigma':float(amp_sigma),'min_persistence_yr':float(min_persistence),'tol_yr':tol,'n_detected':len(flips),'n_ref':len(REF_TRANSITIONS),'n_matched':metrics['n_match'],'recall':metrics['recall'],'precision':metrics['precision'],'f1':metrics['f1'],'median_abs_age_error_yr':metrics['median_abs_age_error_yr'],'mean_abs_age_error_yr':metrics['mean_abs_age_error_yr']})
    grid = pd.DataFrame(rows)
    grid_wide = grid.pivot_table(index=['amp_sigma','min_persistence_yr'], columns='tol_yr', values=['f1','n_detected','median_abs_age_error_yr','precision','recall'])
    grid_wide.columns = [f'{a}_{b}' for a,b in grid_wide.columns]
    grid_wide = grid_wide.reset_index()
    grid_wide['score_primary'] = grid_wide['f1_300'] + 0.7*grid_wide['f1_200'] + 0.4*grid_wide['f1_100'] - 0.0004*grid_wide['median_abs_age_error_yr_300'].fillna(1000)
    return grid, grid_wide

def select_best_params(grid_wide):
    return grid_wide.sort_values(['score_primary','f1_300','precision_300','recall_300','f1_200','f1_100'], ascending=[False]*6).iloc[0]

def summarize_proxy(proxy):
    pre = load_selected_series(proxy['selected_csv'])
    full_age_50 = pre['age_yrBP'].to_numpy(float)
    full_resid_50 = pre['filtered_value'].to_numpy(float)
    cal_age, cal_resid = crop_to_ref_window(full_age_50, full_resid_50)
    cal_median = float(np.median(cal_resid))
    grid, grid_wide = compute_grid(cal_age, cal_resid, cal_median)
    best = select_best_params(grid_wide)
    best_amp = float(best['amp_sigma'])
    best_persist = float(best['min_persistence_yr'])
    sel_flips, sel_segs, sel_states, sel_segment_id = apply_detector(cal_age, cal_resid, cal_median, best_amp, best_persist, DT_DETECT)
    base_flips, _, base_states, _ = apply_detector(cal_age, cal_resid, cal_median, 0.0, 0.0, DT_DETECT)
    sel_metrics = {tol: metric_dict(REF_TRANSITIONS, sel_flips, tol) for tol in TOLS}
    base_metrics = {tol: metric_dict(REF_TRANSITIONS, base_flips, tol) for tol in TOLS}
    ref_table, det_table, match_df, unmatched_ref, unmatched_det = build_match_tables(REF_TRANSITIONS, sel_flips, sel_metrics[SELECTED_TOL]['matches'])
    full_median = float(np.median(full_resid_50))
    full_flips, full_segs, full_states, full_segment_id = apply_detector(full_age_50, full_resid_50, full_median, best_amp, best_persist, DT_DETECT)
    return {'proxy':proxy,'selected_precomputed':pre,'series':{'ages_50':full_age_50,'value_hp_50':full_resid_50},'cal_age':cal_age,'cal_resid':cal_resid,'cal_median':cal_median,'grid':grid,'grid_wide':grid_wide,'best':best,'best_amp':best_amp,'best_persist':best_persist,'sel_flips':sel_flips,'sel_segs':sel_segs,'sel_states':sel_states,'sel_segment_id':sel_segment_id,'base_flips':base_flips,'base_states':base_states,'sel_metrics':sel_metrics,'base_metrics':base_metrics,'ref_table':ref_table,'det_table':det_table,'match_df':match_df,'unmatched_ref':unmatched_ref,'unmatched_det':unmatched_det,'full_age_50':full_age_50,'full_resid_50':full_resid_50,'full_flips':full_flips,'full_segs':full_segs,'full_states':full_states,'full_segment_id':full_segment_id}

def plot_heatmaps_combined(results_by_proxy):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    for ax, key, lab in zip(axes, ['ch4','monsoon'], ['a','b']):
        res = results_by_proxy[key]
        piv = res['grid_wide'].pivot(index='min_persistence_yr', columns='amp_sigma', values='f1_300')
        im = ax.imshow(piv.values, aspect='auto', origin='lower')
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels([f'{c:.2f}' for c in piv.columns], rotation=45, ha='right')
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels([str(int(v)) for v in piv.index])
        ax.set_xlabel('Amplitude threshold (σ)')
        ax.set_ylabel('Minimum persistence (yr)')
        x = list(piv.columns).index(res['best_amp']); y = list(piv.index).index(res['best_persist'])
        ax.scatter([x],[y], s=90, facecolor='none', edgecolor='white', linewidth=1.8, zorder=3)
        ax.scatter([x],[y], s=22, color='white', zorder=3)
        ax.text(0.03,0.97,res['proxy']['display'], transform=ax.transAxes, va='top', ha='left')
        add_panel_label(ax, lab)
        cb = fig.colorbar(im, ax=ax, pad=0.02)
        cb.set_label('F1 score at ±300 yr')
    save_fig(fig, 'fig01_calibration_heatmaps_ch4_monsoon.pdf')

def _plot_series_pair_group(ax_main, ax_pair, res, panel_label):
    x = res['cal_age'] / 1000.0
    y = res['cal_resid']
    color = res['proxy']['state_color']

    ax_main.plot(x, y, color='0.15', linewidth=1.0)
    ax_main.axhline(res['cal_median'], color='0.5', linestyle='--', linewidth=0.9)
    y0, y1 = np.nanmin(y), np.nanmax(y)
    yr = y1 - y0 if np.isfinite(y1 - y0) and (y1 - y0) > 0 else 1.0
    y_marker = y0 + 0.08 * yr
    y_top = y1 - 0.02 * yr
    for t in res['sel_flips']:
        xt = t / 1000.0
        ax_main.vlines(xt, y_marker, y_top, color=color, alpha=0.20, linewidth=0.8)
        ax_main.plot(xt, y_marker, marker='o', markersize=3.2, color=color, markeredgewidth=0)

    # ax_main.text(0.03, 0.97, res['proxy']['display'], transform=ax_main.transAxes, va='top', ha='left')
    ax_main.set_ylabel(res['proxy']['line_label'])
    add_panel_label(ax_main, panel_label, x=-0.05)
    ax_main.tick_params(axis='x', labelbottom=False)

    ax_pair.set_ylim(0, 1)
    ax_pair.set_yticks([])
    ax_pair.spines['top'].set_visible(False)
    ax_pair.spines['right'].set_visible(False)
    ax_pair.spines['left'].set_visible(False)
    ax_pair.tick_params(axis='x', labelbottom=False)

    y_det = 0.82
    y_ref = 0.18
    for _, row in res['match_df'].iterrows():
        tr = row['ref_age_yrBP'] / 1000.0
        td = row['det_age_yrBP'] / 1000.0
        ax_pair.plot([td, tr], [y_det, y_ref], color='tab:green', alpha=0.45, linewidth=1.0)
        ax_pair.plot(td, y_det, marker='|', color=color, markersize=10, markeredgewidth=1.7)
        ax_pair.plot(tr, y_ref, marker='|', color='tab:blue', markersize=10, markeredgewidth=1.7)
    if not res['unmatched_det'].empty:
        ax_pair.scatter(res['unmatched_det']['det_age_yrBP'] / 1000.0, np.full(len(res['unmatched_det']), y_det), s=18, color=color)
    if not res['unmatched_ref'].empty:
        ax_pair.scatter(res['unmatched_ref']['ref_age_yrBP'] / 1000.0, np.full(len(res['unmatched_ref']), y_ref), s=18, color='tab:blue')

    ax_pair.text(-0.05, 0.92, f"{res['proxy']['display']}", transform=ax_pair.transAxes, color=color, fontsize=9, ha='left', va='top')
    ax_pair.text(-0.05, 0.08, 'NGRIP', transform=ax_pair.transAxes, color='tab:blue', fontsize=9, ha='left', va='bottom')


def plot_sequence_overview_combined(results_by_proxy):
    ch4 = results_by_proxy['ch4']
    mon = results_by_proxy['monsoon']

    fig = plt.figure(figsize=(11.2, 12.2))
    outer = fig.add_gridspec(5, 1, height_ratios=[2.8, 2.8, 0.95, 0.95, 0.95], hspace=0.26)

    group_a = outer[0].subgridspec(2, 1, height_ratios=[2.35, 0.78], hspace=0.02)
    axa = fig.add_subplot(group_a[0, 0])
    axa_pair = fig.add_subplot(group_a[1, 0], sharex=axa)

    group_b = outer[1].subgridspec(2, 1, height_ratios=[2.35, 0.78], hspace=0.02)
    axb = fig.add_subplot(group_b[0, 0], sharex=axa)
    axb_pair = fig.add_subplot(group_b[1, 0], sharex=axa)

    axc = fig.add_subplot(outer[2, 0], sharex=axa)
    axd = fig.add_subplot(outer[3, 0], sharex=axa)
    axe = fig.add_subplot(outer[4, 0], sharex=axa)

    _plot_series_pair_group(axa, axa_pair, ch4, 'a')
    _plot_series_pair_group(axb, axb_pair, mon, 'b')

    ref_state_on_cal = REF_STATE[np.searchsorted(REF_AGES, ch4['cal_age'])]
    axc.step(ch4['cal_age'] / 1000.0, ref_state_on_cal, where='post', color='tab:blue', linewidth=1.5)
    for t in REF_TRANSITIONS:
        axc.axvline(t / 1000.0, color='tab:blue', alpha=0.24, linewidth=0.8)
    axc.set_ylim(-0.2, 1.2)
    axc.set_yticks([0, 1])
    axc.set_yticklabels(['cold', 'warm'])
    axc.set_ylabel('NGRIP')
    add_panel_label(axc, 'c', x=-0.05)
    axc.tick_params(axis='x', labelbottom=False)

    for ax, res, lab in [(axd, ch4, 'd'), (axe, mon, 'e')]:
        ax.step(res['cal_age'] / 1000.0, res['sel_states'], where='post', color=res['proxy']['state_color'], linewidth=1.5)
        for t in res['sel_flips']:
            ax.axvline(t / 1000.0, color=res['proxy']['state_color'], alpha=0.24, linewidth=0.8)
        ax.set_ylim(-0.2, 1.2)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['low', 'high'])
        ax.set_ylabel(res['proxy']['state_label'])
        add_panel_label(ax, lab, x=-0.05)
        if ax is not axe:
            ax.tick_params(axis='x', labelbottom=False)

    axe.set_xlabel('Age (kyr BP)')
    axa.set_xlim(12, 119)
    save_fig(fig, 'fig02_sequence_overview_ch4_monsoon_combined.pdf')

def plot_pairing_summary_generic(res, out_name):
    fig = plt.figure(figsize=(11.8, 8.0)); gs = fig.add_gridspec(2,2,hspace=0.30,wspace=0.28)
    ax = fig.add_subplot(gs[0,0]); band_x=np.array([12,119]); ax.plot(band_x,band_x,color='0.25',linewidth=1.1); ax.fill_between(band_x, band_x-SELECTED_TOL/1000, band_x+SELECTED_TOL/1000, color='0.88', zorder=0)
    if not res['match_df'].empty:
        sc=ax.scatter(res['match_df']['ref_age_yrBP']/1000,res['match_df']['det_age_yrBP']/1000,c=res['match_df']['abs_age_error_yr'],s=30)
        cb=fig.colorbar(sc, ax=ax, pad=0.02); cb.set_label('|Age error| (yr)')
    ax.set_xlabel('NGRIP transition age (kyr BP)'); ax.set_ylabel(f"{res['proxy']['display']} detected age (kyr BP)"); ax.set_xlim(12,119); ax.set_ylim(12,119); add_panel_label(ax,'a')
    ax = fig.add_subplot(gs[0,1]); ax.step(res['cal_age']/1000, cumulative_count_curve(res['cal_age'], REF_TRANSITIONS), where='post', linewidth=1.6, label='NGRIP reference'); ax.step(res['cal_age']/1000, cumulative_count_curve(res['cal_age'], res['sel_flips']), where='post', linewidth=1.6, label=f"{res['proxy']['display']} detector"); ax.set_xlabel('Age (kyr BP)'); ax.set_ylabel('Cumulative transitions'); ax.legend(frameon=False, loc='upper left'); ax.set_xlim(12,119); add_panel_label(ax,'b')
    ax = fig.add_subplot(gs[1,0]);
    if not res['match_df'].empty:
        ax.axhline(0, color='0.5', linestyle='--', linewidth=1); ax.scatter(res['match_df']['ref_age_yrBP']/1000, res['match_df']['signed_age_error_yr'], s=28)
    ax.set_xlabel('NGRIP transition age (kyr BP)'); ax.set_ylabel('Detected - reference (yr)'); ax.set_xlim(12,119); 
    add_panel_label(ax,'c')
    ax = fig.add_subplot(gs[1,1]); xt=np.arange(len(TOLS)); width=0.36; base_f1=[res['base_metrics'][t]['f1'] for t in TOLS]; sel_f1=[res['sel_metrics'][t]['f1'] for t in TOLS]; base_p=[res['base_metrics'][t]['precision'] for t in TOLS]; sel_p=[res['sel_metrics'][t]['precision'] for t in TOLS]; ax.bar(xt-width/2,base_f1,width=width,label='Baseline F1',alpha=0.9); ax.bar(xt+width/2,sel_f1,width=width,label='Selected F1',alpha=0.9); ax.plot(xt-width/2,base_p,marker='o',linestyle='--',linewidth=1.1,label='Baseline precision'); ax.plot(xt+width/2,sel_p,marker='s',linestyle='--',linewidth=1.1,label='Selected precision'); ax.set_xticks(xt); ax.set_xticklabels([f'±{t}' for t in TOLS]); ax.set_ylim(0,1); ax.set_xlabel('Matching tolerance (yr)'); ax.set_ylabel('Score'); ax.legend(frameon=False,fontsize=8,loc='lower right'); add_panel_label(ax,'d')
    save_fig(fig, out_name)

def plot_windowed_comparison_generic(res, out_name):
    windows=[(12,30),(30,50),(50,70),(70,90),(90,119)]
    fig, axes = plt.subplots(len(windows),1,figsize=(12.2,9.8),sharey=True)
    for ax,(x0,x1),lab in zip(axes,windows,list('abcde')):
        ref_in = REF_TRANSITIONS[(REF_TRANSITIONS/1000>=x0)&(REF_TRANSITIONS/1000<=x1)]
        det_in = res['sel_flips'][(res['sel_flips']/1000>=x0)&(res['sel_flips']/1000<=x1)]
        for t in ref_in: ax.plot([t/1000,t/1000],[0.58,0.98],color='tab:blue',linewidth=1.4)
        for t in det_in: ax.plot([t/1000,t/1000],[0.02,0.42],color=res['proxy']['state_color'],linewidth=1.4)
        subm = res['match_df'][((res['match_df']['ref_age_yrBP']/1000).between(x0-0.5,x1+0.5))|((res['match_df']['det_age_yrBP']/1000).between(x0-0.5,x1+0.5))]
        for _, row in subm.iterrows():
            tr=row['ref_age_yrBP']/1000; td=row['det_age_yrBP']/1000
            if (x0-0.5<=tr<=x1+0.5) and (x0-0.5<=td<=x1+0.5): ax.plot([tr,td],[0.78,0.22],color='tab:green',alpha=0.55,linewidth=1.0)
        ax.text(x0+0.2,0.90,'NGRIP',color='tab:blue',fontsize=9); ax.text(x0+0.2,0.06,res['proxy']['display'],color=res['proxy']['state_color'],fontsize=9); ax.text(0.99,0.92,f'{x0}-{x1} kyr BP', transform=ax.transAxes, ha='right', va='top', fontsize=10); ax.set_xlim(x0,x1); ax.set_ylim(0,1); ax.set_yticks([]); add_panel_label(ax,lab,x=-0.08,y=1.01)
    axes[-1].set_xlabel('Age (kyr BP)'); save_fig(fig, out_name)

def plot_full_records_combined(results_by_proxy):
    ch4=results_by_proxy['ch4']; mon=results_by_proxy['monsoon']
    fig = plt.figure(figsize=(12.2, 9.6)); gs = fig.add_gridspec(4,1,height_ratios=[2.0,0.85,2.0,0.85],hspace=0.12)
    ax0=fig.add_subplot(gs[0,0]); ax0.plot(ch4['full_age_50']/1000,ch4['full_resid_50'],color='0.15',linewidth=0.9); ax0.axvspan(AGE_MIN/1000,AGE_MAX/1000,color='0.94',zorder=0); [ax0.axvline(t/1000,color=ch4['proxy']['state_color'],alpha=0.10,linewidth=0.6) for t in ch4['full_flips']]; ax0.set_ylabel(ch4['proxy']['line_label']); ax0.text(0.03,0.97,ch4['proxy']['display'],transform=ax0.transAxes,va='top',ha='left'); add_panel_label(ax0,'a',x=-0.07); ax0.set_xlim(ch4['full_age_50'].min()/1000,ch4['full_age_50'].max()/1000); ax0.tick_params(axis='x',labelbottom=False)
    ax1=fig.add_subplot(gs[1,0],sharex=ax0); ax1.step(ch4['full_age_50']/1000,ch4['full_states'],where='post',color=ch4['proxy']['state_color'],linewidth=1.4); ax1.axvspan(AGE_MIN/1000,AGE_MAX/1000,color='0.94',zorder=0); ax1.set_ylim(-0.2,1.2); ax1.set_yticks([0,1]); ax1.set_yticklabels(['low','high']); ax1.set_ylabel(ch4['proxy']['state_label']); add_panel_label(ax1,'b',x=-0.07); ax1.tick_params(axis='x',labelbottom=False)
    ax2=fig.add_subplot(gs[2,0],sharex=ax0); ax2.plot(mon['full_age_50']/1000,mon['full_resid_50'],color='0.15',linewidth=0.9); ax2.axvspan(AGE_MIN/1000,AGE_MAX/1000,color='0.94',zorder=0); [ax2.axvline(t/1000,color=mon['proxy']['state_color'],alpha=0.10,linewidth=0.6) for t in mon['full_flips']]; ax2.set_ylabel(mon['proxy']['line_label']); ax2.text(0.03,0.97,mon['proxy']['display'],transform=ax2.transAxes,va='top',ha='left'); add_panel_label(ax2,'c',x=-0.07); ax2.tick_params(axis='x',labelbottom=False)
    ax3=fig.add_subplot(gs[3,0],sharex=ax0); ax3.step(mon['full_age_50']/1000,mon['full_states'],where='post',color=mon['proxy']['state_color'],linewidth=1.4); ax3.axvspan(AGE_MIN/1000,AGE_MAX/1000,color='0.94',zorder=0); ax3.set_ylim(-0.2,1.2); ax3.set_yticks([0,1]); ax3.set_yticklabels(['low','high']); ax3.set_ylabel(mon['proxy']['state_label']); ax3.set_xlabel('Age (kyr BP)'); add_panel_label(ax3,'d',x=-0.07)
    save_fig(fig, 'fig07_full_records_square_wave.pdf')

def main():
    setup_style()
    results_by_proxy = {}
    summary_rows = []
    for proxy in PROXIES:
        res = summarize_proxy(proxy)
        results_by_proxy[proxy['name']] = res
        save_table(res['grid'], f"detector_grid_{proxy['name']}.csv")
        save_table(res['grid_wide'], f"detector_grid_wide_{proxy['name']}.csv")
        save_table(res['match_df'], f"selected_matches_tol300_{proxy['name']}.csv")
        save_table(res['unmatched_ref'], f"selected_unmatched_reference_tol300_{proxy['name']}.csv")
        save_table(res['unmatched_det'], f"selected_extra_detected_tol300_{proxy['name']}.csv")
        save_table(res['det_table'], f"selected_detected_flips_{proxy['name']}.csv")
        metric_rows=[]
        for det_name, metrics_by_tol, flips in [('baseline_crossing', res['base_metrics'], res['base_flips']), (f"selected_{res['best_amp']:.2f}_{int(res['best_persist'])}yr", res['sel_metrics'], res['sel_flips'])]:
            for tol in TOLS:
                metric_rows.append({'detector':det_name,'tol_yr':tol,'n_detected':len(flips),'n_matched':metrics_by_tol[tol]['n_match'],'precision':metrics_by_tol[tol]['precision'],'recall':metrics_by_tol[tol]['recall'],'f1':metrics_by_tol[tol]['f1'],'median_abs_age_error_yr':metrics_by_tol[tol]['median_abs_age_error_yr'],'mean_abs_age_error_yr':metrics_by_tol[tol]['mean_abs_age_error_yr']})
        save_table(pd.DataFrame(metric_rows), f"baseline_vs_selected_{proxy['name']}.csv")
        full_summary = pd.DataFrame([{'proxy':proxy['name'],'amp_sigma':res['best_amp'],'min_persistence_yr':res['best_persist'],'age_min_yrBP':int(res['full_age_50'].min()),'age_max_yrBP':int(res['full_age_50'].max()),'n_transitions_calibration':len(res['sel_flips']),'n_segments_calibration':len(res['sel_segs']),'n_matched_tol300':res['sel_metrics'][300]['n_match'],'precision_tol300':res['sel_metrics'][300]['precision'],'recall_tol300':res['sel_metrics'][300]['recall'],'f1_tol300':res['sel_metrics'][300]['f1'],'median_abs_age_error_tol300_yr':res['sel_metrics'][300]['median_abs_age_error_yr'],'mean_abs_age_error_tol300_yr':res['sel_metrics'][300]['mean_abs_age_error_yr'],'n_transitions_full':len(res['full_flips']),'n_segments_full':len(res['full_segs'])}])
        save_table(full_summary, f"selected_detector_summary_{proxy['name']}.csv")
        summary_rows.extend(full_summary.to_dict(orient='records'))
        print('done summarize', proxy['name'], res['best_amp'], res['best_persist'])
    save_table(pd.DataFrame(summary_rows), 'selected_detector_summary_all_proxies.csv')
    plot_heatmaps_combined(results_by_proxy); print('fig1')
    plot_sequence_overview_combined(results_by_proxy); print('fig2')
    plot_pairing_summary_generic(results_by_proxy['ch4'], 'fig03_ch4_pairing_summary.pdf'); print('fig3')
    plot_windowed_comparison_generic(results_by_proxy['ch4'], 'fig04_ch4_windowed_event_comparison.pdf'); print('fig4')
    plot_pairing_summary_generic(results_by_proxy['monsoon'], 'fig05_monsoon_pairing_summary.pdf'); print('fig5')
    plot_windowed_comparison_generic(results_by_proxy['monsoon'], 'fig06_monsoon_windowed_event_comparison.pdf'); print('fig6')
    plot_full_records_combined(results_by_proxy); print('fig7')

if __name__ == '__main__':
    main()
