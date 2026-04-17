from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from functools import lru_cache

# ============================================================
# 基本路径设置
# - 优先在脚本同级目录下寻找 data/ 与 results/
# - 若输入 Excel 不在 data/ 中，也允许从脚本同级目录直接读取
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
BASE = SCRIPT_DIR / 'data'
OUT = SCRIPT_DIR / 'results'
FIG_DIR = OUT / 'figures'
TAB_DIR = OUT / 'tables'

BASE.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. Greenland reference framework
# ============================================================
GS_START = np.array([
    11703, 14692, 23020, 23340, 27780, 28900, 30840, 32500, 33740, 35480,
    38220, 40160, 41460, 43340, 46860, 49280, 54220, 55000, 55800, 58040,
    58280, 59080, 59440, 64100, 69620, 72340, 76440, 84760, 85060, 90040,
    104040, 104520, 106750, 108280, 115370
])
GS_END = np.array([
    12896, 22900, 23220, 27540, 28600, 30600, 32040, 33360, 34740, 36580,
    39900, 40800, 42240, 44280, 48340, 49600, 54900, 55400, 56500, 58160,
    58560, 59300, 63840, 69400, 70380, 74100, 77760, 84960, 87600, 90140,
    104380, 105440, 106900, 110640, 119140
])
AGE_MIN, AGE_MAX = 11703, 119140
REF_AGES = np.arange(AGE_MIN, AGE_MAX + 1, 1)
REF_STATE = np.ones_like(REF_AGES, dtype=int)
for s, e in zip(GS_START, GS_END):
    REF_STATE[(REF_AGES >= s) & (REF_AGES <= e)] = 0
REF_TRANSITIONS = REF_AGES[np.diff(REF_STATE, prepend=REF_STATE[0]) != 0]


# ============================================================
# 2. User-selected detector parameters
# ============================================================
SELECTED_AMP = 0.27
SELECTED_PERSIST = 100
SELECTED_TOL = 300

DT_INPUT = 10          # 10 yr interpolation for filtering backbone
DT_DETECT = 50         # detector resolution
HP_CUTOFF = 10_000.0   # yr, high-pass cutoff period

AMP_GRID = sorted(set([0.0, 0.1, 0.25, 0.5, 0.75] + [round(x, 2) for x in np.arange(0.20, 0.301, 0.01)]))
PERSIST_GRID = sorted(set([0, 100, 200, 300, 400, 500] + list(range(80, 151, 10))))
TOLS = [100, 200, 300]


# ============================================================
# 3. Utilities
# ============================================================
def resolve_input_file(filename: str) -> Path:
    candidates = [BASE / filename, SCRIPT_DIR / filename]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f'Cannot find {filename}. Expected in either {BASE} or {SCRIPT_DIR}.'
    )


def load_ch4() -> pd.DataFrame:
    xlsx = resolve_input_file('CH4_AICC2023.xlsx')
    raw = pd.read_excel(xlsx)
    required = {'age', 'ch4'}
    if not required.issubset(raw.columns):
        raise ValueError(f'Input file must contain columns {required}; got {list(raw.columns)}')
    raw = raw[['age', 'ch4']].dropna().sort_values('age').reset_index(drop=True)
    return raw


def build_filtered_series(raw: pd.DataFrame):
    # 原始数据中 age 可从负值开始，这里保留 >=1000 yr BP 的主分析区间
    raw_use = raw[raw['age'] >= 1000].copy()

    ages_10 = np.arange(1000, int(np.floor(raw_use['age'].max())) + 1, DT_INPUT)
    ch4_10 = np.interp(ages_10, raw_use['age'].to_numpy(), raw_use['ch4'].to_numpy())

    fs = 1.0 / DT_INPUT
    fc = 1.0 / HP_CUTOFF
    wn = fc / (fs * 0.5)
    b, a = butter(4, wn, btype='highpass')
    ch4_hp_10 = filtfilt(b, a, ch4_10)

    ages_50 = np.arange(int(ages_10.min()), int(ages_10.max()) + 1, DT_DETECT)
    ch4_hp_50 = np.interp(ages_50, ages_10, ch4_hp_10)

    return {
        'raw_use': raw_use,
        'ages_10': ages_10,
        'ch4_10': ch4_10,
        'ch4_hp_10': ch4_hp_10,
        'ages_50': ages_50,
        'ch4_hp_50': ch4_hp_50,
    }


def crop_to_ref_window(ages: np.ndarray, values: np.ndarray):
    mask = (ages > AGE_MIN) & (ages < AGE_MAX)
    return ages[mask], values[mask]


def make_segments(age: np.ndarray, value: np.ndarray, med: float, dt_yr: int):
    state = (value >= med).astype(int)
    idx = np.flatnonzero(np.diff(state) != 0) + 1
    starts = np.r_[0, idx]
    ends = np.r_[idx, len(state)]
    segs = []
    for s, e in zip(starts, ends):
        segs.append({
            'start': int(s),
            'end': int(e),
            'state': int(state[s]),
            'dur_yr': float(age[e - 1] - age[s] + dt_yr),
            'amp': float(np.max(np.abs(value[s:e] - med))),
        })
    return segs


def merge_segment_triplet(left, mid, right):
    return {
        'start': left['start'],
        'end': right['end'],
        'state': left['state'],
        'dur_yr': left['dur_yr'] + mid['dur_yr'] + right['dur_yr'],
        'amp': max(left['amp'], mid['amp'], right['amp']),
    }


def apply_detector(age: np.ndarray, value: np.ndarray, med: float,
                   amp_sigma: float = 0.0,
                   min_persistence_yr: float = 0.0,
                   dt_yr: int = 50):
    segs = make_segments(age, value, med, dt_yr)
    amp_thr = amp_sigma * np.std(value)

    changed = True
    while changed and len(segs) > 1:
        changed = False
        for i, seg in enumerate(segs):
            too_small_amp = seg['amp'] < amp_thr
            too_short = seg['dur_yr'] < min_persistence_yr
            if not (too_small_amp or too_short):
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
                segs = segs[:i - 1] + [merge_segment_triplet(segs[i - 1], seg, segs[i + 1])] + segs[i + 2:]
            break

    flips = np.array([age[s['start']] for s in segs[1:]], dtype=float)
    states = np.zeros_like(age, dtype=int)
    segment_id = np.zeros_like(age, dtype=int)
    for k, seg in enumerate(segs):
        states[seg['start']:seg['end']] = seg['state']
        segment_id[seg['start']:seg['end']] = k
    return flips, segs, states, segment_id


# ordered one-to-one matching
@lru_cache(None)
def _noop():
    return None


def ordered_match(ref: np.ndarray, det: np.ndarray, tol: float):
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
    return {
        'n_match': int(n_match),
        'recall': float(recall),
        'precision': float(precision),
        'f1': float(f1),
        'median_abs_age_error_yr': med_abs_err,
        'mean_abs_age_error_yr': mean_abs_err,
        'matches': matches,
    }


def build_match_tables(ref_transitions, det_transitions, matches):
    ref_table = pd.DataFrame({'ref_idx': np.arange(len(ref_transitions)), 'ref_age_yrBP': ref_transitions})
    det_table = pd.DataFrame({'det_idx': np.arange(len(det_transitions)), 'det_age_yrBP': det_transitions})
    match_rows = []
    for i, j, err in matches:
        match_rows.append({
            'ref_idx': i,
            'ref_age_yrBP': ref_transitions[i],
            'det_idx': j,
            'det_age_yrBP': det_transitions[j],
            'signed_age_error_yr': det_transitions[j] - ref_transitions[i],
            'abs_age_error_yr': err,
        })
    match_df = pd.DataFrame(match_rows)
    unmatched_ref = ref_table[~ref_table['ref_idx'].isin(match_df['ref_idx'])].copy() if not match_df.empty else ref_table.copy()
    unmatched_det = det_table[~det_table['det_idx'].isin(match_df['det_idx'])].copy() if not match_df.empty else det_table.copy()
    return ref_table, det_table, match_df, unmatched_ref, unmatched_det


def cumulative_count_curve(all_ages, transitions):
    transitions = np.asarray(transitions)
    return np.searchsorted(np.sort(transitions), all_ages, side='right')


def save_table(df: pd.DataFrame, name: str):
    path = TAB_DIR / name
    df.to_csv(path, index=False)
    return path


def save_square_wave_to_data(ages, values, states, segment_id, flips, out_name: str):
    flip_set = set(np.asarray(flips, int).tolist())
    out = pd.DataFrame({
        'age_yrBP': ages.astype(int),
        'ch4_hp': values,
        'mcv_state': states.astype(int),
        'segment_id': segment_id.astype(int),
        'is_transition_age': np.array([1 if int(a) in flip_set else 0 for a in ages], dtype=int),
    })
    out_path = BASE / out_name
    out.to_csv(out_path, index=False)
    pd.DataFrame({'transition_age_yrBP': np.asarray(flips, int)}).to_csv(
        BASE / out_name.replace('.csv', '_transitions.csv'), index=False
    )
    return out_path


# ============================================================
# 4. Plot helpers
# ============================================================
def setup_style():
    plt.rcParams.update({
        'font.size': 10,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': False,
    })


def plot_heatmap(grid_wide: pd.DataFrame):
    sub = grid_wide.copy()
    piv = sub.pivot(index='min_persistence_yr', columns='amp_sigma', values='f1_300')
    fig, ax = plt.subplots(figsize=(8.6, 5.9))
    im = ax.imshow(piv.values, aspect='auto', origin='lower')
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([f'{c:.2f}' for c in piv.columns], rotation=45, ha='right')
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([str(int(v)) for v in piv.index])
    ax.set_xlabel('Amplitude threshold (σ)')
    ax.set_ylabel('Minimum persistence (yr)')
    ax.set_title('Calibration grid on the 11.7–119.1 ka window\n(F1 at ±300 yr)')

    # 高亮用户选择参数
    x = list(piv.columns).index(SELECTED_AMP)
    y = list(piv.index).index(SELECTED_PERSIST)
    ax.scatter([x], [y], s=90, facecolor='none', edgecolor='white', linewidth=1.8, zorder=3)
    ax.scatter([x], [y], s=22, color='white', zorder=3)

    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label('F1 score')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig01_calibration_heatmap_selected.png', dpi=240)
    plt.close(fig)


def plot_sequence_overview(cal_age, cal_resid, cal_med, ref_states_on_cal, det_states, ref_trans, det_trans,
                           match_df, unmatched_ref, unmatched_det):
    fig = plt.figure(figsize=(12.6, 8.8))
    gs = fig.add_gridspec(4, 1, height_ratios=[2.4, 0.8, 0.8, 1.3], hspace=0.15)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(cal_age / 1000, cal_resid, color='0.15', linewidth=1.0)
    ax0.axhline(cal_med, color='0.5', linestyle='--', linewidth=1)
    for t in det_trans:
        ax0.axvline(t / 1000, color='tab:red', alpha=0.20, linewidth=0.9)
    ax0.set_ylabel('Filtered CH$_4$')
    ax0.set_title('Selected detector on filtered CH$_4$ residual, with reference and detected MCV sequences')

    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    ax1.step(cal_age / 1000, ref_states_on_cal, where='post', color='tab:blue', linewidth=1.6)
    for t in ref_trans:
        ax1.axvline(t / 1000, color='tab:blue', alpha=0.35, linewidth=1)
    ax1.set_ylim(-0.2, 1.2)
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['cold', 'warm'])
    ax1.set_ylabel('NGRIP')

    ax2 = fig.add_subplot(gs[2, 0], sharex=ax0)
    ax2.step(cal_age / 1000, det_states, where='post', color='tab:red', linewidth=1.6)
    for t in det_trans:
        ax2.axvline(t / 1000, color='tab:red', alpha=0.35, linewidth=1)
    ax2.set_ylim(-0.2, 1.2)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['cold', 'warm'])
    ax2.set_ylabel('CH$_4$')

    ax3 = fig.add_subplot(gs[3, 0], sharex=ax0)
    ax3.set_ylim(0, 1)
    ax3.set_yticks([])
    for _, row in match_df.iterrows():
        tr = row['ref_age_yrBP'] / 1000
        td = row['det_age_yrBP'] / 1000
        ax3.plot([tr, td], [0.80, 0.20], color='tab:green', alpha=0.45, linewidth=1.0)
        ax3.plot(tr, 0.80, marker='|', color='tab:blue', markersize=11, markeredgewidth=1.8)
        ax3.plot(td, 0.20, marker='|', color='tab:red', markersize=11, markeredgewidth=1.8)
    if not unmatched_ref.empty:
        ax3.scatter(unmatched_ref['ref_age_yrBP'] / 1000, np.full(len(unmatched_ref), 0.80),
                    s=18, color='tab:blue', label='unmatched NGRIP')
    if not unmatched_det.empty:
        ax3.scatter(unmatched_det['det_age_yrBP'] / 1000, np.full(len(unmatched_det), 0.20),
                    s=18, color='tab:red', label='extra CH$_4$ detection')
    ax3.text(12.2, 0.86, 'NGRIP transitions', color='tab:blue', fontsize=9)
    ax3.text(12.2, 0.06, 'CH$_4$ transitions', color='tab:red', fontsize=9)
    ax3.set_xlabel('Age (kyr BP)')
    ax3.set_ylabel('pairing')
    ax3.set_xlim(12, 119)

    for ax in [ax0, ax1, ax2, ax3]:
        ax.tick_params(axis='x', labelbottom=ax is ax3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig02_selected_sequence_overview.png', dpi=240)
    plt.close(fig)


def plot_pairing_summary(cal_age, ref_trans, det_trans, match_df, base_metrics, sel_metrics):
    fig = plt.figure(figsize=(12.0, 8.6))
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.25)

    # A. matched age scatter
    ax = fig.add_subplot(gs[0, 0])
    band_x = np.array([12, 119])
    ax.plot(band_x, band_x, color='0.2', linewidth=1.2)
    ax.fill_between(band_x, band_x - SELECTED_TOL / 1000, band_x + SELECTED_TOL / 1000,
                    color='0.85', zorder=0)
    if not match_df.empty:
        sc = ax.scatter(match_df['ref_age_yrBP'] / 1000, match_df['det_age_yrBP'] / 1000,
                        c=match_df['abs_age_error_yr'], s=34)
        cb = fig.colorbar(sc, ax=ax, pad=0.02)
        cb.set_label('|Age error| (yr)')
    ax.set_xlabel('NGRIP transition age (kyr BP)')
    ax.set_ylabel('CH$_4$ detected transition age (kyr BP)')
    ax.set_title('Matched events: how close are they to the 1:1 line?')
    ax.set_xlim(12, 119)
    ax.set_ylim(12, 119)

    # B. cumulative count curves
    ax = fig.add_subplot(gs[0, 1])
    cum_ref = cumulative_count_curve(cal_age, ref_trans)
    cum_det = cumulative_count_curve(cal_age, det_trans)
    ax.step(cal_age / 1000, cum_ref, where='post', linewidth=1.8, label='NGRIP reference')
    ax.step(cal_age / 1000, cum_det, where='post', linewidth=1.8, label='CH$_4$ selected detector')
    ax.set_xlabel('Age (kyr BP)')
    ax.set_ylabel('Cumulative number of transitions')
    ax.set_title('Cumulative event count through time')
    ax.legend(frameon=False)
    ax.set_xlim(12, 119)

    # C. error vs age
    ax = fig.add_subplot(gs[1, 0])
    if not match_df.empty:
        ax.axhline(0, color='0.5', linestyle='--', linewidth=1)
        ax.scatter(match_df['ref_age_yrBP'] / 1000, match_df['signed_age_error_yr'], s=28)
    ax.set_xlabel('NGRIP transition age (kyr BP)')
    ax.set_ylabel('Detected - reference (yr)')
    ax.set_title('Signed age error versus time')
    ax.set_xlim(12, 119)

    # D. baseline vs selected metrics
    ax = fig.add_subplot(gs[1, 1])
    xt = np.arange(len(TOLS))
    width = 0.36
    base_f1 = [base_metrics[t]['f1'] for t in TOLS]
    sel_f1 = [sel_metrics[t]['f1'] for t in TOLS]
    base_p = [base_metrics[t]['precision'] for t in TOLS]
    sel_p = [sel_metrics[t]['precision'] for t in TOLS]
    ax.bar(xt - width/2, base_f1, width=width, label='Baseline F1', alpha=0.9)
    ax.bar(xt + width/2, sel_f1, width=width, label='Selected F1', alpha=0.9)
    ax.plot(xt - width/2, base_p, marker='o', linestyle='--', linewidth=1.2, label='Baseline precision')
    ax.plot(xt + width/2, sel_p, marker='s', linestyle='--', linewidth=1.2, label='Selected precision')
    ax.set_xticks(xt)
    ax.set_xticklabels([f'±{t}' for t in TOLS])
    ax.set_ylim(0, 1)
    ax.set_xlabel('Matching tolerance (yr)')
    ax.set_ylabel('Score')
    ax.set_title('Selected detector improves agreement relative to baseline')
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig03_pairing_summary.png', dpi=240)
    plt.close(fig)


def plot_windowed_comparison(ref_trans, det_trans, match_df):
    windows = [(12, 30), (30, 50), (50, 70), (70, 90), (90, 119)]
    fig, axes = plt.subplots(len(windows), 1, figsize=(13.2, 10.8), sharey=True)
    for ax, (x0, x1) in zip(axes, windows):
        ref_in = ref_trans[(ref_trans / 1000 >= x0) & (ref_trans / 1000 <= x1)]
        det_in = det_trans[(det_trans / 1000 >= x0) & (det_trans / 1000 <= x1)]

        for t in ref_in:
            ax.plot([t / 1000, t / 1000], [0.58, 0.98], color='tab:blue', linewidth=1.5)
        for t in det_in:
            ax.plot([t / 1000, t / 1000], [0.02, 0.42], color='tab:red', linewidth=1.5)

        subm = match_df[
            ((match_df['ref_age_yrBP'] / 1000).between(x0 - 0.5, x1 + 0.5)) |
            ((match_df['det_age_yrBP'] / 1000).between(x0 - 0.5, x1 + 0.5))
        ]
        for _, row in subm.iterrows():
            tr = row['ref_age_yrBP'] / 1000
            td = row['det_age_yrBP'] / 1000
            if (x0 - 0.5 <= tr <= x1 + 0.5) and (x0 - 0.5 <= td <= x1 + 0.5):
                ax.plot([tr, td], [0.78, 0.22], color='tab:green', alpha=0.55, linewidth=1.1)

        ax.text(x0 + 0.2, 0.90, 'NGRIP', color='tab:blue', fontsize=9)
        ax.text(x0 + 0.2, 0.06, 'CH$_4$', color='tab:red', fontsize=9)
        ax.set_xlim(x0, x1)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_title(f'{x0}–{x1} kyr BP')

    axes[-1].set_xlabel('Age (kyr BP)')
    fig.suptitle('Window-by-window event comparison: matched pairs linked in green', y=0.995)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig04_windowed_event_comparison.png', dpi=240)
    plt.close(fig)


def plot_full_ch4(full_age_10, full_hp_10, full_age_50, full_state_50, full_flips):
    fig = plt.figure(figsize=(13.0, 7.8))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.5, 1.0], hspace=0.12)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(full_age_10 / 1000, full_hp_10, color='0.2', linewidth=0.8)
    ax0.axvspan(AGE_MIN / 1000, AGE_MAX / 1000, color='0.92', zorder=0)
    for t in full_flips:
        ax0.axvline(t / 1000, color='tab:red', alpha=0.10, linewidth=0.6)
    ax0.set_ylabel('Filtered CH$_4$')
    ax0.set_title('Selected MCV detector applied to the full filtered CH$_4$ record')
    ax0.set_xlim(full_age_10.min() / 1000, full_age_10.max() / 1000)

    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    ax1.step(full_age_50 / 1000, full_state_50, where='post', color='tab:red', linewidth=1.5)
    ax1.axvspan(AGE_MIN / 1000, AGE_MAX / 1000, color='0.92', zorder=0)
    ax1.set_ylim(-0.2, 1.2)
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['cold', 'warm'])
    ax1.set_xlabel('Age (kyr BP)')
    ax1.set_ylabel('MCV')

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig05_full_ch4_square_wave.png', dpi=240)
    plt.close(fig)


# ============================================================
# 5. Main workflow
# ============================================================
def main():
    setup_style()
    raw = load_ch4()
    series = build_filtered_series(raw)

    # --- Calibration window ---
    cal_age, cal_resid = crop_to_ref_window(series['ages_50'], series['ch4_hp_50'])
    cal_median = float(np.median(cal_resid))

    # grid scan on calibration window
    rows = []
    for amp_sigma in AMP_GRID:
        for min_persistence in PERSIST_GRID:
            flips, segs, states, segment_id = apply_detector(
                cal_age, cal_resid, cal_median,
                amp_sigma=amp_sigma,
                min_persistence_yr=min_persistence,
                dt_yr=DT_DETECT,
            )
            for tol in TOLS:
                metrics = metric_dict(REF_TRANSITIONS, flips, tol)
                rows.append({
                    'dt_yr': DT_DETECT,
                    'amp_sigma': amp_sigma,
                    'min_persistence_yr': min_persistence,
                    'tol_yr': tol,
                    'n_detected': len(flips),
                    'n_ref': len(REF_TRANSITIONS),
                    'n_matched': metrics['n_match'],
                    'recall': metrics['recall'],
                    'precision': metrics['precision'],
                    'f1': metrics['f1'],
                    'median_abs_age_error_yr': metrics['median_abs_age_error_yr'],
                    'mean_abs_age_error_yr': metrics['mean_abs_age_error_yr'],
                    'addition_ratio_vs_ref': (len(flips) / len(REF_TRANSITIONS) - 1) if len(REF_TRANSITIONS) else np.nan,
                })
    grid = pd.DataFrame(rows)
    save_table(grid, 'detector_grid.csv')

    grid_wide = grid.pivot_table(
        index=['amp_sigma', 'min_persistence_yr'],
        columns='tol_yr',
        values=['f1', 'n_detected', 'median_abs_age_error_yr', 'precision', 'recall']
    )
    grid_wide.columns = [f'{a}_{b}' for a, b in grid_wide.columns]
    grid_wide = grid_wide.reset_index()
    grid_wide['n_detected'] = grid_wide['n_detected_300']
    grid_wide['score_primary'] = (
        grid_wide['f1_300'] + 0.7 * grid_wide['f1_200'] + 0.4 * grid_wide['f1_100']
        - 0.0004 * grid_wide['median_abs_age_error_yr_300'].fillna(1000)
    )
    save_table(grid_wide, 'detector_grid_wide.csv')

    best_rows = []
    for tol in TOLS:
        sub = grid[grid['tol_yr'] == tol].sort_values(['f1', 'precision', 'recall'], ascending=[False, False, False])
        best_rows.append(sub.iloc[0])
    best_by_tol = pd.DataFrame(best_rows)
    save_table(best_by_tol, 'best_by_tolerance.csv')

    # selected detector on calibration window
    sel_flips, sel_segs, sel_states, sel_segment_id = apply_detector(
        cal_age, cal_resid, cal_median,
        amp_sigma=SELECTED_AMP,
        min_persistence_yr=SELECTED_PERSIST,
        dt_yr=DT_DETECT,
    )
    base_flips, base_segs, base_states, base_segment_id = apply_detector(
        cal_age, cal_resid, cal_median,
        amp_sigma=0.0,
        min_persistence_yr=0.0,
        dt_yr=DT_DETECT,
    )

    sel_metrics = {tol: metric_dict(REF_TRANSITIONS, sel_flips, tol) for tol in TOLS}
    base_metrics = {tol: metric_dict(REF_TRANSITIONS, base_flips, tol) for tol in TOLS}

    ref_table, det_table, match_df, unmatched_ref, unmatched_det = build_match_tables(
        REF_TRANSITIONS, sel_flips, sel_metrics[SELECTED_TOL]['matches']
    )
    save_table(match_df, 'selected_matches_tol300.csv')
    save_table(unmatched_ref, 'selected_unmatched_reference_tol300.csv')
    save_table(unmatched_det, 'selected_extra_detected_tol300.csv')
    save_table(det_table, 'selected_detected_flips.csv')

    seg_rows = []
    for k, seg in enumerate(sel_segs):
        seg_rows.append({
            'segment_idx': k,
            'state': seg['state'],
            'start_age_yrBP': cal_age[seg['start']],
            'end_age_yrBP': cal_age[seg['end'] - 1],
            'duration_yr': seg['dur_yr'],
            'max_abs_amp': seg['amp'],
        })
    save_table(pd.DataFrame(seg_rows), 'selected_segments.csv')

    metric_rows = []
    for det_name, metrics_by_tol, flips in [
        ('baseline_crossing', base_metrics, base_flips),
        ('selected_0.27_100yr', sel_metrics, sel_flips),
    ]:
        for tol in TOLS:
            metric_rows.append({
                'detector': det_name,
                'tol_yr': tol,
                'n_detected': len(flips),
                'n_matched': metrics_by_tol[tol]['n_match'],
                'precision': metrics_by_tol[tol]['precision'],
                'recall': metrics_by_tol[tol]['recall'],
                'f1': metrics_by_tol[tol]['f1'],
                'median_abs_age_error_yr': metrics_by_tol[tol]['median_abs_age_error_yr'],
                'mean_abs_age_error_yr': metrics_by_tol[tol]['mean_abs_age_error_yr'],
            })
    save_table(pd.DataFrame(metric_rows), 'baseline_vs_selected.csv')

    selected_summary = pd.DataFrame([
        {
            'amp_sigma': SELECTED_AMP,
            'min_persistence_yr': SELECTED_PERSIST,
            'n_detected_calibration': len(sel_flips),
            'n_reference': len(REF_TRANSITIONS),
            'n_matched_tol300': sel_metrics[300]['n_match'],
            'precision_tol300': sel_metrics[300]['precision'],
            'recall_tol300': sel_metrics[300]['recall'],
            'f1_tol300': sel_metrics[300]['f1'],
            'median_abs_age_error_tol300_yr': sel_metrics[300]['median_abs_age_error_yr'],
            'mean_abs_age_error_tol300_yr': sel_metrics[300]['mean_abs_age_error_yr'],
        }
    ])
    save_table(selected_summary, 'selected_detector_summary.csv')

    # plots for calibration window
    ref_state_on_cal = REF_STATE[np.searchsorted(REF_AGES, cal_age)]
    plot_heatmap(grid_wide)
    plot_sequence_overview(
        cal_age=cal_age,
        cal_resid=cal_resid,
        cal_med=cal_median,
        ref_states_on_cal=ref_state_on_cal,
        det_states=sel_states,
        ref_trans=REF_TRANSITIONS,
        det_trans=sel_flips,
        match_df=match_df,
        unmatched_ref=unmatched_ref,
        unmatched_det=unmatched_det,
    )
    plot_pairing_summary(
        cal_age=cal_age,
        ref_trans=REF_TRANSITIONS,
        det_trans=sel_flips,
        match_df=match_df,
        base_metrics=base_metrics,
        sel_metrics=sel_metrics,
    )
    plot_windowed_comparison(REF_TRANSITIONS, sel_flips, match_df)

    # --- Full CH4 record ---
    full_age_50 = series['ages_50']
    full_resid_50 = series['ch4_hp_50']
    full_median = float(np.median(full_resid_50))
    full_flips, full_segs, full_states, full_segment_id = apply_detector(
        full_age_50, full_resid_50, full_median,
        amp_sigma=SELECTED_AMP,
        min_persistence_yr=SELECTED_PERSIST,
        dt_yr=DT_DETECT,
    )

    # 输出完整序列到 data/
    save_square_wave_to_data(
        ages=full_age_50,
        values=full_resid_50,
        states=full_states,
        segment_id=full_segment_id,
        flips=full_flips,
        out_name='ch4_mcv_square_wave_selected_full_50yr.csv',
    )

    # 同时输出校准窗口内序列，便于和 NGRIP 做局部对照
    save_square_wave_to_data(
        ages=cal_age,
        values=cal_resid,
        states=sel_states,
        segment_id=sel_segment_id,
        flips=sel_flips,
        out_name='ch4_mcv_square_wave_selected_calibration_50yr.csv',
    )

    full_summary = pd.DataFrame([
        {
            'amp_sigma': SELECTED_AMP,
            'min_persistence_yr': SELECTED_PERSIST,
            'age_min_yrBP': int(full_age_50.min()),
            'age_max_yrBP': int(full_age_50.max()),
            'n_transitions_full': len(full_flips),
            'n_segments_full': len(full_segs),
        }
    ])
    save_table(full_summary, 'full_record_summary.csv')
    save_table(pd.DataFrame({'transition_age_yrBP': full_flips.astype(int)}), 'full_selected_transitions.csv')

    plot_full_ch4(
        full_age_10=series['ages_10'],
        full_hp_10=series['ch4_hp_10'],
        full_age_50=full_age_50,
        full_state_50=full_states,
        full_flips=full_flips,
    )

    print('Done.')
    print(f'Selected detector: amp_sigma={SELECTED_AMP}, min_persistence={SELECTED_PERSIST} yr')
    print(f'Calibration-window detected transitions: {len(sel_flips)}')
    print(f'Matched transitions at ±300 yr: {sel_metrics[300]["n_match"]} / {len(REF_TRANSITIONS)}')
    print(f'Full-record detected transitions: {len(full_flips)}')
    print(f'Full square-wave CSV: {BASE / "ch4_mcv_square_wave_selected_full_50yr.csv"}')


if __name__ == '__main__':
    main()
