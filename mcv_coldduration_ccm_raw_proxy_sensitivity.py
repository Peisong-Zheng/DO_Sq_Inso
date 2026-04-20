#!/usr/bin/env python3
from __future__ import annotations
import os
import gc
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter1d

# ============================================================
# Paths (all relative)
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
BASE = SCRIPT_DIR / 'data'
OUT = SCRIPT_DIR / 'results' / 'mcv_coldduration_ccm_raw_proxy_sensitivity'
FIG = OUT / 'figures'
TAB = OUT / 'tables'
for p in [BASE, OUT, FIG, TAB]:
    p.mkdir(parents=True, exist_ok=True)

# ============================================================
# User settings
# ============================================================
# Sensitivity experiment core switch:
#   "median" -> split filtered target at its median (default)
#   "mean"   -> split filtered target at its mean
THRESHOLD_STAT = 'median'

CH4_FILE = 'CH4_AICC2023.xlsx'
MONSOON_FILE = 'monsoon.xlsx'
PRE_FILE = 'pre_800_inter100.txt'
OBL_FILE = 'obl_800_inter100.txt'
ECC_FILE = 'ecc_1000_inter100.txt'

# Filtering/state construction settings (kept aligned with the earlier MCV workflow)
DT_INPUT = 10
DT_STATE = 50
HP_CUTOFF = 10_000.0
HP_ORDER = 4
AGE_MIN_YR_BP = 1_000

# CCM settings (same research content as v11)
DOWNSAMPLE_YR = 1000
E = 4
TAU = -2
N_SURROGATES = 200
LAG_RANGE_KYR = np.arange(-8, 9, 1)
WINDOW_YR = 20_000
COLD_DURATION_SMOOTH_POINTS = 0
LAG_LIBRARY_SIZE = 300
LAG_BOOTSTRAP_LIBS = 100
LAG_GAUSS_SIGMA = 1.0
LAG_LIBRARY_FRACTION_CAP = 0.60
RNG_SEED = 42
N_NEIGH = E + 1
LIB_POINTS = 6

PROXY_SPECS = {
    'ch4': {'file': CH4_FILE, 'value_col': 'ch4', 'label': 'CH$_4$', 'color': 'tab:red'},
    'monsoon': {'file': MONSOON_FILE, 'value_col': 'd18O', 'label': 'Monsoon d$^{18}$O', 'color': 'tab:orange'},
}

# ============================================================
# IO + filtering helpers
# ============================================================
def resolve_input_file(filename: str) -> Path:
    candidates = [BASE / filename, SCRIPT_DIR / filename, Path('/mnt/data') / filename]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f'Cannot find {filename} in {candidates}')


def load_target_series(proxy: str) -> pd.DataFrame:
    spec = PROXY_SPECS[proxy]
    raw = pd.read_excel(resolve_input_file(spec['file']))
    required = {'age', spec['value_col']}
    if not required.issubset(raw.columns):
        raise ValueError(f'{spec["file"]} must contain columns {required}; got {list(raw.columns)}')
    out = raw[['age', spec['value_col']]].dropna().copy()
    out['age'] = pd.to_numeric(out['age'], errors='coerce')
    out[spec['value_col']] = pd.to_numeric(out[spec['value_col']], errors='coerce')
    out = out.dropna().sort_values('age').reset_index(drop=True)
    # monsoon ages are kyr BP, CH4 ages are already yr BP
    if out['age'].abs().max() < 10_000:
        out['age'] = out['age'] * 1000.0
    return out


def build_filtered_target(raw: pd.DataFrame, value_col: str):
    raw_use = raw[raw['age'] >= AGE_MIN_YR_BP].copy()
    ages_10 = np.arange(AGE_MIN_YR_BP, int(np.floor(raw_use['age'].max())) + 1, DT_INPUT)
    vals_10 = np.interp(ages_10, raw_use['age'].to_numpy(float), raw_use[value_col].to_numpy(float))

    fs = 1.0 / DT_INPUT
    fc = 1.0 / HP_CUTOFF
    wn = fc / (fs * 0.5)
    b, a = butter(HP_ORDER, wn, btype='highpass')
    vals_hp_10 = filtfilt(b, a, vals_10)

    ages_50 = np.arange(int(ages_10.min()), int(ages_10.max()) + 1, DT_STATE)
    vals_hp_50 = np.interp(ages_50, ages_10, vals_hp_10)

    return {
        'raw_use': raw_use,
        'ages_10': ages_10,
        'values_10': vals_10,
        'values_hp_10': vals_hp_10,
        'ages_50': ages_50,
        'values_hp_50': vals_hp_50,
    }


def choose_center(x: np.ndarray, method: str) -> float:
    if method == 'median':
        return float(np.nanmedian(x))
    if method == 'mean':
        return float(np.nanmean(x))
    raise ValueError("THRESHOLD_STAT must be 'median' or 'mean'")


def build_state_from_filtered(filtered_values: np.ndarray, method: str):
    center = choose_center(filtered_values, method)
    state = np.where(filtered_values >= center, 1, 0).astype(int)  # 1 warm, 0 cold
    return state, center

# ============================================================
# Cold-duration helpers
# ============================================================
def centered_smooth_with_edge_handling(x, win_pts):
    x = np.asarray(x, float)
    if win_pts <= 1:
        return x.copy()
    return pd.Series(x).rolling(window=win_pts, center=True, min_periods=1).mean().to_numpy()


def moving_sum_centered(x, win_pts):
    y = np.convolve(x, np.ones(win_pts, float), mode='valid')
    h = win_pts // 2
    return y, h


def moving_mean_centered(x, win_pts):
    y, h = moving_sum_centered(x, win_pts)
    return y / win_pts, h


def extract_intervals_from_binary(age, cold):
    age = np.asarray(age, float)
    cold = np.asarray(cold, bool)
    trans = np.flatnonzero(np.diff(cold.astype(int)) != 0) + 1
    starts = np.r_[0, trans]
    ends = np.r_[trans, len(age)]
    rows = []
    dt = np.median(np.diff(age))
    for s, e in zip(starts, ends):
        if not cold[s]:
            continue
        rows.append({
            'young_age_yrBP': age[s],
            'old_age_yrBP': age[e - 1],
            'duration_yr': age[e - 1] - age[s] + dt,
            'start_truncated': bool(s == 0),
            'end_truncated': bool(e == len(age)),
        })
    return pd.DataFrame(rows)


def extract_warm_intervals_from_binary(age, cold):
    return extract_intervals_from_binary(age, ~np.asarray(cold, bool))


def local_mean_durations(centers, intervals_df, window_yr):
    vals = np.full_like(centers, np.nan, dtype=float)
    half = window_yr / 2
    for i, c in enumerate(centers):
        mask = ((intervals_df['young_age_yrBP'] >= c - half) &
                (intervals_df['old_age_yrBP'] <= c + half))
        if mask.any():
            vals[i] = intervals_df.loc[mask, 'duration_yr'].mean()
    return vals


def compute_window_stats(age, cold, warmings, cold_intervals, warm_intervals, window_yr):
    dt = np.median(np.diff(age))
    win_pts = int(round(window_yr / dt))
    if win_pts % 2 == 0:
        win_pts += 1
    e_series, h = moving_sum_centered(warmings.astype(float), win_pts)
    p_series, _ = moving_mean_centered(cold.astype(float), win_pts)
    centers = age[h:len(age) - h]
    warm_full = warm_intervals.loc[
        ~warm_intervals['start_truncated'] & ~warm_intervals['end_truncated']
    ].copy()
    return pd.DataFrame({
        'age_yrBP': centers,
        'warming_count_20k': e_series,
        'cold_fraction_20k': p_series,
        'mean_cold_duration_yr': local_mean_durations(centers, cold_intervals, window_yr),
        'mean_warm_duration_yr': local_mean_durations(centers, warm_full, window_yr),
    })


def prepare_proxy_sequence_from_raw_threshold(proxy: str):
    spec = PROXY_SPECS[proxy]
    raw = load_target_series(proxy)
    filt = build_filtered_target(raw, spec['value_col'])
    state, center = build_state_from_filtered(filt['values_hp_50'], THRESHOLD_STAT)
    age = filt['ages_50'].astype(float)
    cold = (state == 0)
    cold_intervals = extract_intervals_from_binary(age, cold)
    warm_intervals = extract_warm_intervals_from_binary(age, cold)

    warmings = np.zeros_like(age, int)
    for a in cold_intervals['young_age_yrBP'].values:
        warmings[np.argmin(np.abs(age - a))] = 1
    if cold[0]:
        warmings[0] = 0

    state_df = pd.DataFrame({
        'age_yrBP': age.astype(int),
        'filtered_value': filt['values_hp_50'],
        'raw_threshold_center': center,
        'warm_cold_state': state.astype(int),
        'mcv_state': state.astype(int),
    })
    window_stats = compute_window_stats(age, cold, warmings, cold_intervals, warm_intervals, WINDOW_YR)
    return {
        'raw_target': raw,
        'filtered': filt,
        'state_df': state_df,
        'center_value': center,
        'cold_intervals': cold_intervals,
        'warm_intervals': warm_intervals,
        'window_stats': window_stats,
    }

# ============================================================
# Orbital helpers
# ============================================================
def load_orbital_series(path, name):
    df = pd.read_csv(path, sep=r'\s+', header=None, engine='python')
    out = df.iloc[:, :2].copy()
    out.columns = ['age_raw', name]
    out['age_yrBP'] = out['age_raw'].abs() * 1000.0
    out[name] = pd.to_numeric(out[name], errors='coerce')
    return out[['age_yrBP', name]].dropna().sort_values('age_yrBP').reset_index(drop=True)


def interp_to_grid(age, src_age, src_val):
    order = np.argsort(src_age)
    return np.interp(age, np.asarray(src_age, float)[order], np.asarray(src_val, float)[order])


def zscore(x):
    x = np.asarray(x, float)
    sd = np.nanstd(x)
    return (x - np.nanmean(x)) / sd if sd > 0 else np.zeros_like(x)


def build_dataset_for_proxy(proxy_name):
    prep = prepare_proxy_sequence_from_raw_threshold(proxy_name)
    stats = prep['window_stats'].copy()
    step = max(1, int(round(DOWNSAMPLE_YR / DT_STATE)))
    stats = stats.iloc[::step].reset_index(drop=True)
    n_rows_before_dropna = len(stats)

    for name, filename in [('pre', PRE_FILE), ('obl', OBL_FILE), ('ecc', ECC_FILE)]:
        df = load_orbital_series(resolve_input_file(filename), name)
        stats[name] = interp_to_grid(stats['age_yrBP'].values, df['age_yrBP'].values, df[name].values)

    stats['mean_cold_duration_kyr_raw'] = stats['mean_cold_duration_yr'] / 1000.0
    smooth_pts = COLD_DURATION_SMOOTH_POINTS if COLD_DURATION_SMOOTH_POINTS > 0 else 1
    stats['mean_cold_duration_kyr'] = centered_smooth_with_edge_handling(stats['mean_cold_duration_kyr_raw'].to_numpy(), smooth_pts)
    stats = stats.dropna(subset=['mean_cold_duration_kyr', 'pre', 'obl', 'ecc']).copy()
    stats = stats.sort_values('age_yrBP', ascending=False).reset_index(drop=True)
    stats['Time'] = np.arange(1, len(stats) + 1)
    stats.attrs['n_rows_before_dropna'] = n_rows_before_dropna
    stats.attrs['center_value'] = prep['center_value']
    stats.attrs['state_n'] = len(prep['state_df'])
    return stats, prep

# ============================================================
# CCM core
# ============================================================
def build_embedding(x, E, tau):
    lag = abs(int(tau))
    start = (E - 1) * lag
    valid_idx = np.arange(start, len(x))
    emb = np.column_stack([x[valid_idx - j * lag] for j in range(E)]).astype(np.float32)
    return valid_idx, emb


def pearson_r(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5:
        return np.nan
    aa = a[m]
    bb = b[m]
    if np.std(aa) == 0 or np.std(bb) == 0:
        return np.nan
    return float(np.corrcoef(aa, bb)[0, 1])


def simplex_predict(dist_sub, lib_target):
    n_pred, n_lib = dist_sub.shape
    if n_lib < 2:
        return None
    k = min(N_NEIGH, n_lib - 1)
    if k < 2:
        return None
    idx = np.argpartition(dist_sub, kth=k - 1, axis=1)[:, :k]
    d_nei = np.take_along_axis(dist_sub, idx, axis=1)
    order = np.argsort(d_nei, axis=1)
    idx = np.take_along_axis(idx, order, axis=1)
    d_nei = np.take_along_axis(d_nei, order, axis=1)
    tar_nei = lib_target[idx]
    d1 = np.maximum(d_nei[:, [0]], 1e-12)
    w = np.exp(-d_nei / d1)
    return np.sum(w * tar_nei, axis=1) / np.sum(w, axis=1)


def libsizes_for_n(n_valid):
    vals = np.unique(np.round(np.linspace(max(E + 5, int(0.1 * n_valid)), int(0.9 * n_valid), LIB_POINTS)).astype(int))
    return vals[vals >= E + 2]


def ccm_curve_and_nulls_fast(source_series, target_series, E, tau, n_surrogates, rng_seed):
    valid_idx, manifold = build_embedding(source_series, E, tau)
    libs = libsizes_for_n(len(valid_idx))
    rng = np.random.default_rng(rng_seed)
    subsets = {int(L): np.sort(rng.choice(valid_idx, size=int(L), replace=False)) for L in libs}
    distmat = cdist(manifold, manifold, metric='euclidean').astype(np.float32)

    obs = []
    for L in libs:
        subset = subsets[int(L)]
        lib_pos = np.searchsorted(valid_idx, subset)
        dsub = distmat[:, lib_pos].copy()
        dsub[lib_pos, np.arange(len(subset))] = np.inf
        preds = simplex_predict(dsub, target_series[subset])
        obs.append(pearson_r(preds, target_series[valid_idx]) if preds is not None else np.nan)
    obs = np.array(obs, float)

    null_curves = np.zeros((n_surrogates, len(libs)), float)
    for s in range(n_surrogates):
        perm = np.random.default_rng(rng_seed + 1000 + s).permutation(target_series)
        for j, L in enumerate(libs):
            subset = subsets[int(L)]
            lib_pos = np.searchsorted(valid_idx, subset)
            dsub = distmat[:, lib_pos].copy()
            dsub[lib_pos, np.arange(len(subset))] = np.inf
            preds = simplex_predict(dsub, perm[subset])
            null_curves[s, j] = pearson_r(preds, perm[valid_idx]) if preds is not None else np.nan

    curve_df = pd.DataFrame({
        'LibSize': libs.astype(int),
        'rho_obs': obs,
        'rho_null_median': np.nanpercentile(null_curves, 50, axis=0),
        'rho_null_lo95': np.nanpercentile(null_curves, 2.5, axis=0),
        'rho_null_hi95': np.nanpercentile(null_curves, 97.5, axis=0),
    })
    p = float((np.sum(null_curves[:, -1] >= obs[-1]) + 1) / (n_surrogates + 1)) if np.isfinite(obs[-1]) else np.nan
    del distmat, null_curves
    gc.collect()
    return curve_df, float(obs[-1]) if np.isfinite(obs[-1]) else np.nan, p


def choose_effective_lag_library_size(n_valid, lag_values, E, tau, requested, frac_cap):
    valid_idx, _ = build_embedding(np.zeros(n_valid, float), E, tau)
    min_rows = min(np.sum((valid_idx + lag >= 0) & (valid_idx + lag < n_valid)) for lag in np.asarray(lag_values, int))
    min_needed = max(E + 5, N_NEIGH + 1)
    capped = int(np.floor(frac_cap * min_rows))
    effective = min(requested, max(min_needed, capped))
    effective = min(effective, max(min_needed, min_rows - 1))
    return {
        'n_valid': int(len(valid_idx)),
        'min_rows_across_lags': int(min_rows),
        'requested_library_size': int(requested),
        'effective_library_size': int(effective),
    }


def ccm_lag_curve_vannes_style(source_series, target_series, E, tau, lag_values, library_size, n_boot, gauss_sigma, rng_seed):
    rng = np.random.default_rng(rng_seed)
    valid_idx, manifold = build_embedding(source_series, E, tau)
    lag_values = np.asarray(lag_values, int)
    curves = np.full((n_boot, len(lag_values)), np.nan, dtype=float)

    for j, lag in enumerate(lag_values):
        row_idx = valid_idx[(valid_idx + lag >= 0) & (valid_idx + lag < len(target_series))]
        if len(row_idx) < max(library_size, E + 5):
            continue
        row_pos = np.searchsorted(valid_idx, row_idx)
        manif_rows = manifold[row_pos]
        target_shift = target_series[row_idx + lag]
        dist = cdist(manif_rows, manif_rows, metric='euclidean').astype(np.float32)
        for b in range(n_boot):
            lib_local = np.sort(rng.choice(len(row_idx), size=library_size, replace=False))
            dsub = dist[:, lib_local].copy()
            membership = np.full(len(row_idx), -1, dtype=int)
            membership[lib_local] = np.arange(len(lib_local))
            pred_has_self = membership >= 0
            dsub[pred_has_self, membership[pred_has_self]] = np.inf
            preds = simplex_predict(dsub, target_shift[lib_local])
            curves[b, j] = pearson_r(preds, target_shift) if preds is not None else np.nan

    smooth_curves = gaussian_filter1d(curves, sigma=gauss_sigma, axis=1, mode='nearest')
    med = np.nanmedian(smooth_curves, axis=0)
    if np.all(~np.isfinite(med)):
        summary = {'best_lag_kyr': np.nan, 'best_rho': np.nan}
    else:
        best_idx = int(np.nanargmax(med))
        summary = {'best_lag_kyr': int(lag_values[best_idx]), 'best_rho': float(med[best_idx])}

    curve_df = pd.DataFrame({
        'lag_kyr': lag_values.astype(float),
        'rho_median_smooth': med,
        'rho_lo5_smooth': np.nanpercentile(smooth_curves, 5, axis=0),
        'rho_hi95_smooth': np.nanpercentile(smooth_curves, 95, axis=0),
        'rho_mean_raw': np.nanmean(curves, axis=0),
    })
    return curve_df, summary

# ============================================================
# Plot helpers
# ============================================================
def setup_style():
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': False})


def plot_threshold_diagnostics(prep_by_proxy):
    fig, axes = plt.subplots(4, 1, figsize=(12.5, 10.0), dpi=180, sharex=False)
    for proxy, prep in prep_by_proxy.items():
        spec = PROXY_SPECS[proxy]
        color = spec['color']
        label = spec['label']
        axes[0].plot(prep['filtered']['ages_10'] / 1000.0, prep['filtered']['values_hp_10'], color=color, lw=0.9, alpha=0.8, label=label)
        axes[1].plot(prep['state_df']['age_yrBP'] / 1000.0, prep['state_df']['filtered_value'], color=color, lw=1.0, alpha=0.8, label=label)
        axes[1].axhline(prep['center_value'], color=color, lw=1.0, ls='--', alpha=0.9)
        axes[2].step(prep['state_df']['age_yrBP'] / 1000.0, prep['state_df']['warm_cold_state'], where='post', color=color, lw=1.2, label=label)
        axes[3].plot(prep['window_stats']['age_yrBP'] / 1000.0, prep['window_stats']['mean_cold_duration_yr'] / 1000.0, color=color, lw=1.2, label=label)
    axes[0].set_title(f'Raw-threshold sensitivity experiment: 10-kyr high-pass targets split by {THRESHOLD_STAT}')
    axes[0].set_ylabel('HP target\n(10-yr grid)')
    axes[1].set_ylabel('HP target\n(50-yr grid)')
    axes[2].set_ylabel('state\n0 cold / 1 warm')
    axes[3].set_ylabel('cold dur.\n(kyr)')
    axes[3].set_xlabel('Age (kyr BP)')
    axes[0].legend(frameon=True, fontsize=8, ncol=2)
    axes[2].set_ylim(-0.15, 1.15)
    for ax in axes:
        ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(FIG / 'fig01_threshold_diagnostics_raw_split.png', bbox_inches='tight')
    plt.close(fig)


def plot_inputs_comparison(dataset_by_proxy):
    fig, axes = plt.subplots(5, 1, figsize=(12, 9.2), dpi=180, sharex=True)
    for proxy, stats in dataset_by_proxy.items():
        spec = PROXY_SPECS[proxy]
        color = spec['color']
        label = spec['label']
        axes[0].plot(stats['age_yrBP'] / 1000, stats['mean_cold_duration_kyr_raw'], color=color, lw=1.1, alpha=0.45)
        axes[0].plot(stats['age_yrBP'] / 1000, stats['mean_cold_duration_kyr'], color=color, lw=1.8, label=label)
        for ax, col in zip(axes[1:], ['pre', 'obl', 'ecc', 'warming_count_20k']):
            ax.plot(stats['age_yrBP'] / 1000, stats[col], color=color, lw=1.0, alpha=0.75)
    axes[0].set_ylabel('cold dur.\n(kyr)')
    axes[0].legend(frameon=True, fontsize=8, ncol=2)
    for ax, lab in zip(axes[1:], ['pre', 'obl', 'ecc', 'warmings/20k']):
        ax.set_ylabel(lab)
    axes[0].set_title(
        f'Cold-duration inputs from raw-threshold states ({THRESHOLD_STAT}; Δt={DOWNSAMPLE_YR} yr, E={E}, tau={TAU}, '
        f'row1 null={N_SURROGATES}, lag bootlibs={LAG_BOOTSTRAP_LIBS}, req.lib={LAG_LIBRARY_SIZE}, '
        f'smooth_pts={COLD_DURATION_SMOOTH_POINTS})'
    )
    axes[-1].set_xlabel('Age (kyr BP)')
    axes[-1].invert_xaxis()
    fig.tight_layout()
    fig.savefig(FIG / 'fig02_inputs_cold_duration_and_orbitals_comparison.png', bbox_inches='tight')
    plt.close(fig)


def plot_ccm_comparison(results_by_proxy, std_curves_by_proxy, lag_curves_by_proxy):
    drivers = [('pre', 'precession'), ('obl', 'obliquity'), ('ecc', 'eccentricity')]
    col_fwd = '#1f77b4'
    col_rev = '#d62728'
    fig, axes = plt.subplots(4, 3, figsize=(15.5, 14.0), dpi=150)
    for p_row, proxy in enumerate(['ch4', 'monsoon']):
        res = results_by_proxy[proxy]
        for j, (driver, label) in enumerate(drivers):
            rr = res.loc[res.driver == driver].iloc[0]
            ax = axes[2 * p_row, j]
            df_f = std_curves_by_proxy[proxy][(driver, 'fwd')]
            df_r = std_curves_by_proxy[proxy][(driver, 'rev')]
            ax.fill_between(df_f['LibSize'], df_f['rho_null_lo95'], df_f['rho_null_hi95'], color=col_fwd, alpha=0.15)
            ax.plot(df_f['LibSize'], df_f['rho_null_median'], '--', color=col_fwd, lw=1)
            ax.plot(df_f['LibSize'], df_f['rho_obs'], 'o-', color=col_fwd, lw=1.6, label='cold duration xmap orbital')
            ax.fill_between(df_r['LibSize'], df_r['rho_null_lo95'], df_r['rho_null_hi95'], color=col_rev, alpha=0.15)
            ax.plot(df_r['LibSize'], df_r['rho_null_median'], '--', color=col_rev, lw=1)
            ax.plot(df_r['LibSize'], df_r['rho_obs'], 's-', color=col_rev, lw=1.6, label='orbital xmap cold duration')
            ax.set_title(f'{PROXY_SPECS[proxy]["label"]} · {label}\nCCM: p_fwd={rr.p_metric_xmap_orbital:.3f}, p_rev={rr.p_orbital_xmap_metric:.3f}')
            if j == 0:
                ax.set_ylabel('CCM skill ρ')
                if p_row == 0:
                    ax.legend(frameon=True, fontsize=8)
            ax.set_xlabel('Library size')
            ax.grid(alpha=0.2, ls='--')

            ax = axes[2 * p_row + 1, j]
            df_f = lag_curves_by_proxy[proxy][(driver, 'fwd')]
            df_r = lag_curves_by_proxy[proxy][(driver, 'rev')]
            ax.fill_between(df_f['lag_kyr'], df_f['rho_lo5_smooth'], df_f['rho_hi95_smooth'], color=col_fwd, alpha=0.18)
            ax.plot(df_f['lag_kyr'], df_f['rho_median_smooth'], 'o-', color=col_fwd, lw=1.8, label='cold duration xmap orbital')
            ax.fill_between(df_r['lag_kyr'], df_r['rho_lo5_smooth'], df_r['rho_hi95_smooth'], color=col_rev, alpha=0.18)
            ax.plot(df_r['lag_kyr'], df_r['rho_median_smooth'], 's-', color=col_rev, lw=1.8, label='orbital xmap cold duration')
            ax.axvline(0, color='0.4', ls='--', lw=1)
            if np.isfinite(rr.best_lag_metric_xmap_orbital_kyr):
                ax.axvline(rr.best_lag_metric_xmap_orbital_kyr, color=col_fwd, ls=':', lw=1.2)
            if np.isfinite(rr.best_lag_orbital_xmap_metric_kyr):
                ax.axvline(rr.best_lag_orbital_xmap_metric_kyr, color=col_rev, ls=':', lw=1.2)
            ax.set_title(f'robust lag-test: best_fwd={rr.best_lag_metric_xmap_orbital_kyr:.0f} kyr, best_rev={rr.best_lag_orbital_xmap_metric_kyr:.0f} kyr\neffective lib={int(rr.lag_library_effective)}')
            if j == 0:
                ax.set_ylabel('Smoothed CCM skill ρ')
                if p_row == 0:
                    ax.legend(frameon=True, fontsize=8)
            ax.set_xlabel('Cross-map lag (kyr)')
            ax.grid(alpha=0.2, ls='--')
    fig.suptitle('Cold-duration focused CCM and lag-test from raw-threshold CH$_4$/monsoon states (bidirectional)', y=0.995)
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.05, top=0.93, wspace=0.22, hspace=0.34)
    fig.savefig(FIG / 'fig03_coldduration_ccm_and_lagtest_comparison_4x3.png')
    plt.close(fig)


def plot_proxy_driver_contrast(results_by_proxy):
    drivers = ['pre', 'obl', 'ecc']
    labels = ['precession', 'obliquity', 'eccentricity']
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.5), dpi=160)
    x = np.arange(len(drivers))
    width = 0.36
    metrics = [
        ('rho_metric_xmap_orbital', 'CCM ρ (cold duration xmap orbital)'),
        ('rho_orbital_xmap_metric', 'CCM ρ (orbital xmap cold duration)'),
        ('best_lag_metric_xmap_orbital_rho', 'Best lag-test ρ (cold duration xmap orbital)'),
        ('best_lag_orbital_xmap_metric_rho', 'Best lag-test ρ (orbital xmap cold duration)'),
    ]
    for ax, (col, ylabel) in zip(axes.flat, metrics):
        ch4 = results_by_proxy['ch4'].set_index('driver').loc[drivers, col].to_numpy(float)
        mon = results_by_proxy['monsoon'].set_index('driver').loc[drivers, col].to_numpy(float)
        ax.bar(x - width/2, ch4, width, label='CH$_4$', color=PROXY_SPECS['ch4']['color'], alpha=0.85)
        ax.bar(x + width/2, mon, width, label='Monsoon d$^{18}$O', color=PROXY_SPECS['monsoon']['color'], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', alpha=0.2, ls='--')
    axes[0, 0].legend(frameon=True, fontsize=8)
    fig.suptitle('CH$_4$ versus monsoon: raw-threshold sensitivity summary across orbital drivers', y=0.98)
    fig.tight_layout()
    fig.savefig(FIG / 'fig04_proxy_driver_summary_comparison.png')
    plt.close(fig)

# ============================================================
# Main workflow
# ============================================================
def run_proxy(proxy, proxy_seed_offset):
    stats, prep = build_dataset_for_proxy(proxy)
    stats.to_csv(TAB / f'aligned_coldduration_orbitals_for_ccm_lagtest_{proxy}.csv', index=False)
    prep['state_df'].to_csv(TAB / f'raw_threshold_state_{proxy}.csv', index=False)
    prep['cold_intervals'].to_csv(TAB / f'raw_threshold_cold_intervals_{proxy}.csv', index=False)
    prep['warm_intervals'].to_csv(TAB / f'raw_threshold_warm_intervals_{proxy}.csv', index=False)
    metric_series = zscore(stats['mean_cold_duration_kyr'].to_numpy())
    drivers = [('pre', 'precession'), ('obl', 'obliquity'), ('ecc', 'eccentricity')]

    lib_info = choose_effective_lag_library_size(
        n_valid=len(metric_series), lag_values=LAG_RANGE_KYR,
        E=E, tau=TAU, requested=LAG_LIBRARY_SIZE,
        frac_cap=LAG_LIBRARY_FRACTION_CAP
    )

    rows = []
    std_curves = {}
    lag_curves = {}

    for j, (driver, label) in enumerate(drivers):
        print(f'proxy={proxy}, driver={driver}', flush=True)
        dseries = zscore(stats[driver].to_numpy())
        f_curve, f_obs, f_p = ccm_curve_and_nulls_fast(metric_series, dseries, E, TAU, N_SURROGATES, RNG_SEED + proxy_seed_offset + j * 10 + 1)
        r_curve, r_obs, r_p = ccm_curve_and_nulls_fast(dseries, metric_series, E, TAU, N_SURROGATES, RNG_SEED + proxy_seed_offset + j * 10 + 2)
        lag_f, sum_f = ccm_lag_curve_vannes_style(metric_series, dseries, E, TAU, LAG_RANGE_KYR, lib_info['effective_library_size'], LAG_BOOTSTRAP_LIBS, LAG_GAUSS_SIGMA, RNG_SEED + proxy_seed_offset + j * 100 + 11)
        lag_r, sum_r = ccm_lag_curve_vannes_style(dseries, metric_series, E, TAU, LAG_RANGE_KYR, lib_info['effective_library_size'], LAG_BOOTSTRAP_LIBS, LAG_GAUSS_SIGMA, RNG_SEED + proxy_seed_offset + j * 100 + 22)
        std_curves[(driver, 'fwd')] = f_curve
        std_curves[(driver, 'rev')] = r_curve
        lag_curves[(driver, 'fwd')] = lag_f
        lag_curves[(driver, 'rev')] = lag_r
        rows.append({
            'proxy': proxy,
            'driver': driver,
            'driver_label': label,
            'threshold_stat': THRESHOLD_STAT,
            'threshold_center_value': stats.attrs['center_value'],
            'n_rows_after_dropna': len(stats),
            'lag_library_requested': lib_info['requested_library_size'],
            'lag_library_effective': lib_info['effective_library_size'],
            'lag_library_min_rows_across_lags': lib_info['min_rows_across_lags'],
            'rho_metric_xmap_orbital': f_obs,
            'p_metric_xmap_orbital': f_p,
            'rho_orbital_xmap_metric': r_obs,
            'p_orbital_xmap_metric': r_p,
            'best_lag_metric_xmap_orbital_kyr': sum_f['best_lag_kyr'],
            'best_lag_metric_xmap_orbital_rho': sum_f['best_rho'],
            'best_lag_orbital_xmap_metric_kyr': sum_r['best_lag_kyr'],
            'best_lag_orbital_xmap_metric_rho': sum_r['best_rho'],
        })
        gc.collect()
    res = pd.DataFrame(rows)
    res.to_csv(TAB / f'coldduration_ccm_lagtest_results_{proxy}.csv', index=False)
    return {'stats': stats, 'prep': prep, 'results': res, 'std_curves': std_curves, 'lag_curves': lag_curves, 'lib_info': lib_info}


def main():
    setup_style()
    all_results = {'ch4': run_proxy('ch4', 0), 'monsoon': run_proxy('monsoon', 1000)}
    dataset_by_proxy = {k: v['stats'] for k, v in all_results.items()}
    prep_by_proxy = {k: v['prep'] for k, v in all_results.items()}
    results_by_proxy = {k: v['results'] for k, v in all_results.items()}
    std_curves_by_proxy = {k: v['std_curves'] for k, v in all_results.items()}
    lag_curves_by_proxy = {k: v['lag_curves'] for k, v in all_results.items()}

    comparison_rows = []
    for proxy, bundle in all_results.items():
        s = bundle['stats']
        li = bundle['lib_info']
        comparison_rows.append({
            'proxy': proxy,
            'proxy_label': PROXY_SPECS[proxy]['label'],
            'threshold_stat': THRESHOLD_STAT,
            'threshold_center_value': s.attrs['center_value'],
            'age_min_kyrBP': s['age_yrBP'].min() / 1000.0,
            'age_max_kyrBP': s['age_yrBP'].max() / 1000.0,
            'state_n_50yr': s.attrs['state_n'],
            'n_rows_after_dropna': len(s),
            'lag_library_requested': li['requested_library_size'],
            'lag_library_effective': li['effective_library_size'],
            'lag_library_min_rows_across_lags': li['min_rows_across_lags'],
            'nan_rows_removed_from_metric': int(s.attrs.get('n_rows_before_dropna', len(s)) - len(s)),
        })
    pd.DataFrame(comparison_rows).to_csv(TAB / 'proxy_series_summary.csv', index=False)
    combined = pd.concat([results_by_proxy['ch4'], results_by_proxy['monsoon']], ignore_index=True)
    combined.to_csv(TAB / 'coldduration_ccm_lagtest_results_all_proxies.csv', index=False)

    plot_threshold_diagnostics(prep_by_proxy)
    plot_inputs_comparison(dataset_by_proxy)
    plot_ccm_comparison(results_by_proxy, std_curves_by_proxy, lag_curves_by_proxy)
    plot_proxy_driver_contrast(results_by_proxy)

    with open(OUT / 'summary.txt', 'w', encoding='utf-8') as f:
        f.write('Cold-duration focused CCM and robust lag-test for CH4 and monsoon\n')
        f.write('Sensitivity experiment: no NGRIP-calibrated MCV detector is used.\n')
        f.write(f'Warm/cold states are defined directly from the 10-kyr high-pass target series split by THRESHOLD_STAT={THRESHOLD_STAT}.\n')
        f.write(f'Filtering backbone: {DT_INPUT}-yr interpolation, Butterworth high-pass cutoff={HP_CUTOFF} yr, state grid={DT_STATE} yr.\n')
        f.write(f'Downsample={DOWNSAMPLE_YR} yr, E={E}, tau={TAU}, row1 surrogates={N_SURROGATES}\n')
        f.write(f'Lag range={LAG_RANGE_KYR[0]}..{LAG_RANGE_KYR[-1]} kyr\n')
        f.write(f'Cold-duration smoothing points={COLD_DURATION_SMOOTH_POINTS} (0 means none)\n')
        f.write(f'Lag-test method: Van Nes style bootstrapped library sets (n={LAG_BOOTSTRAP_LIBS}, requested library size={LAG_LIBRARY_SIZE}, fraction cap={LAG_LIBRARY_FRACTION_CAP}) + Gaussian smoothing (sigma={LAG_GAUSS_SIGMA}).\n\n')
        for proxy, bundle in all_results.items():
            f.write(f'Proxy: {proxy} ({PROXY_SPECS[proxy]["label"]})\n')
            f.write(f'  threshold center ({THRESHOLD_STAT}) = {bundle["stats"].attrs["center_value"]:.6g}\n')
            f.write(f'  rows after dropna = {len(bundle["stats"])}\n')
            f.write(f'  effective lag library size = {bundle["lib_info"]["effective_library_size"]} (requested {bundle["lib_info"]["requested_library_size"]})\n')
            for _, r in bundle['results'].iterrows():
                f.write(f'  Driver: {r.driver}\n')
                f.write(f'    CCM   fwd (cold duration xmap orbital): rho={r.rho_metric_xmap_orbital:.3f}, p={r.p_metric_xmap_orbital:.3f}\n')
                f.write(f'    CCM   rev (orbital xmap cold duration): rho={r.rho_orbital_xmap_metric:.3f}, p={r.p_orbital_xmap_metric:.3f}\n')
                f.write(f'    LAG   fwd: best lag={int(r.best_lag_metric_xmap_orbital_kyr)} kyr, rho={r.best_lag_metric_xmap_orbital_rho:.3f}\n')
                f.write(f'    LAG   rev: best lag={int(r.best_lag_orbital_xmap_metric_kyr)} kyr, rho={r.best_lag_orbital_xmap_metric_rho:.3f}\n')
            f.write('\n')
    print('Done.')
    print(f'Outputs written to: {OUT}')


if __name__ == '__main__':
    main()
