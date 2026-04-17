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
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter1d

# ============================================================
# Paths (all relative)
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
BASE = SCRIPT_DIR / 'data'
OUT = SCRIPT_DIR / 'results' / 'mcv_coldduration_ccm_predeamp_lr04_co2_v15'
FIG = OUT / 'figures'
TAB = OUT / 'tables'
for p in [BASE, OUT, FIG, TAB]:
    p.mkdir(parents=True, exist_ok=True)

PROXY_FILES = {
    'ch4': BASE / 'ch4_mcv_square_wave_selected_full_50yr.csv',
    'monsoon': BASE / 'monsoon_mcv_square_wave_selected_full_50yr.csv',
}
PRE_FILE = BASE / 'pre_800_inter100.txt'
ECC_FILE = BASE / 'ecc_1000_inter100.txt'
LR04_FILE = BASE / 'lr04.xlsx'
CO2_FILE = BASE / 'composite_co2.xlsx'

# ============================================================
# User settings
# ============================================================
DOWNSAMPLE_YR = 1000
E = 4
TAU = -2
N_SURROGATES = 200
LAG_RANGE_KYR = np.arange(-8, 9, 1)
WINDOW_YR = 20_000
COLD_DURATION_SMOOTH_POINTS = 0
LAG_LIBRARY_SIZE = 200
LAG_BOOTSTRAP_LIBS = 20
LAG_GAUSS_SIGMA = 1.0
LAG_LIBRARY_FRACTION_CAP = 0.50
PHASE_PLOT_MAX_AGE_KYR = 800
RNG_SEED = 42
N_NEIGH = E + 1
LIB_POINTS = 6
LR04_SMOOTH_WINDOW_KYR = 10

PROXY_LABELS = {'ch4': 'CH$_4$', 'monsoon': 'Monsoon d$^{18}$O'}
DRIVER_META = [
    ('pre_phase_like', 'precession de-amplitude'),
    ('lr04_10k', 'LR04 10-kyr smooth'),
    ('co2_10k', 'CO$_2$ 10-kyr smooth'),
]

# ============================================================
# Helpers for cold-duration metric construction
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


def prepare_proxy_sequence(path):
    df = pd.read_csv(path).sort_values('age_yrBP')
    age = df['age_yrBP'].to_numpy(float)
    cold = (df['mcv_state'].to_numpy() == 0)
    cold_intervals = extract_intervals_from_binary(age, cold)
    warm_intervals = extract_warm_intervals_from_binary(age, cold)

    warmings = np.zeros_like(age, int)
    for a in cold_intervals['young_age_yrBP'].values:
        warmings[np.argmin(np.abs(age - a))] = 1
    if cold[0]:
        warmings[0] = 0

    return {'window_stats': compute_window_stats(age, cold, warmings, cold_intervals, warm_intervals, WINDOW_YR)}

# ============================================================
# Driver loading and transformations
# ============================================================
def load_orbital_series(path, name):
    df = pd.read_csv(path, sep=r'\s+', header=None, engine='python')
    out = df.iloc[:, :2].copy()
    out.columns = ['age_raw', name]
    out['age_yrBP'] = out['age_raw'].abs() * 1000.0
    out[name] = pd.to_numeric(out[name], errors='coerce')
    return out[['age_yrBP', name]].dropna().sort_values('age_yrBP').reset_index(drop=True)


def load_lr04_series_smoothed(path, window_kyr=10):
    raw = pd.read_excel(path)
    if 'Time (ka)' not in raw.columns or 'Benthic d18O (per mil)' not in raw.columns:
        raise ValueError(f'Unexpected LR04 columns: {list(raw.columns)}')
    out = raw[['Time (ka)', 'Benthic d18O (per mil)']].copy()
    out.columns = ['age_ka', 'lr04_raw']
    out['age_yrBP'] = pd.to_numeric(out['age_ka'], errors='coerce') * 1000.0
    out['lr04_raw'] = pd.to_numeric(out['lr04_raw'], errors='coerce')
    out = out[['age_yrBP', 'lr04_raw']].dropna().sort_values('age_yrBP').reset_index(drop=True)

    age_grid = np.arange(int(np.ceil(out['age_yrBP'].min() / 1000.0)) * 1000,
                         int(np.floor(out['age_yrBP'].max() / 1000.0)) * 1000 + 1,
                         1000)
    lr04_1k = np.interp(age_grid, out['age_yrBP'].to_numpy(float), out['lr04_raw'].to_numpy(float))
    win_pts = max(1, int(round(window_kyr)))
    lr04_10k = centered_smooth_with_edge_handling(lr04_1k, win_pts)
    return pd.DataFrame({'age_yrBP': age_grid.astype(float), 'lr04_1k': lr04_1k, 'lr04_10k': lr04_10k})


def load_co2_series_smoothed(path, window_kyr=10):
    raw = pd.read_excel(path)
    age_col = None
    value_col = None
    for c in raw.columns:
        cs = str(c).strip()
        if cs == 'Gasage (yr BP)':
            age_col = c
        if cs == 'CO2 (ppmv)':
            value_col = c
    if age_col is None or value_col is None:
        raise ValueError(f'Unexpected CO2 columns: {list(raw.columns)}')
    out = raw[[age_col, value_col]].copy()
    out.columns = ['age_raw', 'co2_raw']
    out['age_yrBP'] = pd.to_numeric(out['age_raw'], errors='coerce').abs()
    out['co2_raw'] = pd.to_numeric(out['co2_raw'], errors='coerce')
    out = out[['age_yrBP', 'co2_raw']].dropna().sort_values('age_yrBP').reset_index(drop=True)

    age_grid = np.arange(int(np.ceil(out['age_yrBP'].min() / 1000.0)) * 1000,
                         int(np.floor(out['age_yrBP'].max() / 1000.0)) * 1000 + 1,
                         1000)
    co2_1k = np.interp(age_grid, out['age_yrBP'].to_numpy(float), out['co2_raw'].to_numpy(float))
    win_pts = max(1, int(round(window_kyr)))
    co2_10k = centered_smooth_with_edge_handling(co2_1k, win_pts)
    return pd.DataFrame({'age_yrBP': age_grid.astype(float), 'co2_1k': co2_1k, 'co2_10k': co2_10k})


def interp_to_grid(age, src_age, src_val):
    order = np.argsort(src_age)
    return np.interp(age, np.asarray(src_age, float)[order], np.asarray(src_val, float)[order])


def zscore(x):
    x = np.asarray(x, float)
    sd = np.nanstd(x)
    return (x - np.nanmean(x)) / sd if sd > 0 else np.zeros_like(x)


def extract_pre_phase_by_deamplitude(pre_df, ecc_df):
    merged = pre_df.merge(ecc_df, on='age_yrBP', how='inner').copy()
    merged = merged.sort_values('age_yrBP').reset_index(drop=True)
    pre = merged['pre'].to_numpy(float)
    ecc = merged['ecc'].to_numpy(float)

    ecc_floor = np.nanpercentile(ecc, 5)
    ecc_safe = np.maximum(ecc, ecc_floor)
    ecc_ref = float(np.nanmedian(ecc_safe))

    pre_deamp_unit = pre / ecc_safe
    pre_phase_like = pre_deamp_unit * ecc_ref
    # normalize just enough so visual comparison is on a comparable vertical range
    pre_phase_like_rescaled = pre_phase_like * (np.nanstd(pre) / np.nanstd(pre_phase_like) if np.nanstd(pre_phase_like) > 0 else 1.0)

    merged['ecc_safe'] = ecc_safe
    merged['pre_envelope_proxy'] = ecc_safe / ecc_ref
    merged['pre_phase_like_raw'] = pre_phase_like
    merged['pre_phase_like'] = pre_phase_like_rescaled
    merged['pre_phase_like_z'] = zscore(pre_phase_like_rescaled)
    merged['pre_z'] = zscore(pre)
    merged['pre_deamp_unit'] = pre_deamp_unit
    return merged


def build_driver_library():
    pre_df = load_orbital_series(PRE_FILE, 'pre')
    ecc_df = load_orbital_series(ECC_FILE, 'ecc')
    lr04_df = load_lr04_series_smoothed(LR04_FILE, window_kyr=LR04_SMOOTH_WINDOW_KYR)
    co2_df = load_co2_series_smoothed(CO2_FILE, window_kyr=LR04_SMOOTH_WINDOW_KYR)
    pre_phase_df = extract_pre_phase_by_deamplitude(pre_df, ecc_df)
    return {'pre_raw': pre_df, 'ecc_raw': ecc_df, 'pre_phase': pre_phase_df, 'lr04': lr04_df, 'co2': co2_df}


def build_dataset_for_proxy(proxy_name, driver_lib):
    stats = prepare_proxy_sequence(PROXY_FILES[proxy_name])['window_stats'].copy()
    step = max(1, int(round(DOWNSAMPLE_YR / 50)))
    stats = stats.iloc[::step].reset_index(drop=True)
    n_rows_before_dropna = len(stats)

    stats['pre_phase_like'] = interp_to_grid(
        stats['age_yrBP'].values,
        driver_lib['pre_phase']['age_yrBP'].values,
        driver_lib['pre_phase']['pre_phase_like'].values,
    )
    stats['lr04_10k'] = interp_to_grid(
        stats['age_yrBP'].values,
        driver_lib['lr04']['age_yrBP'].values,
        driver_lib['lr04']['lr04_10k'].values,
    )
    stats['co2_10k'] = interp_to_grid(
        stats['age_yrBP'].values,
        driver_lib['co2']['age_yrBP'].values,
        driver_lib['co2']['co2_10k'].values,
    )
    stats['mean_cold_duration_kyr_raw'] = stats['mean_cold_duration_yr'] / 1000.0
    smooth_pts = COLD_DURATION_SMOOTH_POINTS if COLD_DURATION_SMOOTH_POINTS > 0 else 1
    stats['mean_cold_duration_kyr'] = centered_smooth_with_edge_handling(stats['mean_cold_duration_kyr_raw'].to_numpy(), smooth_pts)
    stats = stats.dropna(subset=['mean_cold_duration_kyr', 'pre_phase_like', 'lr04_10k', 'co2_10k']).copy()
    stats = stats.sort_values('age_yrBP', ascending=False).reset_index(drop=True)
    stats['Time'] = np.arange(1, len(stats) + 1)
    stats.attrs['n_rows_before_dropna'] = n_rows_before_dropna
    return stats

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
        'n_valid_embedding_rows': int(len(valid_idx)),
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


def plot_pre_phase_diagnostic(pre_phase_df):
    df = pre_phase_df[pre_phase_df['age_yrBP'] <= PHASE_PLOT_MAX_AGE_KYR * 1000].copy()
    x = df['age_yrBP'].to_numpy() / 1000.0
    fig, axes = plt.subplots(3, 1, figsize=(13.0, 9.0), dpi=180, sharex=True)
    axes[0].plot(x, df['pre'], color='black', lw=1.0, label='raw precession')
    axes[0].plot(x, df['pre_phase_like'], color='tab:orange', lw=1.0, label='de-amplitude precession')
    axes[0].set_ylabel('pre value')
    axes[0].legend(frameon=True, fontsize=8)
    axes[0].set_title('Precession de-amplitude diagnostic (divide by eccentricity envelope)')
    axes[1].plot(x, df['ecc'], color='tab:red', lw=1.0, label='eccentricity')
    axes[1].plot(x, df['ecc_safe'], color='tab:purple', lw=1.0, ls='--', label='ecc used as envelope')
    axes[1].set_ylabel('ecc')
    axes[1].legend(frameon=True, fontsize=8)
    axes[2].plot(x, df['pre_z'], color='0.3', lw=1.0, label='raw pre z')
    axes[2].plot(x, df['pre_phase_like_z'], color='tab:green', lw=1.0, label='de-amplitude pre z')
    axes[2].set_ylabel('z-score')
    axes[2].set_xlabel('Age (kyr BP)')
    axes[2].invert_xaxis()
    axes[2].legend(frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / 'fig01_pre_deamplitude_diagnostic.png', bbox_inches='tight')
    plt.close(fig)


def plot_driver_inputs(driver_lib):
    pre_df = driver_lib['pre_phase'].copy()
    lr04_df = driver_lib['lr04'].copy()
    co2_df = driver_lib['co2'].copy()
    pre_df = pre_df[pre_df['age_yrBP'] <= PHASE_PLOT_MAX_AGE_KYR * 1000]
    lr04_df = lr04_df[lr04_df['age_yrBP'] <= PHASE_PLOT_MAX_AGE_KYR * 1000]
    co2_df = co2_df[co2_df['age_yrBP'] <= PHASE_PLOT_MAX_AGE_KYR * 1000]
    fig, axes = plt.subplots(3, 1, figsize=(12.5, 9.0), dpi=180)
    axes[0].plot(pre_df['age_yrBP'] / 1000, pre_df['pre'], color='0.4', lw=0.9, label='raw pre')
    axes[0].plot(pre_df['age_yrBP'] / 1000, pre_df['pre_phase_like'], color='tab:orange', lw=1.0, label='de-amplitude pre')
    axes[0].set_ylabel('pre-like driver')
    axes[0].set_title('Drivers used in CCM')
    axes[0].legend(frameon=True, fontsize=8)
    axes[0].invert_xaxis()
    axes[1].plot(lr04_df['age_yrBP'] / 1000, lr04_df['lr04_1k'], color='0.7', lw=0.8, label='LR04 1-kyr')
    axes[1].plot(lr04_df['age_yrBP'] / 1000, lr04_df['lr04_10k'], color='black', lw=1.1, label='LR04 10-kyr smooth')
    axes[1].set_ylabel('LR04 (‰)')
    axes[1].legend(frameon=True, fontsize=8)
    axes[1].invert_xaxis()
    axes[2].plot(co2_df['age_yrBP'] / 1000, co2_df['co2_1k'], color='0.7', lw=0.8, label='CO$_2$ 1-kyr')
    axes[2].plot(co2_df['age_yrBP'] / 1000, co2_df['co2_10k'], color='tab:purple', lw=1.1, label='CO$_2$ 10-kyr smooth')
    axes[2].set_ylabel('CO$_2$ (ppmv)')
    axes[2].set_xlabel('Age (kyr BP)')
    axes[2].legend(frameon=True, fontsize=8)
    axes[2].invert_xaxis()
    fig.tight_layout()
    fig.savefig(FIG / 'fig02_driver_inputs_predeamp_lr04_co2.png', bbox_inches='tight')
    plt.close(fig)


def plot_ccm_comparison(results_by_proxy, std_curves_by_proxy, lag_curves_by_proxy):
    col_fwd = '#1f77b4'
    col_rev = '#d62728'
    ncols = len(DRIVER_META)
    fig, axes = plt.subplots(4, ncols, figsize=(5.8 * ncols, 13.6), dpi=150)
    for p_row, proxy in enumerate(['ch4', 'monsoon']):
        res = results_by_proxy[proxy]
        for j, (driver, label) in enumerate(DRIVER_META):
            rr = res.loc[res.driver == driver].iloc[0]
            ax = axes[2 * p_row, j]
            df_f = std_curves_by_proxy[proxy][(driver, 'fwd')]
            df_r = std_curves_by_proxy[proxy][(driver, 'rev')]
            ax.fill_between(df_f['LibSize'], df_f['rho_null_lo95'], df_f['rho_null_hi95'], color=col_fwd, alpha=0.15)
            ax.plot(df_f['LibSize'], df_f['rho_null_median'], '--', color=col_fwd, lw=1)
            ax.plot(df_f['LibSize'], df_f['rho_obs'], 'o-', color=col_fwd, lw=1.6, label='cold duration xmap driver')
            ax.fill_between(df_r['LibSize'], df_r['rho_null_lo95'], df_r['rho_null_hi95'], color=col_rev, alpha=0.15)
            ax.plot(df_r['LibSize'], df_r['rho_null_median'], '--', color=col_rev, lw=1)
            ax.plot(df_r['LibSize'], df_r['rho_obs'], 's-', color=col_rev, lw=1.6, label='driver xmap cold duration')
            ax.set_title(f'{PROXY_LABELS[proxy]} · {label}\nCCM: p_fwd={rr.p_metric_xmap_driver:.3f}, p_rev={rr.p_driver_xmap_metric:.3f}')
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
            ax.plot(df_f['lag_kyr'], df_f['rho_median_smooth'], 'o-', color=col_fwd, lw=1.8, label='cold duration xmap driver')
            ax.fill_between(df_r['lag_kyr'], df_r['rho_lo5_smooth'], df_r['rho_hi95_smooth'], color=col_rev, alpha=0.18)
            ax.plot(df_r['lag_kyr'], df_r['rho_median_smooth'], 's-', color=col_rev, lw=1.8, label='driver xmap cold duration')
            ax.axvline(0, color='0.4', ls='--', lw=1)
            if np.isfinite(rr.best_lag_metric_xmap_driver_kyr):
                ax.axvline(rr.best_lag_metric_xmap_driver_kyr, color=col_fwd, ls=':', lw=1.2)
            if np.isfinite(rr.best_lag_driver_xmap_metric_kyr):
                ax.axvline(rr.best_lag_driver_xmap_metric_kyr, color=col_rev, ls=':', lw=1.2)
            ax.set_title(f'lag-test: best_fwd={rr.best_lag_metric_xmap_driver_kyr:.0f} kyr, best_rev={rr.best_lag_driver_xmap_metric_kyr:.0f} kyr\neffective lib={int(rr.lag_library_effective)}')
            if j == 0:
                ax.set_ylabel('Smoothed CCM skill ρ')
                if p_row == 0:
                    ax.legend(frameon=True, fontsize=8)
            ax.set_xlabel('Cross-map lag (kyr)')
            ax.grid(alpha=0.2, ls='--')
    fig.suptitle('Cold-duration CCM and lag-test with de-amplitude PRE + smoothed LR04 + smoothed CO$_2$ (bidirectional)', y=0.995)
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.05, top=0.92, wspace=0.24, hspace=0.34)
    fig.savefig(FIG / 'fig03_coldduration_ccm_lagtest_predeamp_lr04_co2_4x3.png')
    plt.close(fig)


def plot_summary(results_by_proxy):
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.2), dpi=160)
    drivers = [d for d, _ in DRIVER_META]
    labels = [lab for _, lab in DRIVER_META]
    x = np.arange(len(drivers))
    width = 0.36
    metric_specs = [
        ('rho_metric_xmap_driver', 'CCM ρ (cold duration xmap driver)'),
        ('rho_driver_xmap_metric', 'CCM ρ (driver xmap cold duration)'),
        ('best_lag_metric_xmap_driver_rho', 'Best lag-test ρ (cold duration xmap driver)'),
        ('best_lag_driver_xmap_metric_rho', 'Best lag-test ρ (driver xmap cold duration)'),
    ]
    for ax, (col, title) in zip(axes.flat, metric_specs):
        y_ch4 = results_by_proxy['ch4'].set_index('driver').loc[drivers, col].to_numpy(float)
        y_mon = results_by_proxy['monsoon'].set_index('driver').loc[drivers, col].to_numpy(float)
        ax.bar(x - width / 2, y_ch4, width=width, label='CH$_4$')
        ax.bar(x + width / 2, y_mon, width=width, label='Monsoon')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha='right')
        ax.set_title(title)
        ax.set_ylabel('ρ')
        ax.grid(axis='y', alpha=0.2, ls='--')
    axes[0, 0].legend(frameon=True)
    fig.tight_layout()
    fig.savefig(FIG / 'fig04_proxy_driver_summary_predeamp_lr04_co2.png', bbox_inches='tight')
    plt.close(fig)

# ============================================================
# Main
# ============================================================
def run_proxy(proxy_name, driver_lib):
    print(f'Running proxy: {proxy_name}', flush=True)
    stats = build_dataset_for_proxy(proxy_name, driver_lib)
    stats.to_csv(TAB / f'aligned_metric_and_drivers_{proxy_name}.csv', index=False)
    metric_series = zscore(stats['mean_cold_duration_kyr'].to_numpy())

    rows = []
    std_curves = {}
    lag_curves = {}
    lib_summary_rows = []

    for j, (driver, label) in enumerate(DRIVER_META):
        dseries = zscore(stats[driver].to_numpy())
        lag_lib_info = choose_effective_lag_library_size(
            n_valid=len(metric_series),
            lag_values=LAG_RANGE_KYR,
            E=E,
            tau=TAU,
            requested=LAG_LIBRARY_SIZE,
            frac_cap=LAG_LIBRARY_FRACTION_CAP,
        )
        lib_summary_rows.append({
            'proxy': proxy_name,
            'driver': driver,
            'driver_label': label,
            'n_rows_before_dropna': int(stats.attrs.get('n_rows_before_dropna', len(stats))),
            'n_rows_after_dropna': int(len(stats)),
            **lag_lib_info,
        })

        f_curve, f_obs, f_p = ccm_curve_and_nulls_fast(metric_series, dseries, E, TAU, N_SURROGATES, RNG_SEED + j * 10 + 1 + (0 if proxy_name == 'ch4' else 1000))
        r_curve, r_obs, r_p = ccm_curve_and_nulls_fast(dseries, metric_series, E, TAU, N_SURROGATES, RNG_SEED + j * 10 + 2 + (0 if proxy_name == 'ch4' else 1000))
        lag_f, sum_f = ccm_lag_curve_vannes_style(metric_series, dseries, E, TAU, LAG_RANGE_KYR, lag_lib_info['effective_library_size'], LAG_BOOTSTRAP_LIBS, LAG_GAUSS_SIGMA, RNG_SEED + j * 100 + 11 + (0 if proxy_name == 'ch4' else 1000))
        lag_r, sum_r = ccm_lag_curve_vannes_style(dseries, metric_series, E, TAU, LAG_RANGE_KYR, lag_lib_info['effective_library_size'], LAG_BOOTSTRAP_LIBS, LAG_GAUSS_SIGMA, RNG_SEED + j * 100 + 22 + (0 if proxy_name == 'ch4' else 1000))

        std_curves[(driver, 'fwd')] = f_curve
        std_curves[(driver, 'rev')] = r_curve
        lag_curves[(driver, 'fwd')] = lag_f
        lag_curves[(driver, 'rev')] = lag_r

        f_curve.to_csv(TAB / f'ccm_curve_{proxy_name}_{driver}_fwd.csv', index=False)
        r_curve.to_csv(TAB / f'ccm_curve_{proxy_name}_{driver}_rev.csv', index=False)
        lag_f.to_csv(TAB / f'lag_curve_{proxy_name}_{driver}_fwd.csv', index=False)
        lag_r.to_csv(TAB / f'lag_curve_{proxy_name}_{driver}_rev.csv', index=False)

        rows.append({
            'proxy': proxy_name,
            'proxy_label': PROXY_LABELS[proxy_name],
            'driver': driver,
            'driver_label': label,
            'rho_metric_xmap_driver': f_obs,
            'p_metric_xmap_driver': f_p,
            'rho_driver_xmap_metric': r_obs,
            'p_driver_xmap_metric': r_p,
            'best_lag_metric_xmap_driver_kyr': sum_f['best_lag_kyr'],
            'best_lag_metric_xmap_driver_rho': sum_f['best_rho'],
            'best_lag_driver_xmap_metric_kyr': sum_r['best_lag_kyr'],
            'best_lag_driver_xmap_metric_rho': sum_r['best_rho'],
            'lag_library_requested': lag_lib_info['requested_library_size'],
            'lag_library_effective': lag_lib_info['effective_library_size'],
            'lag_min_rows_across_lags': lag_lib_info['min_rows_across_lags'],
        })
        gc.collect()

    res = pd.DataFrame(rows)
    res.to_csv(TAB / f'coldduration_ccm_lagtest_results_{proxy_name}.csv', index=False)
    lib_summary_df = pd.DataFrame(lib_summary_rows)
    return stats, res, std_curves, lag_curves, lib_summary_df


def main():
    setup_style()
    driver_lib = build_driver_library()
    plot_pre_phase_diagnostic(driver_lib['pre_phase'])
    plot_driver_inputs(driver_lib)
    driver_lib['pre_phase'].to_csv(TAB / 'pre_deamplitude_diagnostic_series_native.csv', index=False)
    driver_lib['lr04'].to_csv(TAB / 'lr04_1k_10k_native.csv', index=False)
    driver_lib['co2'].to_csv(TAB / 'co2_1k_10k_native.csv', index=False)

    results_by_proxy = {}
    std_curves_by_proxy = {}
    lag_curves_by_proxy = {}
    proxy_summaries = []

    for proxy in ['ch4', 'monsoon']:
        _, res, std_curves, lag_curves, proxy_summary = run_proxy(proxy, driver_lib)
        results_by_proxy[proxy] = res
        std_curves_by_proxy[proxy] = std_curves
        lag_curves_by_proxy[proxy] = lag_curves
        proxy_summaries.append(proxy_summary)

    all_res = pd.concat([results_by_proxy['ch4'], results_by_proxy['monsoon']], ignore_index=True)
    all_res.to_csv(TAB / 'coldduration_ccm_lagtest_results_all_proxies_predeamp_lr04_co2.csv', index=False)
    proxy_summary_df = pd.concat(proxy_summaries, ignore_index=True)
    proxy_summary_df.to_csv(TAB / 'proxy_series_summary_predeamp_lr04_co2.csv', index=False)

    plot_ccm_comparison(results_by_proxy, std_curves_by_proxy, lag_curves_by_proxy)
    plot_summary(results_by_proxy)

    with open(OUT / 'summary.txt', 'w', encoding='utf-8') as f:
        f.write('Cold-duration CCM and robust lag-test with de-amplitude PRE + LR04 10-kyr smooth + CO2 10-kyr smooth\n')
        f.write(f'Downsample={DOWNSAMPLE_YR} yr, E={E}, tau={TAU}, row1 surrogates={N_SURROGATES}\n')
        f.write(f'Lag range={LAG_RANGE_KYR[0]}..{LAG_RANGE_KYR[-1]} kyr\n')
        f.write(f'Lag-test boot libraries={LAG_BOOTSTRAP_LIBS}, requested library size={LAG_LIBRARY_SIZE}\n')
        f.write(f'PRE transform: pre / ecc (with ecc floor at 5th percentile), then rescaled to pre std.\n')
        f.write(f'LR04 processing: interpolated to 1-kyr grid, then centered {LR04_SMOOTH_WINDOW_KYR}-kyr moving mean\n')
        f.write(f'CO2 processing: interpolated to 1-kyr grid, then centered {LR04_SMOOTH_WINDOW_KYR}-kyr moving mean\n\n')
        for proxy in ['ch4', 'monsoon']:
            f.write(f'Proxy: {proxy}\n')
            sub = results_by_proxy[proxy]
            for _, r in sub.iterrows():
                f.write(f"  Driver: {r.driver}\n")
                f.write(f"    CCM fwd: rho={r.rho_metric_xmap_driver:.3f}, p={r.p_metric_xmap_driver:.3f}\n")
                f.write(f"    CCM rev: rho={r.rho_driver_xmap_metric:.3f}, p={r.p_driver_xmap_metric:.3f}\n")
                f.write(f"    LAG fwd: best lag={r.best_lag_metric_xmap_driver_kyr:.0f} kyr, rho={r.best_lag_metric_xmap_driver_rho:.3f}\n")
                f.write(f"    LAG rev: best lag={r.best_lag_driver_xmap_metric_kyr:.0f} kyr, rho={r.best_lag_driver_xmap_metric_rho:.3f}\n")
            f.write('\n')

    print('Done.')
    print(all_res)


if __name__ == '__main__':
    main()
