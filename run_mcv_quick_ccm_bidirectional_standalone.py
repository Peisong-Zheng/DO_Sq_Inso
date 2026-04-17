#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# ============================================================
# Basic paths (relative to script location)
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / 'data'
RESULTS_DIR = SCRIPT_DIR / 'results' / 'mcv_quick_ccm_outputs_v7'
FIG_DIR = RESULTS_DIR / 'figures'
TAB_DIR = RESULTS_DIR / 'tables'
for p in [RESULTS_DIR, FIG_DIR, TAB_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# ============================================================
# Inputs
# ============================================================
CH4_STATE_FILE = 'ch4_mcv_square_wave_selected_full_50yr.csv'
PRE_FILE = 'pre_800_inter100.txt'
OBL_FILE = 'obl_800_inter100.txt'
ECC_FILE = 'ecc_1000_inter100.txt'

# ============================================================
# Fixed-parameter quick CCM settings
# ============================================================
DOWNSAMPLE_YR = 1000
E = 4
TAU = -2
N_SURROGATES = 200
RNG_SEED = 42
LIB_POINTS = 6
N_NEIGH = E + 1
WINDOW_YR = 20_000


# ============================================================
# Time-series utilities (ported from benchmark workflow)
# ============================================================


def resolve_input_file(filename: str) -> Path:
    candidates = [DATA_DIR / filename, SCRIPT_DIR / filename]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Cannot find {filename}. Expected in either {DATA_DIR} or {SCRIPT_DIR}.")
def load_orbital_series(path: Path, value_name: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r'\s+', header=None, engine='python')
    out = df.iloc[:, :2].copy()
    out.columns = ['age_raw', value_name]
    out['age_yrBP'] = out['age_raw'].abs() * 1000.0
    out[value_name] = pd.to_numeric(out[value_name], errors='coerce')
    return out[['age_yrBP', value_name]].dropna().sort_values('age_yrBP').reset_index(drop=True)


def interp_to_grid(age_yr, src_age, src_val):
    src_age = np.asarray(src_age, float)
    src_val = np.asarray(src_val, float)
    order = np.argsort(src_age)
    src_age = src_age[order]
    src_val = src_val[order]
    mask = np.isfinite(src_age) & np.isfinite(src_val)
    return np.interp(age_yr, src_age[mask], src_val[mask])


def zscore(x):
    x = np.asarray(x, float)
    sd = np.nanstd(x)
    return (x - np.nanmean(x)) / sd if sd > 0 else np.zeros_like(x)


def moving_sum_centered(arr, win_pts):
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr[None, :]
        squeeze = True
    else:
        squeeze = False
    if win_pts % 2 == 0:
        win_pts += 1
    h = win_pts // 2
    padded = np.pad(arr, ((0, 0), (1, 0)), mode='constant', constant_values=0)
    cs = np.cumsum(padded, axis=1)
    out = cs[:, win_pts:] - cs[:, :-win_pts]
    if squeeze:
        return out[0], h
    return out, h


def moving_mean_centered(arr, win_pts):
    sums, h = moving_sum_centered(arr, win_pts)
    return sums / win_pts, h


def extract_intervals_from_binary(age_asc, cold_bool_asc):
    age = np.asarray(age_asc)
    cold = np.asarray(cold_bool_asc).astype(int)
    change = np.diff(np.r_[0, cold, 0])
    starts = np.where(change == 1)[0]
    ends = np.where(change == -1)[0] - 1
    return pd.DataFrame({
        'young_age_yrBP': age[starts],
        'old_age_yrBP': age[ends],
        'duration_yr': age[ends] - age[starts]
    })


def extract_warm_intervals_from_binary(age_asc, cold_bool_asc):
    age = np.asarray(age_asc)
    warm = (~np.asarray(cold_bool_asc)).astype(int)
    change = np.diff(np.r_[0, warm, 0])
    starts = np.where(change == 1)[0]
    ends = np.where(change == -1)[0] - 1
    return pd.DataFrame({
        'young_age_yrBP': age[starts],
        'old_age_yrBP': age[ends],
        'duration_yr': age[ends] - age[starts],
        'start_truncated': starts == 0,
        'end_truncated': ends == len(age) - 1,
    })


def local_mean_durations(window_centers, intervals_df, window_yr):
    mids = 0.5 * (intervals_df['young_age_yrBP'].values + intervals_df['old_age_yrBP'].values)
    dur = intervals_df['duration_yr'].values
    out = np.full(len(window_centers), np.nan)
    half = window_yr / 2
    for i, c in enumerate(window_centers):
        mask = (mids >= c - half) & (mids <= c + half)
        if np.any(mask):
            out[i] = np.mean(dur[mask])
    return out


def compute_window_stats(age_asc, cold_bool_asc, warming_indicator_asc, cold_intervals_df, warm_intervals_df, window_yr):
    dt = np.median(np.diff(age_asc))
    win_pts = int(round(window_yr / dt))
    if win_pts % 2 == 0:
        win_pts += 1
    Eseries, h = moving_sum_centered(warming_indicator_asc.astype(float), win_pts)
    Pseries, _ = moving_mean_centered(cold_bool_asc.astype(float), win_pts)
    centers = age_asc[h: len(age_asc) - h]

    warm_full = warm_intervals_df.loc[
        ~warm_intervals_df['start_truncated'] & ~warm_intervals_df['end_truncated']
    ].copy()
    Tcold = local_mean_durations(centers, cold_intervals_df, window_yr)
    Twarm = local_mean_durations(centers, warm_full, window_yr)
    return pd.DataFrame({
        'age_yrBP': centers,
        'warming_count_20k': Eseries,
        'cold_fraction_20k': Pseries,
        'mean_cold_duration_yr': Tcold,
        'mean_warm_duration_yr': Twarm,
    })


def prepare_ch4_sequence(path_csv: Path):
    df = pd.read_csv(path_csv).sort_values('age_yrBP')
    required = {'age_yrBP', 'mcv_state'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'{path_csv.name} missing required columns: {sorted(missing)}')

    age_asc = df['age_yrBP'].to_numpy(dtype=float)
    cold = (df['mcv_state'].to_numpy() == 0)

    cold_intervals = extract_intervals_from_binary(age_asc, cold)
    warm_intervals = extract_warm_intervals_from_binary(age_asc, cold)

    warming_indicator = np.zeros_like(age_asc, dtype=int)
    cooling_indicator = np.zeros_like(age_asc, dtype=int)
    for a in cold_intervals['young_age_yrBP'].values:
        warming_indicator[np.argmin(np.abs(age_asc - a))] = 1
    for a in cold_intervals['old_age_yrBP'].values:
        cooling_indicator[np.argmin(np.abs(age_asc - a))] = 1
    if cold[0]:
        warming_indicator[0] = 0
    if cold[-1]:
        cooling_indicator[-1] = 0

    stats = compute_window_stats(age_asc, cold, warming_indicator, cold_intervals, warm_intervals, WINDOW_YR)
    return {
        'age_asc': age_asc,
        'cold_asc': cold,
        'warming_indicator_asc': warming_indicator,
        'cooling_indicator_asc': cooling_indicator,
        'window_stats': stats,
        'ch4_hp': df['ch4_hp'].to_numpy(dtype=float) if 'ch4_hp' in df.columns else None,
        'mcv_state': df['mcv_state'].to_numpy(dtype=int),
    }


# ============================================================
# CCM helpers
# ============================================================
def build_embedding(x: np.ndarray, E: int, tau: int):
    lag = abs(int(tau))
    start = (E - 1) * lag
    valid_idx = np.arange(start, len(x))
    emb = np.column_stack([x[valid_idx - j * lag] for j in range(E)])
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


def libsizes_for_n(n_valid: int) -> np.ndarray:
    vals = np.unique(np.round(np.linspace(max(E + 5, int(0.1 * n_valid)), int(0.9 * n_valid), LIB_POINTS)).astype(int))
    vals = vals[vals >= E + 2]
    return vals


def simplex_rho_from_dist_fast(distmat: np.ndarray, target_full: np.ndarray, valid_idx: np.ndarray, lib_subset: np.ndarray):
    lib_subset = np.asarray(lib_subset, int)
    lib_pos = np.searchsorted(valid_idx, lib_subset)
    dsub = distmat[:, lib_pos].copy()

    # exclude self matches where prediction point is also in library
    rowpos = lib_pos
    colpos = np.arange(len(lib_subset))
    dsub[rowpos, colpos] = np.inf

    finite_counts = np.isfinite(dsub).sum(axis=1)
    if np.nanmax(finite_counts) < 2:
        return np.nan
    k = min(N_NEIGH, len(lib_subset) - 1)
    if k < 2:
        return np.nan

    idx = np.argpartition(dsub, kth=k - 1, axis=1)[:, :k]
    d_nei = np.take_along_axis(dsub, idx, axis=1)
    order = np.argsort(d_nei, axis=1)
    idx = np.take_along_axis(idx, order, axis=1)
    d_nei = np.take_along_axis(d_nei, order, axis=1)

    lib_target = target_full[lib_subset]
    tar_nei = lib_target[idx]
    d1 = np.maximum(d_nei[:, [0]], 1e-12)
    w = np.exp(-d_nei / d1)
    preds = np.sum(w * tar_nei, axis=1) / np.sum(w, axis=1)
    return pearson_r(preds, target_full[valid_idx])


def ccm_curve_and_nulls_fast(source_series: np.ndarray, target_series: np.ndarray, E: int, tau: int, n_surrogates: int, rng_seed: int):
    valid_idx, manifold = build_embedding(source_series, E, tau)
    libs = libsizes_for_n(len(valid_idx))
    rng = np.random.default_rng(rng_seed)
    subsets = {int(L): np.sort(rng.choice(valid_idx, size=int(L), replace=False)) for L in libs}
    distmat = cdist(manifold, manifold, metric='euclidean')

    obs = []
    for L in libs:
        subset = subsets[int(L)]
        obs.append(simplex_rho_from_dist_fast(distmat, target_series, valid_idx, subset))
    obs = np.array(obs, float)

    null_curves = np.zeros((n_surrogates, len(libs)), float)
    for s in range(n_surrogates):
        perm = np.random.default_rng(rng_seed + 1000 + s).permutation(target_series)
        for j, L in enumerate(libs):
            null_curves[s, j] = simplex_rho_from_dist_fast(distmat, perm, valid_idx, subsets[int(L)])

    null_lo = np.nanpercentile(null_curves, 2.5, axis=0)
    null_med = np.nanpercentile(null_curves, 50, axis=0)
    null_hi = np.nanpercentile(null_curves, 97.5, axis=0)
    obs_terminal = float(obs[-1])
    p = float((np.sum(null_curves[:, -1] >= obs_terminal) + 1) / (n_surrogates + 1))
    curve_df = pd.DataFrame({
        'LibSize': libs.astype(int),
        'rho_obs': obs,
        'rho_null_median': null_med,
        'rho_null_lo95': null_lo,
        'rho_null_hi95': null_hi,
    })
    return curve_df, obs_terminal, p


# ============================================================
# Dataset builder
# ============================================================
def build_dataset():
    seq = prepare_ch4_sequence(resolve_input_file(CH4_STATE_FILE))
    stats = seq['window_stats'].copy()
    step = max(1, int(round(DOWNSAMPLE_YR / 50)))  # original state file is 50 yr
    stats = stats.iloc[::step].reset_index(drop=True)

    for name, path in [('pre', PRE_FILE), ('obl', OBL_FILE), ('ecc', ECC_FILE)]:
        df = load_orbital_series(resolve_input_file(path), name)
        stats[name] = interp_to_grid(stats['age_yrBP'].values, df['age_yrBP'].values, df[name].values)

    stats['warmings'] = stats['warming_count_20k']
    stats['cold_fraction'] = stats['cold_fraction_20k']
    stats['mean_cold_duration_kyr'] = stats['mean_cold_duration_yr'] / 1000.0

    # internal CCM order: oldest -> youngest in abstract time coordinate
    stats = stats.sort_values('age_yrBP', ascending=False).reset_index(drop=True)
    stats['Time'] = np.arange(1, len(stats) + 1)
    return stats


# ============================================================
# Main
# ============================================================
def main():
    stats = build_dataset()
    stats.to_csv(TAB_DIR / 'aligned_stats_orbitals_for_ccm.csv', index=False)

    metrics = [
        ('warmings', 'warmings (/20 kyr)'),
        ('cold_fraction', 'cold fraction'),
        ('mean_cold_duration_kyr', 'mean cold duration (kyr)'),
    ]
    drivers = [('pre', 'pre'), ('obl', 'obl'), ('ecc', 'ecc')]
    col_fwd = '#1f77b4'
    col_rev = '#d62728'

    # Input check figure
    fig, axes = plt.subplots(6, 1, figsize=(12, 10), dpi=180, sharex=True)
    cols = ['warmings', 'cold_fraction', 'mean_cold_duration_kyr', 'pre', 'obl', 'ecc']
    labels = ['warmings', 'cold frac', 'mean cold dur (kyr)', 'pre', 'obl', 'ecc']
    colors_list = ['black', 'black', 'black', 'tab:blue', 'tab:green', 'tab:red']
    for ax, col, lab, cc in zip(axes, cols, labels, colors_list):
        ax.plot(stats['age_yrBP'] / 1000, stats[col], color=cc, lw=1)
        ax.set_ylabel(lab)
    axes[0].set_title(
        f'Inputs for CCM (Δt={DOWNSAMPLE_YR} yr, E={E}, tau={TAU}, surrogates={N_SURROGATES}). '
        'Internal order oldest→youngest.'
    )
    axes[-1].set_xlabel('Age (kyr BP)')
    axes[-1].invert_xaxis()
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig01_timeseries_and_inputs.png', bbox_inches='tight')
    plt.close(fig)

    result_rows = []
    curve_store = {}
    for mi, (metric, _) in enumerate(metrics):
        for dj, (driver, _) in enumerate(drivers):
            print(f'Running bidirectional CCM for {metric} and {driver}...', flush=True)
            metric_series = zscore(stats[metric].to_numpy())
            driver_series = zscore(stats[driver].to_numpy())
            f_curve, f_obs, f_p = ccm_curve_and_nulls_fast(metric_series, driver_series, E, TAU, N_SURROGATES, RNG_SEED + mi * 100 + dj * 10 + 1)
            r_curve, r_obs, r_p = ccm_curve_and_nulls_fast(driver_series, metric_series, E, TAU, N_SURROGATES, RNG_SEED + mi * 100 + dj * 10 + 2)
            result_rows.append({
                'metric': metric,
                'driver': driver,
                'E': E,
                'tau': TAU,
                'downsample_yr': DOWNSAMPLE_YR,
                'n_surrogates': N_SURROGATES,
                'terminal_library_size': int(f_curve['LibSize'].iloc[-1]),
                'rho_metric_xmap_orbital': f_obs,
                'p_metric_xmap_orbital': f_p,
                'sig_metric_xmap_orbital': bool(f_p < 0.05),
                'rho_orbital_xmap_metric': r_obs,
                'p_orbital_xmap_metric': r_p,
                'sig_orbital_xmap_metric': bool(r_p < 0.05),
            })
            curve_store[(metric, driver, 'fwd')] = f_curve
            curve_store[(metric, driver, 'rev')] = r_curve

    res = pd.DataFrame(result_rows)
    res.to_csv(TAB_DIR / 'ccm_results_bidirectional.csv', index=False)

    # Main figure: bidirectional skill curves with null 95% CI
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), dpi=180)
    for i, (metric, _) in enumerate(metrics):
        for j, (driver, _) in enumerate(drivers):
            ax = axes[i, j]
            df_f = curve_store[(metric, driver, 'fwd')]
            df_r = curve_store[(metric, driver, 'rev')]
            ax.fill_between(df_f['LibSize'], df_f['rho_null_lo95'], df_f['rho_null_hi95'], color=col_fwd, alpha=0.15)
            ax.plot(df_f['LibSize'], df_f['rho_null_median'], '--', color=col_fwd, lw=1.0)
            ax.plot(df_f['LibSize'], df_f['rho_obs'], 'o-', color=col_fwd, lw=1.6, label='metric xmap orbital')
            ax.fill_between(df_r['LibSize'], df_r['rho_null_lo95'], df_r['rho_null_hi95'], color=col_rev, alpha=0.15)
            ax.plot(df_r['LibSize'], df_r['rho_null_median'], '--', color=col_rev, lw=1.0)
            ax.plot(df_r['LibSize'], df_r['rho_obs'], 's-', color=col_rev, lw=1.6, label='orbital xmap metric')
            rf = res.loc[(res.metric == metric) & (res.driver == driver)].iloc[0]
            ax.set_title(f'{metric} ↔ {driver}\nFwd p={rf.p_metric_xmap_orbital:.3f}, Rev p={rf.p_orbital_xmap_metric:.3f}')
            ax.grid(alpha=0.2, ls='--')
            if i == 2:
                ax.set_xlabel('Library size')
            if j == 0:
                ax.set_ylabel('CCM skill ρ')
            if i == 0 and j == 0:
                ax.legend(frameon=True, fontsize=8)
    fig.suptitle('Bidirectional fixed-parameter CCM skill curves with permutation-null 95% CI', y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig02_ccm_skill_curves_bidirectional_with_null95.png', bbox_inches='tight')
    plt.close(fig)

    # Summary heatmaps
    f_rho = res.pivot(index='metric', columns='driver', values='rho_metric_xmap_orbital').loc[[m[0] for m in metrics], [d[0] for d in drivers]]
    r_rho = res.pivot(index='metric', columns='driver', values='rho_orbital_xmap_metric').loc[[m[0] for m in metrics], [d[0] for d in drivers]]
    f_p = res.pivot(index='metric', columns='driver', values='p_metric_xmap_orbital').loc[[m[0] for m in metrics], [d[0] for d in drivers]]
    r_p = res.pivot(index='metric', columns='driver', values='p_orbital_xmap_metric').loc[[m[0] for m in metrics], [d[0] for d in drivers]]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=180)
    mats = [
        (f_rho, 'metric xmap orbital: terminal ρ', 'viridis', None),
        (r_rho, 'orbital xmap metric: terminal ρ', 'viridis', None),
        (f_p, 'metric xmap orbital: p-value', 'magma_r', (0, 1)),
        (r_p, 'orbital xmap metric: p-value', 'magma_r', (0, 1)),
    ]
    for ax, (mat, title, cmap, lims) in zip(axes.ravel(), mats):
        kwargs = {'cmap': cmap, 'aspect': 'auto'}
        if lims is not None:
            kwargs['vmin'], kwargs['vmax'] = lims
        im = ax.imshow(mat.values, **kwargs)
        ax.set_xticks(range(3), [d[0] for d in drivers])
        ax.set_yticks(range(3), [m[0] for m in metrics])
        ax.set_title(title)
        for ii in range(3):
            for jj in range(3):
                val = mat.values[ii, jj]
                txt = f'{val:.3f}'
                if 'p-value' in title and val < 0.05:
                    txt += '*'
                color = 'white' if (lims is not None and val < 0.5) or (lims is None and val > np.nanmax(mat.values) / 2) else 'black'
                ax.text(jj, ii, txt, ha='center', va='center', color=color)
        plt.colorbar(im, ax=ax, shrink=0.82)
    fig.suptitle('Summary of bidirectional fixed-parameter CCM tests', y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig03_summary_heatmaps_bidirectional.png', bbox_inches='tight')
    plt.close(fig)

    with open(RESULTS_DIR / 'summary.txt', 'w', encoding='utf-8') as f:
        f.write('Self-contained bidirectional fixed-parameter CCM quick test\n')
        f.write(f'Downsample={DOWNSAMPLE_YR} yr, E={E}, tau={TAU}, surrogates={N_SURROGATES}.\n')
        f.write('metric xmap orbital implies evidence for orbital -> metric; orbital xmap metric implies reverse direction.\n')
        f.write('This script is self-contained and no longer depends on mcv_sequence_falsification_benchmark_v1.py.\n\n')
        f.write(res.to_string(index=False))

    print('Done.')


if __name__ == '__main__':
    main()
