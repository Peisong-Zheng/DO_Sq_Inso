#!/usr/bin/env python3
from __future__ import annotations

import string
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import coherence, csd, welch

# =========================
# 用户可优先修改的参数
# =========================
WINDOW_YR = 20_000            # cold duration 的滑动窗口长度（年）
DOWNSAMPLE_YR = 1000          # 频谱分析前重采样到 1 kyr
FS = 1 / 1000.0              # 采样频率：1 kyr 一个样点 -> cycles / year
PERIOD_XLIM = (10, 150)      # 频谱图横轴：周期范围（kyr）
PERIOD_MARKERS = {'pre': 23, 'obl': 41, 'ecc': 100}

# 绘图风格（尽量靠近简洁的 GRL 风格）
FONT_SIZE = 11
LINE_WIDTH = 1.25
GRID_ALPHA = 0.18
FIG_DPI = 300

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR_CANDIDATES = [
    SCRIPT_DIR / 'data',   # 推荐：数据放在脚本同级 data/ 文件夹
    SCRIPT_DIR,            # 或者直接和脚本放在同一文件夹
    Path.cwd() / 'data',
    Path.cwd(),
    Path('/mnt/data'),     # 当前 ChatGPT 容器环境
]

OUT_DIR = SCRIPT_DIR / 'results' / 'cold_duration_cross_spectra'
FIG_DIR = OUT_DIR / 'figures'
TAB_DIR = OUT_DIR / 'tables'
for p in [OUT_DIR, FIG_DIR, TAB_DIR]:
    p.mkdir(parents=True, exist_ok=True)

PROXY_FILES = {
    'ch4': 'ch4_mcv_square_wave_selected_full_50yr.csv',
    'monsoon': 'monsoon_mcv_square_wave_selected_full_50yr.csv',
}

DRIVER_FILES = {
    'pre': 'pre_800_inter100.txt',
    'obl': 'obl_800_inter100.txt',
    'ecc': 'ecc_1000_inter100.txt',
    'lr04': 'lr04.xlsx',
    'co2': 'composite_co2.xlsx',
    'at': 'AT.csv',
}

PROXY_LABELS = {
    'ch4': 'CH$_4$ cold duration',
    'monsoon': 'Monsoon cold duration',
}

DRIVER_LABELS = {
    'pre': 'Precession',
    'obl': 'Obliquity',
    'ecc': 'Eccentricity',
    'lr04': 'LR04',
    'co2': 'CO$_2$',
    'at': 'AT',
}

DRIVER_ORDER = ['pre', 'obl', 'ecc', 'lr04', 'co2', 'at']


def resolve_file(name: str) -> Path:
    """在多个候选目录中寻找文件，便于以后在本机上直接迁移运行。"""
    for base in DATA_DIR_CANDIDATES:
        p = base / name
        if p.exists():
            return p
    raise FileNotFoundError(f'Cannot find file: {name}')


def normalize_col(s: str) -> str:
    return str(s).strip().lower().replace('\n', ' ')


def pick_column(df: pd.DataFrame, candidates: list[str]) -> str:
    norm_map = {normalize_col(c): c for c in df.columns}
    for cand in candidates:
        cand_norm = normalize_col(cand)
        if cand_norm in norm_map:
            return norm_map[cand_norm]
    # 宽松匹配：只要候选字符串被包含即可
    for cand in candidates:
        cand_norm = normalize_col(cand)
        for k, v in norm_map.items():
            if cand_norm in k:
                return v
    raise KeyError(f'Cannot find column among candidates: {candidates}. Available: {list(df.columns)}')


def style_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=FONT_SIZE)
    ax.grid(True, ls='--', lw=0.5, alpha=GRID_ALPHA)


def add_panel_label(ax, label: str):
    # panel label 放在坐标轴外侧左上方，避免遮挡数据区
    ax.text(-0.16, 1.04, label, transform=ax.transAxes, ha='left', va='bottom',
            fontsize=FONT_SIZE + 1, fontweight='bold', clip_on=False)


def add_period_markers(ax):
    colors = {'pre': '0.45', 'obl': '0.45', 'ecc': '0.45'}
    for key, p in PERIOD_MARKERS.items():
        ax.axvline(p, color=colors[key], lw=0.9, ls='--', zorder=0)
    ymin, ymax = ax.get_ylim()
    ytxt = ymin + 0.92 * (ymax - ymin)
    for key, p in PERIOD_MARKERS.items():
        ax.text(p, ytxt, key, rotation=90, ha='right', va='top', fontsize=FONT_SIZE - 2,
                color='0.35', bbox=dict(facecolor='white', edgecolor='none', pad=0.6, alpha=0.75))


def zscore(x):
    x = np.asarray(x, float)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd == 0:
        return x * np.nan
    return (x - np.nanmean(x)) / sd


def interp_to_grid(target_age, src_age, src_val):
    order = np.argsort(src_age)
    src_age = np.asarray(src_age, float)[order]
    src_val = np.asarray(src_val, float)[order]
    return np.interp(target_age, src_age, src_val)


# =========================
# cold duration 计算部分（沿用原脚本思路）
# =========================
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


def moving_sum_centered(x, win_pts):
    y = np.convolve(x, np.ones(win_pts, float), mode='valid')
    h = win_pts // 2
    return y, h


def moving_mean_centered(x, win_pts):
    y, h = moving_sum_centered(x, win_pts)
    return y / win_pts, h


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


def prepare_proxy_sequence(path: Path):
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


# =========================
# 各驱动变量读取函数
# =========================
def load_orbital_series(path: Path, name: str):
    df = pd.read_csv(path, sep=r'\s+', header=None, engine='python')
    out = df.iloc[:, :2].copy()
    out.columns = ['age_raw', name]
    out['age_yrBP'] = out['age_raw'].abs() * 1000.0
    out[name] = pd.to_numeric(out[name], errors='coerce')
    return out[['age_yrBP', name]].dropna().sort_values('age_yrBP').reset_index(drop=True)


def load_lr04_series(path: Path):
    df = pd.read_excel(path)
    age_col = pick_column(df, ['Time (ka)', 'time'])
    val_col = pick_column(df, ['Benthic d18O (per mil)', 'benthic d18o', 'd18o'])
    out = df[[age_col, val_col]].copy()
    out.columns = ['age_raw', 'lr04']
    out['age_yrBP'] = pd.to_numeric(out['age_raw'], errors='coerce') * 1000.0
    out['lr04'] = pd.to_numeric(out['lr04'], errors='coerce')
    return out[['age_yrBP', 'lr04']].dropna().sort_values('age_yrBP').reset_index(drop=True)


def load_co2_series(path: Path):
    df = pd.read_excel(path)
    age_col = pick_column(df, ['Gasage (yr BP)', 'gasage', 'age'])
    val_col = pick_column(df, ['CO2 (ppmv)', 'co2'])
    out = df[[age_col, val_col]].copy()
    out.columns = ['age_raw', 'co2']
    out['age_yrBP'] = pd.to_numeric(out['age_raw'], errors='coerce').abs()
    out['co2'] = pd.to_numeric(out['co2'], errors='coerce')
    return out[['age_yrBP', 'co2']].dropna().sort_values('age_yrBP').reset_index(drop=True)


def load_at_series(path: Path):
    df = pd.read_csv(path)
    age_col = pick_column(df, ['age'])
    val_col = pick_column(df, ['AT', 'temperature'])
    out = df[[age_col, val_col]].copy()
    out.columns = ['age_raw', 'at']
    out['age_yrBP'] = pd.to_numeric(out['age_raw'], errors='coerce').abs()
    out['at'] = pd.to_numeric(out['at'], errors='coerce')
    return out[['age_yrBP', 'at']].dropna().sort_values('age_yrBP').reset_index(drop=True)


def load_driver_series(name: str) -> pd.DataFrame:
    path = resolve_file(DRIVER_FILES[name])
    if name in ['pre', 'obl', 'ecc']:
        return load_orbital_series(path, name)
    if name == 'lr04':
        return load_lr04_series(path)
    if name == 'co2':
        return load_co2_series(path)
    if name == 'at':
        return load_at_series(path)
    raise ValueError(name)


# =========================
# 构建统一分析数据表
# =========================
def build_dataset(proxy_name: str):
    stats = prepare_proxy_sequence(resolve_file(PROXY_FILES[proxy_name]))['window_stats'].copy()
    step = max(1, int(round(DOWNSAMPLE_YR / 50)))
    stats = stats.iloc[::step].reset_index(drop=True)

    for name in DRIVER_ORDER:
        df = load_driver_series(name)
        stats[name] = interp_to_grid(stats['age_yrBP'].values, df['age_yrBP'].values, df[name].values)

    stats['mean_cold_duration_kyr'] = stats['mean_cold_duration_yr'] / 1000.0
    keep_cols = ['mean_cold_duration_kyr'] + DRIVER_ORDER
    stats = stats.dropna(subset=keep_cols).copy()
    stats = stats.sort_values('age_yrBP', ascending=False).reset_index(drop=True)
    return stats


# =========================
# 频谱、互功率谱、相干谱
# =========================
def get_spectral_params(x, y=None):
    if y is None:
        n = len(x)
    else:
        n = min(len(x), len(y))
    nperseg = min(256, n)
    noverlap = nperseg // 2
    return nperseg, noverlap


def compute_auto_spectrum(series):
    x = zscore(series)
    nperseg, noverlap = get_spectral_params(x)
    f, pxx = welch(x, fs=FS, window='hann', detrend='linear',
                   nperseg=nperseg, noverlap=noverlap, scaling='density')
    m = f > 0
    return pd.DataFrame({
        'frequency_cpy': f[m],
        'period_kyr': 1 / f[m] / 1000.0,
        'power': pxx[m],
    })


def compute_cross_spectrum(x, y):
    x = zscore(x)
    y = zscore(y)
    nperseg, noverlap = get_spectral_params(x, y)
    f, pxy = csd(x, y, fs=FS, window='hann', detrend='linear',
                 nperseg=nperseg, noverlap=noverlap, scaling='density')
    m = f > 0
    return pd.DataFrame({
        'frequency_cpy': f[m],
        'period_kyr': 1 / f[m] / 1000.0,
        'cross_power_abs': np.abs(pxy[m]),
        'cross_power_real': np.real(pxy[m]),
        'cross_power_imag': np.imag(pxy[m]),
        'phase_rad': np.angle(pxy[m]),
    })


def compute_coherence_spectrum(x, y):
    x = zscore(x)
    y = zscore(y)
    nperseg, noverlap = get_spectral_params(x, y)
    f, coh = coherence(x, y, fs=FS, window='hann', detrend='linear',
                       nperseg=nperseg, noverlap=noverlap)
    m = f > 0
    return pd.DataFrame({
        'frequency_cpy': f[m],
        'period_kyr': 1 / f[m] / 1000.0,
        'coherence': coh[m],
    })


def local_peak(spec_df: pd.DataFrame, col: str, pmin: float, pmax: float):
    sub = spec_df[(spec_df['period_kyr'] >= pmin) & (spec_df['period_kyr'] <= pmax)].copy()
    if len(sub) == 0:
        return {'peak_period_kyr': np.nan, 'peak_value': np.nan}
    row = sub.iloc[sub[col].values.argmax()]
    return {'peak_period_kyr': float(row['period_kyr']), 'peak_value': float(row[col])}


# =========================
# 绘图函数
# =========================
def plot_power_spectra(auto_specs: dict[str, pd.DataFrame]):
    fig, axes = plt.subplots(2, 1, figsize=(5.4, 5.2), dpi=FIG_DPI, sharex=True)
    panel_labels = iter(string.ascii_lowercase)

    for ax, proxy in zip(axes, ['ch4', 'monsoon']):
        df = auto_specs[proxy].sort_values('period_kyr')
        m = (df['period_kyr'] >= PERIOD_XLIM[0]) & (df['period_kyr'] <= PERIOD_XLIM[1])
        ax.plot(df.loc[m, 'period_kyr'], df.loc[m, 'power'], color='black', lw=LINE_WIDTH)
        ax.set_xlim(*PERIOD_XLIM)
        ax.set_ylabel('Power', fontsize=FONT_SIZE)
        ax.text(0.885, 0.16, PROXY_LABELS[proxy], transform=ax.transAxes,
                ha='right', va='top', fontsize=FONT_SIZE)
        style_axes(ax)
        ax.set_ylim(bottom=0)
        add_period_markers(ax)
        add_panel_label(ax, next(panel_labels))

    axes[-1].set_xlabel('Period (kyr)', fontsize=FONT_SIZE)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig01_cold_duration_power_spectra_linear.pdf', bbox_inches='tight')
    plt.close(fig)


def plot_grid_spectra(spec_dict: dict[tuple[str, str], pd.DataFrame], value_col: str,
                      y_label: str, output_name: str, coherence_mode: bool = False):
    # 版式改为上下两组：
    # 上半部分 CH4：2×3；下半部分 Monsoon：2×3；两组间留空白，更适合论文排版
    fig = plt.figure(figsize=(7.2, 9.4), dpi=FIG_DPI)
    gs = fig.add_gridspec(nrows=5, ncols=3, height_ratios=[1, 1, 0.18, 1, 1], hspace=0.58, wspace=0.30)
    panel_labels = iter(string.ascii_lowercase)

    axes = {
        ('ch4', 'pre'): fig.add_subplot(gs[0, 0]),
        ('ch4', 'obl'): fig.add_subplot(gs[0, 1]),
        ('ch4', 'ecc'): fig.add_subplot(gs[0, 2]),
        ('ch4', 'lr04'): fig.add_subplot(gs[1, 0]),
        ('ch4', 'co2'): fig.add_subplot(gs[1, 1]),
        ('ch4', 'at'): fig.add_subplot(gs[1, 2]),
        ('monsoon', 'pre'): fig.add_subplot(gs[3, 0]),
        ('monsoon', 'obl'): fig.add_subplot(gs[3, 1]),
        ('monsoon', 'ecc'): fig.add_subplot(gs[3, 2]),
        ('monsoon', 'lr04'): fig.add_subplot(gs[4, 0]),
        ('monsoon', 'co2'): fig.add_subplot(gs[4, 1]),
        ('monsoon', 'at'): fig.add_subplot(gs[4, 2]),
    }

    # 每组顶行加列标题
    for drv in ['pre', 'obl', 'ecc']:
        axes[('ch4', drv)].set_title(DRIVER_LABELS[drv], fontsize=FONT_SIZE, pad=8)
        axes[('monsoon', drv)].set_title(DRIVER_LABELS[drv], fontsize=FONT_SIZE, pad=8)
    for drv in ['lr04', 'co2', 'at']:
        axes[('ch4', drv)].set_title(DRIVER_LABELS[drv], fontsize=FONT_SIZE, pad=8)
        axes[('monsoon', drv)].set_title(DRIVER_LABELS[drv], fontsize=FONT_SIZE, pad=8)

    # 组标签：水平放在每组上方左侧，读起来更自然
    fig.text(0.065, 1.025, 'CH$_4$ cold duration', ha='left', va='top', fontsize=FONT_SIZE + 1)
    fig.text(0.065, 0.495, 'Monsoon cold duration', ha='left', va='top', fontsize=FONT_SIZE + 1)

    for proxy in ['ch4', 'monsoon']:
        for drv in DRIVER_ORDER:
            ax = axes[(proxy, drv)]
            df = spec_dict[(proxy, drv)].sort_values('period_kyr')
            m = (df['period_kyr'] >= PERIOD_XLIM[0]) & (df['period_kyr'] <= PERIOD_XLIM[1])
            ax.plot(df.loc[m, 'period_kyr'], df.loc[m, value_col], color='black', lw=LINE_WIDTH)
            ax.set_xlim(*PERIOD_XLIM)
            if coherence_mode:
                ax.set_ylim(0, 1.02)
            else:
                ax.set_ylim(bottom=0)
            style_axes(ax)
            add_period_markers(ax)
            add_panel_label(ax, next(panel_labels))
            if drv in ['pre', 'lr04']:
                ax.set_ylabel(y_label, fontsize=FONT_SIZE)
            if drv in ['lr04', 'co2', 'at']:
                ax.set_xlabel('Period (kyr)', fontsize=FONT_SIZE)

    fig.subplots_adjust(left=0.10, right=0.985, top=0.95, bottom=0.06)
    fig.savefig(FIG_DIR / output_name, bbox_inches='tight')
    plt.close(fig)


# =========================
# 主程序
# =========================
def main():
    datasets = {proxy: build_dataset(proxy) for proxy in PROXY_FILES}

    # 保存对齐后的分析数据，便于你后续复核
    for proxy, df in datasets.items():
        df.to_csv(TAB_DIR / f'aligned_dataset_{proxy}.csv', index=False)

    auto_specs: dict[str, pd.DataFrame] = {}
    cross_specs: dict[tuple[str, str], pd.DataFrame] = {}
    coherence_specs: dict[tuple[str, str], pd.DataFrame] = {}

    for proxy, df in datasets.items():
        auto_specs[proxy] = compute_auto_spectrum(df['mean_cold_duration_kyr'])
        auto_specs[proxy].sort_values('period_kyr').to_csv(
            TAB_DIR / f'auto_spectrum_{proxy}_cold_duration.csv', index=False
        )

        for drv in DRIVER_ORDER:
            cross_specs[(proxy, drv)] = compute_cross_spectrum(df['mean_cold_duration_kyr'], df[drv])
            cross_specs[(proxy, drv)].sort_values('period_kyr').to_csv(
                TAB_DIR / f'cross_spectrum_{proxy}_cold_duration_vs_{drv}.csv', index=False
            )

            coherence_specs[(proxy, drv)] = compute_coherence_spectrum(df['mean_cold_duration_kyr'], df[drv])
            coherence_specs[(proxy, drv)].sort_values('period_kyr').to_csv(
                TAB_DIR / f'coherence_{proxy}_cold_duration_vs_{drv}.csv', index=False
            )

    # 峰值摘要表：auto / cross / coherence 都汇总一下
    rows = []
    for proxy, auto in auto_specs.items():
        for band, center in PERIOD_MARKERS.items():
            hw = 8 if band != 'ecc' else 30
            pk = local_peak(auto, 'power', center - hw, center + hw)
            rows.append({'proxy': proxy, 'spectrum': 'auto_cold_duration', 'driver_band': band, **pk})

    for (proxy, drv), spec in cross_specs.items():
        center = PERIOD_MARKERS.get(drv, np.nan)
        if np.isfinite(center):
            hw = 8 if drv != 'ecc' else 30
            pk = local_peak(spec, 'cross_power_abs', center - hw, center + hw)
            rows.append({'proxy': proxy, 'spectrum': f'cross_with_{drv}', 'driver_band': drv, **pk})

    for (proxy, drv), spec in coherence_specs.items():
        center = PERIOD_MARKERS.get(drv, np.nan)
        if np.isfinite(center):
            hw = 8 if drv != 'ecc' else 30
            pk = local_peak(spec, 'coherence', center - hw, center + hw)
            rows.append({'proxy': proxy, 'spectrum': f'coherence_with_{drv}', 'driver_band': drv, **pk})

    pd.DataFrame(rows).to_csv(TAB_DIR / 'spectral_band_peak_summary.csv', index=False)

    # 出图
    plot_power_spectra(auto_specs)
    plot_grid_spectra(cross_specs, value_col='cross_power_abs', y_label='|Cross power|',
                      output_name='fig02_cold_duration_cross_power_spectra_2x6.pdf', coherence_mode=False)
    plot_grid_spectra(coherence_specs, value_col='coherence', y_label='Coherence',
                      output_name='fig03_cold_duration_coherence_2x6.pdf', coherence_mode=True)

    print('Done.')
    print(f'Outputs written to: {OUT_DIR}')


if __name__ == '__main__':
    main()
