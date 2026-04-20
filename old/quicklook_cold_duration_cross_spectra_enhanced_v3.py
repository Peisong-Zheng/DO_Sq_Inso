#!/usr/bin/env python3
from __future__ import annotations

import math
import os
from pathlib import Path

# 让 matplotlib 使用可写缓存目录，避免不同环境下卡在 font cache
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import coherence, csd, welch

# ==========================================================
# 配置区：你以后若想改参数，优先看这里
# ==========================================================
WINDOW_YR = 20_000          # 计算局地 cold duration 的滑动窗口（yr）
DOWNSAMPLE_YR = 1_000       # 做谱分析前的目标采样间隔（yr）
FS = 1 / 1000.0             # 1 kyr 采样 -> cycles per year
PERIOD_XLIM = (10, 150)     # 图中展示的周期范围（kyr）
NPERSEG_MAX = 256           # Welch / CSD / coherence 的最大分段长度
NOVERLAP_FRAC = 0.5         # 分段重叠比例
COH_DISPLAY_THRESHOLD = 0.2 # phase 图中突出显示的 coherence 阈值
ALPHA_COH = 0.05            # coherence 近似显著性阈值的显著性水平

PERIOD_MARKERS = {
    "pre": 23,
    "obl": 41,
    "ecc": 100,
}
BAND_HALF_WIDTH = {
    "pre": 8,
    "obl": 8,
    "ecc": 30,
}

PROXY_LABELS = {
    "ch4": r"CH$_4$",
    "monsoon": r"Monsoon d$^{18}$O",
}
DRIVER_LABELS = {
    "pre": "Precession",
    "obl": "Obliquity",
    "ecc": "Eccentricity",
}

# ==========================================================
# 路径解析：优先兼容你的 data/ 路径，也兼容当前 /mnt/data 测试环境
# ==========================================================
THIS_DIR = Path(__file__).resolve().parent
SEARCH_DIRS = [
    Path("data"),
    THIS_DIR / "data",
    THIS_DIR,
    Path("/mnt/data"),
]


def resolve_file(name: str) -> Path:
    """在多个候选目录中寻找文件，便于脚本在不同环境直接运行。"""
    candidates = []
    for root in SEARCH_DIRS:
        p = root / name
        candidates.append(p)
        if p.exists():
            return p
    raise FileNotFoundError(f"Cannot find {name}. Tried: {candidates}")


PROXY_FILES = {
    "ch4": resolve_file("ch4_mcv_square_wave_selected_full_50yr.csv"),
    "monsoon": resolve_file("monsoon_mcv_square_wave_selected_full_50yr.csv"),
}
ORBITAL_FILES = {
    "pre": resolve_file("pre_800_inter100.txt"),
    "obl": resolve_file("obl_800_inter100.txt"),
    "ecc": resolve_file("ecc_1000_inter100.txt"),
}

# 用一个新的输出目录，避免与已有 root 权限目录冲突
OUT_DIR = THIS_DIR / "cold_duration_cross_spectra_enhanced_outputs"
FIG_DIR = OUT_DIR / "figures"
TAB_DIR = OUT_DIR / "tables"
for p in [OUT_DIR, FIG_DIR, TAB_DIR]:
    p.mkdir(parents=True, exist_ok=True)


# ==========================================================
# 数据准备部分：沿用你原始脚本的 cold/warm 区间与滑动统计逻辑
# ==========================================================
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
        rows.append(
            {
                "young_age_yrBP": age[s],
                "old_age_yrBP": age[e - 1],
                "duration_yr": age[e - 1] - age[s] + dt,
                "start_truncated": bool(s == 0),
                "end_truncated": bool(e == len(age)),
            }
        )
    return pd.DataFrame(rows)


def extract_warm_intervals_from_binary(age, cold):
    return extract_intervals_from_binary(age, ~np.asarray(cold, bool))


def moving_sum_centered(x, win_pts):
    y = np.convolve(x, np.ones(win_pts, float), mode="valid")
    h = win_pts // 2
    return y, h


def moving_mean_centered(x, win_pts):
    y, h = moving_sum_centered(x, win_pts)
    return y / win_pts, h


def local_mean_durations(centers, intervals_df, window_yr):
    vals = np.full_like(centers, np.nan, dtype=float)
    half = window_yr / 2
    for i, c in enumerate(centers):
        mask = (
            (intervals_df["young_age_yrBP"] >= c - half)
            & (intervals_df["old_age_yrBP"] <= c + half)
        )
        if mask.any():
            vals[i] = intervals_df.loc[mask, "duration_yr"].mean()
    return vals


def compute_window_stats(age, cold, warmings, cold_intervals, warm_intervals, window_yr):
    dt = np.median(np.diff(age))
    win_pts = int(round(window_yr / dt))
    if win_pts % 2 == 0:
        win_pts += 1

    e_series, h = moving_sum_centered(warmings.astype(float), win_pts)
    p_series, _ = moving_mean_centered(cold.astype(float), win_pts)
    centers = age[h : len(age) - h]
    warm_full = warm_intervals.loc[
        ~warm_intervals["start_truncated"] & ~warm_intervals["end_truncated"]
    ].copy()

    return pd.DataFrame(
        {
            "age_yrBP": centers,
            "warming_count_20k": e_series,
            "cold_fraction_20k": p_series,
            "mean_cold_duration_yr": local_mean_durations(centers, cold_intervals, window_yr),
            "mean_warm_duration_yr": local_mean_durations(centers, warm_full, window_yr),
        }
    )


def prepare_proxy_sequence(path: Path):
    df = pd.read_csv(path).sort_values("age_yrBP")
    age = df["age_yrBP"].to_numpy(float)
    cold = df["mcv_state"].to_numpy() == 0

    cold_intervals = extract_intervals_from_binary(age, cold)
    warm_intervals = extract_warm_intervals_from_binary(age, cold)

    # 每个 cold interval 的开始年龄，作为 warming 事件位置
    warmings = np.zeros_like(age, dtype=int)
    for a in cold_intervals["young_age_yrBP"].values:
        warmings[np.argmin(np.abs(age - a))] = 1
    if cold[0]:
        warmings[0] = 0

    return {
        "window_stats": compute_window_stats(
            age, cold, warmings, cold_intervals, warm_intervals, WINDOW_YR
        )
    }


def load_orbital_series(path: Path, name: str):
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    out = df.iloc[:, :2].copy()
    out.columns = ["age_raw", name]
    out["age_yrBP"] = out["age_raw"].abs() * 1000.0
    out[name] = pd.to_numeric(out[name], errors="coerce")
    return out[["age_yrBP", name]].dropna().sort_values("age_yrBP").reset_index(drop=True)


def interp_to_grid(age, src_age, src_val):
    order = np.argsort(src_age)
    return np.interp(age, np.asarray(src_age, float)[order], np.asarray(src_val, float)[order])


def zscore(x):
    x = np.asarray(x, float)
    return (x - np.nanmean(x)) / np.nanstd(x)


def build_dataset(proxy_name: str):
    stats = prepare_proxy_sequence(PROXY_FILES[proxy_name])["window_stats"].copy()

    # 原始 proxy 是 50 yr 采样，这里下采样到 1 kyr 做谱分析
    step = max(1, int(round(DOWNSAMPLE_YR / 50)))
    stats = stats.iloc[::step].reset_index(drop=True)

    for name, path in ORBITAL_FILES.items():
        df = load_orbital_series(path, name)
        stats[name] = interp_to_grid(stats["age_yrBP"].values, df["age_yrBP"].values, df[name].values)

    stats["mean_cold_duration_kyr"] = stats["mean_cold_duration_yr"] / 1000.0
    stats = stats.dropna(subset=["mean_cold_duration_kyr", "pre", "obl", "ecc"]).copy()

    # 年龄从老到新排列，更接近通常时间序列习惯；谱本身不受整体翻转影响
    stats = stats.sort_values("age_yrBP", ascending=False).reset_index(drop=True)
    return stats


# ==========================================================
# 谱分析部分
# ==========================================================
def get_spectral_kwargs(n: int):
    nperseg = min(NPERSEG_MAX, n)
    if nperseg < 8:
        raise ValueError(f"Series too short for spectral analysis: n={n}")
    noverlap = int(round(nperseg * NOVERLAP_FRAC))
    noverlap = min(noverlap, nperseg - 1)
    return {
        "fs": FS,
        "window": "hann",
        "detrend": "linear",
        "nperseg": nperseg,
        "noverlap": noverlap,
        "scaling": "density",
    }


def estimate_num_segments(n: int, nperseg: int, noverlap: int) -> int:
    step = nperseg - noverlap
    if n < nperseg or step <= 0:
        return 1
    return 1 + (n - nperseg) // step


def approximate_coherence_siglevel(num_segments: int, alpha: float = ALPHA_COH) -> float:
    """
    近似 95% coherence 阈值（零假设下）:
    C_alpha = 1 - alpha^(1/(L-1))
    其中 L 为近似独立分段数。
    注意：Welch + overlap + Hann 窗会使它只是近似值。
    """
    if num_segments <= 1:
        return float("nan")
    return 1.0 - alpha ** (1.0 / (num_segments - 1.0))


def compute_auto_spectrum(series):
    x = zscore(series)
    kw = get_spectral_kwargs(len(x))
    f, pxx = welch(x, **kw)
    m = f > 0
    spec = pd.DataFrame(
        {
            "frequency_cpy": f[m],
            "period_kyr": 1.0 / f[m] / 1000.0,
            "power": pxx[m],
        }
    )
    meta = {
        "n_samples": len(x),
        "nperseg": kw["nperseg"],
        "noverlap": kw["noverlap"],
        "num_segments": estimate_num_segments(len(x), kw["nperseg"], kw["noverlap"]),
    }
    return spec, meta


def compute_cross_products(x, y):
    """
    返回互谱相关结果。

    这里采用 scipy.signal.csd(x, y)，其相位定义来自 conj(FFT(x)) * FFT(y)。
    因此若 x=冷期时长, y=轨道驱动，则：
    phase > 0 近似表示 y（驱动）领先 x（冷期时长）；
    phase < 0 近似表示 y 落后 x。
    """
    x = zscore(x)
    y = zscore(y)
    kw = get_spectral_kwargs(min(len(x), len(y)))

    f1, pxx = welch(x, **kw)
    f2, pyy = welch(y, **kw)
    f3, pxy = csd(x, y, **kw)
    kw_coh = {k: v for k, v in kw.items() if k != "scaling"}
    f4, coh = coherence(x, y, **kw_coh)

    if not (np.allclose(f1, f2) and np.allclose(f1, f3) and np.allclose(f1, f4)):
        raise RuntimeError("Frequency grids from welch/csd/coherence are inconsistent.")

    m = f1 > 0
    f = f1[m]
    phase_rad = np.angle(pxy[m])
    phase_deg = np.degrees(phase_rad)

    # 用包裹相位换算等效时滞；范围约为 ±半个周期
    lag_yr = phase_rad / (2.0 * np.pi * f)
    lag_kyr = lag_yr / 1000.0

    spec = pd.DataFrame(
        {
            "frequency_cpy": f,
            "period_kyr": 1.0 / f / 1000.0,
            "power_x": pxx[m],
            "power_y": pyy[m],
            "cross_power_abs": np.abs(pxy[m]),
            "cross_power_real": np.real(pxy[m]),   # cospectrum
            "cross_power_imag": np.imag(pxy[m]),   # quadrature spectrum
            "coherence": coh[m],                   # magnitude-squared coherence
            "phase_rad": phase_rad,
            "phase_deg": phase_deg,
            "lag_kyr": lag_kyr,
        }
    )

    num_segments = estimate_num_segments(len(x), kw["nperseg"], kw["noverlap"])
    meta = {
        "n_samples": len(x),
        "nperseg": kw["nperseg"],
        "noverlap": kw["noverlap"],
        "num_segments": num_segments,
        "approx_coherence_sig95": approximate_coherence_siglevel(num_segments, ALPHA_COH),
    }
    return spec, meta


def subset_period_range(df: pd.DataFrame, xlim=PERIOD_XLIM):
    return df[(df["period_kyr"] >= xlim[0]) & (df["period_kyr"] <= xlim[1])].copy()


def local_peak(spec_df: pd.DataFrame, col: str, pmin: float, pmax: float):
    sub = spec_df[(spec_df["period_kyr"] >= pmin) & (spec_df["period_kyr"] <= pmax)].copy()
    if len(sub) == 0:
        return None
    return sub.iloc[sub[col].to_numpy().argmax()].copy()


def build_band_summary_row(proxy: str, driver: str, auto_spec: pd.DataFrame, cross_spec: pd.DataFrame):
    center = PERIOD_MARKERS[driver]
    hw = BAND_HALF_WIDTH[driver]

    auto_peak = local_peak(auto_spec, "power", center - hw, center + hw)
    cross_peak = local_peak(cross_spec, "cross_power_abs", center - hw, center + hw)
    coh_peak = local_peak(cross_spec, "coherence", center - hw, center + hw)

    out = {
        "proxy": proxy,
        "driver": driver,
        "band_center_kyr": center,
        "band_half_width_kyr": hw,
        "auto_power_peak_period_kyr": np.nan,
        "auto_power_peak_value": np.nan,
        "cross_power_peak_period_kyr": np.nan,
        "cross_power_peak_value": np.nan,
        "coherence_at_cross_power_peak": np.nan,
        "phase_deg_at_cross_power_peak": np.nan,
        "lag_kyr_at_cross_power_peak": np.nan,
        "coherence_peak_period_kyr": np.nan,
        "coherence_peak_value": np.nan,
        "phase_deg_at_coherence_peak": np.nan,
        "lag_kyr_at_coherence_peak": np.nan,
        "cross_power_at_coherence_peak": np.nan,
    }

    if auto_peak is not None:
        out["auto_power_peak_period_kyr"] = float(auto_peak["period_kyr"])
        out["auto_power_peak_value"] = float(auto_peak["power"])
    if cross_peak is not None:
        out["cross_power_peak_period_kyr"] = float(cross_peak["period_kyr"])
        out["cross_power_peak_value"] = float(cross_peak["cross_power_abs"])
        out["coherence_at_cross_power_peak"] = float(cross_peak["coherence"])
        out["phase_deg_at_cross_power_peak"] = float(cross_peak["phase_deg"])
        out["lag_kyr_at_cross_power_peak"] = float(cross_peak["lag_kyr"])
    if coh_peak is not None:
        out["coherence_peak_period_kyr"] = float(coh_peak["period_kyr"])
        out["coherence_peak_value"] = float(coh_peak["coherence"])
        out["phase_deg_at_coherence_peak"] = float(coh_peak["phase_deg"])
        out["lag_kyr_at_coherence_peak"] = float(coh_peak["lag_kyr"])
        out["cross_power_at_coherence_peak"] = float(coh_peak["cross_power_abs"])

    return out


# ==========================================================
# 绘图辅助函数
# ==========================================================
def add_period_markers(ax):
    colors = {"pre": "tab:blue", "obl": "tab:green", "ecc": "tab:red"}
    for key, p in PERIOD_MARKERS.items():
        ax.axvline(p, color=colors[key], lw=0.9, ls="--", alpha=0.7)

    ymin, ymax = ax.get_ylim()
    if ax.get_yscale() == "log":
        ytext = 10 ** (np.log10(ymin) + 0.92 * (np.log10(ymax) - np.log10(ymin)))
    else:
        ytext = ymin + 0.92 * (ymax - ymin)

    for key, p in PERIOD_MARKERS.items():
        ax.text(
            p,
            ytext,
            key,
            color=colors[key],
            rotation=90,
            ha="right",
            va="top",
            fontsize=8,
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1.2),
        )


def style_common_axis(ax, title: str, ylabel: str | None = None, xlabel: str | None = None):
    ax.set_title(title)
    ax.grid(alpha=0.22, ls="--")
    ax.set_xlim(*PERIOD_XLIM)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xlabel is not None:
        ax.set_xlabel(xlabel)


# ==========================================================
# 主程序
# ==========================================================
def main():
    datasets = {proxy: build_dataset(proxy) for proxy in PROXY_FILES}
    auto_specs = {}
    auto_meta = {}
    cross_specs = {}
    cross_meta = {}
    summary_rows = []

    # 1) 计算频谱 / 互谱 / coherence / phase
    for proxy, df in datasets.items():
        auto_specs[proxy], auto_meta[proxy] = compute_auto_spectrum(df["mean_cold_duration_kyr"])
        auto_specs[proxy].sort_values("period_kyr").to_csv(
            TAB_DIR / f"auto_spectrum_{proxy}_cold_duration.csv", index=False
        )

        for drv in ["pre", "obl", "ecc"]:
            cross_specs[(proxy, drv)], cross_meta[(proxy, drv)] = compute_cross_products(
                df["mean_cold_duration_kyr"], df[drv]
            )
            cross_specs[(proxy, drv)].sort_values("period_kyr").to_csv(
                TAB_DIR / f"cross_products_{proxy}_cold_duration_vs_{drv}.csv", index=False
            )
            summary_rows.append(
                build_band_summary_row(proxy, drv, auto_specs[proxy], cross_specs[(proxy, drv)])
            )

    # 2) 输出元数据
    meta_rows = []
    for proxy, meta in auto_meta.items():
        meta_rows.append({"kind": "auto", "proxy": proxy, "driver": "", **meta})
    for (proxy, drv), meta in cross_meta.items():
        meta_rows.append({"kind": "cross", "proxy": proxy, "driver": drv, **meta})
    pd.DataFrame(meta_rows).to_csv(TAB_DIR / "spectral_metadata.csv", index=False)

    # 3) 输出带宽峰值摘要
    pd.DataFrame(summary_rows).to_csv(TAB_DIR / "spectral_band_peak_summary_enhanced.csv", index=False)

    # ------------------------------------------------------
    # 图 1：cold duration 自身功率谱
    # ------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.2), dpi=180, sharex=True)
    for ax, proxy in zip(axes, ["ch4", "monsoon"]):
        df = subset_period_range(auto_specs[proxy]).sort_values("period_kyr")
        ax.semilogy(df["period_kyr"], df["power"], color="black", lw=1.25)
        style_common_axis(ax, f"{PROXY_LABELS[proxy]} cold-duration spectrum", ylabel="Power")
        add_period_markers(ax)
    axes[-1].set_xlabel("Period (kyr)")
    fig.suptitle("Power spectrum of cold duration", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG_DIR / "fig01_cold_duration_power_spectra.png", bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------
    # 图 2：互功率谱
    # ------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.8), dpi=180, sharex=True)
    for i, proxy in enumerate(["ch4", "monsoon"]):
        for j, drv in enumerate(["pre", "obl", "ecc"]):
            ax = axes[i, j]
            df = subset_period_range(cross_specs[(proxy, drv)]).sort_values("period_kyr")
            ax.semilogy(df["period_kyr"], df["cross_power_abs"], color="black", lw=1.15)
            style_common_axis(ax, f"{PROXY_LABELS[proxy]} × {DRIVER_LABELS[drv]}")
            add_period_markers(ax)
            if j == 0:
                ax.set_ylabel(r"|Cross power|")
            if i == 1:
                ax.set_xlabel("Period (kyr)")
    fig.suptitle("Cross-power spectra: cold duration vs orbital drivers", y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG_DIR / "fig02_cold_duration_cross_power_spectra.png", bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------
    # 图 3：coherence
    # ------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.8), dpi=180, sharex=True, sharey=True)
    for i, proxy in enumerate(["ch4", "monsoon"]):
        for j, drv in enumerate(["pre", "obl", "ecc"]):
            ax = axes[i, j]
            df = subset_period_range(cross_specs[(proxy, drv)]).sort_values("period_kyr")
            sig95 = cross_meta[(proxy, drv)]["approx_coherence_sig95"]
            ax.plot(df["period_kyr"], df["coherence"], color="black", lw=1.15)
            if np.isfinite(sig95):
                ax.axhline(sig95, color="0.55", lw=0.9, ls=":", label=f"approx. 95% = {sig95:.2f}")
            ax.set_ylim(0, 1.02)
            style_common_axis(ax, f"{PROXY_LABELS[proxy]} × {DRIVER_LABELS[drv]}")
            add_period_markers(ax)
            if j == 0:
                ax.set_ylabel("Coherence")
            if i == 1:
                ax.set_xlabel("Period (kyr)")
            if i == 0 and j == 2 and np.isfinite(sig95):
                ax.legend(loc="lower right", frameon=False, fontsize=8)
    fig.suptitle("Magnitude-squared coherence", y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG_DIR / "fig03_cold_duration_coherence.png", bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------
    # 图 4：phase（灰色=全部频率；黑色=coherence 较高部分）
    # 正值表示 driver 领先 cold duration
    # ------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.8), dpi=180, sharex=True, sharey=True)
    for i, proxy in enumerate(["ch4", "monsoon"]):
        for j, drv in enumerate(["pre", "obl", "ecc"]):
            ax = axes[i, j]
            df = subset_period_range(cross_specs[(proxy, drv)]).sort_values("period_kyr")
            mask = df["coherence"] >= COH_DISPLAY_THRESHOLD
            ax.plot(df["period_kyr"], df["phase_deg"], color="0.75", lw=0.9)
            ax.plot(df.loc[mask, "period_kyr"], df.loc[mask, "phase_deg"], color="black", lw=1.2)
            ax.axhline(0, color="0.5", lw=0.8, ls=":")
            ax.set_ylim(-180, 180)
            ax.set_yticks(np.arange(-180, 181, 90))
            style_common_axis(ax, f"{PROXY_LABELS[proxy]} × {DRIVER_LABELS[drv]}")
            add_period_markers(ax)
            if j == 0:
                ax.set_ylabel("Phase (deg; + driver leads)")
            if i == 1:
                ax.set_xlabel("Period (kyr)")
    fig.suptitle(
        f"Cross-spectral phase (black: coherence ≥ {COH_DISPLAY_THRESHOLD:.2f}; positive = driver leads)",
        y=0.985,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG_DIR / "fig04_cold_duration_phase.png", bbox_inches="tight")
    plt.close(fig)

    print("Done.")
    print(f"Output directory: {OUT_DIR}")
    print("Figures:")
    for f in sorted(FIG_DIR.glob("*.png")):
        print("  ", f)
    print("Tables:")
    for f in sorted(TAB_DIR.glob("*.csv")):
        print("  ", f)


if __name__ == "__main__":
    main()
