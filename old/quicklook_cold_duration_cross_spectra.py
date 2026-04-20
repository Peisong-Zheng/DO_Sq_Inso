#!/usr/bin/env python3
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import welch, csd


DATA_DIR = Path("data/")
OUT_DIR = Path("results/cold_duration_cross_spectra")
FIG_DIR = OUT_DIR / "figures"
TAB_DIR = OUT_DIR / "tables"

WINDOW_YR = 20_000
DOWNSAMPLE_YR = 1000
FS = 1 / 1000.0  # 1 kyr sampled series -> cycles per year
PERIOD_MARKERS = {"pre": 23, "obl": 41, "ecc": 100}
PERIOD_XLIM = (10, 150)

PROXY_FILES = {
    "ch4": DATA_DIR / "ch4_mcv_square_wave_selected_full_50yr.csv",
    "monsoon": DATA_DIR / "monsoon_mcv_square_wave_selected_full_50yr.csv",
}
PROXY_LABELS = {"ch4": "CH$_4$", "monsoon": "Monsoon d$^{18}$O"}
DRIVER_LABELS = {"pre": "Precession", "obl": "Obliquity", "ecc": "Eccentricity"}

def resolve_file(name: str) -> Path:
    for p in [DATA_DIR / name]:
        if p.exists():
            return p
    raise FileNotFoundError(name)

ORBITAL_FILES = {
    "pre": resolve_file("pre_800_inter100.txt"),
    "obl": resolve_file("obl_800_inter100.txt"),
    "ecc": resolve_file("ecc_1000_inter100.txt"),
}

for p in [OUT_DIR, FIG_DIR, TAB_DIR]:
    p.mkdir(parents=True, exist_ok=True)

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
            "young_age_yrBP": age[s],
            "old_age_yrBP": age[e - 1],
            "duration_yr": age[e - 1] - age[s] + dt,
            "start_truncated": bool(s == 0),
            "end_truncated": bool(e == len(age)),
        })
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
        mask = ((intervals_df["young_age_yrBP"] >= c - half) &
                (intervals_df["old_age_yrBP"] <= c + half))
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
    centers = age[h:len(age) - h]
    warm_full = warm_intervals.loc[
        ~warm_intervals["start_truncated"] & ~warm_intervals["end_truncated"]
    ].copy()
    return pd.DataFrame({
        "age_yrBP": centers,
        "warming_count_20k": e_series,
        "cold_fraction_20k": p_series,
        "mean_cold_duration_yr": local_mean_durations(centers, cold_intervals, window_yr),
        "mean_warm_duration_yr": local_mean_durations(centers, warm_full, window_yr),
    })

def prepare_proxy_sequence(path: Path):
    df = pd.read_csv(path).sort_values("age_yrBP")
    age = df["age_yrBP"].to_numpy(float)
    cold = (df["mcv_state"].to_numpy() == 0)
    cold_intervals = extract_intervals_from_binary(age, cold)
    warm_intervals = extract_warm_intervals_from_binary(age, cold)
    warmings = np.zeros_like(age, int)
    for a in cold_intervals["young_age_yrBP"].values:
        warmings[np.argmin(np.abs(age - a))] = 1
    if cold[0]:
        warmings[0] = 0
    return {"window_stats": compute_window_stats(age, cold, warmings, cold_intervals, warm_intervals, WINDOW_YR)}

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
    step = max(1, int(round(DOWNSAMPLE_YR / 50)))
    stats = stats.iloc[::step].reset_index(drop=True)
    for name, path in ORBITAL_FILES.items():
        df = load_orbital_series(path, name)
        stats[name] = interp_to_grid(stats["age_yrBP"].values, df["age_yrBP"].values, df[name].values)
    stats["mean_cold_duration_kyr"] = stats["mean_cold_duration_yr"] / 1000.0
    stats = stats.dropna(subset=["mean_cold_duration_kyr", "pre", "obl", "ecc"]).copy()
    stats = stats.sort_values("age_yrBP", ascending=False).reset_index(drop=True)
    return stats

def compute_auto_spectrum(series):
    x = zscore(series)
    nperseg = min(256, len(x))
    noverlap = nperseg // 2
    f, pxx = welch(x, fs=FS, window="hann", detrend="linear",
                   nperseg=nperseg, noverlap=noverlap, scaling="density")
    m = f > 0
    return pd.DataFrame({
        "frequency_cpy": f[m],
        "period_kyr": 1 / f[m] / 1000.0,
        "power": pxx[m],
    })

def compute_cross_spectrum(x, y):
    x = zscore(x)
    y = zscore(y)
    nperseg = min(256, len(x), len(y))
    noverlap = nperseg // 2
    f, pxy = csd(x, y, fs=FS, window="hann", detrend="linear",
                 nperseg=nperseg, noverlap=noverlap, scaling="density")
    m = f > 0
    return pd.DataFrame({
        "frequency_cpy": f[m],
        "period_kyr": 1 / f[m] / 1000.0,
        "cross_power_abs": np.abs(pxy[m]),
        "cross_power_real": np.real(pxy[m]),
        "cross_power_imag": np.imag(pxy[m]),
    })

def local_peak(spec_df: pd.DataFrame, col: str, pmin: float, pmax: float):
    sub = spec_df[(spec_df["period_kyr"] >= pmin) & (spec_df["period_kyr"] <= pmax)].copy()
    if len(sub) == 0:
        return {"peak_period_kyr": np.nan, "peak_value": np.nan}
    row = sub.iloc[sub[col].values.argmax()]
    return {"peak_period_kyr": float(row["period_kyr"]), "peak_value": float(row[col])}

def add_period_markers(ax):
    colors = {"pre": "tab:blue", "obl": "tab:green", "ecc": "tab:red"}
    for key, p in PERIOD_MARKERS.items():
        ax.axvline(p, color=colors[key], lw=0.9, ls="--", alpha=0.7)
    y = 10 ** (np.log10(ax.get_ylim()[0]) + 0.92 * (np.log10(ax.get_ylim()[1]) - np.log10(ax.get_ylim()[0])))
    for key, p in PERIOD_MARKERS.items():
        ax.text(
            p, y, key, color=colors[key], rotation=90, ha="right", va="top", fontsize=8,
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1.2)
        )

def main():
    datasets = {proxy: build_dataset(proxy) for proxy in PROXY_FILES}
    auto_specs = {}
    cross_specs = {}

    for proxy, df in datasets.items():
        auto_specs[proxy] = compute_auto_spectrum(df["mean_cold_duration_kyr"])
        auto_specs[proxy].sort_values("period_kyr").to_csv(TAB_DIR / f"auto_spectrum_{proxy}_cold_duration.csv", index=False)
        for drv in ["pre", "obl", "ecc"]:
            cross_specs[(proxy, drv)] = compute_cross_spectrum(df["mean_cold_duration_kyr"], df[drv])
            cross_specs[(proxy, drv)].sort_values("period_kyr").to_csv(
                TAB_DIR / f"cross_spectrum_{proxy}_cold_duration_vs_{drv}.csv", index=False
            )

    rows = []
    for proxy, auto in auto_specs.items():
        for band, center in PERIOD_MARKERS.items():
            hw = 8 if band != "ecc" else 30
            pk = local_peak(auto, "power", center - hw, center + hw)
            rows.append({"proxy": proxy, "spectrum": "auto_cold_duration", "driver_band": band, **pk})
    for (proxy, drv), spec in cross_specs.items():
        center = PERIOD_MARKERS[drv]
        hw = 8 if drv != "ecc" else 30
        pk = local_peak(spec, "cross_power_abs", center - hw, center + hw)
        rows.append({"proxy": proxy, "spectrum": f"cross_with_{drv}", "driver_band": drv, **pk})
    pd.DataFrame(rows).to_csv(TAB_DIR / "spectral_band_peak_summary.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.2), dpi=180, sharex=True)
    for ax, proxy in zip(axes, ["ch4", "monsoon"]):
        df = auto_specs[proxy].sort_values("period_kyr")
        m = (df["period_kyr"] >= PERIOD_XLIM[0]) & (df["period_kyr"] <= PERIOD_XLIM[1])
        ax.semilogy(df.loc[m, "period_kyr"], df.loc[m, "power"], color="black", lw=1.2)
        ax.set_ylabel("Power")
        ax.set_title(f"{PROXY_LABELS[proxy]} cold-duration spectrum")
        ax.grid(alpha=0.2, ls="--")
        ax.set_xlim(*PERIOD_XLIM)
        add_period_markers(ax)
    axes[-1].set_xlabel("Period (kyr)")
    fig.suptitle("Quick-look power spectrum of cold duration", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG_DIR / "fig01_cold_duration_power_spectra.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.8), dpi=180, sharex=True, sharey=False)
    for i, proxy in enumerate(["ch4", "monsoon"]):
        for j, drv in enumerate(["pre", "obl", "ecc"]):
            ax = axes[i, j]
            df = cross_specs[(proxy, drv)].sort_values("period_kyr")
            m = (df["period_kyr"] >= PERIOD_XLIM[0]) & (df["period_kyr"] <= PERIOD_XLIM[1])
            ax.semilogy(df.loc[m, "period_kyr"], df.loc[m, "cross_power_abs"], color="black", lw=1.1)
            ax.grid(alpha=0.2, ls="--")
            ax.set_xlim(*PERIOD_XLIM)
            ax.set_title(f"{PROXY_LABELS[proxy]} × {DRIVER_LABELS[drv]}")
            add_period_markers(ax)
            if j == 0:
                ax.set_ylabel(r"|Cross power|")
            if i == 1:
                ax.set_xlabel("Period (kyr)")
    fig.suptitle("Quick-look cross-power spectra: cold duration vs orbital drivers", y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG_DIR / "fig02_cold_duration_cross_power_spectra.png", bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    main()
