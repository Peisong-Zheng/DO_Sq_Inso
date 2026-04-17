#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mainline transfer-entropy workflow for two paleoclimate targets:
1) Antarctic-stacked CH4
2) Monsoon d18O

This script intentionally keeps only the research mainline:
- uniform resampling
- high-pass filtering
- alignment with orbital forcing (pre / obl / ecc)
- overall bidirectional TE surrogate tests

Deliberately excluded:
- local TE
- lag scans / dt scans / other sensitivity experiments
- wavelet TE / CCM / event decomposition

Directory convention
--------------------
Place inputs under data/ and run from the project root:
    data/CH4_AICC2023.xlsx
    data/monsoon.xlsx
    data/pre_800_inter100.txt
    data/obl_800_inter100.txt
    data/ecc_1000_inter100.txt

Outputs are written under results/.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt

try:
    from pyinform import transfer_entropy
except ImportError as exc:  # pragma: no cover
    raise ImportError("pyinform is required. Install with: pip install pyinform") from exc


# ============================================================
# User-editable configuration
# ============================================================
DATA_DIR = Path("data")
RESULTS_DIR = Path("results")

CH4_FILE = "CH4_AICC2023.xlsx"
MONSOON_FILE = "monsoon.xlsx"
PRE_FILE = "pre_800_inter100.txt"
OBL_FILE = "obl_800_inter100.txt"
ECC_FILE = "ecc_1000_inter100.txt"

UNIFORM_STEP_YR = 10               # resampling step before filtering
TE_STEP_YR = 50                    # resampling step for TE analysis
CUTOFF_PERIOD_YR = 10_000          # high-pass cutoff period
BUTTER_ORDER = 4
FORCING_BINS = 6
TARGET_BINS = 2
TE_K = 1
N_SURROGATES = 1000
ALPHA = 0.05
DISCRETIZATION_METHOD = "hist"    # "hist" | "quantile" | "kmeans"
RNG_SEED = 42

# Keep the two target series directly comparable.
# Monsoon is ~641 kyr long, so here we deliberately truncate both at 640 kyr.
ANALYSIS_AGE_MIN_YR_BP = 1_000
ANALYSIS_AGE_MAX_YR_BP = 640_000

# To reproduce the old notebook more faithfully, target interpolation onto the
# TE grid defaults to nearest-neighbor, matching interpolate_data_forcing().
TARGET_INTERP_KIND_FOR_TE = "nearest"   # "nearest" | "linear"


@dataclass
class Config:
    data_dir: Path = DATA_DIR
    results_dir: Path = RESULTS_DIR
    ch4_file: str = CH4_FILE
    monsoon_file: str = MONSOON_FILE
    pre_file: str = PRE_FILE
    obl_file: str = OBL_FILE
    ecc_file: str = ECC_FILE
    uniform_step_yr: int = UNIFORM_STEP_YR
    te_step_yr: int = TE_STEP_YR
    cutoff_period_yr: int = CUTOFF_PERIOD_YR
    butter_order: int = BUTTER_ORDER
    forcing_bins: int = FORCING_BINS
    target_bins: int = TARGET_BINS
    te_k: int = TE_K
    n_surrogates: int = N_SURROGATES
    alpha: float = ALPHA
    discretization_method: str = DISCRETIZATION_METHOD
    rng_seed: int = RNG_SEED
    analysis_age_min_yr_bp: int = ANALYSIS_AGE_MIN_YR_BP
    analysis_age_max_yr_bp: int = ANALYSIS_AGE_MAX_YR_BP
    target_interp_kind_for_te: str = TARGET_INTERP_KIND_FOR_TE


# ============================================================
# I/O helpers
# ============================================================
def ensure_dirs(cfg: Config) -> None:
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    (cfg.results_dir / "ch4").mkdir(parents=True, exist_ok=True)
    (cfg.results_dir / "monsoon").mkdir(parents=True, exist_ok=True)


def load_target_series(path: Path, value_col: str) -> pd.DataFrame:
    """
    Load a target series and normalize its age axis to years BP increasing with age.

    Rules:
    - CH4 ages are already in years BP.
    - monsoon ages appear to be in kyr BP, so if max age < 10,000 we convert to years.
    - slight negative ages around 0 are tolerated; later ANALYSIS_AGE_MIN removes them.
    """
    df = pd.read_excel(path)
    required = {"age", value_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")

    out = df[["age", value_col]].copy()
    out["age"] = pd.to_numeric(out["age"], errors="coerce")
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
    out = out.dropna().reset_index(drop=True)

    # Convert kyr BP to yr BP when needed.
    if out["age"].abs().max() < 10_000:
        out["age"] = out["age"] * 1000.0

    out = out.sort_values("age").reset_index(drop=True)
    return out


def load_orbital_series(path: Path, value_name: str) -> pd.DataFrame:
    """
    Original orbital tables are given from negative kyr to 0.
    Convert to positive yr BP and sort age from young to old (ascending BP).
    """
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    if df.shape[1] < 2:
        raise ValueError(f"{path.name} must have at least two columns")
    out = df.iloc[:, :2].copy()
    out.columns = ["age_raw", value_name]
    out["age"] = out["age_raw"].abs() * 1000.0
    out[value_name] = pd.to_numeric(out[value_name], errors="coerce")
    out = out[["age", value_name]].dropna().sort_values("age").reset_index(drop=True)
    return out


# ============================================================
# Signal processing
# ============================================================
def resample_to_uniform_grid(
    df: pd.DataFrame,
    value_col: str,
    step_yr: float,
    age_min_yr: float,
    age_max_yr: float,
) -> pd.DataFrame:
    if step_yr <= 0:
        raise ValueError("step_yr must be positive")
    if age_max_yr <= age_min_yr:
        raise ValueError("age_max_yr must be greater than age_min_yr")

    src = df[["age", value_col]].dropna().sort_values("age").reset_index(drop=True)
    new_age = np.arange(age_min_yr, age_max_yr + step_yr, step_yr, dtype=float)
    new_values = np.interp(new_age, src["age"].to_numpy(), src[value_col].to_numpy())
    return pd.DataFrame({"age": new_age, value_col: new_values})


def highpass_filter_series(
    df_uniform: pd.DataFrame,
    value_col: str,
    cutoff_period_yr: float,
    butter_order: int,
    output_col: str,
) -> pd.DataFrame:
    ages = df_uniform["age"].to_numpy(dtype=float)
    values = df_uniform[value_col].to_numpy(dtype=float)
    dt = float(np.median(np.diff(ages)))
    if dt <= 0:
        raise ValueError("Uniform age grid must be strictly increasing")

    fs = 1.0 / dt
    fc = 1.0 / cutoff_period_yr
    wn = fc / (0.5 * fs)
    if not (0 < wn < 1):
        raise ValueError(f"Invalid Butterworth cutoff Wn={wn:.6f}")

    b, a = butter(butter_order, wn, btype="highpass")
    filt = filtfilt(b, a, values)
    return pd.DataFrame({"age": ages, output_col: filt})


# ============================================================
# Alignment helpers
# ============================================================
def align_target_with_forcing(
    df_target_filt: pd.DataFrame,
    target_col: str,
    orbital: Dict[str, pd.DataFrame],
    te_step_yr: float,
    age_min_yr: float,
    age_max_yr: float,
    target_interp_kind: str = "nearest",
) -> pd.DataFrame:
    """
    Align one filtered target with orbital forcing on a common TE grid.

    To stay close to the old notebook's interpolate_data_forcing(), the target
    interpolation defaults to nearest-neighbor rather than linear.
    """
    overlap_min = max(
        age_min_yr,
        float(df_target_filt["age"].min()),
        *(float(df["age"].min()) for df in orbital.values()),
    )
    overlap_max = min(
        age_max_yr,
        float(df_target_filt["age"].max()),
        *(float(df["age"].max()) for df in orbital.values()),
    )
    if overlap_max <= overlap_min:
        raise ValueError("No overlapping age range between target and forcing data")

    new_age = np.arange(overlap_min, overlap_max + te_step_yr, te_step_yr, dtype=float)
    aligned = pd.DataFrame({"age": new_age})

    f_target = interp1d(
        df_target_filt["age"].to_numpy(),
        df_target_filt[target_col].to_numpy(),
        kind=target_interp_kind,
        bounds_error=False,
        fill_value="extrapolate",
    )
    aligned[target_col] = f_target(new_age)

    for name, df in orbital.items():
        f = interp1d(
            df["age"].to_numpy(),
            df[name].to_numpy(),
            kind="nearest",
            bounds_error=False,
            fill_value="extrapolate",
        )
        aligned[name] = f(new_age)

    return aligned


# ============================================================
# TE helpers
# ============================================================
def discretize_target_like_original(y: np.ndarray, bins: int, method: str) -> np.ndarray:
    if method == "quantile":
        ybins = np.quantile(y, np.linspace(0, 1, bins + 1))
        return np.digitize(y, ybins) - 1
    if method == "hist":
        ybins = np.histogram_bin_edges(y, bins=bins)
        return np.digitize(y, ybins) - 1
    if method == "kmeans":
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=bins, n_init=10, random_state=0)
        return km.fit_predict(y.reshape(-1, 1))
    raise ValueError("method must be 'hist', 'quantile', or 'kmeans'")


def transfer_entropy_surrogate_test_original_style(
    forcing: np.ndarray,
    target: np.ndarray,
    *,
    k: int,
    forcing_bins: int,
    target_bins: int,
    n_surrogates: int,
    alpha: float,
    discretization_method: str,
    rng: np.random.Generator,
) -> Tuple[dict, dict]:
    """
    Rebuild the old main TE significance function as closely as practical.

    Important features intentionally preserved from the original sq_ana.py:
    - reverse both input series so TE is evaluated oldest -> youngest
    - discretize x and y using hist / quantile / kmeans rules
    - compute TE on the FULL discretized sequences, not on x[:-1], y[1:]
    - build surrogate nulls by permuting the already-discretized x and y
    """
    x = np.asarray(forcing, dtype=float)[::-1]
    y = np.asarray(target, dtype=float)[::-1]

    if discretization_method == "quantile":
        xbins = np.quantile(x, np.linspace(0, 1, forcing_bins + 1))
        ybins = np.quantile(y, np.linspace(0, 1, target_bins + 1))
        y_disc = np.digitize(y, ybins) - 1
    elif discretization_method == "hist":
        xbins = np.histogram_bin_edges(x, bins=forcing_bins)
        ybins = np.histogram_bin_edges(y, bins=target_bins)
        y_disc = np.digitize(y, ybins) - 1
    elif discretization_method == "kmeans":
        xbins = np.histogram_bin_edges(x, bins=forcing_bins)
        ybins = None
        y_disc = discretize_target_like_original(y, target_bins, "kmeans")
    else:
        raise ValueError("Unknown discretization_method")

    x_disc = np.digitize(x, xbins) - 1

    te_xy = float(transfer_entropy(x_disc, y_disc, k=k))
    te_yx = float(transfer_entropy(y_disc, x_disc, k=k))

    null_xy = np.zeros(n_surrogates, dtype=float)
    null_yx = np.zeros(n_surrogates, dtype=float)
    for i in range(n_surrogates):
        xs = rng.permutation(x_disc)
        ys = rng.permutation(y_disc)
        null_xy[i] = float(transfer_entropy(xs, y_disc, k=k))
        null_yx[i] = float(transfer_entropy(ys, x_disc, k=k))

    p_xy = float((np.sum(null_xy >= te_xy) + 1) / (n_surrogates + 1))
    p_yx = float((np.sum(null_yx >= te_yx) + 1) / (n_surrogates + 1))

    result = {
        "te_xy": te_xy,
        "p_xy": p_xy,
        "te_yx": te_yx,
        "p_yx": p_yx,
        "sig_xy": bool(p_xy < alpha),
        "sig_yx": bool(p_yx < alpha),
        "sig_unidirectional": bool((p_xy < alpha) and not (p_yx < alpha)),
    }
    diagnostics = {
        "x_disc": x_disc,
        "y_disc": y_disc,
        "null_xy": null_xy,
        "null_yx": null_yx,
    }
    return result, diagnostics


# ============================================================
# Plotting
# ============================================================
def plot_filtered_series(df_uniform: pd.DataFrame, df_filt: pd.DataFrame, raw_col: str, filt_col: str, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4), dpi=160)
    ax.plot(df_uniform["age"], df_uniform[raw_col], color="0.7", lw=0.8, label=f"Raw {raw_col}")
    ax.plot(df_filt["age"], df_filt[filt_col], color="black", lw=1.0, label=f"High-pass {raw_col}")
    ax.invert_xaxis()
    ax.set_xlabel("Age (yr BP)")
    ax.set_ylabel(raw_col)
    ax.set_title(title)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_aligned_series(df_aligned: pd.DataFrame, target_col: str, target_label: str, out_path: Path) -> None:
    cols = [target_col, "pre", "obl", "ecc"]
    labels = [target_label, "Precession", "Obliquity", "Eccentricity"]
    colors = ["black", "tab:blue", "tab:green", "tab:red"]
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), dpi=160, sharex=True)
    for ax, col, label, color in zip(axes, cols, labels, colors):
        ax.plot(df_aligned["age"], df_aligned[col], color=color, lw=0.9)
        ax.set_ylabel(label)
    axes[0].set_title(f"Aligned series used for overall TE: {target_label}")
    axes[-1].set_xlabel("Age (yr BP)")
    axes[-1].invert_xaxis()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_null_histograms(summary_rows: pd.DataFrame, diagnostics_by_driver: Dict[str, dict], target_label: str, out_path: Path) -> None:
    drivers = list(diagnostics_by_driver.keys())
    fig, axes = plt.subplots(len(drivers), 1, figsize=(7, 3.2 * len(drivers)), dpi=160)
    if len(drivers) == 1:
        axes = [axes]

    for ax, driver in zip(axes, drivers):
        diag = diagnostics_by_driver[driver]
        row = summary_rows.loc[summary_rows["driver"] == driver].iloc[0]
        ax.hist(diag["null_xy"], bins=25, alpha=0.7, color="#CC6677", edgecolor="white", label=f"Null {driver}→{target_label}")
        ax.axvline(row["te_xy"], color="#882255", lw=2, label=f"Empirical {driver}→{target_label}, p={row['p_xy']:.3f}")
        ax.hist(diag["null_yx"], bins=25, alpha=0.6, color="#88CCEE", edgecolor="white", label=f"Null {target_label}→{driver}")
        ax.axvline(row["te_yx"], color="#44AA99", lw=2, label=f"Empirical {target_label}→{driver}, p={row['p_yx']:.3f}")
        ax.set_xlabel("Transfer entropy (bits)")
        ax.set_ylabel("Count")
        ax.set_title(driver)
        ax.legend(frameon=True, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)




def plot_cross_target_comparison(all_summary: pd.DataFrame, out_path: Path) -> None:
    """
    Make CH4 and monsoon directly comparable on a single figure.

    Panel A: forcing -> target TE
    Panel B: target -> forcing TE
    Colors separate CH4 and monsoon; stars mark p < alpha.
    """
    drivers = ["pre", "obl", "ecc"]
    targets = [
        ("ch4", "CH4", "#4C72B0"),
        ("monsoon", "Monsoon d18O", "#DD8452"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), dpi=180, sharey=False)
    width = 0.34
    x = np.arange(len(drivers))

    for ax, te_col, p_col, title in [
        (axes[0], "te_xy", "p_xy", "Forcing → target"),
        (axes[1], "te_yx", "p_yx", "Target → forcing"),
    ]:
        ymax = 0.0
        for idx, (target_name, label, color) in enumerate(targets):
            subset = (
                all_summary.loc[all_summary["target_name"] == target_name]
                .set_index("driver")
                .loc[drivers]
                .reset_index()
            )
            xpos = x + (idx - 0.5) * width
            vals = subset[te_col].to_numpy(dtype=float)
            ps = subset[p_col].to_numpy(dtype=float)
            bars = ax.bar(xpos, vals, width=width, color=color, alpha=0.85, label=label)
            ymax = max(ymax, float(np.nanmax(np.abs(vals))))

            for bar, p in zip(bars, ps):
                if p >= 0.05:
                    continue
                y = bar.get_height()
                offset = 0.03 * (ymax if ymax > 0 else 1.0)
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    y + (offset if y >= 0 else -offset),
                    '*',
                    ha='center',
                    va='bottom' if y >= 0 else 'top',
                    fontsize=13,
                    fontweight='bold',
                )

        ax.axhline(0, color='0.3', lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([d.upper() for d in drivers])
        ax.set_title(title)
        ax.set_xlabel('Orbital driver')
        ax.grid(axis='y', linestyle='--', alpha=0.25)

    axes[0].set_ylabel('Transfer entropy (bits)')
    axes[0].legend(frameon=True)
    fig.suptitle('Direct comparison of CH4 and monsoon d18O TE results', y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


# ============================================================
# Dataset-specific pipeline
# ============================================================
def run_one_target(
    *,
    target_name: str,
    target_label: str,
    target_file: str,
    target_value_col: str,
    cfg: Config,
    orbital: Dict[str, pd.DataFrame],
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, Dict[str, Path]]:
    out_dir = cfg.results_dir / target_name
    out_dir.mkdir(parents=True, exist_ok=True)

    df_raw = load_target_series(cfg.data_dir / target_file, target_value_col)

    df_uniform = resample_to_uniform_grid(
        df_raw,
        value_col=target_value_col,
        step_yr=cfg.uniform_step_yr,
        age_min_yr=cfg.analysis_age_min_yr_bp,
        age_max_yr=cfg.analysis_age_max_yr_bp,
    )
    filt_col = f"filt_{target_value_col}"
    df_filt = highpass_filter_series(
        df_uniform,
        value_col=target_value_col,
        cutoff_period_yr=cfg.cutoff_period_yr,
        butter_order=cfg.butter_order,
        output_col=filt_col,
    )
    df_aligned = align_target_with_forcing(
        df_target_filt=df_filt,
        target_col=filt_col,
        orbital=orbital,
        te_step_yr=cfg.te_step_yr,
        age_min_yr=cfg.analysis_age_min_yr_bp,
        age_max_yr=cfg.analysis_age_max_yr_bp,
        target_interp_kind=cfg.target_interp_kind_for_te,
    )

    rows = []
    diagnostics_by_driver: Dict[str, dict] = {}
    for driver in ["pre", "obl", "ecc"]:
        res, diag = transfer_entropy_surrogate_test_original_style(
            forcing=df_aligned[driver].to_numpy(),
            target=df_aligned[filt_col].to_numpy(),
            k=cfg.te_k,
            forcing_bins=cfg.forcing_bins,
            target_bins=cfg.target_bins,
            n_surrogates=cfg.n_surrogates,
            alpha=cfg.alpha,
            discretization_method=cfg.discretization_method,
            rng=rng,
        )
        res.update({
            "target_name": target_name,
            "target_label": target_label,
            "driver": driver,
            "discretization_method": cfg.discretization_method,
            "forcing_bins": cfg.forcing_bins,
            "target_bins": cfg.target_bins,
            "te_k": cfg.te_k,
            "n_aligned": len(df_aligned),
            "age_min_yr_bp": float(df_aligned["age"].min()),
            "age_max_yr_bp": float(df_aligned["age"].max()),
        })
        rows.append(res)
        diagnostics_by_driver[driver] = diag

    df_summary = pd.DataFrame(rows)

    out_paths = {
        "uniform_csv": out_dir / f"{target_name}_uniform.csv",
        "filtered_csv": out_dir / f"{target_name}_filtered.csv",
        "aligned_csv": out_dir / f"{target_name}_aligned.csv",
        "summary_csv": out_dir / f"{target_name}_te_summary.csv",
        "filtered_fig": out_dir / f"fig_{target_name}_filtered.png",
        "aligned_fig": out_dir / f"fig_{target_name}_aligned.png",
        "null_fig": out_dir / f"fig_{target_name}_te_nulls.png",
    }

    df_uniform.to_csv(out_paths["uniform_csv"], index=False)
    df_filt.to_csv(out_paths["filtered_csv"], index=False)
    df_aligned.to_csv(out_paths["aligned_csv"], index=False)
    df_summary.to_csv(out_paths["summary_csv"], index=False)

    plot_filtered_series(
        df_uniform=df_uniform,
        df_filt=df_filt,
        raw_col=target_value_col,
        filt_col=filt_col,
        title=f"{target_label}: raw and high-pass filtered series",
        out_path=out_paths["filtered_fig"],
    )
    plot_aligned_series(
        df_aligned=df_aligned,
        target_col=filt_col,
        target_label=target_label,
        out_path=out_paths["aligned_fig"],
    )
    plot_null_histograms(
        summary_rows=df_summary,
        diagnostics_by_driver=diagnostics_by_driver,
        target_label=target_label,
        out_path=out_paths["null_fig"],
    )
    return df_summary, out_paths


# ============================================================
# Main
# ============================================================
def main() -> None:
    cfg = Config()
    ensure_dirs(cfg)
    rng = np.random.default_rng(cfg.rng_seed)

    orbital = {
        "pre": load_orbital_series(cfg.data_dir / cfg.pre_file, "pre"),
        "obl": load_orbital_series(cfg.data_dir / cfg.obl_file, "obl"),
        "ecc": load_orbital_series(cfg.data_dir / cfg.ecc_file, "ecc"),
    }

    ch4_summary, ch4_paths = run_one_target(
        target_name="ch4",
        target_label="CH4 millennial variability",
        target_file=cfg.ch4_file,
        target_value_col="ch4",
        cfg=cfg,
        orbital=orbital,
        rng=rng,
    )
    monsoon_summary, monsoon_paths = run_one_target(
        target_name="monsoon",
        target_label="Monsoon d18O millennial variability",
        target_file=cfg.monsoon_file,
        target_value_col="d18O",
        cfg=cfg,
        orbital=orbital,
        rng=rng,
    )

    all_summary = pd.concat([ch4_summary, monsoon_summary], ignore_index=True)
    all_summary_path = cfg.results_dir / "te_summary_all.csv"
    all_summary.to_csv(all_summary_path, index=False)

    comparison_fig = cfg.results_dir / "fig_te_cross_target_comparison.png"
    plot_cross_target_comparison(all_summary, comparison_fig)

    metadata = {
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()},
        "ch4_outputs": {k: str(v) for k, v in ch4_paths.items()},
        "monsoon_outputs": {k: str(v) for k, v in monsoon_paths.items()},
        "combined_summary": str(all_summary_path),
        "comparison_figure": str(comparison_fig),
    }
    with open(cfg.results_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("Done. Combined summary written to:", all_summary_path)


if __name__ == "__main__":
    main()
