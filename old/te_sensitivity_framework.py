#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sensitivity-analysis framework for orbital-forcing transfer entropy (TE)
against two paleoclimate targets:

1) Antarctic-stacked CH4
2) Monsoon d18O

Design principles
-----------------
- Keep everything in a single readable .py file.
- Put all user-adjustable settings at the top.
- Reuse the same preprocessing / TE definitions across targets so CH4 and
  monsoon results remain directly comparable.
- Organize experiments by layers: baseline, one-dimensional scans,
  two-dimensional grids, and null-model comparison.
- Export the important figures in both PNG and PDF formats, styled for GRL-like
  supplementary figures (panel letters, larger fonts, no grid, no titles by
  default).

Expected input files under data/
--------------------------------
    data/CH4_AICC2023.xlsx
    data/monsoon.xlsx
    data/pre_800_inter100.txt
    data/obl_800_inter100.txt
    data/ecc_1000_inter100.txt

Outputs
-------
Written under results/sensitivity/
"""

from __future__ import annotations

import json
import string
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt

try:
    from pyinform import transfer_entropy
except ImportError as exc:  # pragma: no cover
    raise ImportError("pyinform is required. Install with: pip install pyinform") from exc


# =============================================================================
# User-editable configuration
# =============================================================================
DATA_DIR = Path("data")
RESULTS_DIR = Path("results") / "sensitivity"

CH4_FILE = "CH4_AICC2023.xlsx"
MONSOON_FILE = "monsoon.xlsx"
PRE_FILE = "pre_800_inter100.txt"
OBL_FILE = "obl_800_inter100.txt"
ECC_FILE = "ecc_1000_inter100.txt"

# ------------------------- analysis window & preprocessing --------------------
ANALYSIS_AGE_MIN_YR_BP = 1_000
ANALYSIS_AGE_MAX_YR_BP = 640_000
UNIFORM_STEP_YR = 10               # initial uniform interpolation before filtering
DEFAULT_TE_STEP_YR = 50            # TE input resolution for baseline analyses
TARGET_INTERP_KIND_FOR_TE = "nearest"  # keep close to the original notebook
BUTTER_ORDER = 4
DEFAULT_CUTOFF_PERIOD_YR = 10_000
TARGET_BINS = 2                    # fixed warm/cold state representation

# ------------------------------ TE settings ----------------------------------
DEFAULT_FORCING_BINS = 6
DEFAULT_K = 1
ALPHA = 0.05
RNG_SEED = 42

# IMPORTANT: use only 'hist' and 'quantile' in the main robustness workflow.
# kmeans is intentionally excluded from the main script to keep the results
# easier to interpret and compare.
BASELINE_DISCRETIZATION = "hist"          # "hist" | "quantile"
DISCRETIZATION_OPTIONS = ("hist", "quantile")

# Null models.
# shuffle: weakest null, kept mainly as a baseline.
# phase: preserves the power spectrum but not the exact amplitude distribution.
# iaaft: stricter null that approximately preserves both.
BASELINE_NULL_MODEL = "shuffle"
NULL_MODELS_COMPARISON = ("shuffle", "phase", "iaaft")
SCAN_NULL_MODELS = ("shuffle",)
GRID_NULL_MODEL = "shuffle"

# Surrogate counts and IAAFT iterations.
# These values can be computationally expensive. Increase / decrease here.
N_SURROGATES_BASELINE = 150
N_SURROGATES_SCAN = 60
N_SURROGATES_GRID = 48
N_SURROGATES_NULL_SHUFFLE = 100
N_SURROGATES_NULL_PHASE = 60
N_SURROGATES_NULL_IAAFT = 10
IAAFT_MAX_ITER = 20

# ------------------------------ experiment toggles ---------------------------
RUN_BASELINE = True
RUN_DT_SCAN = True
RUN_CUTOFF_SCAN = True
RUN_SHIFT_SCAN = True
RUN_BINS_K_GRID = True
RUN_DISCRETIZATION_BINS_GRID = True
RUN_NULL_MODEL_COMPARISON = True

# ------------------------------ parameter grids ------------------------------
DT_GRID_YR = (10, 20, 30, 40, 50, 75, 100, 125, 150, 200)
CUTOFF_GRID_YR = (8_000, 9_000, 10_000, 11_000, 12_000)
SHIFT_GRID_YR = (-20_000, -10_000, -5_000, 0, 5_000, 10_000, 20_000)
FORCING_BINS_GRID = (4, 5, 6, 7, 8)
K_GRID = (1, 2, 3)

# ------------------------------ figure settings ------------------------------
SAVE_DPI = 300
FIG_EXTENSIONS = ("png", "pdf")
FONT_FAMILY = "Arial"
LOG_PROGRESS = True
BASE_FONT_SIZE = 10.5
PANEL_LABEL_SIZE = 11.5
LINEWIDTH = 1.4
MARKERSIZE = 5.0
SCAN_MARKER_AREA = 34
SCAN_MARKER_AREA_SIG = 54


@dataclass
class Config:
    data_dir: Path = DATA_DIR
    results_dir: Path = RESULTS_DIR
    ch4_file: str = CH4_FILE
    monsoon_file: str = MONSOON_FILE
    pre_file: str = PRE_FILE
    obl_file: str = OBL_FILE
    ecc_file: str = ECC_FILE
    analysis_age_min_yr_bp: int = ANALYSIS_AGE_MIN_YR_BP
    analysis_age_max_yr_bp: int = ANALYSIS_AGE_MAX_YR_BP
    uniform_step_yr: int = UNIFORM_STEP_YR
    default_te_step_yr: int = DEFAULT_TE_STEP_YR
    target_interp_kind_for_te: str = TARGET_INTERP_KIND_FOR_TE
    butter_order: int = BUTTER_ORDER
    default_cutoff_period_yr: int = DEFAULT_CUTOFF_PERIOD_YR
    target_bins: int = TARGET_BINS
    default_forcing_bins: int = DEFAULT_FORCING_BINS
    default_k: int = DEFAULT_K
    alpha: float = ALPHA
    rng_seed: int = RNG_SEED
    baseline_discretization: str = BASELINE_DISCRETIZATION
    baseline_null_model: str = BASELINE_NULL_MODEL
    n_surrogates_baseline: int = N_SURROGATES_BASELINE
    n_surrogates_scan: int = N_SURROGATES_SCAN
    n_surrogates_grid: int = N_SURROGATES_GRID
    n_surrogates_null_shuffle: int = N_SURROGATES_NULL_SHUFFLE
    n_surrogates_null_phase: int = N_SURROGATES_NULL_PHASE
    n_surrogates_null_iaaft: int = N_SURROGATES_NULL_IAAFT
    iaaft_max_iter: int = IAAFT_MAX_ITER


# =============================================================================
# Plot styling helpers
# =============================================================================
def _choose_font_family() -> str:
    available = {f.name for f in font_manager.fontManager.ttflist}
    for candidate in (FONT_FAMILY, "Helvetica", "Arial Unicode MS", "DejaVu Sans"):
        if candidate in available:
            return candidate
    return "DejaVu Sans"


def log_message(cfg: Config | None, message: str) -> None:
    if cfg is None or LOG_PROGRESS:
        print(message, flush=True)


def apply_grl_style() -> None:
    """Apply a clean publication-like plotting style."""
    chosen_font = _choose_font_family()
    mpl.rcParams.update({
        "font.family": [chosen_font, "DejaVu Sans"],
        "font.size": BASE_FONT_SIZE,
        "axes.labelsize": BASE_FONT_SIZE,
        "axes.titlesize": BASE_FONT_SIZE,
        "xtick.labelsize": BASE_FONT_SIZE - 0.5,
        "ytick.labelsize": BASE_FONT_SIZE - 0.5,
        "legend.fontsize": BASE_FONT_SIZE - 0.8,
        "axes.linewidth": 0.9,
        "lines.linewidth": LINEWIDTH,
        "savefig.dpi": SAVE_DPI,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def style_axis(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.tick_params(direction="out", length=3.5, width=0.9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def add_panel_labels(axes: Sequence[plt.Axes], x: float = -0.12, y: float = 1.03) -> None:
    for ax, label in zip(axes, string.ascii_lowercase):
        ax.text(x, y, label, transform=ax.transAxes,
                fontsize=PANEL_LABEL_SIZE, fontweight="bold",
                va="bottom", ha="left")


def save_figure(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in FIG_EXTENSIONS:
        fig.savefig(out_base.with_suffix(f".{ext}"), bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Data loading
# =============================================================================
def ensure_dirs(cfg: Config) -> None:
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    (cfg.results_dir / "csv").mkdir(parents=True, exist_ok=True)
    (cfg.results_dir / "figures").mkdir(parents=True, exist_ok=True)
    (cfg.results_dir / "tables").mkdir(parents=True, exist_ok=True)


def load_target_series(path: Path, value_col: str) -> pd.DataFrame:
    """
    Normalize age to years BP, sorted from young to old (ascending BP).

    Notes
    -----
    - CH4 file already mixes near-modern negative values with large positive yr BP.
    - monsoon ages are stored in kyr BP and need converting to yr BP.
    - slight negative ages near the present are tolerated and later excluded by the
      selected analysis window.
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

    # Convert kyr BP -> yr BP when the absolute age range is clearly kyr-scale.
    if out["age"].abs().max() < 10_000:
        out["age"] = out["age"] * 1000.0

    return out.sort_values("age").reset_index(drop=True)


def load_orbital_series(path: Path, value_name: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    if df.shape[1] < 2:
        raise ValueError(f"{path.name} must have at least two columns")
    out = df.iloc[:, :2].copy()
    out.columns = ["age_raw", value_name]
    out["age"] = out["age_raw"].abs() * 1000.0
    out[value_name] = pd.to_numeric(out[value_name], errors="coerce")
    return out[["age", value_name]].dropna().sort_values("age").reset_index(drop=True)


# =============================================================================
# Signal preprocessing
# =============================================================================
def resample_to_uniform_grid(
    df: pd.DataFrame,
    value_col: str,
    step_yr: float,
    age_min_yr: float,
    age_max_yr: float,
) -> pd.DataFrame:
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
    fs = 1.0 / dt
    fc = 1.0 / cutoff_period_yr
    wn = fc / (0.5 * fs)
    if not (0 < wn < 1):
        raise ValueError(f"Invalid Butterworth cutoff Wn={wn:.6f}")
    b, a = butter(butter_order, wn, btype="highpass")
    filt = filtfilt(b, a, values)
    return pd.DataFrame({"age": ages, output_col: filt})


def prepare_target_filtered(
    *,
    df_raw: pd.DataFrame,
    value_col: str,
    output_col: str,
    cfg: Config,
    cutoff_period_yr: float,
) -> pd.DataFrame:
    df_uniform = resample_to_uniform_grid(
        df_raw,
        value_col=value_col,
        step_yr=cfg.uniform_step_yr,
        age_min_yr=cfg.analysis_age_min_yr_bp,
        age_max_yr=cfg.analysis_age_max_yr_bp,
    )
    return highpass_filter_series(
        df_uniform,
        value_col=value_col,
        cutoff_period_yr=cutoff_period_yr,
        butter_order=cfg.butter_order,
        output_col=output_col,
    )


def align_target_with_forcing(
    *,
    df_target_filt: pd.DataFrame,
    target_col: str,
    orbital: Dict[str, pd.DataFrame],
    te_step_yr: float,
    age_min_yr: float,
    age_max_yr: float,
    target_interp_kind: str,
    forcing_shift_yr: float = 0.0,
) -> pd.DataFrame:
    """
    Align one filtered target with orbital forcing onto a common TE grid.

    forcing_shift_yr is used for negative-control experiments.
    Positive shift means the forcing age axis is moved to older ages.
    """
    overlap_min = max(
        age_min_yr,
        float(df_target_filt["age"].min()),
        *(float(df["age"].min() + forcing_shift_yr) for df in orbital.values()),
    )
    overlap_max = min(
        age_max_yr,
        float(df_target_filt["age"].max()),
        *(float(df["age"].max() + forcing_shift_yr) for df in orbital.values()),
    )
    if overlap_max <= overlap_min:
        raise ValueError("No overlapping age range between target and forcing")

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
        shifted_age = df["age"].to_numpy(dtype=float) + forcing_shift_yr
        f_forcing = interp1d(
            shifted_age,
            df[name].to_numpy(dtype=float),
            kind="nearest",
            bounds_error=False,
            fill_value="extrapolate",
        )
        aligned[name] = f_forcing(new_age)

    return aligned


# =============================================================================
# Discretization and surrogate generation
# =============================================================================
def _sanitize_edges(edges: np.ndarray, bins: int, series: np.ndarray) -> np.ndarray:
    edges = np.asarray(edges, dtype=float)
    edges = np.unique(edges)
    if len(edges) < bins + 1:
        smin = float(np.nanmin(series))
        smax = float(np.nanmax(series))
        if np.isclose(smin, smax):
            smax = smin + 1e-9
        edges = np.linspace(smin, smax, bins + 1)
    # Ensure the edges fully cover the data by a tiny epsilon.
    eps = 1e-12 * max(1.0, np.nanstd(series))
    edges[0] -= eps
    edges[-1] += eps
    return edges


def make_bin_edges(series: np.ndarray, bins: int, method: str) -> np.ndarray:
    if method == "hist":
        edges = np.histogram_bin_edges(series, bins=bins)
    elif method == "quantile":
        edges = np.quantile(series, np.linspace(0, 1, bins + 1))
    else:
        raise ValueError("method must be 'hist' or 'quantile'")
    return _sanitize_edges(edges, bins, series)


def apply_bin_edges(series: np.ndarray, edges: np.ndarray, bins: int) -> np.ndarray:
    disc = np.digitize(series, edges) - 1
    return np.clip(disc, 0, bins - 1).astype(np.int32)


def stable_seed(*parts: object, base_seed: int) -> int:
    import hashlib
    joined = "||".join(map(str, parts)).encode("utf-8")
    digest = hashlib.blake2b(joined, digest_size=8).hexdigest()
    return (base_seed + int(digest, 16)) % (2**32 - 1)


def n_surrogates_for_null(cfg: Config, method: str) -> int:
    if method == "shuffle":
        return cfg.n_surrogates_null_shuffle
    if method == "phase":
        return cfg.n_surrogates_null_phase
    if method == "iaaft":
        return cfg.n_surrogates_null_iaaft
    raise ValueError(f"Unknown surrogate method: {method}")


def phase_randomized_surrogate(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Generate a phase-randomized surrogate preserving the power spectrum."""
    x = np.asarray(x, dtype=float)
    n = x.size
    x_centered = x - np.mean(x)
    spectrum = np.fft.rfft(x_centered)
    amps = np.abs(spectrum)

    random_phases = rng.uniform(0, 2 * np.pi, len(spectrum))
    random_phases[0] = 0.0
    if n % 2 == 0 and len(random_phases) > 1:
        random_phases[-1] = 0.0

    surrogate = np.fft.irfft(amps * np.exp(1j * random_phases), n=n)
    surrogate += np.mean(x)
    return surrogate


def iaaft_surrogate(x: np.ndarray, rng: np.random.Generator, max_iter: int) -> np.ndarray:
    """Approximate IAAFT surrogate preserving amplitude distribution and spectrum."""
    x = np.asarray(x, dtype=float)
    n = x.size
    sorted_x = np.sort(x)
    target_amp = np.abs(np.fft.rfft(x - np.mean(x)))

    y = rng.permutation(x)
    last = y.copy()
    for _ in range(max_iter):
        # Impose the target power spectrum.
        y_fft = np.fft.rfft(y - np.mean(y))
        phase = np.exp(1j * np.angle(y_fft))
        y = np.fft.irfft(target_amp * phase, n=n)

        # Impose the original amplitude distribution by rank ordering.
        order = np.argsort(y)
        y_ranked = np.empty_like(y)
        y_ranked[order] = sorted_x
        y = y_ranked

        if np.allclose(y, last, rtol=0, atol=1e-12):
            break
        last = y.copy()

    return y


def make_source_surrogate(
    x: np.ndarray,
    *,
    method: str,
    rng: np.random.Generator,
    iaaft_max_iter: int,
) -> np.ndarray:
    if method == "shuffle":
        return rng.permutation(x)
    if method == "phase":
        return phase_randomized_surrogate(x, rng)
    if method == "iaaft":
        return iaaft_surrogate(x, rng, max_iter=iaaft_max_iter)
    raise ValueError("Unknown surrogate method")


# =============================================================================
# TE computation
# =============================================================================
def _chronological_arrays(forcing: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reverse young->old BP arrays to oldest->youngest chronological order.

    This preserves the logic used in the original notebook / sq_ana.py TE tests.
    """
    x = np.asarray(forcing, dtype=float)[::-1]
    y = np.asarray(target, dtype=float)[::-1]
    valid = np.isfinite(x) & np.isfinite(y)
    return x[valid], y[valid]


def evaluate_te(
    forcing: np.ndarray,
    target: np.ndarray,
    *,
    k: int,
    forcing_bins: int,
    target_bins: int,
    discretization_method: str,
    surrogate_method: str,
    n_surrogates: int,
    alpha: float,
    rng: np.random.Generator,
    iaaft_max_iter: int,
) -> dict:
    """
    Compute bidirectional TE and one-sided surrogate significance.

    TE is evaluated on chronological series (oldest -> youngest). For null_yx,
    only the source series (the target treated as source) is surrogated, while
    the destination remains unchanged, mirroring the null_xy logic.
    """
    x, y = _chronological_arrays(forcing, target)
    if len(x) <= k + 2:
        raise ValueError("Time series too short for the requested TE history length")

    x_edges = make_bin_edges(x, forcing_bins, discretization_method)
    y_edges = make_bin_edges(y, target_bins, discretization_method)

    x_disc = apply_bin_edges(x, x_edges, forcing_bins)
    y_disc = apply_bin_edges(y, y_edges, target_bins)

    te_xy = float(transfer_entropy(x_disc, y_disc, k=k))
    te_yx = float(transfer_entropy(y_disc, x_disc, k=k))

    null_xy = np.zeros(n_surrogates, dtype=float)
    null_yx = np.zeros(n_surrogates, dtype=float)

    for i in range(n_surrogates):
        xs = make_source_surrogate(x, method=surrogate_method, rng=rng, iaaft_max_iter=iaaft_max_iter)
        ys = make_source_surrogate(y, method=surrogate_method, rng=rng, iaaft_max_iter=iaaft_max_iter)
        xs_disc = apply_bin_edges(xs, x_edges, forcing_bins)
        ys_disc = apply_bin_edges(ys, y_edges, target_bins)
        null_xy[i] = float(transfer_entropy(xs_disc, y_disc, k=k))
        null_yx[i] = float(transfer_entropy(ys_disc, x_disc, k=k))

    p_xy = float((np.sum(null_xy >= te_xy) + 1) / (n_surrogates + 1))
    p_yx = float((np.sum(null_yx >= te_yx) + 1) / (n_surrogates + 1))
    delta_te = te_xy - te_yx

    return {
        "te_xy": te_xy,
        "te_yx": te_yx,
        "delta_te": delta_te,
        "p_xy": p_xy,
        "p_yx": p_yx,
        "sig_xy": bool(p_xy < alpha),
        "sig_yx": bool(p_yx < alpha),
        "uni_sig": bool((p_xy < alpha) and (p_yx >= alpha)),
        "null_xy_mean": float(np.mean(null_xy)),
        "null_yx_mean": float(np.mean(null_yx)),
        "null_xy_q95": float(np.quantile(null_xy, 0.95)),
        "null_yx_q95": float(np.quantile(null_yx, 0.95)),
    }


# =============================================================================
# Experiment wrappers
# =============================================================================
def load_all_inputs(cfg: Config) -> Tuple[Dict[str, Tuple[pd.DataFrame, str, str]], Dict[str, pd.DataFrame]]:
    targets = {
        "ch4": (load_target_series(cfg.data_dir / cfg.ch4_file, "ch4"), "ch4", "CH$_4$"),
        "monsoon": (load_target_series(cfg.data_dir / cfg.monsoon_file, "d18O"), "d18O", r"Monsoon $\delta^{18}$O"),
    }
    orbital = {
        "pre": load_orbital_series(cfg.data_dir / cfg.pre_file, "pre"),
        "obl": load_orbital_series(cfg.data_dir / cfg.obl_file, "obl"),
        "ecc": load_orbital_series(cfg.data_dir / cfg.ecc_file, "ecc"),
    }
    return targets, orbital


def build_filtered_targets(cfg: Config, targets: Dict[str, Tuple[pd.DataFrame, str, str]], cutoff_period_yr: float) -> Dict[str, Tuple[pd.DataFrame, str, str]]:
    out = {}
    for target_name, (df_raw, value_col, label) in targets.items():
        filt_col = f"filt_{value_col}"
        df_filt = prepare_target_filtered(
            df_raw=df_raw,
            value_col=value_col,
            output_col=filt_col,
            cfg=cfg,
            cutoff_period_yr=cutoff_period_yr,
        )
        out[target_name] = (df_filt, filt_col, label)
    return out


def run_single_condition(
    *,
    cfg: Config,
    filtered_targets: Dict[str, Tuple[pd.DataFrame, str, str]],
    orbital: Dict[str, pd.DataFrame],
    te_step_yr: float,
    cutoff_period_yr: float,
    discretization_method: str,
    forcing_bins: int,
    k: int,
    surrogate_method: str,
    n_surrogates: int,
    forcing_shift_yr: float,
    experiment_type: str,
    parameter_name: str,
    parameter_value: float | str,
) -> pd.DataFrame:
    rows: List[dict] = []
    base_seed = int(cfg.rng_seed)

    log_message(cfg, f"[{experiment_type}] {parameter_name}={parameter_value} | null={surrogate_method} | n_surr={n_surrogates}")

    for target_idx, (target_name, (df_filt, filt_col, label)) in enumerate(filtered_targets.items()):
        aligned = align_target_with_forcing(
            df_target_filt=df_filt,
            target_col=filt_col,
            orbital=orbital,
            te_step_yr=te_step_yr,
            age_min_yr=cfg.analysis_age_min_yr_bp,
            age_max_yr=cfg.analysis_age_max_yr_bp,
            target_interp_kind=cfg.target_interp_kind_for_te,
            forcing_shift_yr=forcing_shift_yr,
        )
        for driver_idx, driver in enumerate(("pre", "obl", "ecc")):
            seed = stable_seed(experiment_type, parameter_name, parameter_value, surrogate_method, target_name, driver, base_seed=base_seed)
            rng = np.random.default_rng(seed)
            res = evaluate_te(
                aligned[driver].to_numpy(),
                aligned[filt_col].to_numpy(),
                k=k,
                forcing_bins=forcing_bins,
                target_bins=cfg.target_bins,
                discretization_method=discretization_method,
                surrogate_method=surrogate_method,
                n_surrogates=n_surrogates,
                alpha=cfg.alpha,
                rng=rng,
                iaaft_max_iter=cfg.iaaft_max_iter,
            )
            rows.append({
                "target": target_name,
                "target_label": label,
                "driver": driver,
                "experiment_type": experiment_type,
                "parameter_name": parameter_name,
                "parameter_value": parameter_value,
                "te_step_yr": te_step_yr,
                "cutoff_period_yr": cutoff_period_yr,
                "discretization_method": discretization_method,
                "forcing_bins": forcing_bins,
                "target_bins": cfg.target_bins,
                "k": k,
                "surrogate_method": surrogate_method,
                "n_surrogates": n_surrogates,
                "forcing_shift_yr": forcing_shift_yr,
                **res,
            })
    return pd.DataFrame(rows)


def run_baseline(cfg: Config, filtered_targets, orbital) -> pd.DataFrame:
    return run_single_condition(
        cfg=cfg,
        filtered_targets=filtered_targets,
        orbital=orbital,
        te_step_yr=cfg.default_te_step_yr,
        cutoff_period_yr=cfg.default_cutoff_period_yr,
        discretization_method=cfg.baseline_discretization,
        forcing_bins=cfg.default_forcing_bins,
        k=cfg.default_k,
        surrogate_method=cfg.baseline_null_model,
        n_surrogates=cfg.n_surrogates_baseline,
        forcing_shift_yr=0.0,
        experiment_type="baseline",
        parameter_name="baseline",
        parameter_value="baseline",
    )


def run_dt_scan(cfg: Config, filtered_targets, orbital) -> pd.DataFrame:
    dfs = []
    for surrogate in SCAN_NULL_MODELS:
        for dt in DT_GRID_YR:
            dfs.append(run_single_condition(
                cfg=cfg,
                filtered_targets=filtered_targets,
                orbital=orbital,
                te_step_yr=float(dt),
                cutoff_period_yr=cfg.default_cutoff_period_yr,
                discretization_method=cfg.baseline_discretization,
                forcing_bins=cfg.default_forcing_bins,
                k=cfg.default_k,
                surrogate_method=surrogate,
                n_surrogates=cfg.n_surrogates_scan,
                forcing_shift_yr=0.0,
                experiment_type="dt_scan",
                parameter_name="dt_yr",
                parameter_value=float(dt),
            ))
    return pd.concat(dfs, ignore_index=True)


def run_cutoff_scan(cfg: Config, targets, orbital) -> pd.DataFrame:
    dfs = []
    for cutoff in CUTOFF_GRID_YR:
        filtered_targets = build_filtered_targets(cfg, targets, cutoff_period_yr=float(cutoff))
        for surrogate in SCAN_NULL_MODELS:
            dfs.append(run_single_condition(
                cfg=cfg,
                filtered_targets=filtered_targets,
                orbital=orbital,
                te_step_yr=cfg.default_te_step_yr,
                cutoff_period_yr=float(cutoff),
                discretization_method=cfg.baseline_discretization,
                forcing_bins=cfg.default_forcing_bins,
                k=cfg.default_k,
                surrogate_method=surrogate,
                n_surrogates=cfg.n_surrogates_scan,
                forcing_shift_yr=0.0,
                experiment_type="cutoff_scan",
                parameter_name="cutoff_period_yr",
                parameter_value=float(cutoff),
            ))
    return pd.concat(dfs, ignore_index=True)


def run_shift_scan(cfg: Config, filtered_targets, orbital) -> pd.DataFrame:
    dfs = []
    for shift in SHIFT_GRID_YR:
        for surrogate in SCAN_NULL_MODELS:
            dfs.append(run_single_condition(
                cfg=cfg,
                filtered_targets=filtered_targets,
                orbital=orbital,
                te_step_yr=cfg.default_te_step_yr,
                cutoff_period_yr=cfg.default_cutoff_period_yr,
                discretization_method=cfg.baseline_discretization,
                forcing_bins=cfg.default_forcing_bins,
                k=cfg.default_k,
                surrogate_method=surrogate,
                n_surrogates=cfg.n_surrogates_scan,
                forcing_shift_yr=float(shift),
                experiment_type="shift_scan",
                parameter_name="forcing_shift_yr",
                parameter_value=float(shift),
            ))
    return pd.concat(dfs, ignore_index=True)


def run_bins_k_grid(cfg: Config, filtered_targets, orbital) -> pd.DataFrame:
    dfs = []
    for k in K_GRID:
        for bins in FORCING_BINS_GRID:
            dfs.append(run_single_condition(
                cfg=cfg,
                filtered_targets=filtered_targets,
                orbital=orbital,
                te_step_yr=cfg.default_te_step_yr,
                cutoff_period_yr=cfg.default_cutoff_period_yr,
                discretization_method=cfg.baseline_discretization,
                forcing_bins=int(bins),
                k=int(k),
                surrogate_method=GRID_NULL_MODEL,
                n_surrogates=cfg.n_surrogates_grid,
                forcing_shift_yr=0.0,
                experiment_type="bins_k_grid",
                parameter_name="forcing_bins_k",
                parameter_value=f"bins={bins},k={k}",
            ))
    return pd.concat(dfs, ignore_index=True)


def run_discretization_bins_grid(cfg: Config, filtered_targets, orbital) -> pd.DataFrame:
    dfs = []
    for method in DISCRETIZATION_OPTIONS:
        for bins in FORCING_BINS_GRID:
            dfs.append(run_single_condition(
                cfg=cfg,
                filtered_targets=filtered_targets,
                orbital=orbital,
                te_step_yr=cfg.default_te_step_yr,
                cutoff_period_yr=cfg.default_cutoff_period_yr,
                discretization_method=method,
                forcing_bins=int(bins),
                k=cfg.default_k,
                surrogate_method=GRID_NULL_MODEL,
                n_surrogates=cfg.n_surrogates_grid,
                forcing_shift_yr=0.0,
                experiment_type="discretization_bins_grid",
                parameter_name="discretization_bins",
                parameter_value=f"{method}|bins={bins}",
            ))
    return pd.concat(dfs, ignore_index=True)


def run_null_model_comparison(cfg: Config, filtered_targets, orbital) -> pd.DataFrame:
    dfs = []
    for surrogate in NULL_MODELS_COMPARISON:
        dfs.append(run_single_condition(
            cfg=cfg,
            filtered_targets=filtered_targets,
            orbital=orbital,
            te_step_yr=cfg.default_te_step_yr,
            cutoff_period_yr=cfg.default_cutoff_period_yr,
            discretization_method=cfg.baseline_discretization,
            forcing_bins=cfg.default_forcing_bins,
            k=cfg.default_k,
            surrogate_method=surrogate,
            n_surrogates=n_surrogates_for_null(cfg, surrogate),
            forcing_shift_yr=0.0,
            experiment_type="null_model_comparison",
            parameter_name="null_model",
            parameter_value=surrogate,
        ))
    return pd.concat(dfs, ignore_index=True)


# =============================================================================
# Plotting of experiment outputs
# =============================================================================


# =============================================================================
# Plotting of experiment outputs
# =============================================================================
def _marker_face(row: pd.Series) -> str:
    return "black" if bool(row["uni_sig"]) else "white"


def plot_baseline_comparison(df: pd.DataFrame, out_base: Path) -> None:
    apply_grl_style()
    fig, axes = plt.subplots(2, 3, figsize=(8.2, 4.8), sharey="row")
    axes_flat = axes.ravel()

    for col_idx, driver in enumerate(("pre", "obl", "ecc")):
        for row_idx, target in enumerate(("ch4", "monsoon")):
            ax = axes[row_idx, col_idx]
            sub = df[(df["driver"] == driver) & (df["target"] == target)].iloc[0]
            xs = np.array([0, 1])
            ys = np.array([sub["te_xy"], sub["te_yx"]])
            ax.plot(xs, ys, color="black", lw=1.0)
            ax.scatter([0], [sub["te_xy"]], s=34, facecolor="black" if sub["sig_xy"] else "white", edgecolor="black", zorder=3)
            ax.scatter([1], [sub["te_yx"]], s=34, facecolor="black" if sub["sig_yx"] else "white", edgecolor="black", zorder=3)
            ax.axhline(0, color="0.75", lw=0.8)
            ax.set_xticks(xs)
            ax.set_xticklabels([f"{driver}$\\to$target", f"target$\\to${driver}"] if row_idx == 1 else [])
            if col_idx == 0:
                ax.set_ylabel(f"{sub['target_label']}\nTE (bits)")
            style_axis(ax)

    add_panel_labels(axes_flat)
    fig.tight_layout()
    save_figure(fig, out_base)


def plot_scan_panels(df: pd.DataFrame, x_col: str, x_label: str, out_base: Path) -> None:
    apply_grl_style()
    fig, axes = plt.subplots(3, 2, figsize=(8.3, 7.3), sharex="col", constrained_layout=True)
    axes_flat = axes.ravel()
    c_xy = "#1f77b4"
    c_yx = "#d95f02"

    for row_idx, driver in enumerate(("pre", "obl", "ecc")):
        for col_idx, target in enumerate(("ch4", "monsoon")):
            ax = axes[row_idx, col_idx]
            sub = df[(df["driver"] == driver) & (df["target"] == target)].sort_values(x_col)
            if sub.empty:
                continue

            ax.plot(sub[x_col], sub["te_xy"], color=c_xy, lw=1.5, label="forcing→target")
            ax.plot(sub[x_col], sub["te_yx"], color=c_yx, lw=1.5, ls="--", label="target→forcing")

            sig_xy = sub["sig_xy"].to_numpy(dtype=bool)
            sig_yx = sub["sig_yx"].to_numpy(dtype=bool)
            xvals = sub[x_col].to_numpy()
            y_xy = sub["te_xy"].to_numpy()
            y_yx = sub["te_yx"].to_numpy()

            ax.scatter(xvals[~sig_xy], y_xy[~sig_xy], s=SCAN_MARKER_AREA,
                       marker="o", facecolor="white", edgecolor=c_xy,
                       linewidth=1.0, zorder=4)
            ax.scatter(xvals[sig_xy], y_xy[sig_xy], s=SCAN_MARKER_AREA_SIG,
                       marker="o", facecolor=c_xy, edgecolor="black",
                       linewidth=0.6, zorder=5)

            ax.scatter(xvals[~sig_yx], y_yx[~sig_yx], s=SCAN_MARKER_AREA,
                       marker="s", facecolor="white", edgecolor=c_yx,
                       linewidth=1.0, zorder=4)
            ax.scatter(xvals[sig_yx], y_yx[sig_yx], s=SCAN_MARKER_AREA_SIG,
                       marker="s", facecolor=c_yx, edgecolor="black",
                       linewidth=0.6, zorder=5)

            ax.axhline(0, color="0.75", lw=0.8)
            if row_idx == 2:
                ax.set_xlabel(x_label)
            if col_idx == 0:
                ax.set_ylabel(f"{driver}\nTE (bits)")
            if row_idx == 0:
                ax.text(0.02, 0.98, r"CH$_4$" if target == "ch4" else r"Monsoon $\delta^{18}$O",
                        transform=ax.transAxes, ha="left", va="top")
            style_axis(ax)

    line_xy = Line2D([0], [0], color=c_xy, lw=1.5, label="forcing→target")
    line_yx = Line2D([0], [0], color=c_yx, lw=1.5, ls="--", label="target→forcing")
    sig_handle = Line2D([0], [0], marker="o", color="black", lw=0,
                        markerfacecolor="black", markersize=6.5,
                        label="significant")
    nonsig_handle = Line2D([0], [0], marker="o", color="black", lw=0,
                           markerfacecolor="white", markersize=6.5,
                           label="not significant")
    fig.legend([line_xy, line_yx, sig_handle, nonsig_handle],
               ["forcing→target", "target→forcing", "significant", "not significant"],
               loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.52, 1.02))
    add_panel_labels(axes_flat)
    save_figure(fig, out_base)


def _pivot_heatmap(df: pd.DataFrame, target: str, driver: str, value_col: str, row_key: str, col_key: str) -> pd.DataFrame:
    sub = df[(df["target"] == target) & (df["driver"] == driver)].copy()
    return sub.pivot(index=row_key, columns=col_key, values=value_col).sort_index().sort_index(axis=1)


def _add_heatmap_grid(ax: plt.Axes, nrows: int, ncols: int) -> None:
    ax.set_xticks(np.arange(-0.5, ncols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nrows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)


def plot_bins_k_heatmap_sig(df: pd.DataFrame, out_base_sig: Path) -> None:
    apply_grl_style()
    fig, axes = plt.subplots(3, 2, figsize=(7.9, 7.8), constrained_layout=True)
    sig_cmap = ListedColormap(["#e9edf2", "#2b8cbe"])
    sig_norm = BoundaryNorm([-0.5, 0.5, 1.5], sig_cmap.N)
    last_im = None

    for row_idx, driver in enumerate(("pre", "obl", "ecc")):
        for col_idx, target in enumerate(("ch4", "monsoon")):
            ax = axes[row_idx, col_idx]
            piv_sig = _pivot_heatmap(df.assign(uni_sig=df["uni_sig"].astype(int)), target, driver, "uni_sig", "k", "forcing_bins")
            last_im = ax.imshow(piv_sig.values, origin="lower", aspect="auto", cmap=sig_cmap, norm=sig_norm)
            ax.set_xticks(np.arange(len(piv_sig.columns)))
            ax.set_xticklabels(piv_sig.columns)
            ax.set_yticks(np.arange(len(piv_sig.index)))
            ax.set_yticklabels(piv_sig.index)
            _add_heatmap_grid(ax, *piv_sig.values.shape)
            if row_idx == 2:
                ax.set_xlabel("forcing bins")
            if col_idx == 0:
                ax.set_ylabel(f"{driver}\n$k$")
            style_axis(ax)

    add_panel_labels(axes.ravel())
    cbar = fig.colorbar(last_im, ax=axes, fraction=0.025, pad=0.02, ticks=[0, 1])
    cbar.ax.set_yticklabels(["no", "yes"])
    cbar.set_label("unidirectional significance")
    save_figure(fig, out_base_sig)


def plot_discretization_bins_heatmap_sig(df: pd.DataFrame, out_base_sig: Path) -> None:
    apply_grl_style()
    fig, axes = plt.subplots(3, 2, figsize=(7.9, 7.8), constrained_layout=True)
    sig_cmap = ListedColormap(["#f0f0f0", "#7fcdbb"])
    sig_norm = BoundaryNorm([-0.5, 0.5, 1.5], sig_cmap.N)
    last_im = None

    for row_idx, driver in enumerate(("pre", "obl", "ecc")):
        for col_idx, target in enumerate(("ch4", "monsoon")):
            ax = axes[row_idx, col_idx]
            sub = df[(df["target"] == target) & (df["driver"] == driver)].copy()
            piv_sig = sub.assign(uni_sig=sub["uni_sig"].astype(int)).pivot(index="discretization_method", columns="forcing_bins", values="uni_sig")
            piv_sig = piv_sig.sort_index().sort_index(axis=1)
            last_im = ax.imshow(piv_sig.values, origin="lower", aspect="auto", cmap=sig_cmap, norm=sig_norm)
            ax.set_xticks(np.arange(len(piv_sig.columns)))
            ax.set_xticklabels(piv_sig.columns)
            ax.set_yticks(np.arange(len(piv_sig.index)))
            ax.set_yticklabels(piv_sig.index)
            _add_heatmap_grid(ax, *piv_sig.values.shape)
            if row_idx == 2:
                ax.set_xlabel("forcing bins")
            if col_idx == 0:
                ax.set_ylabel(f"{driver}\nmethod")
            style_axis(ax)

    add_panel_labels(axes.ravel())
    cbar = fig.colorbar(last_im, ax=axes, fraction=0.025, pad=0.02, ticks=[0, 1])
    cbar.ax.set_yticklabels(["no", "yes"])
    cbar.set_label("unidirectional significance")
    save_figure(fig, out_base_sig)


def plot_null_model_comparison(df: pd.DataFrame, out_base: Path) -> None:
    apply_grl_style()
    fig, axes = plt.subplots(3, 2, figsize=(8.2, 7.2), sharex=True, sharey=True, constrained_layout=True)
    width = 0.34
    x = np.arange(len(NULL_MODELS_COMPARISON))
    thresh = -np.log10(ALPHA)
    c_xy = "#4c78a8"
    c_yx = "#f58518"

    for row_idx, driver in enumerate(("pre", "obl", "ecc")):
        for col_idx, target in enumerate(("ch4", "monsoon")):
            ax = axes[row_idx, col_idx]
            sub = df[(df["driver"] == driver) & (df["target"] == target)].copy()
            sub["surrogate_method"] = pd.Categorical(sub["surrogate_method"], categories=NULL_MODELS_COMPARISON, ordered=True)
            sub = sub.sort_values("surrogate_method")
            y1 = -np.log10(np.clip(sub["p_xy"].to_numpy(dtype=float), 1e-12, 1.0))
            y2 = -np.log10(np.clip(sub["p_yx"].to_numpy(dtype=float), 1e-12, 1.0))
            ax.bar(x - width/2, y1, width, color=c_xy, label="forcing→target")
            ax.bar(x + width/2, y2, width, color=c_yx, label="target→forcing")
            ax.axhline(thresh, color="0.55", lw=0.9, ls="--")
            for i, (_, row) in enumerate(sub.iterrows()):
                if bool(row["sig_xy"]):
                    ax.text(x[i]-width/2, y1[i]+0.05, "*", ha="center", va="bottom", fontsize=11)
                if bool(row["sig_yx"]):
                    ax.text(x[i]+width/2, y2[i]+0.05, "*", ha="center", va="bottom", fontsize=11)
            ax.set_xticks(x)
            ax.set_xticklabels(NULL_MODELS_COMPARISON)
            if row_idx == 2:
                ax.set_xlabel("null model")
            if col_idx == 0:
                ax.set_ylabel(f"{driver}\n-log$_{10}$(p)")
            if row_idx == 0:
                ax.text(0.02, 0.98, r"CH$_4$" if target == "ch4" else r"Monsoon $\delta^{18}$O", transform=ax.transAxes, ha="left", va="top")
            style_axis(ax)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.52, 1.01))
    add_panel_labels(axes.ravel())
    save_figure(fig, out_base)


# =============================================================================
# Main driver
# =============================================================================


# =============================================================================
# Main driver
# =============================================================================
def main() -> None:
    cfg = Config()
    ensure_dirs(cfg)
    apply_grl_style()

    # Save config so the run is fully documented.
    cfg_json = {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()}
    with open(cfg.results_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg_json, f, indent=2)

    targets, orbital = load_all_inputs(cfg)
    filtered_targets_default = build_filtered_targets(cfg, targets, cutoff_period_yr=cfg.default_cutoff_period_yr)

    all_frames: List[pd.DataFrame] = []

    if RUN_BASELINE:
        log_message(cfg, "Starting baseline analysis...")
        df = run_baseline(cfg, filtered_targets_default, orbital)
        df.to_csv(cfg.results_dir / "csv" / "baseline.csv", index=False)
        plot_baseline_comparison(df, cfg.results_dir / "figures" / "fig_baseline_comparison")
        all_frames.append(df)

    if RUN_DT_SCAN:
        log_message(cfg, "Starting dt scan...")
        df = run_dt_scan(cfg, filtered_targets_default, orbital)
        df.to_csv(cfg.results_dir / "csv" / "dt_scan.csv", index=False)
        plot_scan_panels(df[df["surrogate_method"] == SCAN_NULL_MODELS[0]], "parameter_value", "TE input step (yr)", cfg.results_dir / "figures" / "fig_dt_scan")
        all_frames.append(df)

    if RUN_CUTOFF_SCAN:
        log_message(cfg, "Starting cutoff scan...")
        df = run_cutoff_scan(cfg, targets, orbital)
        df.to_csv(cfg.results_dir / "csv" / "cutoff_scan.csv", index=False)
        plot_scan_panels(df[df["surrogate_method"] == SCAN_NULL_MODELS[0]], "parameter_value", "High-pass cutoff (yr)", cfg.results_dir / "figures" / "fig_cutoff_scan")
        all_frames.append(df)

    if RUN_SHIFT_SCAN:
        log_message(cfg, "Starting shift scan...")
        df = run_shift_scan(cfg, filtered_targets_default, orbital)
        df.to_csv(cfg.results_dir / "csv" / "shift_scan.csv", index=False)
        plot_scan_panels(df[df["surrogate_method"] == SCAN_NULL_MODELS[0]], "parameter_value", "Forcing age shift (yr)", cfg.results_dir / "figures" / "fig_shift_scan")
        all_frames.append(df)

    if RUN_BINS_K_GRID:
        log_message(cfg, "Starting forcing-bins × k grid...")
        df = run_bins_k_grid(cfg, filtered_targets_default, orbital)
        df.to_csv(cfg.results_dir / "csv" / "bins_k_grid.csv", index=False)
        plot_bins_k_heatmap_sig(df, cfg.results_dir / "figures" / "fig_bins_k_grid_uni_sig")
        all_frames.append(df)

    if RUN_DISCRETIZATION_BINS_GRID:
        log_message(cfg, "Starting discretization × bins grid...")
        df = run_discretization_bins_grid(cfg, filtered_targets_default, orbital)
        df.to_csv(cfg.results_dir / "csv" / "discretization_bins_grid.csv", index=False)
        plot_discretization_bins_heatmap_sig(df, cfg.results_dir / "figures" / "fig_discretization_bins_grid_uni_sig")
        all_frames.append(df)

    if RUN_NULL_MODEL_COMPARISON:
        log_message(cfg, "Starting null-model comparison...")
        df = run_null_model_comparison(cfg, filtered_targets_default, orbital)
        df.to_csv(cfg.results_dir / "csv" / "null_model_comparison.csv", index=False)
        plot_null_model_comparison(df, cfg.results_dir / "figures" / "fig_null_model_comparison")
        all_frames.append(df)

    if all_frames:
        df_all = pd.concat(all_frames, ignore_index=True)
        df_all.to_csv(cfg.results_dir / "tables" / "all_results_long.csv", index=False)


if __name__ == "__main__":
    main()
