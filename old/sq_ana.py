import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import groupby
from scipy.interpolate import interp1d
from scipy.stats import ttest_ind
from pyinform import transfer_entropy

from toolbox import sq_ana as sa
import importlib
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed





import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
from scipy import stats
from scipy.signal import find_peaks

import numpy as np







import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans








import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# def local_te_stay_flip_by_bin(
#     forcing, sq,
#     k=1,
#     forcing_bins=6, sq_bins=2,
#     sq_method='hist',          # 'hist' | 'quantile' | 'kmeans'
#     xbins=None, ybins=None,    # optionally pass precomputed edges for consistency
#     alpha=0.5,                 # Laplace smoothing
#     reverse=True,              # match your TE function’s reverse
#     if_plot=True, dpi=120
# ):
#     """
#     Compute local TE (k=1) from forcing -> sq, and decompose mean TE
#     per forcing-bin into contributions from 'stay' vs 'flip' in sq.

#     Returns
#     -------
#     out : dict with keys
#         - xbins, ybins           : bin edges used for forcing and sq (None for kmeans y)
#         - te_mean_bin            : mean local TE per forcing bin (all events)
#         - te_mean_stay_bin       : mean local TE per bin, restricted to stays
#         - te_mean_flip_bin       : mean local TE per bin, restricted to flips
#         - te_sum_bin             : sum of local TE per bin (all events)
#         - te_sum_stay_bin        : sum of local TE per bin for stays
#         - te_sum_flip_bin        : sum of local TE per bin for flips
#         - count_stay_bin         : number of stay events per bin
#         - count_flip_bin         : number of flip events per bin
#         - local_te               : 1D array of local TE for each t (length N-1)
#         - x_idx, y_prev, y_next  : discretized sequences used (aligned to N-1 pairs)
#         - counts_tensor          : raw count tensor C[bin, y_prev, y_next]
#     fig : matplotlib Figure (or None)
#     """
#     if k != 1:
#         raise NotImplementedError("This helper currently implements local TE for k=1 only.")

#     # ------------------ prep & optional reversal ------------------
#     x = np.asarray(forcing, float)
#     y = np.asarray(sq, float)

#     if reverse:
#         x = x[::-1].copy()
#         y = y[::-1].copy()

#     # Drop NaNs consistently
#     valid = np.isfinite(x) & np.isfinite(y)
#     x = x[valid]
#     y = y[valid]

#     # ------------------ discretize y (sq) and x ------------------
#     if sq_method == 'quantile':
#         if xbins is None:
#             xbins = np.quantile(x, np.linspace(0, 1, forcing_bins + 1))
#         if ybins is None:
#             ybins = np.quantile(y, np.linspace(0, 1, sq_bins + 1))
#         y_disc = np.digitize(y, ybins) - 1

#     elif sq_method == 'hist':
#         if xbins is None:
#             xbins = np.histogram_bin_edges(x, bins=forcing_bins)
#         if ybins is None:
#             ybins = np.histogram_bin_edges(y, bins=sq_bins)
#         y_disc = np.digitize(y, ybins) - 1

#     elif sq_method == 'kmeans':
#         if xbins is None:
#             xbins = np.histogram_bin_edges(x, bins=forcing_bins)
#         km = KMeans(n_clusters=sq_bins, n_init=10, random_state=0)
#         y_disc = km.fit_predict(y.reshape(-1, 1))
#         ybins = None  # no edges for kmeans
#     else:
#         raise ValueError(f"Unknown sq_method '{sq_method}'")

#     # Discretize forcing using xbins
#     x_disc = np.digitize(x, xbins) - 1

#     B = forcing_bins
#     K = sq_bins

#     # Clip indices to valid range
#     x_disc = np.clip(x_disc, 0, B - 1)
#     y_disc = np.clip(y_disc, 0, K - 1)

#     # ------------------ build one-step pairs ------------------
#     # indices correspond to t=0..N-2 for (y_t, x_t) and t+1 for y_{t+1}
#     y_prev = y_disc[:-1]
#     y_next = y_disc[1:]
#     x_idx  = x_disc[:-1]

#     # ------------------ counts & conditional probabilities ------------------
#     # C[b, s0, s1] = count of (x_bin=b, y_t=s0, y_{t+1}=s1)
#     counts_tensor = np.zeros((B, K, K), dtype=np.int64)
#     for b, s0, s1 in zip(x_idx, y_prev, y_next):
#         counts_tensor[b, s0, s1] += 1

#     # Global counts over x: C_y[s0, s1]
#     counts_y = counts_tensor.sum(axis=0)

#     # Conditional P(y_{t+1} | y_t, x_bin=b) with Laplace smoothing
#     P1 = (counts_tensor + alpha) / (
#         counts_tensor.sum(axis=2, keepdims=True) + alpha * K
#     )

#     # Conditional P(y_{t+1} | y_t) (marginalizing over x)
#     P0 = (counts_y + alpha) / (
#         counts_y.sum(axis=1, keepdims=True) + alpha * K
#     )

#     # ------------------ local TE per time step ------------------
#     # local_te[t] = log2 P1[b, y_prev, y_next] - log2 P0[y_prev, y_next]
#     p1_local = P1[x_idx, y_prev, y_next]
#     p0_local = P0[y_prev, y_next]
#     eps = 1e-15
#     local_te = np.log2(np.maximum(p1_local, eps)) - np.log2(np.maximum(p0_local, eps))

#     # ------------------ per-bin TE decomposition (stay vs flip) ------------------
#     te_mean_bin       = np.full(B, np.nan, float)
#     te_sum_bin        = np.full(B, np.nan, float)
#     te_mean_stay_bin  = np.full(B, np.nan, float)
#     te_mean_flip_bin  = np.full(B, np.nan, float)
#     te_sum_stay_bin   = np.full(B, np.nan, float)
#     te_sum_flip_bin   = np.full(B, np.nan, float)
#     count_stay_bin    = np.zeros(B, dtype=int)
#     count_flip_bin    = np.zeros(B, dtype=int)

#     for b in range(B):
#         idx_b = (x_idx == b)
#         if not np.any(idx_b):
#             continue

#         # all events in this forcing bin
#         te_b = local_te[idx_b]
#         te_mean_bin[b] = np.mean(te_b)
#         te_sum_bin[b]  = np.sum(te_b)

#         # stays vs flips
#         stays_b = (y_next[idx_b] == y_prev[idx_b])
#         flips_b = ~stays_b

#         if np.any(stays_b):
#             te_stay = te_b[stays_b]
#             te_mean_stay_bin[b] = np.mean(te_stay)
#             te_sum_stay_bin[b]  = np.sum(te_stay)
#             count_stay_bin[b]   = te_stay.size

#         if np.any(flips_b):
#             te_flip = te_b[flips_b]
#             te_mean_flip_bin[b] = np.mean(te_flip)
#             te_sum_flip_bin[b]  = np.sum(te_flip)
#             count_flip_bin[b]   = te_flip.size

#     # ------------------ plotting ------------------
#     fig = None
#     if if_plot:
#         fig = plt.figure(figsize=(max(6, 0.9 * B), 4.2), dpi=dpi)
#         ax = plt.gca()

#         x_pos = np.arange(B)
#         width = 0.38

#         # bars: mean TE for stay vs flip
#         b1 = ax.bar(x_pos - width/2, te_mean_stay_bin, width,
#                     label="mean TE (stay)")
#         b2 = ax.bar(x_pos + width/2, te_mean_flip_bin, width,
#                     label="mean TE (flip)")

#         # zero line
#         ax.axhline(0, lw=1.25, color='k', alpha=0.6)

#         # annotate with mean values
#         for i in range(B):
#             vals = [te_mean_stay_bin[i], te_mean_flip_bin[i]]
#             if not np.any(np.isfinite(vals)):
#                 continue
#             y_max = np.nanmax(np.abs(vals))
#             if not np.isfinite(y_max) or y_max == 0:
#                 continue
#             # place label just above the taller bar (by magnitude)
#             j = int(np.nanargmax(np.abs(vals)))
#             y_annot = vals[j]
#             ax.text(
#                 x_pos[i] + (-width/2 if j == 0 else width/2),
#                 y_annot + 0.02 * np.sign(y_annot),
#                 f"{y_annot:+.3f}",
#                 ha='center',
#                 va='bottom' if y_annot >= 0 else 'top',
#                 fontsize=9,
#                 rotation=90
#             )

#         # x tick labels as bin ranges
#         edges = xbins
#         if edges is not None and len(edges) == B + 1:
#             labels = [
#                 f"[{edges[i]:.3f}, {edges[i+1]:.3f})" for i in range(B-1)
#             ] + [f"[{edges[-2]:.3f}, {edges[-1]:.3f}]"]
#         else:
#             labels = [f"bin {i+1}" for i in range(B)]

#         ax.set_xticks(x_pos)
#         ax.set_xticklabels(labels, rotation=30, ha='right')
#         ax.set_ylabel("Mean local TE (bits)")
#         ax.set_title("Mean TE for stay vs flip by forcing bin")
#         ax.legend(frameon=True, loc='upper right')

#         # y-limit padding
#         ymax = np.nanmax(np.abs(np.concatenate([te_mean_stay_bin, te_mean_flip_bin])))
#         if np.isfinite(ymax) and ymax > 0:
#             ax.set_ylim(-1.1 * ymax, 1.1 * ymax)

#         plt.tight_layout()
#         plt.show()

#     out = dict(
#         xbins=xbins, ybins=ybins,
#         te_mean_bin=te_mean_bin, te_sum_bin=te_sum_bin,
#         te_mean_stay_bin=te_mean_stay_bin, te_sum_stay_bin=te_sum_stay_bin,
#         te_mean_flip_bin=te_mean_flip_bin, te_sum_flip_bin=te_sum_flip_bin,
#         count_stay_bin=count_stay_bin, count_flip_bin=count_flip_bin,
#         local_te=local_te,
#         x_idx=x_idx, y_prev=y_prev, y_next=y_next,
#         counts_tensor=counts_tensor,
#     )
#     return out, fig








def local_te_decomposition_by_bin(
    forcing, sq,
    k=1,
    forcing_bins=6, sq_bins=2,
    sq_method='hist',          # 'hist' | 'quantile' | 'kmeans'
    xbins=None, ybins=None,    # optionally pass precomputed edges for consistency
    alpha=0.5,                 # Laplace smoothing
    reverse=True,              # match your TE function’s reverse
    if_plot=True, dpi=120
):
    """
    Decompose transfer entropy (k=1) by forcing-bin and diagnose whether
    positive local TE is associated with 'stay' (y_{t+1}=y_t) or 'flip' (y_{t+1}!=y_t).

    Returns
    -------
    out : dict with keys
        - xbins, ybins : bin edges (xbins for forcing; ybins for sq when method uses edges)
        - te_mean_bin  : mean local TE per forcing bin (shape [B])
        - te_sum_bin   : summed local TE per forcing bin (shape [B])
        - contrib_mask : boolean mask of bins with positive mean local TE
        - p_stay_bin   : P(stay | bin) per bin (shape [B])
        - p_flip_bin   : P(flip | bin) per bin (shape [B])
        - p_stay_baseline_mix : baseline stay prob per bin under the same y_prev mix (shape [B])
        - delta_stay_vs_baseline : p_stay_bin - p_stay_baseline_mix (shape [B])
        - local_te     : array of local TE values for each t (length N-1)
        - x_idx, y_prev, y_next : discretized sequences used (aligned to N-1 pairs)
        - counts_tensor : raw count tensor C[bin, y_prev, y_next]
    fig : matplotlib Figure (or None)
    """
    if k != 1:
        raise NotImplementedError("This helper currently implements local TE for k=1 only.")

    # ------------- prep & discretize (match your TE function) -------------
    x = np.asarray(forcing)
    y = np.asarray(sq)
    if reverse:
        x = x[::-1].copy()
        y = y[::-1].copy()

    # Drop NaNs consistently
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    # # Discretize y (sq)
    # if sq_method == 'quantile':
    #     if ybins is None:
    #         ybins = np.quantile(y, np.linspace(0, 1, sq_bins + 1))
    #     y_disc = np.digitize(y, ybins) - 1
    # elif sq_method == 'hist':
    #     if ybins is None:
    #         ybins = np.histogram_bin_edges(y, bins=sq_bins)
    #     y_disc = np.digitize(y, ybins) - 1
    # elif sq_method == 'kmeans':
    #     km = KMeans(n_clusters=sq_bins, n_init=10, random_state=0)
    #     y_disc = km.fit_predict(y.reshape(-1, 1))
    #     ybins = None  # no edges for kmeans
    # else:
    #     raise ValueError(f"Unknown sq_method '{sq_method}'")



    # # Discretize sq series
    # if sq_method == 'quantile':
    #     xbins = np.quantile(x, np.linspace(0, 1, forcing_bins+1))
    #     ybins = np.quantile(y, np.linspace(0, 1, sq_bins+1))
    #     y_disc = np.digitize(y, ybins) - 1
    # elif sq_method == 'hist':
    #     xbins = np.histogram_bin_edges(x, bins=forcing_bins)
    #     ybins = np.histogram_bin_edges(y, bins=sq_bins)
    #     y_disc = np.digitize(y, ybins) - 1

    # elif sq_method == 'kmeans':
    #     xbins = np.histogram_bin_edges(x, bins=forcing_bins)
    #     # Fit k-means on the values
    #     km = KMeans(n_clusters=sq_bins, n_init=10, random_state=0)
    #     y_disc = km.fit_predict(y.reshape(-1, 1))

    if sq_method == 'quantile':
        if xbins is None:
            xbins = np.quantile(x, np.linspace(0, 1, forcing_bins+1))
        if ybins is None:
            ybins = np.quantile(y, np.linspace(0, 1, sq_bins+1))
        y_disc = np.digitize(y, ybins) - 1
    elif sq_method == 'hist':
        if xbins is None:
            xbins = np.histogram_bin_edges(x, bins=forcing_bins)
        if ybins is None:
            ybins = np.histogram_bin_edges(y, bins=sq_bins)
        y_disc = np.digitize(y, ybins) - 1
    elif sq_method == 'kmeans':
        if xbins is None:
            xbins = np.histogram_bin_edges(x, bins=forcing_bins)
        km = KMeans(n_clusters=sq_bins, n_init=10, random_state=0)
        y_disc = km.fit_predict(y.reshape(-1, 1))
        ybins = None  # no edges



    x_disc = np.digitize(x, xbins) - 1

    # Clip any out-of-range indices (e.g., right-edge hits)
    B = forcing_bins
    K = sq_bins
    x_disc = np.clip(x_disc, 0, B - 1)
    y_disc = np.clip(y_disc, 0, K - 1)

    # Build one-step pairs
    # indices correspond to times t=0..N-2 for (y_t, x_t) and t+1 for y_{t+1}
    y_prev = y_disc[:-1]
    y_next = y_disc[1:]
    x_idx  = x_disc[:-1]

    # ------------- counts & conditional probability tables -------------
    # C[b, s0, s1] = count of (x_bin=b, y_t=s0, y_{t+1}=s1)
    counts_tensor = np.zeros((B, K, K), dtype=np.int64)
    for b, s0, s1 in zip(x_idx, y_prev, y_next):
        counts_tensor[b, s0, s1] += 1

    # Global counts over x: C_y[s0, s1]
    counts_y = counts_tensor.sum(axis=0)

    # Conditional P(y_{t+1} | y_t, x_bin=b) with Laplace smoothing
    P1 = (counts_tensor + alpha) / ((counts_tensor.sum(axis=2, keepdims=True) + alpha * K))

    # Conditional P(y_{t+1} | y_t) (marginalizing x)
    P0 = (counts_y + alpha) / ((counts_y.sum(axis=1, keepdims=True) + alpha * K))

    # ------------- local TE per time step -------------
    # local_te[t] = log2 P1[b, y_prev, y_next] - log2 P0[y_prev, y_next]
    p1_local = P1[x_idx, y_prev, y_next]
    p0_local = P0[y_prev, y_next]
    # Guard against any numerical issues
    eps = 1e-15
    local_te = np.log2(np.maximum(p1_local, eps)) - np.log2(np.maximum(p0_local, eps))

    # ------------- per-bin aggregation & stay/flip diagnostics -------------
    te_mean_bin = np.full(B, np.nan, float)
    te_sum_bin  = np.full(B, np.nan, float)
    p_stay_bin  = np.full(B, np.nan, float)
    p_flip_bin  = np.full(B, np.nan, float)
    p_stay_baseline_mix = np.full(B, np.nan, float)
    delta_stay_vs_baseline = np.full(B, np.nan, float)

    # baseline conditional stay probability given y_prev *mix in that bin*
    # w_b[s0] = fraction of y_prev==s0 within bin b
    for b in range(B):
        idx_b = (x_idx == b)
        if not np.any(idx_b):
            continue

        # TE aggregations
        te_mean_bin[b] = np.mean(local_te[idx_b])
        te_sum_bin[b]  = np.sum(local_te[idx_b])

        # stay/flip within bin
        stays = (y_next[idx_b] == y_prev[idx_b])
        total_b = np.count_nonzero(idx_b)
        p_stay = np.count_nonzero(stays) / total_b
        p_flip = 1.0 - p_stay
        p_stay_bin[b] = p_stay
        p_flip_bin[b] = p_flip

        # mix-aware baseline stay prob:
        # w_b over y_prev
        w_b = np.bincount(y_prev[idx_b], minlength=K).astype(float)
        if w_b.sum() > 0:
            w_b /= w_b.sum()
        # baseline stay prob under same mix: sum_s w_b[s] * P0[s, s]
        p_stay_base_mix = float(np.sum(w_b * np.diag(P0)))
        p_stay_baseline_mix[b] = p_stay_base_mix
        delta_stay_vs_baseline[b] = p_stay - p_stay_base_mix

    contrib_mask = te_mean_bin > 0  # bins with positive mean local TE

    # ------------- plot -------------
    fig = None
    if if_plot:
        fig = plt.figure(figsize=(max(6, 0.9*B), 4.2), dpi=dpi)
        ax = plt.gca()

        # Bars: place "stay" and "flip" per bin; sign determined by mean local TE
        x_pos = np.arange(B)
        width = 0.38

        # Heights signed by TE sign (positive up, negative down)
        signs = np.sign(np.nan_to_num(te_mean_bin, nan=0.0))
        stay_heights = signs * np.nan_to_num(p_stay_bin, nan=0.0)
        flip_heights = signs * np.nan_to_num(p_flip_bin, nan=0.0)

        # default to zero where NaN
        stay_heights = np.where(np.isfinite(stay_heights), stay_heights, 0.0)
        flip_heights = np.where(np.isfinite(flip_heights), flip_heights, 0.0)

        b1 = ax.bar(x_pos - width/2, stay_heights, width, label="P(stay | bin)")
        b2 = ax.bar(x_pos + width/2, flip_heights, width, label="P(flip | bin)")

        # zero line
        ax.axhline(0, lw=1.25, color='k', alpha=0.6)

        # annotate with mean local TE per bin
        for i in range(B):
            if not np.isfinite(te_mean_bin[i]): 
                continue
            y_annot = (abs(stay_heights[i]) + abs(flip_heights[i])) * (1 if signs[i] >= 0 else -1)
            # nudge outside the taller bar
            if y_annot == 0:
                y_annot = 0.02 if signs[i] >= 0 else -0.02
            ax.text(i, y_annot,
                    f"⟨ℓ⟩={te_mean_bin[i]:+.3f}",
                    ha='center', va='bottom' if signs[i]>=0 else 'top',
                    fontsize=9, rotation=90)

        # x tick labels as bin ranges
        edges = xbins
        if edges is not None and len(edges) == B+1:
            labels = [f"[{edges[i]:.3f}, {edges[i+1]:.3f})" for i in range(B-1)] + [f"[{edges[-2]:.3f}, {edges[-1]:.3f}]"]
        else:
            labels = [f"bin {i+1}" for i in range(B)]

        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=30, ha='right')
        ax.set_ylabel("Conditional probability (signed by mean local TE)")
        ax.set_title("Stay/Flip by forcing bin (bars up if bin mean local TE > 0, else down)")
        ax.legend(frameon=True, loc='upper right')
        # y-limit padding
        ymax = 1.05 * max(1e-6, np.nanmax(np.abs([stay_heights, flip_heights])))
        ax.set_ylim(-ymax, ymax)

        plt.tight_layout()
        plt.show()

    out = dict(
        xbins=xbins, ybins=ybins,
        te_mean_bin=te_mean_bin, te_sum_bin=te_sum_bin,
        contrib_mask=contrib_mask,
        p_stay_bin=p_stay_bin, p_flip_bin=p_flip_bin,
        p_stay_baseline_mix=p_stay_baseline_mix,
        delta_stay_vs_baseline=delta_stay_vs_baseline,
        local_te=local_te,
        x_idx=x_idx, y_prev=y_prev, y_next=y_next,
        counts_tensor=counts_tensor
    )
    return out, fig

import numpy as np
import matplotlib.pyplot as plt

def plot_decomposition(
    out,
    normalize='mean',         # 'mean' or 'sum'
    show_driver_label=True,
    stay_color=None,          # defaults from tab10 if None
    flip_color=None,
    dpi=140
):
    """
    TE decomposition with two right-hand y-axes:
      - Right axis (inner):  TE contribution of 'stay' (bars)
      - Right axis (outer):  TE contribution of 'flip' (bars)
      - Left axis:           total (stay+flip) as hollow diamonds
    """

    # -------- Unpack --------
    x_idx   = out['x_idx']
    y_prev  = out['y_prev']
    y_next  = out['y_next']
    lte     = out['local_te']
    te_mean = out['te_mean_bin']        # length B
    B = len(te_mean)
    if B == 0:
        raise ValueError("te_mean_bin is empty.")

    # -------- Colors --------
    tab10 = plt.get_cmap("tab10").colors
    stay_color = stay_color or tab10[0]   # blue-ish
    flip_color = flip_color or tab10[3]   # red-ish

    # -------- Compute TE contributions per bin --------
    te_stay = np.full(B, np.nan)
    te_flip = np.full(B, np.nan)
    n_inbin = np.zeros(B, dtype=int)

    for b in range(B):
        mask_b = (x_idx == b)
        n_b = int(np.count_nonzero(mask_b))
        n_inbin[b] = n_b
        if n_b == 0:
            continue

        stay_b = mask_b & (y_next == y_prev)
        flip_b = mask_b & (y_next != y_prev)

        if normalize == 'sum':
            te_stay[b] = np.nansum(lte[stay_b])
            te_flip[b] = np.nansum(lte[flip_b])
        elif normalize == 'mean':
            te_stay[b] = np.nansum(lte[stay_b]) / n_b
            te_flip[b] = np.nansum(lte[flip_b]) / n_b
        else:
            raise ValueError("normalize must be 'mean' or 'sum'")

    total = te_stay + te_flip

    # -------- Helper: symmetric limits with margin --------
    def sym_limits(vals, pad=0.15, fallback=1.0):
        finite = np.asarray(vals)[np.isfinite(vals)]
        m = np.max(np.abs(finite)) if finite.size else 0.0
        if m == 0:
            m = fallback
        M = m * (1 + pad)
        return -M, M

    # -------- Figure & axes --------
    fig, axL = plt.subplots(figsize=(max(8, 1.0 * B), 5.2), dpi=dpi)
    axR_stay = axL.twinx()
    axR_flip = axL.twinx()
    axR_flip.spines['right'].set_position(('axes', 1.18))
    axR_flip.spines['right'].set_visible(True)
    fig.subplots_adjust(right=0.82)

    # X ticks
    x = np.arange(B)
    axL.set_xticks(x)
    axL.set_xticklabels([str(i+1) for i in range(B)], ha='center')
    axL.set_xlabel('forcing bin')

    # Y limits (give bars comfy headroom so labels stay inside)
    L_lo, L_hi       = sym_limits(total,   pad=0.10)
    stay_lo, stay_hi = sym_limits(te_stay, pad=0.20)
    flip_lo, flip_hi = sym_limits(te_flip, pad=0.20)

    axL.set_ylim(L_lo, L_hi)
    axR_stay.set_ylim(stay_lo, stay_hi)
    axR_flip.set_ylim(flip_lo, flip_hi)

    # Zero line
    axL.axhline(0, color='k', lw=0.8, alpha=0.6)

    # -------- Bars --------
    width = 0.68
    bars_stay = axR_stay.bar(
        x, te_stay, width,
        color=stay_color, alpha=0.6,
        edgecolor='white', linewidth=0.6,
        label='TE contrib: stay', zorder=3
    )
    bars_flip = axR_flip.bar(
        x, te_flip, width,
        color=flip_color, alpha=0.6,
        edgecolor='white', linewidth=0.6,
        label='TE contrib: flip', zorder=4
    )

    # -------- Total (hollow diamonds) --------
    axL.plot(
        x, total,
        linestyle='None', marker='D', markersize=6.5,
        markerfacecolor='none', markeredgecolor='black', markeredgewidth=1.5,
        zorder=6, label='Mean local TE (total)'
    )

    # -------- Labels & style --------
    ylab = "Mean local TE (total)" if normalize == 'mean' else "Total TE per bin (sum)"
    axL.set_ylabel(ylab)
    axR_stay.set_ylabel("TE contrib (stay)", color=stay_color)
    axR_flip.set_ylabel("TE contrib (flip)", color=flip_color)

    axR_stay.tick_params(axis='y', colors=stay_color)
    axR_flip.tick_params(axis='y', colors=flip_color)
    axR_stay.spines['right'].set_color(stay_color)
    axR_flip.spines['right'].set_color(flip_color)

    axL.yaxis.grid(True, linestyle='--', linewidth=0.6, alpha=0.35)
    axL.set_axisbelow(True)

    # -------- Simple driver labels (no transforms) --------
    if show_driver_label:
        for i in range(B):
            ts = te_stay[i]; tf = te_flip[i]
            if not (np.isfinite(ts) or np.isfinite(tf)):
                continue

            # Treat NaNs as -inf so the other wins; compare raw values (as before)
            ts_cmp = ts if np.isfinite(ts) else -np.inf
            tf_cmp = tf if np.isfinite(tf) else -np.inf

            if ts_cmp >= tf_cmp:
                anchor = ts
                ax_for_text = axR_stay
                label_txt = "stay-driven"
            else:
                anchor = tf
                ax_for_text = axR_flip
                label_txt = "flip-driven"

            if not np.isfinite(anchor):
                continue

            dy = 6 if anchor >= 0 else -6
            va = 'bottom' if anchor >= 0 else 'top'
            ax_for_text.annotate(
                label_txt,
                xy=(i, anchor), xycoords=ax_for_text.transData,
                xytext=(0, dy), textcoords='offset points',
                ha='center', va=va, fontsize=9, rotation=90,
                clip_on=True, zorder=7
            )

    # -------- Legend (clean) --------
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([], [], linestyle='None', marker='D', markersize=6.5,
               markerfacecolor='none', markeredgecolor='black', markeredgewidth=1.5,
               label='Mean local TE (total)'),
        Patch(facecolor=stay_color, alpha=0.6, edgecolor='white', linewidth=0.6, label='TE contrib: stay'),
        Patch(facecolor=flip_color, alpha=0.6, edgecolor='white', linewidth=0.6, label='TE contrib: flip'),
    ]
    axL.legend(handles=legend_handles, loc='upper left', frameon=True)

    plt.tight_layout()
    plt.show()

    return {
        "te_contrib_stay": te_stay,
        "te_contrib_flip": te_flip,
        "total": total,
        "n_inbin": n_inbin
    }, fig



def threshold_filtered_square_wave(sq, age, std_multiplier=1.0, min_years=0.0, levels=(-1.0, 1.0)):
    """
    Produce a square-wave signal from `sq` that only flips when:
      (1) the sign around the median changes, AND
      (2) at the flip, at least one side exceeds a global threshold:
          max(|sq[i]-med|, |sq[i-1]-med|) >= std_multiplier * global_std
      (3) accepted flips are ≥ min_years apart along the `age` axis.

    Small wiggles (< threshold) are ignored, so the output stays in the
    previous state. Output levels default to (-1, +1).

    Parameters
    ----------
    sq : array_like
        Signal values (1D).
    age : array_like
        Time/age values (1D, same length as `sq`). Can be ascending or descending.
    std_multiplier : float
        Multiplier for the global standard deviation threshold.
    min_years : float
        Minimum separation between accepted flips, in the same units as `age`.
    levels : tuple(float, float)
        (low_level, high_level) for the square wave.

    Returns
    -------
    wave : np.ndarray
        Square-wave array of shape (N,), with values in `levels`.
    """
    sq  = np.asarray(sq,  float)
    age = np.asarray(age, float)
    if sq.ndim != 1 or age.ndim != 1 or sq.size != age.size:
        raise ValueError("sq and age must be 1D arrays of the same length.")

    med = np.median(sq)
    glob_std = sq.std(ddof=1) + 1e-12
    thr = std_multiplier * glob_std

    # Sign relative to median; candidate flips where sign changes
    sgn = np.where(sq - med >= 0.0, 1, -1)
    candidate_flips = np.where(np.diff(sgn) != 0)[0] + 1

    # If no flips at all, return a constant wave
    if candidate_flips.size == 0:
        start_high = (sgn[0] > 0)
        start_level = levels[1] if start_high else levels[0]
        return np.full_like(sq, fill_value=start_level, dtype=float)

    # Convert min_years to samples (handle descending age with abs(dt))
    dt = float(np.median(np.diff(age)))
    dt = abs(dt) if dt != 0 else 1.0
    min_samp = int(round(min_years / dt)) if min_years > 0 else 0
    min_samp = max(0, min_samp)

    # Keep only threshold-passing, well-spaced flips
    keep = []
    last_idx = -10**9
    for i in candidate_flips:
        v1 = abs(sq[i]   - med)
        v0 = abs(sq[i-1] - med)
        if max(v1, v0) >= thr and (i - last_idx) >= max(1, min_samp):
            keep.append(i)
            last_idx = i
    keep = np.array(keep, dtype=int)

    # Build square wave with flips only at `keep`
    N = sq.size
    wave = np.empty(N, dtype=float)

    # Initial state based on sign at start
    high_state = (sgn[0] > 0)
    low_level, high_level = levels

    bounds = np.concatenate(([0], keep, [N]))
    state_high = high_state
    for k in range(len(bounds) - 1):
        wave[bounds[k]:bounds[k+1]] = high_level if state_high else low_level
        state_high = not state_high  # toggle at each kept flip

    return wave



# ------------------------------
# Event detector 
# ------------------------------
def detect_transitions_threshold(sq, age, std_multiplier=1.0, min_years=0.0):
    """
    Keep an event when the sign around the median flips AND at the flip
    at least one side is beyond a global threshold:
        max(|sq[i]-med|, |sq[i-1]-med|) >= std_multiplier * global_std
    Returns indices into the time series.
    """
    sq  = np.asarray(sq,  float)
    age = np.asarray(age, float)

    med = np.median(sq)
    glob_std = sq.std(ddof=1) + 1e-12
    thr = std_multiplier * glob_std

    # raw sign flips around median
    sgn = np.where(sq - med >= 0.0, 1, -1)
    flips = np.where(np.diff(sgn) != 0)[0] + 1
    if flips.size == 0:
        return flips

    # keep only flips that cross the global threshold
    keep = []
    last_idx = -10**9
    dt = float(np.median(np.diff(age)))
    min_samp = int(round(min_years / dt)) if min_years > 0 else 0

    for i in flips:
        v1 = abs(sq[i]   - med)
        v0 = abs(sq[i-1] - med)
        if max(v1, v0) >= thr:
            if (i - last_idx) >= max(1, min_samp):
                keep.append(i)
                last_idx = i

    return np.array(keep, dtype=int)


# ------------------------------
# Phase from min–max interpolation, wrapped
# ------------------------------
def minmax_phase(age, pre, min_separation_years=8000.0):
    """
    Build a phase that is 0° at a local minimum, 180° at the next maximum,
    360° at the next minimum, etc., using linear interpolation between extrema.
    Returns theta_wrapped in radians in [0, 2π), with NaN before the first and
    after the last usable extremum.
    """
    age = np.asarray(age, float)
    pre = np.asarray(pre, float)
    assert np.all(np.diff(age) > 0), "age must be strictly increasing"

    dt = np.median(np.diff(age))
    min_dist_samples = max(1, int(round(min_separation_years / dt)))

    # extrema
    max_idx, _ = find_peaks(pre,  distance=min_dist_samples)
    min_idx, _ = find_peaks(-pre, distance=min_dist_samples)

    if len(min_idx) + len(max_idx) < 2:
        return np.full_like(pre, np.nan, dtype=float)

    # merge, sort, and enforce alternation starting at a minimum
    idx   = np.concatenate([min_idx, max_idx])
    types = np.concatenate([np.full_like(min_idx, -1), np.full_like(max_idx, +1)])  # -1=min, +1=max
    order = np.argsort(idx)
    idx, types = idx[order], types[order]

    if types[0] == +1:  # start at min
        idx, types = idx[1:], types[1:]
    if len(idx) < 2:
        return np.full_like(pre, np.nan, dtype=float)

    keep = [0]
    for k in range(1, len(idx)):
        if types[k] != types[keep[-1]]:
            keep.append(k)
    idx, types = idx[keep], types[keep]
    if len(idx) < 2:
        return np.full_like(pre, np.nan, dtype=float)

    # assign unwrapped degrees at extrema: 0, 180, 360, ...
    phase_deg_ext = np.arange(len(idx)) * 180.0
    age_ext       = age[idx]

    # interpolate unwrapped; outside -> NaN
    phase_deg_unwrapped = np.interp(age, age_ext, phase_deg_ext, left=np.nan, right=np.nan)

    # wrap to [0, 360) then radians
    phase_deg_wrapped = np.mod(phase_deg_unwrapped, 360.0)
    theta_wrapped = np.deg2rad(phase_deg_wrapped)
    return np.sin(theta_wrapped)


# import hilbert transform
from scipy.signal import hilbert

def nor_phase(df_pre_i, df_ecc_i=None, use_ecc=False):
    """
    return the phase of precession after Hilbert transform-based
    amplitude normalization.
    If use_ecc is True, first regress precession on eccentricity and use the residuals.
    Parameters
    ----------
    df_pre_i : pd.DataFrame
        DataFrame with 'age' and 'pre' columns.
    df_ecc_i : pd.DataFrame, optional
        DataFrame with 'age' and 'ecc' columns, required if use_ecc is True.
    use_ecc : bool, optional
        If True, regress precession on eccentricity and use the residuals for phase calculation.

    Returns
    -------
    pre_norm : np.ndarray
        Normalized precession phase after Hilbert transform.
    """
    if use_ecc:
        if df_ecc_i is None:
            raise ValueError("df_ecc_i must be provided when use_ecc is True.")
        else:
            amp=df_ecc_i['ecc'].values
    else:
        amp = np.abs(hilbert(pre))

    pre = df_pre_i['pre'].values
    pre_norm = pre / amp              # variance ≈ constant

    return pre_norm


def H_phase(df_pre_i, var_name='pre'):
    # get the phase of df_pre_i['pre'] using Hilbert transform
    pre = df_pre_i[var_name].values
    analytic_signal = hilbert(pre)
    phase = np.angle(analytic_signal)

    return np.sin(phase)


def interpolate_df(df, interval):
    # Get column names
    time_col = df.columns[0]
    value_col = df.columns[1]
    
    # Create new regular time grid
    time_min = df[time_col].min()
    time_max = df[time_col].max()
    new_time = np.arange(time_min, time_max + interval, interval)
    
    # Interpolate values onto new time grid
    new_values = np.interp(new_time, df[time_col], df[value_col])
    
    # # Create new dataframe
    # df_interp = pd.DataFrame({
    #     time_col: new_time,
    #     value_col: new_values
    # })
    # return df_interp, using the same column names as the original dataframe
    df_interp = pd.DataFrame({time_col: new_time, value_col: new_values})
    
    return df_interp







import numpy as np, matplotlib.pyplot as plt, pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection
from pyinform import transfer_entropy

# ------------------------------------------------------------
# short helper: build 0/1 sequence from σ-multiplier
# ------------------------------------------------------------
def _state(sig, s_mult, sigma_ref=None):
    thr = s_mult * (sigma_ref if sigma_ref is not None else np.std(sig))
    st  = np.full_like(sig, np.nan, float)
    st[sig <= -thr] = 0
    st[sig >=  thr] = 1
    for i in range(1, len(st)):          # carry forward
        if np.isnan(st[i]): st[i] = st[i-1]
    first = np.where(~np.isnan(st))[0][0]
    st[:first] = st[first]
    return st.astype(int)

# ------------------------------------------------------------
#  main analysis
# ------------------------------------------------------------
def thre_data_ana(
    forcing, sq,
    ages = None,  # optional time axis for plots
    *,
    bins_pre       = 6,
    sigma_mult     = 1.5,
    n_surr         = 200,
    nbins_phase    = 8,
    random_seed    = 0
):
    """
    Transfer-entropy workflow with threshold-defined MCV states.

    Parameters
    ----------
    forcing, sq : 1-D arrays (oldest→youngest or youngest→oldest — any order)
    bins_pre    : number of amplitude bins for the forcing
    sigma_mult  : σ-multiplier that defines ±thr
    n_surr      : # permutation surrogates for global TE test
    nbins_phase : # bins when discretising wavelet phase (unused here but
                  kept for interface homogeneity)
    """
    # ---------- 1. chronological order (oldest → youngest) --------------
    
    f = forcing[::-1]
    s = sq      [::-1]

    if ages is None:
        ages = np.arange(len(f))            # dummy axis for plots

    # ---------- 2. build threshold state -------------------------------
    state = _state(s, sigma_mult)       # 0 = cold, 1 = warm

    # ---------- 3. discretise forcing (equal-width amplitude bins) -----
    bins = np.histogram_bin_edges(f, bins=bins_pre)
    x_disc = np.clip(np.digitize(f, bins) - 1, 0, bins_pre-1)

    # ---------- 4. global TE + surrogates ------------------------------
    te_xy = transfer_entropy(x_disc[:-1], state[1:], k=1)
    te_yx = transfer_entropy(state[:-1],  x_disc[1:], k=1)

    rng = np.random.default_rng(random_seed)
    null_xy = np.empty(n_surr)
    null_yx = np.empty(n_surr)
    for i in range(n_surr):
        null_xy[i] = transfer_entropy(rng.permutation(x_disc)[:-1], state[1:], k=1)
        null_yx[i] = transfer_entropy(rng.permutation(state)[:-1],  x_disc[1:], k=1)

    # two-tailed p
    p_xy = (2*min((null_xy>=te_xy).sum(), (null_xy<=te_xy).sum()) + 1)/(n_surr+1)
    p_yx = (2*min((null_yx>=te_yx).sum(), (null_yx<=te_yx).sum()) + 1)/(n_surr+1)

    # ---------- 5. local TE, P(flip), P(stay) ---------------------------
    local_te = transfer_entropy(x_disc, state, k=1, local=True).flatten()[1:]
    x_prev, y_prev, y_next = x_disc[:-1], state[:-1], state[1:]

    counts = np.zeros((bins_pre, 2, 2), int)
    for xi, yi, zi in zip(x_prev, y_prev, y_next):
        counts[xi, yi, zi] += 1
    totals = counts.sum(axis=2, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        cond_p = counts / totals

    p_flip = np.array([cond_p[x,y,1-y] for x,y in zip(x_prev,y_prev)])
    p_stay = np.array([cond_p[x,y,  y] for x,y in zip(x_prev,y_prev)])

    N = min(len(local_te), len(p_flip))
    t_plot = ages[1:1+N]

    # ---------- 6. figures ---------------------------------------------
    # 6a. TE null distributions  (colours swapped vs. your previous draft)
    plt.figure(figsize=(7,3.3))
    plt.hist(null_xy, bins=25, color='#fd8d3c', alpha=.7, label='null  forcing→MCV')
    plt.hist(null_yx, bins=25, color='#6baed6', alpha=.7, label='null  MCV→forcing')
    plt.axvline(te_xy, color='#a63603', lw=2, label=f'obs forcing→MCV (p={p_xy:.3f})')
    plt.axvline(te_yx, color='#08519c', lw=2, label=f'obs MCV→forcing (p={p_yx:.3f})')
    plt.xlabel('transfer entropy (bits)'); plt.ylabel('count')
    plt.legend(frameon=True); plt.tight_layout()

    # 6b. Stacked time-series plot
    fig = plt.figure(figsize=(10,6))
    gs  = GridSpec(3,1, height_ratios=[0.7,0.9,1.1], hspace=0.07)

    axA = fig.add_subplot(gs[0])
    axA.plot(ages, f, lw=0.8, color='k'); axA.set_ylabel('forcing')
    axA.invert_xaxis(); axA.tick_params(axis='x', labelbottom=False)

    axB = fig.add_subplot(gs[1], sharex=axA)
    axB.plot(t_plot, local_te[:N], lw=0.8, color='C0')
    axB.set_ylabel('local TE'); axB.tick_params(axis='x', labelbottom=False)

    axC = fig.add_subplot(gs[2], sharex=axA)
    lnF, = axC.plot(t_plot, p_flip[:N],  color='C1', label='P(flip)')
    axD   = axC.twinx()
    lnS, = axD.plot(t_plot, p_stay[:N],  color='C2', label='P(stay)')
    axC.set_ylabel('P(flip)', color='C1'); axD.set_ylabel('P(stay)', color='C2')
    axC.set_xlabel('Age index'); axC.legend([lnF,lnS], ['P(flip)','P(stay)'])
    plt.show()

    return {'TE_xy': te_xy, 'p_xy': p_xy,
            'TE_yx': te_yx, 'p_yx': p_yx,
            'local_TE': local_te, 'p_flip': p_flip, 'p_stay': p_stay}
















































# import numpy as np
# import pywt
# import matplotlib.pyplot as plt
# from pyinform import transfer_entropy


# def freq_resolved_te(
#     x, y,                       # 1-D arrays, same length, chronological
#     *,
#     wavelet='cmor1.5-1.0',
#     max_level=None,             # None → use all scales up to Nyquist/2
#     k=1,
#     plot=True,
#     cmap='viridis'
# ):
#     """
#     Frequency-(scale-)resolved transfer entropy matrix:
#         rows   = source scale index
#         cols   = target scale index
#     """
#     if len(x) != len(y):
#         raise ValueError("x and y must have equal length")

#     # --- 1. choose scales automatically --------------------------
#     # use dyadic (1,2,3,... ) until "max_level" or N/2
#     N = len(x)
#     max_level = max_level or (N // 2)
#     scales = np.arange(1, max_level + 1)
#     x=x[::-1]
#     y=y[::-1]   

#     # --- 2. CWT (complex) ---------------------------------------
#     coeffs_x, _ = pywt.cwt(x, scales, wavelet)
#     coeffs_y, _ = pywt.cwt(y, scales, wavelet)

#     # --- 3. discretise phase into 8 bins ------------------------
#     phase_bins = np.linspace(-np.pi, np.pi, 9)
#     disc_x = np.digitize(np.angle(coeffs_x), phase_bins) - 1
#     disc_y = np.digitize(np.angle(coeffs_y), phase_bins) - 1

#     n_sc   = len(scales)
#     te_mat = np.empty((n_sc, n_sc))

#     for i in range(n_sc):              # source scale
#         for j in range(n_sc):          # target scale
#             te_mat[i, j] = transfer_entropy(
#                 disc_x[i, :-1], disc_y[j, 1:], k=k
#             )

#     # --- 4. optional plot ---------------------------------------
#     if plot:
#         plt.figure(figsize=(6, 5))
#         plt.imshow(te_mat, origin='lower', cmap=cmap,
#                    extent=[0, n_sc, 0, n_sc], aspect='auto')
#         plt.colorbar(label='TE (bits)')
#         plt.xlabel('target scale index  (low → high freq)')
#         plt.ylabel('source scale index')
#         plt.title('Frequency-resolved Transfer Entropy')
#         plt.tight_layout()
#         plt.show()

#     return te_mat, scales






# import numpy as np
# import pywt
# import matplotlib.pyplot as plt
# from pyinform import transfer_entropy


# def freq_resolved_te(
#     x, y,                       # 1-D arrays (chronological)
#     *,
#     wavelet='cmor1.5-1.0',
#     max_level=None,
#     k=1,
#     plot=True,
#     cmap='viridis'
# ):
#     """
#     Continuous-wavelet scalogram  +  scale×scale TE matrix.

#     Returns
#     -------
#     te_mat  : 2-D ndarray  [src scale , trg scale]
#     scales  : list of scale numbers (PyWavelets convention)
#     """

#     # ------------------------------------------------------------------
#     # 1. choose dyadic scales  (1 … max_level)
#     # ------------------------------------------------------------------
#     N = len(x)
#     max_level = max_level or (N // 4)          # ≤ N/4 keeps runtime sane
#     scales = np.arange(1, max_level + 1)       # 1 = highest frequency

#     # ------------------------------------------------------------------
#     # 2. complex CWT on both series
#     # ------------------------------------------------------------------
#     coeffs_x, _ = pywt.cwt(x, scales, wavelet)
#     coeffs_y, _ = pywt.cwt(y, scales, wavelet)

#     # absolute power for plotting
#     pow_x = np.abs(coeffs_x) ** 2
#     pow_y = np.abs(coeffs_y) ** 2

#     # ------------------------------------------------------------------
#     # 3. phase → 8-bin symbol sequences
#     # ------------------------------------------------------------------
#     phase_bins = np.linspace(-np.pi, np.pi, 9)
#     disc_x = np.digitize(np.angle(coeffs_x), phase_bins) - 1
#     disc_y = np.digitize(np.angle(coeffs_y), phase_bins) - 1

#     # ------------------------------------------------------------------
#     # 4. TE per (src scale , trg scale)
#     # ------------------------------------------------------------------
#     n_sc = len(scales)
#     te_mat = np.empty((n_sc, n_sc))

#     for i in range(n_sc):
#         for j in range(n_sc):
#             te_mat[i, j] = transfer_entropy(
#                 disc_x[i, :-1],     # X_t  at scale i
#                 disc_y[j, 1:],      # Y_{t+1} at scale j
#                 k=k
#             )

#     # ------------------------------------------------------------------
#     # 5. plots: two scalograms + TE matrix
#     # ------------------------------------------------------------------
#     if plot:
#         fig, ax = plt.subplots(1, 3, figsize=(14, 5),
#                                gridspec_kw={'width_ratios':[1,1,1.2]})

#         im0 = ax[0].imshow(pow_x, origin='lower', aspect='auto',
#                            cmap='hot', extent=[0,N,0,n_sc])
#         ax[0].set_title('Source (pre) scalogram')
#         ax[0].set_ylabel('scale index')
#         plt.colorbar(im0, ax=ax[0], fraction=0.046)

#         im1 = ax[1].imshow(pow_y, origin='lower', aspect='auto',
#                            cmap='hot', extent=[0,N,0,n_sc])
#         ax[1].set_title('Target (sq) scalogram')
#         plt.colorbar(im1, ax=ax[1], fraction=0.046)

#         im2 = ax[2].imshow(te_mat, origin='lower', aspect='auto',
#                            cmap=cmap, extent=[0,n_sc,0,n_sc])
#         ax[2].set_title('Scale×Scale TE (bits)')
#         ax[2].set_xlabel('target scale'); ax[2].set_ylabel('source scale')
#         plt.colorbar(im2, ax=ax[2], fraction=0.046)
#         plt.tight_layout(); plt.show()

#     return te_mat, scales




import numpy as np, pywt, matplotlib.pyplot as plt
from pyinform import transfer_entropy
import os

def freq_resolved_te(
    x, y,
    *,
    wavelet='cmor1.5-1.0',
    sampling_period=10,          # yr per sample
    # ---- source (pre) band -----------------------------------
    src_min_period=20_000,       # yr
    src_max_period=20_000,
    n_src_scales=3,              # if min=max, use 3 scales around it
    # ---- target (sq) band ------------------------------------
    trg_min_period=100,          # yr
    trg_max_period=3_000,
    n_trg_scales=64,
    k=1,
    binsp1=9,
    plot=True,
    cmap='viridis',
    source_vname='Precession',   # e.g. 'pre'
    target_vname='CH₄ MCV',      # e.g. 'sq'
    # ---- NEW I/O OPTIONS -------------------------------------
    if_save=False,
    save_file_path='Figures/',
    panel_label='a',
    v_max=None,
    show_colorbar=True
):
    """
    Wavelet-scale × wavelet-scale TE with separate period bands
    for source (x) and target (y).

    Parameters
    ----------
    if_save : bool
        If True, the TE panel is replotted on its own and written to
        a PDF file (vector format) in *save_file_path*.
    save_file_path : str
        Directory in which to place the PDF. Created if it does not exist.

    Returns
    -------
    te_mat     : (n_src × n_trg) ndarray
    periods_x  : ndarray (len n_src)  [yr]
    periods_y  : ndarray (len n_trg)  [yr]
    """
    # --------- 0. housekeeping -------------------------------------------
    x = x[::-1]            # reverse chronological order
    y = y[::-1]

    if len(x) != len(y):
        raise ValueError("x and y must have equal length")

    # ---------- helper: build scale list ---------------------------------
    fc = pywt.central_frequency(wavelet)

    def build_scales(min_p, max_p, n_sc):
        if np.isclose(min_p, max_p):
            ctr = min_p
            periods = ctr * np.geomspace(0, 1, n_sc)
        else:
            periods = np.geomspace(min_p, max_p, n_sc)
        return periods, periods * fc / sampling_period

    periods_x, scales_x = build_scales(src_min_period, src_max_period,
                                       n_src_scales)
    periods_y, scales_y = build_scales(trg_min_period, trg_max_period,
                                       n_trg_scales)

    # ---------- 1. CWT ----------------------------------------------------
    coeffs_x, _ = pywt.cwt(x, scales_x, wavelet,
                           sampling_period=sampling_period)
    coeffs_y, _ = pywt.cwt(y, scales_y, wavelet,
                           sampling_period=sampling_period)

    pow_x = np.abs(coeffs_x) ** 2
    pow_y = np.abs(coeffs_y) ** 2

    # ---------- 2. discretise phase --------------------------------------
    bins = np.linspace(-np.pi, np.pi, binsp1)
    disc_x = np.digitize(np.angle(coeffs_x), bins) - 1
    disc_y = np.digitize(np.angle(coeffs_y), bins) - 1

    # ---------- 3. TE matrix ---------------------------------------------
    te_mat = np.empty((len(scales_x), len(scales_y)))
    for i in range(len(scales_x)):        # src scale
        for j in range(len(scales_y)):    # trg scale
            te_mat[i, j] = transfer_entropy(
                disc_x[i, :-1], disc_y[j, 1:], k=k
            )

    # ---------- 4. plotting ----------------------------------------------
    if plot or if_save:
        # build extents ----------------------------------------------------
        t_ka = np.arange(len(x)) * sampling_period / 1000   # time axis in ka

        extent_src = [t_ka[0], t_ka[-1],
                      periods_x[0]/1000, periods_x[-1]/1000]
        extent_trg = [t_ka[0], t_ka[-1],
                      periods_y[0]/1000, periods_y[-1]/1000]
        extent_te  = [periods_y[0]/1000, periods_y[-1]/1000,
                      periods_x[0]/1000, periods_x[-1]/1000]

        # master figure ----------------------------------------------------
        if plot:
            fig, ax = plt.subplots(1, 3, figsize=(15, 3.5),
                                   gridspec_kw={'width_ratios':[1,1,1.3]})

            im0 = ax[0].imshow(pow_x, origin='upper', aspect='auto',
                               cmap='hot', extent=extent_src)
            ax[0].set_title(f'source scalogram ({source_vname})')
            ax[0].set_xlabel('time (ka BP)')
            ax[0].set_ylabel('period (ka)')
            plt.colorbar(im0, ax=ax[0], fraction=.046)

            im1 = ax[1].imshow(pow_y, origin='upper', aspect='auto',
                               cmap='hot', extent=extent_trg)
            ax[1].set_title(f'target scalogram ({target_vname})')
            ax[1].set_xlabel('time (ka BP)')
            plt.colorbar(im1, ax=ax[1], fraction=.046)

            im2 = ax[2].imshow(te_mat, origin='upper', aspect='auto',
                               cmap=cmap, extent=extent_te,
                               vmin=te_mat.min(), vmax=te_mat.max())
            ax[2].set_title('TE (source phases → target phases)')
            ax[2].set_xlabel(f'{target_vname} period (kyr)')
            ax[2].set_ylabel(f'{source_vname} period (kyr)')
            plt.colorbar(im2, ax=ax[2], fraction=.046)

            plt.tight_layout()
            plt.show()

        # stand-alone TE panel & save -------------------------------------
        if if_save:
            # ensure directory exists
            os.makedirs(save_file_path, exist_ok=True)

            fig_te, ax_te = plt.subplots(figsize=(4.5, 3.5))
            if v_max is not None:
                im = ax_te.imshow(te_mat, origin='upper', aspect='auto',
                                  cmap=cmap, extent=extent_te,
                                  vmin=te_mat.min(), vmax=v_max)
            else:
                im = ax_te.imshow(te_mat, origin='upper', aspect='auto',
                                cmap=cmap, extent=extent_te,
                                vmin=te_mat.min(), vmax=te_mat.max())
                
            # ax_te.set_title('TE (source → target)')
            ax_te.set_xlabel(f'{target_vname} period (kyr)')
            ax_te.set_ylabel(f'{source_vname} period (kyr)')
            if show_colorbar:
                plt.colorbar(im, ax=ax_te, fraction=.046)
            # add 'a' on top left corner (-0.05, 1.05) in axes coords
            ax_te.text(-0.06, 1.06, panel_label, transform=ax_te.transAxes,
                        fontsize=12, fontweight='bold')

            filename = os.path.join(
                save_file_path,
                f"TE_{source_vname}_to_{target_vname}.pdf"
            )
            
            fig_te.savefig(filename, format='pdf', bbox_inches='tight')
            plt.show()
            plt.close(fig_te)   # free memory

    # ---------------------------------------------------------------------
    return te_mat, periods_x, periods_y

























# def freq_resolved_te(
#     x, y,
#     *,
#     wavelet='cmor1.5-1.0',
#     sampling_period=10,          # yr per sample
#     # ---- source (pre) band -----------------------------------
#     src_min_period=20_000,       # yr
#     src_max_period=20_000,
#     n_src_scales=3,              # if min=max, use 3 scales around it
#     # ---- target (sq) band ------------------------------------
#     trg_min_period=100,          # yr
#     trg_max_period=3_000,
#     n_trg_scales=64,
#     k=1,
#     binsp1=9,
#     plot=True,
#     cmap='viridis',
#     source_vname='Precession',       # e.g. 'pre'
#     target_vname='CH₄ MCV'        # e.g. 'sq'
# ):
#     """
#     Wavelet-scale × wavelet-scale TE with *separate* period bands
#     for source (x) and target (y).

#     Returns
#     -------
#     te_mat     : (n_src × n_trg) array
#     periods_x  : list (len n_src)  [yr]
#     periods_y  : list (len n_trg)  [yr]
#     """
#     x= x[::-1]  # reverse chronological order
#     y= y[::-1]  # reverse chronological order

#     if len(x) != len(y):
#         raise ValueError("x and y must have equal length")

#     # ---------- helper: build scale list -----------------------
#     fc = pywt.central_frequency(wavelet)

#     def build_scales(min_p, max_p, n_sc):
#         if np.isclose(min_p, max_p):
#             ctr = min_p
#             # periods = ctr * np.geomspace(0.8, 1.2, n_sc)
#             periods = ctr * np.geomspace(0, 1, n_sc)
#         else:
#             periods = np.geomspace(min_p, max_p, n_sc)
#         return periods, periods * fc / sampling_period

#     periods_x, scales_x = build_scales(src_min_period, src_max_period,
#                                        n_src_scales)
#     periods_y, scales_y = build_scales(trg_min_period, trg_max_period,
#                                        n_trg_scales)
#     print(scales_y)

#     # ---------- 1. CWT -----------------------------------------
#     coeffs_x, _ = pywt.cwt(x, scales_x, wavelet,
#                            sampling_period=sampling_period)
#     coeffs_y, _ = pywt.cwt(y, scales_y, wavelet,
#                            sampling_period=sampling_period)

#     pow_x = np.abs(coeffs_x) ** 2
#     pow_y = np.abs(coeffs_y) ** 2

#     # Immediately after pow_x is computed:
#     row_max = pow_x.mean(axis=1).argmax()
#     print("max-power row = %d   →  period ≈ %.1f ka"
#         % (row_max, periods_x[row_max]/1000))

#     # ---------- 2. discretise phase ----------------------------
#     bins = np.linspace(-np.pi, np.pi, binsp1)
#     disc_x = np.digitize(np.angle(coeffs_x), bins) - 1
#     disc_y = np.digitize(np.angle(coeffs_y), bins) - 1

#     # ---------- 3. TE matrix -----------------------------------
#     te_mat = np.empty((len(scales_x), len(scales_y)))
#     for i in range(len(scales_x)):        # src scale
#         for j in range(len(scales_y)):    # trg scale
#             te_mat[i, j] = transfer_entropy(
#                 disc_x[i, :-1], disc_y[j, 1:], k=k
#             )

#     # ---------- 4. plots ---------------------------------------
#     if plot:
#         # --- build extents -------------------------------------------------
#         t_ka = np.arange(len(x)) * sampling_period / 1000   # time axis in ka

#         extent_src = [t_ka[0], t_ka[-1],
#                     periods_x[0]/1000, periods_x[-1]/1000]   # low → high period
#         extent_trg = [t_ka[0], t_ka[-1],
#                     periods_y[0]/1000, periods_y[-1]/1000]

#         extent_te  = [periods_y[0]/1000, periods_y[-1]/1000,   # x-axis  (target)
#                     periods_x[0]/1000, periods_x[-1]/1000]   # y-axis  (source)

#         # --- plots ---------------------------------------------------------
#         fig, ax = plt.subplots(1, 3, figsize=(15, 3.5),
#                             gridspec_kw={'width_ratios':[1,1,1.3]})

#         im0 = ax[0].imshow(pow_x, origin='upper', aspect='auto',
#                         cmap='hot', extent=extent_src)
#         ax[0].set_title('source scalogram (pre)')
#         ax[0].set_xlabel('time (ka BP)')
#         ax[0].set_ylabel('period (ka)')
#         plt.colorbar(im0, ax=ax[0], fraction=.046)

#         im1 = ax[1].imshow(pow_y, origin='upper', aspect='auto',
#                         cmap='hot', extent=extent_trg)
#         ax[1].set_title('target scalogram (sq)')
#         ax[1].set_xlabel('time (ka BP)')
#         plt.colorbar(im1, ax=ax[1], fraction=.046)

#         # np.quantile(te_mat,0.5)
#         im2 = ax[2].imshow(te_mat, origin='upper', aspect='auto',
#                         cmap=cmap, extent=extent_te, vmin=te_mat.min() , vmax=te_mat.max())
#         ax[2].set_title('TE  (source phases → target phases)')
#         ax[2].set_xlabel(f'{target_vname} period (kyr)')
#         ax[2].set_ylabel(f'{source_vname} period (kyr)')
#         plt.colorbar(im2, ax=ax[2], fraction=.046)

#         plt.tight_layout(); plt.show()


#     return te_mat, periods_x, periods_y










import numpy as np, pywt, matplotlib.pyplot as plt
from pyinform import transfer_entropy

# ----------------------------------------------------------------------
# 1. Decide on one “global” target-period grid (log-spaced is convenient)
#    Do this once – outside the function – and reuse it.
TRG_PERIODS_GLOBAL = np.geomspace(100,     # 0.1 kyr
                                  3_000,   #   3 kyr
                                  64)      # same as old n_trg_scales
# ----------------------------------------------------------------------

def freq_resolved_te_fixed_grid(
    x, y, *,
    wavelet='cmor1.5-1.0',
    sampling_period=10,          # yr/sample
    # ---- source (pre) band -----------------------------------
    src_min_period=20_000,
    src_max_period=20_000,
    n_src_scales=3,
    # ---- target band *limits* for THIS call ------------------
    trg_min_period=100,          # yr
    trg_max_period=3_000,        # yr
    k=1,
    plot=True,
    cmap='viridis',
    source_vname='Precession',
    target_vname='CH₄ MCV',
    periods_y_global=TRG_PERIODS_GLOBAL,   # <<<< NEW
):
    """Transfer-entropy map with an *immutable* target-period grid."""
    # ----------------------------------------------------------
    x = x[::-1];  y = y[::-1]            # reverse chronological order
    if len(x) != len(y):
        raise ValueError("x and y must have equal length")

    fc = pywt.central_frequency(wavelet)

    # ---------- source scales (unchanged) ---------------------
    if np.isclose(src_min_period, src_max_period):
        periods_x = src_min_period * np.geomspace(0.8, 1.2, n_src_scales)
    else:
        periods_x = np.geomspace(src_min_period, src_max_period,
                                 n_src_scales)
    scales_x = periods_x * fc / sampling_period

    # ---------- target scales (FIXED GRID) --------------------
    periods_y = np.asarray(periods_y_global)
    scales_y  = periods_y * fc / sampling_period

    # keep indices of columns inside the requested band
    col_mask = (periods_y >= trg_min_period) & (periods_y <= trg_max_period)

    # ---------- 1. CWT ----------------------------------------
    coeffs_x, _ = pywt.cwt(x, scales_x, wavelet,
                           sampling_period=sampling_period)
    coeffs_y, _ = pywt.cwt(y, scales_y, wavelet,
                           sampling_period=sampling_period)

    pow_x = np.abs(coeffs_x)**2
    pow_y = np.abs(coeffs_y)**2

    # ---------- 2. phase discretisation -----------------------
    bins = np.linspace(-np.pi, np.pi, 9)
    disc_x = np.digitize(np.angle(coeffs_x), bins) - 1
    disc_y = np.digitize(np.angle(coeffs_y), bins) - 1

    # ---------- 3. TE matrix ----------------------------------
    te_mat = np.empty((len(scales_x), len(scales_y)))
    for i in range(len(scales_x)):          # source scale
        for j in range(len(scales_y)):      # target scale
            te_mat[i, j] = transfer_entropy(
                disc_x[i, :-1], disc_y[j, 1:], k=k
            )

    # mask out-of-band columns so they plot in the right place but
    # do not influence the colour scale
    te_mat_masked = np.where(col_mask, te_mat, np.nan)

    # ---------- 4. plotting -----------------------------------
    if plot:
        t_ka = np.arange(len(x))*sampling_period/1000  # kyr axis
        extent_src = [t_ka[0], t_ka[-1],
                      periods_x[0]/1000, periods_x[-1]/1000]
        extent_trg = [t_ka[0], t_ka[-1],
                      periods_y[0]/1000, periods_y[-1]/1000]
        extent_te  = [periods_y[0]/1000, periods_y[-1]/1000,
                      periods_x[0]/1000, periods_x[-1]/1000]

        fig, ax = plt.subplots(1, 3, figsize=(15, 3.5),
                               gridspec_kw={'width_ratios':[1,1,1.3]})

        im0 = ax[0].imshow(pow_x, origin='upper', aspect='auto',
                           cmap='hot', extent=extent_src)
        ax[0].set_title(f'source scalogram ({source_vname})')
        ax[0].set_xlabel('time (ka BP)');  ax[0].set_ylabel('period (ka)')
        plt.colorbar(im0, ax=ax[0], fraction=.046)

        im1 = ax[1].imshow(pow_y, origin='upper', aspect='auto',
                           cmap='hot', extent=extent_trg)
        ax[1].set_title(f'target scalogram ({target_vname})')
        ax[1].set_xlabel('time (ka BP)')
        plt.colorbar(im1, ax=ax[1], fraction=.046)

        im2 = ax[2].imshow(te_mat_masked, origin='upper', aspect='auto',
                           cmap=cmap, extent=extent_te,
                           vmin=np.nanquantile(te_mat_masked, 0.5),
                           vmax=np.nanmax(te_mat_masked))
        ax[2].set_title('TE  (source phases → target phases)')
        ax[2].set_xlabel(f'{target_vname} period (kyr)')
        ax[2].set_ylabel(f'{source_vname} period (kyr)')
        ax[2].set_xscale('log')   # <<<< log axis removes misleading shifts
        plt.colorbar(im2, ax=ax[2], fraction=.046)

        plt.tight_layout(); plt.show()

    return te_mat_masked, periods_x, periods_y, col_mask






































# def te_matrix_zscore(x, y, n_perm=100, **kwargs):
#     te_obs, px, py = freq_resolved_te(x, y, plot=False, **kwargs)
#     null_stack = np.empty((n_perm, *te_obs.shape))
#     rng = np.random.default_rng(0)

#     for k in range(n_perm):
#         x_perm = rng.permutation(x)
#         null_stack[k], _, _ = freq_resolved_te(x_perm, y, plot=False, **kwargs)

#     mu  = null_stack.mean(axis=0)
#     sig = null_stack.std (axis=0)
#     z   = (te_obs - mu) / sig
#     vmax = np.nanpercentile(abs(z), 99)

#     # plot
#     plt.imshow(z, origin='lower', cmap='bwr', vmin=-vmax, vmax=vmax,
#                extent=[py[0]/1e3, py[-1]/1e3, px[0]/1e3, px[-1]/1e3])
#     plt.colorbar(label='z-score')
#     plt.xlabel('target period (ka)'); plt.ylabel('source period (ka)')
#     plt.title('Scale×scale TE  (z-score vs permutations)')
#     plt.tight_layout(); plt.show()
#     return z


# from joblib import Parallel, delayed
# import numpy as np, matplotlib.pyplot as plt

# def _perm_te(x, y, rng, kwargs):
#     """worker: compute TE matrix for one permutation of x"""
#     x_perm = rng.permutation(x)
#     te_mat, _, _ = freq_resolved_te(x_perm, y, plot=False, **kwargs)
#     return te_mat

# def te_matrix_zscore(
#     x, y,
#     n_perm      = 100,
#     n_jobs      = -1,        # -1 → use all available cores
#     random_seed = 0,
#     **kwargs                  # passed straight to freq_resolved_te
# ):
#     """
#     Permutation test for the scale×scale TE map.
#     Returns z-score matrix; also draws it.
#     """
#     x = x[::-1]  # reverse chronological order
#     y = y[::-1]  # reverse chronological order
#     # 1) observed matrix
#     te_obs, p_src, p_trg = freq_resolved_te(x, y, plot=False, **kwargs)

#     # 2) parallel surrogates
#     rng_master = np.random.default_rng(random_seed)
#     seeds = rng_master.integers(0, 2**32-1, size=n_perm)

#     null_list = Parallel(n_jobs=n_jobs, prefer='processes')(
#         delayed(_perm_te)(x, y,
#                           np.random.default_rng(s),  # private RNG per worker
#                           kwargs)
#         for s in seeds
#     )
#     null_stack = np.stack(null_list, axis=0)

#     # 3) z-score
#     mu  = null_stack.mean(axis=0)
#     sig = null_stack.std (axis=0)
#     z   = (te_obs - mu) / sig

#     # 4) plot
#     vmax = np.nanpercentile(np.abs(z), 99)
#     plt.imshow(z, origin='lower', cmap='bwr',
#                vmin=-vmax, vmax=vmax,
#                extent=[p_trg[0]/1e3, p_trg[-1]/1e3,
#                        p_src[0]/1e3, p_src[-1]/1e3])
#     plt.colorbar(label='z-score')
#     plt.xlabel('target period (ka)')
#     plt.ylabel('source period (ka)')
#     plt.title(f'Scale×scale TE  (z, {n_perm} perms, two-tailed)')
#     plt.tight_layout(); plt.show()

#     return z





















# from dit import Distribution
# from dit.other import pid
# from itertools import product


# def pid_unique_redundant(
#     x, y, z,          # 1-D int arrays (|alphabet| ≤ 16 recommended)
#     k=1,              # use x_t , y_t to predict z_{t+1}
#     pid_method='wb2018'
# ):
#     """
#     PID of {X,Y} → Z_{t+1}.  Returns dict with
#         'unique_x' , 'unique_y' , 'redundant' , 'synergy'
#     in bits.
#     """
#     # build joint symbols for a first-order Markov set-up
#     xs, ys, zs = x[:-1], y[:-1], z[1:]

#     # alphabet tuples (x,y,z)
#     joint = list(zip(xs, ys, zs))
#     outcomes, counts = np.unique(joint, return_counts=True, axis=0)
#     probs = counts / counts.sum()
#     outcomes = [tuple(o) for o in outcomes]

#     d = Distribution(outcomes, probs)
#     d.set_rv_names(('X','Y','Z'))

#     P = pid.pid(d, pid_method, ((0,), (1,)), (2,))
#     return {k: P[k] for k in ('UIX', 'UIY', 'RXY', 'SIXY')}
















































# reload module in case of updates
def mc_TE_heatmap_inter(
    df_sq,
    target_column='sq',
    intervals=None,
    niter=100,
    n_surr=100,
    alpha=0.05,
    k=1,
    gbins=None,
    p_thresh=0.9,
    n_jobs=-1,
    forcing_var='pre',  # 'pre' or 'obl'
    if_plot=False,
    plot=True
):
    """
    Perform Monte Carlo tests of transfer entropy between 'sq' and a chosen forcing variable
    over a range of interpolation intervals and multiple forcing_bins, then plot a heatmap.

    Parameters
    ----------
    df_sq : pandas.DataFrame
        Input data frame containing 'sq' and forcing variables.
    intervals : list of int, optional
        Interpolation intervals to test (default=10,20,...,100).
    niter : int, optional
        Number of Monte Carlo repetitions (default=100).
    n_surr : int, optional
        Number of surrogates per TE test (default=100).
    alpha : float, optional
        Significance level for surrogate test (default=0.05).
    k : int, optional
        History length for transfer entropy (default=1).
    gbins : list of int, optional
        Forcing bins to test (default=range(2,11)).
    p_thresh : float, optional
        Threshold for highlighting cells in heatmap (default=0.9).
    n_jobs : int, optional
        Number of parallel jobs (default=-1 for all cores).
    forcing_var : str, optional
        Which forcing variable to test: 'pre' or 'obl' (default='pre').
    if_plot : bool, optional
        Whether to plot intermediate TE results (default=False).
    plot : bool, optional
        Whether to display the heatmap (default=True).

    Returns
    -------
    fractions : numpy.ndarray
        Matrix of fraction significant values (len(intervals) x len(gbins)).
    fig, ax : matplotlib objects (if plot=True)
        Heatmap Figure and Axes.
    """
    # defaults
    if intervals is None:
        intervals = list(range(10, 101, 10))
    if gbins is None:
        gbins = list(range(2, 11))
    if forcing_var not in ('pre', 'obl'):
        raise ValueError("forcing_var must be 'pre' or 'obl'.")

    # reload toolbox
    importlib.reload(sa)

    # 1) Prepare data for each interval
    data_list = []  # list of (forcing, sq) pairs
    for interval in intervals:
        df_sq_i, df_pre_i, df_obl_i = sa.interpolate_data_forcing(
            df_sq.copy(), interval=interval, if_plot=if_plot
        )
        if forcing_var == 'pre':
            forcing = df_pre_i['pre'].values
        else:
            forcing = df_obl_i['obl'].values
        sq_vals = df_sq_i[target_column].values
        data_list.append((forcing, sq_vals))

    # 2) Worker for one MC iteration
    def one_mc_iter(data_list, gbins, n_surr, alpha, k):
        counts = np.zeros((len(data_list), len(gbins)), dtype=int)
        for i, (forcing, sq_vals) in enumerate(data_list):
            for j, bins in enumerate(gbins):
                sig, _ = sa.transfer_entropy_surrogate_test(
                    forcing, sq_vals,
                    k=k,
                    forcing_bins=bins,
                    n_surr=n_surr,
                    p=alpha,
                    if_plot=False
                )
                counts[i, j] = int(sig)
        return counts

    # 3) Parallel Monte Carlo
    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(one_mc_iter)(data_list, gbins, n_surr, alpha, k)
        for _ in range(niter)
    )
    total = np.stack(results).sum(axis=0)
    fractions = total / niter

    # 4) Plot
    if plot:
        fig, ax = plt.subplots(figsize=(8, 4))
        im = ax.imshow(fractions, origin='lower', aspect='auto', cmap='viridis', vmin=0, vmax=1)
        ax.set_xticks(np.arange(len(gbins)))
        ax.set_xticklabels(gbins)
        ax.set_xlabel('forcing_bins')
        ax.set_yticks(np.arange(len(intervals)))
        ax.set_yticklabels(intervals)
        ax.set_ylabel('interpolation interval')
        ax.set_title(f'TE(signif) for sq vs {forcing_var} across intervals')
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Fraction significant')
        # highlight
        for i, j in zip(*np.where(fractions > p_thresh)):
            rect = plt.Rectangle((j-0.5, i-0.5),1,1, fill=False, edgecolor='white', linewidth=2)
            ax.add_patch(rect)
        # annotate
        for i in range(len(intervals)):
            for j in range(len(gbins)):
                pct = fractions[i, j] * 100
                color = 'white' if fractions[i, j] > 0.5 else 'black'
                ax.text(j, i, f"{pct:.0f}%", ha='center', va='center', color=color, fontsize=8)
        plt.tight_layout()
        plt.show()
        return fractions, fig, ax

    return fractions








import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from  scipy.signal import butter, filtfilt


def bandpass_ch4_d18o(
    df_ch4_interp,                 # cols: age , ch4   (common grid)
    df_d18o_interp,                # cols: age , d18O (same grid)
    *,
    low_period = 300,              # [yr]  remove > low_period  (low-cut)
    high_period = 3_000,           # [yr]  remove < high_period (high-cut)
    butter_order = 4,
    flip_sign   = False,           # flip δ18O so “warm = high”
    plot        = True
):
    """
    Butterworth **band-pass** the pre-interpolated CH4 and δ18O series.

    Pass-band = [ 1/high_period  …  1/low_period ]   (Hz)

    Returns
    -------
    df_filt_ch4   : DataFrame['age','filt_ch4']
    df_filt_d18O  : DataFrame['age','filt_d18O']
    """
    # ---------- 1. ensure same age axis ----------------
    ages = df_ch4_interp['age'].values
    if not np.allclose(ages, df_d18o_interp['age'].values):
        raise ValueError("df_ch4_interp and df_d18o_interp must share the same age vector")

    ch4_vals  = df_ch4_interp['ch4'].values
    d18o_vals = df_d18o_interp['d18O'].values
    if flip_sign:
        d18o_vals = -d18o_vals

    # ---------- 2. design Butterworth band-pass --------
    dt = np.median(np.diff(ages))        # yr
    fs = 1.0 / dt                        # samples per yr
    # convert periods [yr] → frequencies [Hz]
    f_low  = 1.0 / high_period           # lower edge (remove slower than this)
    f_high = 1.0 / low_period            # upper edge (remove faster than this)
    Wn = np.array([f_low, f_high]) / (fs * 0.5)   # normalised to Nyquist

    if np.any(Wn <= 0) or np.any(Wn >= 1) or Wn[0] >= Wn[1]:
        raise ValueError("Band edges out of range; check low_period / high_period.")

    b, a = butter(butter_order, Wn, btype='bandpass')
    filt_ch4  = filtfilt(b, a, ch4_vals)
    filt_d18o = filtfilt(b, a, d18o_vals)

    # ---------- 3. wrap to DataFrames ------------------
    df_filt_ch4  = pd.DataFrame({'age': ages, 'filt_ch4' : filt_ch4})
    df_filt_d18O = pd.DataFrame({'age': ages, 'filt_d18O': filt_d18o})

    # ---------- 4. optional quick plots ----------------
    if plot:
        band_lab = f'{low_period//1_000:.1f}–{high_period//1_000:.1f} ka'
        for name, orig, filt in [
            ('CH4',  ch4_vals,  filt_ch4),
            ('δ18O', d18o_vals, filt_d18o)
        ]:
            plt.figure(figsize=(10,3))
            plt.plot(ages, orig, alpha=0.35, label=f'{name} (raw)')
            plt.plot(ages, filt, lw=1.6,    label=f'{name} ({band_lab} band-pass)')
            plt.gca().invert_xaxis()
            plt.xlabel('Age (yr BP)'); plt.ylabel(name)
            plt.legend(); plt.tight_layout(); plt.show()

    return df_filt_ch4, df_filt_d18O






















import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

def highpass_ch4_d18O(
    df_ch4_interp,                 # ← already on common grid; cols: age , ch4
    df_d18O_interp,                # ← same grid;        cols: age , d18O
    *,
    cutoff_period=10_000,          # remove variability slower than this [yr]
    butter_order=4,
    flip_sign=False,                # flip δ18O so “warm = high”
    plot=True
):
    """
    High–pass the pre-interpolated CH4 and δ18O series (> cutoff_period).

    Returns
    -------
    df_filt_ch4   : DataFrame[age , filt_ch4]
    df_filt_d18O  : DataFrame[age , filt_d18O]
    """

    # -------- 1. sanity: same age grid -----------------
    ages = df_ch4_interp['age'].values
    if not np.allclose(ages, df_d18O_interp['age'].values):
        raise ValueError('df_ch4_interp and df_d18o_interp must share the same age vector')

    ch4_vals  = df_ch4_interp['ch4'].values
    d18o_vals = df_d18O_interp['d18O'].values
    if flip_sign:
        d18o_vals = -d18o_vals

    # -------- 2. Butterworth high-pass -----------------
    # dt [yr] from median spacing
    dt = np.median(np.diff(ages))
    fs = 1.0 / dt                       # samples per year
    fc = 1.0 / cutoff_period            # Hz
    Wn = fc / (fs * 0.5)                # normalised to Nyquist

    b, a = butter(butter_order, Wn, btype='highpass')
    filt_ch4  = filtfilt(b, a, ch4_vals)
    filt_d18o = filtfilt(b, a, d18o_vals)

    # -------- 3. wrap to DataFrames --------------------
    df_filt_ch4  = pd.DataFrame({'age': ages, 'filt_ch4' : filt_ch4})
    df_filt_d18O = pd.DataFrame({'age': ages, 'filt_d18O': filt_d18o})

    # -------- 4. quick visual check -------------------
    if plot:
        for name, orig, filt in [
            ('CH4',  ch4_vals,  filt_ch4),
            ('δ18O', d18o_vals, filt_d18o)
        ]:
            plt.figure(figsize=(10,3))
            plt.plot(ages, orig,  alpha=0.4, label=f'{name} (raw)')
            plt.plot(ages, filt,  lw=1.8,   label=f'{name} (> 1/{cutoff_period/1_000:.0f} ka)')
            plt.gca().invert_xaxis()
            plt.xlabel('Age (yr BP)'); plt.ylabel(name)
            plt.legend(); plt.tight_layout(); plt.show()

    return df_filt_ch4, df_filt_d18O


















def lowpass_ch4_d18O(
    df_ch4_interp,                 # ← already on common grid; cols: age, ch4
    df_d18O_interp,                # ← same grid;        cols: age, d18O
    *,
    cutoff_period=10_000,          # retain variability slower than this [yr]
    butter_order=4,
    flip_sign=False,               # flip δ18O so “warm = high”
    plot=True
):
    """
    Low–pass the pre-interpolated CH4 and δ18O series (< cutoff_period).

    Parameters
    ----------
    df_ch4_interp : pd.DataFrame
        Columns ['age', 'ch4'] on a uniform age grid.
    df_d18O_interp : pd.DataFrame
        Columns ['age', 'd18O'] on the same grid as df_ch4_interp.
    cutoff_period : float
        Cutoff period in years; only variability slower than this is retained.
    butter_order : int
        Order of the Butterworth filter.
    flip_sign : bool
        If True, invert the sign of the d18O series before filtering.
    plot : bool
        If True, display before/after comparison plots.

    Returns
    -------
    df_lp_ch4 : pd.DataFrame
        Columns ['age', 'filt_ch4'] containing the low-passed CH4.
    df_lp_d18O : pd.DataFrame
        Columns ['age', 'filt_d18O'] containing the low-passed δ18O.
    """

    # 1. sanity check: same age grid
    ages = df_ch4_interp["age"].values
    if not np.allclose(ages, df_d18O_interp["age"].values):
        raise ValueError("df_ch4_interp and df_d18O_interp must share the same age vector")

    ch4_vals  = df_ch4_interp["ch4"].values
    d18o_vals = df_d18O_interp["d18O"].values
    if flip_sign:
        d18o_vals = -d18o_vals

    # 2. Butterworth low-pass
    dt = np.median(np.diff(ages))     # sampling interval [yr]
    fs = 1.0 / dt                      # sampling frequency [1/yr]
    fc = 1.0 / cutoff_period           # cutoff frequency [1/yr]
    Wn = fc / (0.5 * fs)               # normalised cutoff (Nyquist=0.5*fs)

    b, a = butter(butter_order, Wn, btype="lowpass")
    filt_ch4  = filtfilt(b, a, ch4_vals)
    filt_d18o = filtfilt(b, a, d18o_vals)

    # 3. wrap into DataFrames
    df_lp_ch4   = pd.DataFrame({"age": ages, "filt_ch4": filt_ch4})
    df_lp_d18O  = pd.DataFrame({"age": ages, "filt_d18O": filt_d18o})

    # 4. optional quick plot
    if plot:
        for name, orig, filt in [
            ("CH4",  ch4_vals,  filt_ch4),
            ("δ18O", d18o_vals, filt_d18o)
        ]:
            plt.figure(figsize=(10, 3))
            plt.plot(ages, orig, alpha=0.4, label=f"{name} (raw)")
            plt.plot(ages, filt, lw=1.8,
                     label=f"{name} (> 1/{cutoff_period/1_000:.0f} ka)")
            plt.gca().invert_xaxis()
            plt.xlabel("Age (yr BP)")
            plt.ylabel(name)
            plt.legend()
            plt.tight_layout()
            plt.show()

    return df_lp_ch4, df_lp_d18O





































# reload module in case of updates
def mc_TE_heatmap(
    pre,
    sq,
    niter=100,
    n_surr=100,
    alpha=0.05,
    ks=None,
    gbins=None,
    sq_method='hist',  # 'hist' or 'phase'
    p_thresh=0.9,
    n_jobs=-1,
    if_plot=False,
    plot=True
):

    # default parameter lists
    if ks is None:
        ks = [1, 2, 3, 4, 5, 6]
    if gbins is None:
        gbins = list(range(2, 11))


    # 2) Worker for one MC iteration
    def one_mc_iter(pre, sq, ks, gbins, n_surr, alpha):
        local_counts = np.zeros((len(ks), len(gbins)), dtype=int)
        for i, k in enumerate(ks):
            for j, b in enumerate(gbins):
                sig, fig,te = transfer_entropy_surrogate_test(
                    pre, sq,
                    k=k,
                    forcing_bins=b,
                    n_surr=n_surr,
                    p=alpha,
                    if_plot=False,
                    sq_method=sq_method 
                )
                local_counts[i, j] = int(sig)
        return local_counts

    # 3) Parallel Monte Carlo
    results = Parallel(n_jobs=n_jobs, backend="loky")(  
        delayed(one_mc_iter)(pre, sq, ks, gbins, n_surr, alpha)
        for _ in range(niter)
    )
    total_counts = np.stack(results, axis=0).sum(axis=0)
    fractions = total_counts / niter

    # 4) Plot heatmap
    if plot:
        fig, ax = plt.subplots(figsize=(8, 4))
        im = ax.imshow(
            fractions,
            origin='lower',
            aspect='auto',
            cmap='viridis',
            vmin=0, vmax=1
        )
        ax.set_xticks(np.arange(len(gbins)))
        ax.set_xticklabels(gbins)
        ax.set_xlabel('forcing_bins')
        ax.set_yticks(np.arange(len(ks)))
        ax.set_yticklabels(ks)
        ax.set_ylabel('history length k')
        ax.set_title(f'Fraction significant over {niter} Monte Carlo runs')
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Fraction significant')
        # white boxes where above threshold
        for i, j in zip(*np.where(fractions > p_thresh)):
            rect = plt.Rectangle(
                (j - 0.5, i - 0.5), 1, 1,
                fill=False, edgecolor='white', linewidth=2
            )
            ax.add_patch(rect)
        # annotate
        for i in range(len(ks)):
            for j in range(len(gbins)):
                pct = fractions[i, j] * 100
                color = 'white' if fractions[i, j] > 0.5 else 'black'
                ax.text(j, i, f"{pct:.0f}%", ha='center', va='center',
                        color=color, fontsize=8)
        plt.tight_layout()
        plt.show()
        return fractions, fig, ax

    return fractions




import numpy as np
import matplotlib.pyplot as plt

def moving_TE(
    df_pre,
    df_sq,
    forcing_column='pre',
    target_column='sq',
    time_column='age',
    nbins_pre=4,
    nbins_target=2,
    window_length=500,
):
    # --- 1) Extract raw numpy arrays and reverse if needed
    pre_raw = df_pre[forcing_column].values[::-1]
    sq_raw  = df_sq[target_column].values[::-1]
    t        = df_pre[time_column].values[::-1]
    N = len(pre_raw)

    # --- 1.1) Find “low-pre” periods (below median here)
    q25   = np.quantile(pre_raw, 0.5)
    low   = pre_raw < q25
    edges = np.diff(low.astype(int))
    starts = list(np.where(edges == 1)[0] + 1)
    ends   = list(np.where(edges == -1)[0] + 1)
    if low[0]:   starts.insert(0, 0)
    if low[-1]:  ends.append(N)
    low_periods = list(zip(starts, ends))

    # --- 2) Discretize once, globally
    bins_pre = np.histogram_bin_edges(pre_raw, bins=nbins_pre)
    pre_disc = np.digitize(pre_raw, bins_pre) - 1
    sq_disc  = (sq_raw > 0).astype(int)

    # --- 3) Compute local TE in a moving window
    local_te = np.full(N, np.nan)
    for i in range(1, N):
        start = max(0, i - window_length + 1)
        # counts for this window
        c_xyz = {}
        c_xy  = {}
        c_zy  = {}
        c_z   = {}

        # collect counts from j=start+1 .. i
        for j in range(start+1, i+1):
            x_t   = sq_disc[j]
            x_tm1 = sq_disc[j-1]
            y_tm1 = pre_disc[j-1]

            c_xyz[(x_t, x_tm1, y_tm1)] = c_xyz.get((x_t, x_tm1, y_tm1), 0) + 1
            c_xy[(x_t, x_tm1)]         = c_xy.get((x_t, x_tm1), 0) + 1
            c_zy[(x_tm1, y_tm1)]       = c_zy.get((x_tm1, y_tm1), 0) + 1
            c_z[x_tm1]                 = c_z.get(x_tm1, 0) + 1

        x_t   = sq_disc[i]
        x_tm1 = sq_disc[i-1]
        y_tm1 = pre_disc[i-1]

        # only compute if we've seen that history in this window
        if c_zy.get((x_tm1, y_tm1), 0) > 0 and c_z.get(x_tm1, 0) > 0:
            num = c_xyz[(x_t, x_tm1, y_tm1)] / c_zy[(x_tm1, y_tm1)]
            den = c_xy[(x_t, x_tm1)]         / c_z[x_tm1]
            local_te[i] = np.log2(num / den)

    # --- 4) Plot setup with two panels
    fig = plt.figure(figsize=(12, 4))
    gs  = fig.add_gridspec(2, 1, height_ratios=[0.8, 1.6], hspace=0)

    # Top panel: pre & sq
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(t, pre_raw, label=f'{forcing_column} (raw)')
    ax0.set_ylabel(forcing_column)
    ax0.tick_params(axis='x', which='both', labelbottom=False)
    ax0.set_xlim(t[1], t[-1])

    ax0b = ax0.twinx()
    ax0b.plot(t, sq_raw, color='C1', linestyle='-', marker='', label=f'{target_column} (raw)')
    ax0b.set_ylabel(target_column)

    # Shade low-pre
    for start, end in low_periods:
        ax0.axvspan(t[start], t[end-1], color='gray', alpha=0.3)
        ax0b.axvspan(t[start], t[end-1], color='gray', alpha=0.3)

    # Draw bin boundaries as horizontal lines
    for y in bins_pre:
        ax0.axhline(y=y, color='k', linestyle='--', alpha=0.5)

    # combine legends
    # lns = ax0.get_lines() + ax0b.get_lines()
    # labs = [l.get_label() for l in lns]
    # ax0.legend(lns, labs, loc='upper right')

    all_lines = ax0.get_lines() + ax0b.get_lines()
    good_lines = [line for line in all_lines if not line.get_label().startswith('_')]
    good_labels = [line.get_label() for line in good_lines]
    ax0.legend(good_lines, good_labels, loc='upper right')

    # Bottom panel: local TE
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax1.plot(t, np.concatenate([[np.nan], local_te[1:]]), color='C2', label='Local TE')
    ax1.set_xlabel(time_column)
    ax1.set_ylabel('Local TE (bits)')
    ax1.legend(loc='upper right')

    # Shade low-pre on bottom panel
    for start, end in low_periods:
        ax1.axvspan(t[start], t[end-1], color='gray', alpha=0.3)

    plt.show()

    return local_te











import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from matplotlib import cm, colors


def count_sq_pre_contexts_3d(
    sq, pre, nbins_pre: int = 6, cmap_name: str = "jet"
):
    """
    Count the occurrences of each (sq_t, pre_t-bin) pair and visualise
    them in a colour-coded 3-D bar chart.

    The colour of every bar is taken from a warm–cold colormap so that
    taller bars appear brighter/warmer; bars of equal height share the
    same colour.

    Parameters
    ----------
    sq, pre : 1-D array-like (same length)
        Time-series for the square wave and the forcing (precession, etc.).
    nbins_pre : int, default 6
        Number of equal-width bins for the forcing amplitude.
    cmap_name : str, default "coolwarm"
        Any Matplotlib colormap name.

    Returns
    -------
    counts : ndarray shape (2, nbins_pre)
    """
    # --- 0. prep & sanity check -----------------------------------------
    sq = np.asarray(sq).ravel()
    pre = np.asarray(pre).ravel()
    if sq.shape != pre.shape:
        raise ValueError("sq and pre must have the same length")

    # --- 1. discretise ---------------------------------------------------
    sq_disc = (sq > 0).astype(int)                     # 0 / 1
    bins_pre = np.histogram_bin_edges(pre, bins=nbins_pre)
    pre_disc = np.digitize(pre, bins_pre) - 1          # 0 … nbins_pre
    pre_disc = np.clip(pre_disc, 0, nbins_pre - 1)

    # --- 2. tally counts[i, j] ------------------------------------------
    counts = np.zeros((2, nbins_pre), dtype=int)
    for i, j in zip(sq_disc, pre_disc):
        counts[i, j] += 1

    # --- 3. pretty-print table ------------------------------------------
    df = pd.DataFrame(
        counts,
        index=[r"sq=0", r"sq=1"],
        columns=[f"pre={k}" for k in range(nbins_pre)],
    )
    print(df.to_string())

    # --- 4. 3-D bar plot -------------------------------------------------
    fig = plt.figure(figsize=(7.2, 4.8))
    ax = fig.add_subplot(111, projection="3d")

    # bar positions & sizes
    _xs, _ys = np.meshgrid(np.arange(nbins_pre), [0, 1])
    xs = _xs.ravel()
    ys = _ys.ravel()
    zs = np.zeros_like(xs)
    dx = dy = 0.75
    dz = counts.T.ravel().astype(float)

    # colour mapping (same colour ↔ same height)
    cmap = cm.get_cmap(cmap_name)
    norm = colors.Normalize(vmin=dz.min(), vmax=dz.max())
    bar_colors = cmap(norm(dz))

    ax.bar3d(xs, ys, zs, dx, dy, dz, color=bar_colors, shade=True)

    # axis labelling
    ax.set_xlabel("pre bin")
    ax.set_ylabel("sq bin")
    ax.set_zlabel("counts")
    ax.set_xticks(np.arange(nbins_pre) + dx / 2)
    ax.set_xticklabels([f"{k}" for k in range(nbins_pre)])
    ax.set_yticks([0.4, 1.4])
    ax.set_yticklabels(["0", "1"])

    # --- 5. add colour-bar without blocking z-tick labels ---------------
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    fig.colorbar(
        mappable,
        ax=ax,
        shrink=0.6,
        pad=0.22,        # push further right so it doesn't overlap z-ticks
        label="counts",
    )

    # ax.set_title("3-D context counts: (sq_t, pre_t-bin)")
    # plt.tight_layout()
    plt.show()

    return counts















import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def _count_tensor(pre_disc, sq_disc, nbins_pre):
    """helper: counts[x_bin, y_prev, z_next]"""
    x_idx, y_idx, z_idx = pre_disc[:-1], sq_disc[:-1], sq_disc[1:]
    counts = np.zeros((nbins_pre, 2, 2), dtype=int)
    for xi, yi, zi in zip(x_idx, y_idx, z_idx):
        counts[xi, yi, zi] += 1
    return counts

def _delta_from_counts(counts):
    N_flip = counts[:, 0, 1] + counts[:, 1, 0]
    N_stay = counts[:, 0, 0] + counts[:, 1, 1]
    N_tot  = N_flip + N_stay
    with np.errstate(divide='ignore', invalid='ignore'):
        p_flip = np.divide(N_flip, N_tot, where=N_tot > 0)
        p_stay = np.divide(N_stay, N_tot, where=N_tot > 0)
    return p_stay - p_flip           # Δ_j for all bins

def prob_prebins_diffbar_surr(
    df_pre, df_sq,
    forcing_column='pre',
    target_column='sq',
    time_column='age',
    nbins_pre=6,
    n_surr=100,
    alpha=0.05,
    random_state=None,
    y_min=0.6,
    y_max=1.0
):
    """
    Bar plot of Δ = P(stay)-P(flip) per pre-bin
    + surrogate mean ±1σ and permutation p-value.
    Surrogates are generated by permuting the pre-bin sequence.
    """
    rng = np.random.default_rng(random_state)

    # ---------- 1) prep time series ----------
    pre_raw = df_pre[forcing_column].values[::-1]
    sq_raw  = df_sq[target_column].values[::-1]

    bins_pre = np.histogram_bin_edges(pre_raw, bins=nbins_pre)
    pre_disc = np.clip(np.digitize(pre_raw, bins_pre) - 1, 0, nbins_pre-1)
    sq_disc  = (sq_raw > 0).astype(int)

    # ---------- 2) observed Δ ----------
    counts_obs = _count_tensor(pre_disc, sq_disc, nbins_pre)
    delta_obs  = _delta_from_counts(counts_obs)        # shape (nbins_pre,)

    # ---------- 3) surrogates ----------
    delta_surr = np.zeros((n_surr, nbins_pre))
    for s in range(n_surr):
        pre_perm = rng.permutation(pre_disc)
        cnt_s    = _count_tensor(pre_perm, sq_disc, nbins_pre)
        delta_surr[s] = _delta_from_counts(cnt_s)

    mu_surr  = np.nanmean(delta_surr, axis=0)
    sd_surr  = np.nanstd (delta_surr, axis=0)

    # # p-value: two-sided tail
    # pvals = np.array([
    #     (np.sum(np.abs(delta_surr[:,j]) >= abs(delta_obs[j])) + 1)
    #     / (n_surr + 1)
    #     for j in range(nbins_pre)
    # ])

    pvals = np.empty(nbins_pre)
    for j in range(nbins_pre):
        if delta_obs[j] >= mu_surr[j]:
            tail = np.sum(delta_surr[:, j] >= delta_obs[j])
        else:
            tail = np.sum(delta_surr[:, j] <= delta_obs[j])
        pvals[j] = (tail + 1) / (n_surr + 1)



    # ---------- 4) plot ----------
    x = np.arange(nbins_pre)
    colors = np.where(delta_obs >= 0, 'C3', 'C2')   # green / red
    edgecols = np.where(pvals < alpha, 'k', 'grey')
    hatches  = ['' if p<alpha else '////' for p in pvals]



    # Prepare a red→blue colormap
    cmap = plt.get_cmap('coolwarm')  # reversed so low→red, high→blue
    cmap_arrows = plt.get_cmap('coolwarm_r')  # for arrows
    norm = plt.Normalize(0, nbins_pre - 1)
    colors = cmap(norm(np.arange(nbins_pre)))
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    
    # 1) Draw bars with the gradient colors
    for xi, d, c, ec, ht in zip(x, delta_obs, colors, edgecols, hatches):
        ax.bar(xi, d, color=c, edgecolor=ec, hatch=ht, linewidth=1.4)
    
    ax.errorbar(x, mu_surr, yerr=sd_surr, fmt='o', color='k',
                capsize=4, label='surrogate mean ±1σ')
    # ax.axhline(0, color='k', lw=4)

    # annotate Δ and p
    for xi, d, pv in zip(x, delta_obs, pvals):
        ax.text(xi, d + 0.02*np.sign(d),
                f'{d:+.2f}\n(p={pv:.3f})',
                ha='center',
                va='bottom' if d>=0 else 'top',
                fontsize=8)
    # … (labels, title, annotations of Δ and p remain the same) …
    ax.set_ylim(y_min, y_max)

    ax.set_xticks(x)
    ax.set_xlabel(f'{forcing_column} bin (0 … {nbins_pre-1})')
    ax.set_ylabel('Δ  =  P(stay) − P(flip)')
    # 2) Replace the two annotate-arrows with coloured LineCollections
    y0, y1 = ax.get_ylim()
    y_arrow = y0 + 0.03*(y1 - y0)

    # set the linewidth of spines
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    # helper to draw a gradient arrow from x_start→x_end
    def gradient_arrow(x_start, x_end, y, cmap, norm, ax, head_size=100, dir='left'):
        # make (say) 100 small segments
        xs = np.linspace(x_start, x_end, 200)
        ys = np.full_like(xs, y)
        pts = np.array([xs, ys]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap=cmap, norm=norm, linewidth=5)

        # color each segment by its mid-bin index
        mid_bins = (xs*(nbins_pre-1)/(nbins_pre-1)).clip(0, nbins_pre-1)
        lc.set_array(mid_bins)
        ax.add_collection(lc)
        c0 = cmap(norm(x_start))
        c1 = cmap(norm(x_end))

        # 3) draw small triangle‐markers at each end
        if dir == 'left':
            ax.scatter([x_end], [y],
                    marker='<',
                    s=head_size,     # head_size is area in pts^2
                    color=c1,
                    zorder=lc.get_zorder()+1)
        else:
            ax.scatter([x_end],   [y],
                    marker='>',
                    s=head_size,
                    color=c1,
                    zorder=lc.get_zorder()+1)

    
    # left arrow: from bin mid → bin 0 (decreasing pre)
    gradient_arrow(nbins_pre/2, 0, y_arrow, cmap_arrows, norm, ax, dir='left')
    ax.text(nbins_pre/6, y_arrow + 0.06*(y1-y0),
            'Precession ↓, Insolation ↑',
            ha='center', va='top', fontsize=10)
    
    # right arrow: from bin mid → last bin (increasing pre)
    gradient_arrow(nbins_pre/2, nbins_pre-1, y_arrow, cmap_arrows, norm, ax, dir='right')
    ax.text(nbins_pre*4/6, y_arrow + 0.06*(y1-y0),
            'Precession ↑, Insolation ↓',
            ha='center', va='top', fontsize=10)
    
    # extend the y-axis so your arrows are fully visible
    # ax.set_ylim(y0 - 0.12*(y1 - y0), y1)
    
    plt.tight_layout()
    plt.show()
    
    return delta_obs, mu_surr, sd_surr, pvals











import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# ─────────────────────────── helpers ───────────────────────────────────────
def _count_tensor(pre_disc, sq_disc, nbins_pre):
    """counts[x_bin, y_prev, z_next]"""
    x_idx, y_idx, z_idx = pre_disc[:-1], sq_disc[:-1], sq_disc[1:]
    counts = np.zeros((nbins_pre, 2, 2), dtype=int)
    for xi, yi, zi in zip(x_idx, y_idx, z_idx):
        counts[xi, yi, zi] += 1
    return counts

def _events_from_counts(counts, window, dt, *, relative):
    """Return events per bin (absolute or relative)."""
    N_flip = counts[:, 0, 1] + counts[:, 1, 0]              # flips
    N_tot  = counts.sum(axis=(1, 2))                        # all transitions
    with np.errstate(divide='ignore', invalid='ignore'):
        p_flip = np.divide(N_flip, N_tot, where=N_tot > 0)  # P(flip)_j

    events_abs = (window * p_flip) / dt                     # expected # flips

    # smooth the events_abs with a 1000 window running mean, make sure the smoothed data is in the same length
    # print(events_abs.shape)
    # events_abs = np.convolve(events_abs, np.ones(1000)/1000, mode='same')

    if relative:
        mean_val = events_abs.mean()
        if mean_val == 0:
            raise ValueError("Mean of events is zero; cannot form relative values.")
        return 100.0 * events_abs / mean_val
    else:
        return events_abs

# ───────────────────────── main function ───────────────────────────────────
def prob_prebins_events_surr_relative(
    df_pre, df_sq,
    forcing_column='pre',
    target_column='sq',
    time_column='age',
    sq_method='hist',  
    nbins_pre=6,
    n_surr=100,
    alpha=0.05,
    random_state=None,
    window=20_000,                 # yrs
    relative=True,
    y_min=None,
    y_max=None,
    dpi=100,
    panel_label=None,
    save_figure=False,
    proxy_name=None,
    show_legend=False,
):
    """
    Bar plot of expected # flip events per precession-bin
    + surrogate mean ±1 σ and permutation p-value.

    Surrogates are generated by permuting the pre-bin sequence.

    Parameters
    ----------
    df_pre, df_sq : pandas.DataFrame
        Original time-series (must share the same length and time axis).
    window : int
        Length of the averaging window in years used to convert probabilities
        to event counts.
    relative : bool, default True
        If True, rescale the event series so its mean is 100 %.
    """
    rng = np.random.default_rng(random_state)

    # ---------- 1) prep time series ----------
    pre_raw = df_pre[forcing_column].values[::-1]
    sq_raw  = df_sq[target_column].values[::-1]

    sq_raw = sq_raw -np.median(sq_raw)

    # time step (yrs) → dt
    t = df_pre[time_column].values[::-1]
    dt = float(np.abs(np.median(np.diff(t))))

    if sq_method == 'hist':

        bins_pre = np.histogram_bin_edges(pre_raw, bins=nbins_pre)
        
        # sq_bins = np.histogram_bin_edges(sq_raw, bins=2)
        # sq_disc = np.digitize(sq_raw, sq_bins) - 1
    elif sq_method == 'quantile':
        # --- quantile binning -----------------------------------------------------
        percentiles = np.linspace(0, 1, nbins_pre + 1)
        bins_pre    = np.quantile(pre_raw, percentiles)
        bins_pre    = np.unique(bins_pre)               # drop duplicates from ties
        nbins_eff   = len(bins_pre) - 1
        if nbins_eff < nbins_pre:
            print(f'Warning: only {nbins_eff} distinct bins; adjusting.')





    pre_disc = np.clip(np.digitize(pre_raw, bins_pre) - 1, 0, nbins_pre - 1)
    sq_disc  = (sq_raw > 0).astype(int)

    # ---------- 2) observed events ----------
    counts_obs   = _count_tensor(pre_disc, sq_disc, nbins_pre)
    events_obs   = _events_from_counts(counts_obs, window, dt, relative=relative)
    # print(events_obs.shape)
    # ---------- 3) surrogates ----------
    events_surr = np.zeros((n_surr, nbins_pre))
    for s in range(n_surr):
        # pre_perm     = rng.permutation(pre_disc)
        # counts_surr  = _count_tensor(pre_perm, sq_disc, nbins_pre)
        sq_disc_ran = rng.permutation(sq_disc)
        counts_surr  = _count_tensor(pre_disc, sq_disc_ran, nbins_pre)
        events_surr[s] = _events_from_counts(counts_surr, window, dt,
                                             relative=relative)
        # events_surr[s] = np.convolve( events_surr[s], np.ones(1000)/1000, mode='same')

    mu_surr = np.nanmean(events_surr, axis=0)
    sd_surr = np.nanstd (events_surr, axis=0)


    
    # events_abs = np.convolve(events_abs, np.ones(1000)/1000, mode='same')


    # p-value: two-sided tail
    pvals = np.empty(nbins_pre)
    for j in range(nbins_pre):
        if events_obs[j] >= mu_surr[j]:
            tail = np.sum(events_surr[:, j] >= events_obs[j])
        else:
            tail = np.sum(events_surr[:, j] <= events_obs[j])
        pvals[j] = (tail + 1) / (n_surr + 1)

    # ---------- 4) plot ----------
    x = np.arange(nbins_pre)

    # gradient colours for bars
    cmap   = plt.get_cmap('coolwarm')
    cmap_a = plt.get_cmap('coolwarm')   # for gradient arrows
    norm   = plt.Normalize(0, nbins_pre - 1)
    barcol = cmap(norm(x))
    edgecol = np.where(pvals < alpha, 'k', 'grey')
    hatches = ['' if p < alpha else '////' for p in pvals]

    fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi)

    # bars
    for xi, ev, c, ec, ht in zip(x, events_obs, barcol, edgecol, hatches):
        ax.bar(xi, ev, color=c, edgecolor=ec, hatch=ht, linewidth=1.4)

    # surrogate mean ±1 σ
    ax.errorbar(x, mu_surr, yerr=2*sd_surr, fmt='o', color='k',
                capsize=4, label='surrogate mean ±1σ')

    # lo, hi = np.percentile(events_surr, [2.5, 97.5], axis=0)
    # ax.errorbar(x, mu_surr, yerr=[mu_surr - lo, hi - mu_surr],
    #             fmt='o', color='k', capsize=4,
    #             label='surrogate mean (95 % envelope)')

    # # annotate value & p
    # for xi, ev, pv in zip(x, events_obs, pvals):
    #     ax.text(xi, ev + 0.06*np.sign(ev),
    #             f'{ev:.2f}\n(p={pv:.3f})',
    #             ha='center',
    #             va='bottom' if ev >= 0 else 'top',
    #             fontsize=8)

    # axis limits
    if y_min is None or y_max is None:
        pad = 0.08 * (events_obs.max() - events_obs.min()) 
        y_min = events_obs.min() - pad if y_min is None else y_min
        y_max = events_obs.max() + pad if y_max is None else y_max
    ax.set_ylim(y_min, y_max)

    ax.set_xticks(x)
    ax.set_xlabel(f'{forcing_column} bin (0 … {nbins_pre-1})')
    if relative:
        ax.set_ylabel('Flip events (% of the mean)')
        ax.axhline(100, color='grey', ls='--', lw=0.8, zorder=0)
    else:
        ax.set_ylabel(f'Flip events / {window:,} yr')

    # ── gradient arrows indicating precession direction ────────────────
    def gradient_arrow(x_start, x_end, y, cmap, norm, ax, head_size=100, dir='left'):
        xs = np.linspace(x_start, x_end, 200)
        ys = np.full_like(xs, y)
        pts = np.array([xs, ys]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap=cmap, norm=norm, linewidth=5)
        lc.set_array(xs)                      # colour by x
        ax.add_collection(lc)

        ax.scatter([x_end], [y],
                   marker='<' if dir == 'left' else '>',
                   s=head_size,
                   color=cmap(norm(x_end)),
                   zorder=lc.get_zorder() + 1)

    y0, y1 = ax.get_ylim()
    y_arrow = y1 + 0.1 * (y1 - y0)
    # reset the ylim
    ax.set_ylim(y0, y_arrow + 0.1 * (y1 - y0))
    gradient_arrow(nbins_pre/2, 0,           y_arrow, cmap_a, norm, ax, dir='left')
    gradient_arrow(nbins_pre/2, nbins_pre-1, y_arrow, cmap_a, norm, ax, dir='right')

    # ax.text(nbins_pre/6,   y_arrow + 0.06*(y1 - y0),
    #         'Precession ↓, Insolation ↑',
    #         ha='center', va='top', fontsize=9)
    # ax.text(nbins_pre*4/6, y_arrow + 0.06*(y1 - y0),
    #         'Precession ↑, Insolation ↓',
    #         ha='center', va='top', fontsize=9)
    

    ax.text(nbins_pre/6,   y_arrow + 0.06*(y1 - y0),
            'Precession ↓',
            ha='center', va='top', fontsize=9)
    ax.text(nbins_pre*4/6, y_arrow + 0.06*(y1 - y0),
            'Precession ↑',
            ha='center', va='top', fontsize=9)
    
    # ── panel label if provided ───────────────────────────────────────
    if panel_label is not None:
        ax.text(-0.06, 1.1, panel_label,
                transform=ax.transAxes, fontsize=12,
                ha='left', va='top', fontweight='bold')

    # thicker axes
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    # set title to proxy name, make it bold
    ax.set_title(f"{proxy_name}" if proxy_name else "Proxy Data", fontweight='bold')

    # set the position of legend
    if show_legend:
        ax.legend(loc='upper left', bbox_to_anchor=(0.01, 0.8))

    plt.tight_layout()
    plt.show()

    # if save figure, save it as .pdf in vector fomart to /Figure_saved, merge the proxy name into the file name
    if save_figure and proxy_name is not None:
        fig.savefig(f'Figures/{proxy_name}_prebins_events.pdf',
                    bbox_inches='tight', dpi=dpi, transparent=True)
        print(f"Figure saved as 'Figure_saved/{proxy_name}_prebins_events_surr_relative.pdf'")

    return events_obs, mu_surr, sd_surr, pvals







































import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection



# ---------- NEW helper: Δ = P(warm-stay) – P(cold-stay) ----------
def _staydiff_from_counts(counts):
    """return Δ_j for all pre-bins"""
    # warm-stay   P(1→1 | pre)
    warm_tot = counts[:, 1, 1] + counts[:, 1, 0]
    cold_tot = counts[:, 0, 0] + counts[:, 0, 1]

    with np.errstate(divide='ignore', invalid='ignore'):
        p_warm = np.divide(counts[:, 1, 1], warm_tot, where=warm_tot > 0)
        p_cold = np.divide(counts[:, 0, 0], cold_tot, where=cold_tot > 0)

    return p_warm - p_cold, p_warm, p_cold


# =================================================================
#  MAIN FUNCTION
# =================================================================
def prob_prebins_staydiff_surr(
    df_pre, df_sq,
    forcing_column='pre',
    target_column='sq',
    nbins_pre=6,
    n_surr=100,
    alpha=0.05,
    random_state=None
):
    """
    Δ = P(warm→warm) − P(cold→cold) for each pre-bin
    compared to permutation surrogates.
    """
    rng = np.random.default_rng(random_state)

    # ---------- 1) prepare series ----------
    pre_raw = df_pre[forcing_column].values[::-1]
    sq_raw  = df_sq [target_column].values[::-1]

    bins_pre = np.histogram_bin_edges(pre_raw, bins=nbins_pre)
    pre_disc = np.clip(np.digitize(pre_raw, bins=bins_pre) - 1, 0, nbins_pre-1)
    sq_disc  = (sq_raw > 0).astype(int)

    # ---------- 2) observed Δ ----------
    counts_obs = _count_tensor(pre_disc, sq_disc, nbins_pre)
    delta_obs, p_warm, p_cold = _staydiff_from_counts(counts_obs)

    # ---------- 3) surrogates ----------
    delta_surr = np.zeros((n_surr, nbins_pre))
    for k in range(n_surr):
        counts_k = _count_tensor(rng.permutation(pre_disc), sq_disc, nbins_pre)
        delta_surr[k] = _staydiff_from_counts(counts_k)[0]

    mu_surr = np.nanmean(delta_surr, axis=0)
    sd_surr = np.nanstd (delta_surr, axis=0)

    # p-values (one-sided toward observed sign)
    pvals = np.empty(nbins_pre)
    for j in range(nbins_pre):
        if delta_obs[j] >= mu_surr[j]:
            tail = np.sum(delta_surr[:, j]>= delta_obs[j])
        else:
            tail = np.sum(delta_surr[:, j] <= delta_obs[j])
        pvals[j] = (tail + 1)/(n_surr + 1)

    # ---------- 4) plot ----------
    x = np.arange(nbins_pre)
    cmap  = plt.get_cmap('coolwarm')
    cmapA = plt.get_cmap('coolwarm_r')
    colors = cmap(np.linspace(0, 1, nbins_pre))
    edgecols = np.where(pvals < alpha, 'k', 'grey')
    hatches  = ['' if p < alpha else '////' for p in pvals]

    fig, ax = plt.subplots(figsize=(6, 4))

    for xi, d, c, ec, ht in zip(x, delta_obs, colors, edgecols, hatches):
        ax.bar(xi, d, color=c, edgecolor=ec, hatch=ht, linewidth=1.4)

    ax.errorbar(x, mu_surr, yerr=sd_surr, fmt='o', color='k',
                capsize=4, label='surrogate mean ±1σ')

    for xi, d, pv in zip(x, delta_obs, pvals):
        ax.text(xi, d + 0.01*np.sign(d),
                f'{d:+.2f}\n(p={pv:.3f})',
                ha='center', va='bottom' if d>=0 else 'top', fontsize=8)

    ax.set_xticks(x)
    ax.set_xlabel(f'{forcing_column} bin (0 … {nbins_pre-1})')
    ax.set_ylabel('Δ  =  P(warm stay) − P(cold stay)')
    ax.set_ylim(-0.04, 0.04)

    # gradient arrows as in your previous version ------------------
    y0, y1 = ax.get_ylim();   y_arrow = y0 - 0.1*(y1 - y0)

    def gradient_arrow(x0, x1, y, cmap, ax, head='left'):
        xs = np.linspace(x0, x1, 200)
        ys = np.full_like(xs, y)
        segs = np.stack([xs, ys], axis=-1).reshape(-1, 1, 2)
        segs = np.concatenate([segs[:-1], segs[1:]], axis=1)
        lc = LineCollection(segs, cmap=cmap, norm=plt.Normalize(0, nbins_pre-1),
                            linewidth=5)
        lc.set_array(xs)
        ax.add_collection(lc)
        ax.scatter([x1], [y], marker='<' if head=='left' else '>',
                   s=90, color=cmap(xs[-1]/(nbins_pre-1)))

    gradient_arrow(nbins_pre/2, 0, y_arrow, cmapA, ax, 'left')
    gradient_arrow(nbins_pre/2, nbins_pre-1, y_arrow, cmapA, ax, 'right')
    ax.text(nbins_pre/6,  y_arrow+0.01, 'Precession ↓', ha='center', va='top')
    ax.text(nbins_pre*5/6,y_arrow+0.01, 'Precession ↑', ha='center', va='top')
    ax.set_ylim(y_arrow-0.01, 0.05)

    plt.legend(loc='upper right'); plt.tight_layout(); plt.show()

    return delta_obs, mu_surr, sd_surr, pvals






import numpy as np
import matplotlib.pyplot as plt

def local_stay_split(
    df_pre, df_sq,
    forcing_column='pre',
    target_column='sq',
    time_column='age',
    nbins_pre=4,
    smooth_win=200
):
    # ---------- 1) reverse to chronological order ----------
    pre_raw = df_pre[forcing_column].values[::-1]
    sq_raw  = df_sq[target_column].values[::-1]
    t       = df_pre[time_column].values[::-1]

    # ---------- 2) low-pre shading (optional) ----------
    q50   = np.quantile(pre_raw, 0.5)
    low   = pre_raw < q50
    edges = np.diff(low.astype(int))
    starts = list(np.where(edges == 1)[0] + 1)
    ends   = list(np.where(edges == -1)[0] + 1)
    if low[0]:  starts.insert(0, 0)
    if low[-1]: ends.append(len(low))
    low_periods = list(zip(starts, ends))

    # ---------- 3) discretise ----------
    bins_pre = np.histogram_bin_edges(pre_raw, bins=nbins_pre)
    pre_disc = np.clip(np.digitize(pre_raw, bins_pre) - 1, 0, nbins_pre-1)
    sq_disc  = (sq_raw > 0).astype(int)          # 0 = cold, 1 = warm

    x_idx, y_idx, z_idx = pre_disc[:-1], sq_disc[:-1], sq_disc[1:]

    # ---------- 4) counts tensor ----------
    counts = np.zeros((nbins_pre, 2, 2), dtype=int)
    for xi, yi, zi in zip(x_idx, y_idx, z_idx):
        counts[xi, yi, zi] += 1

    # ---------- 5) conditional-prob tensor ----------
    totals = counts.sum(axis=2, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        cond_prob = counts / totals               # P(z | x, y)

    # ---------- helper: centred running mean that ignores NaNs ----------
    def smooth_nan(arr, win):
        arr      = np.asarray(arr, float)
        mask     = ~np.isnan(arr)
        arr_f    = np.where(mask, arr, 0.0)
        kern     = np.ones(win)
        num      = np.convolve(arr_f, kern, mode='same')
        denom    = np.convolve(mask.astype(float), kern, mode='same')
        out      = np.divide(num, denom, where=denom>0)
        out[denom == 0] = np.nan
        return out

    # ---------- 6) build the four local probability series ----------
    cold_stay_raw = np.array([
        cond_prob[x, 0, 0] if y == 0 else np.nan
        for x, y in zip(x_idx, y_idx)
    ])
    warm_stay_raw = np.array([
        cond_prob[x, 1, 1] if y == 1 else np.nan
        for x, y in zip(x_idx, y_idx)
    ])
    cold_stay = smooth_nan(cold_stay_raw, smooth_win)
    warm_stay = smooth_nan(warm_stay_raw, smooth_win)

    # ---------- 7) plotting ----------
    fig = plt.figure(figsize=(12, 4))
    gs  = fig.add_gridspec(2, 1, height_ratios=[0.8, 1.6], hspace=0)

    # (a) raw signals
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(t, pre_raw, label=forcing_column)
    ax0b = ax0.twinx()
    ax0b.plot(t, sq_raw, color='C1', label=target_column)
    for s, e in low_periods:
        ax0 .axvspan(t[s], t[e-1], color='grey', alpha=0.3)
        ax0b.axvspan(t[s], t[e-1], color='grey', alpha=0.3)
    ax0.set_ylabel(forcing_column)
    ax0b.set_ylabel(target_column)
    ax0.tick_params(axis='x', labelbottom=False)
    lns = ax0.get_lines() + ax0b.get_lines()
    ax0.legend(lns, [l.get_label() for l in lns], loc='upper right')
    ax0.set_xlim(t[1], t[-1])

    # (b) persistence probabilities
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ln1, = ax1.plot(t[1:], cold_stay, color='C0', label='P(cold→cold)')
    ax1.set_ylabel('P(cold stay)', color='C0')
    ax1.tick_params(axis='y', colors='C0')


    ax1b = ax1.twinx()
    ln2, = ax1b.plot(t[1:], warm_stay, color='C3', label='P(warm→warm)')
    ax1b.set_ylabel('P(warm stay)', color='C3')
    ax1b.tick_params(axis='y', colors='C3')

    for s, e in low_periods:
        ax1 .axvspan(t[s], t[e-1], color='grey', alpha=0.3)
        ax1b.axvspan(t[s], t[e-1], color='grey', alpha=0.3)

    ax1.set_xlabel(time_column)
    ax1.legend([ln1, ln2], ['cold persistence', 'warm persistence'],
               loc='upper right')

    plt.show()

    return cold_stay, warm_stay



























import numpy as np
import matplotlib.pyplot as plt

def local_prob(
    df_pre,
    df_sq,
    forcing_column='pre',
    target_column='sq',
    time_column='age',
    nbins_pre=4,
    smooth_win=200
):
    # 1) raw series (reverse → old→young if needed)
    pre_raw = df_pre[forcing_column].values[::-1]
    sq_raw  = df_sq[target_column].values[::-1]
    t       = df_pre[time_column].values[::-1]

    # 2) mark "low-pre" periods (here: below 50 % quantile as in your code)
    q50   = np.quantile(pre_raw, 0.5)
    low   = pre_raw < q50
    edges = np.diff(low.astype(int))
    starts = list(np.where(edges == 1)[0] + 1)
    ends   = list(np.where(edges == -1)[0] + 1)
    if low[0]:  starts.insert(0, 0)
    if low[-1]: ends.append(len(low))
    low_periods = list(zip(starts, ends))

    # 3) discretise
    bins_pre = np.histogram_bin_edges(pre_raw, bins=nbins_pre)
    pre_disc = np.digitize(pre_raw, bins_pre) - 1        # 0…nbins_pre-1
    pre_disc = np.clip(pre_disc, 0, nbins_pre-1)
    sq_disc  = (sq_raw > 0).astype(int)                  # 0/1

    # past indices (length N-1) and future state
    x_idx = pre_disc[:-1]
    y_idx = sq_disc[:-1]
    z_idx = sq_disc[1:]

    # 4) build 3-D count tensor  (pre_bin, sq_prev, sq_next)
    counts = np.zeros((nbins_pre, 2, 2), dtype=int)
    for xi, yi, zi in zip(x_idx, y_idx, z_idx):
        counts[xi, yi, zi] += 1

    # 5) convert to conditional-prob tensor  P(z | x,y)
    cond_prob = np.zeros_like(counts, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        totals = counts.sum(axis=2, keepdims=True)
        cond_prob = np.divide(counts, totals,
                              where=totals>0)          # avoid /0

    # def smooth(arr, win=10):
    #     """Simple centred moving-average, same length as input."""
    #     kern = np.ones(win) / win
    #     return np.convolve(arr, kern, mode='same')

    def smooth(arr, win=10, mode='edge'):
        """
        Centre-aligned moving average with correct length for
        both even and odd window sizes.  Edge values are extended
        ('edge') or mirrored ('reflect').
        """
        arr  = np.asarray(arr)
        L    = len(arr)
        left = win // 2               # floor
        right = win - left - 1        # ensures left+right = win-1
        arr_p = np.pad(arr, (left, right), mode=mode)
        kern  = np.ones(win) / win
        sm    = np.convolve(arr_p, kern, mode='valid')   # length == L
        return sm




    # 6) time-series of flip / stay probabilities
    # flip_prob = np.array([cond_prob[x, y, 1-y] for x, y in zip(x_idx, y_idx)])
    # stay_prob = np.array([cond_prob[x, y,   y ] for x, y in zip(x_idx, y_idx)])

    flip_prob_raw = np.array([cond_prob[x, y, 1-y] for x, y in zip(x_idx, y_idx)])
    stay_prob_raw = np.array([cond_prob[x, y,   y ] for x, y in zip(x_idx, y_idx)])
    flip_prob = smooth(flip_prob_raw, win=200)
    stay_prob = smooth(stay_prob_raw, win=200)


    # -------- PLOT --------
    fig = plt.figure(figsize=(12, 4))
    gs  = fig.add_gridspec(2, 1, height_ratios=[0.8, 1.6], hspace=0)

    # top: original signals
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(t, pre_raw, label=f'{forcing_column} (raw)')
    ax0.set_ylabel(forcing_column)
    ax0b = ax0.twinx()
    ax0b.plot(t, sq_raw, color='C1', label=f'{target_column} (raw)')
    ax0b.set_ylabel(target_column)
    # shade low-pre epochs
    for s, e in low_periods:
        ax0.axvspan(t[s], t[e-1], color='gray', alpha=0.3)
        ax0b.axvspan(t[s], t[e-1], color='gray', alpha=0.3)
    lns = ax0.get_lines() + ax0b.get_lines()
    ax0.legend(lns, [l.get_label() for l in lns], loc='upper right')
    ax0.tick_params(axis='x', which='both', labelbottom=False)
    ax0.set_xlim(t[1], t[-1])


    ax1  = fig.add_subplot(gs[1], sharex=ax0)

    # left axis = flip probability
    ln1, = ax1.plot(t[1:], flip_prob, color='C2', label='P(flip | pre,sq)')
    ax1.set_ylabel('P(flip)', color='C2')
    flip_min, flip_max = flip_prob.min(), flip_prob.max()
    pad = 0.05 * (flip_max - flip_min + 1e-9)
    ax1.set_ylim(flip_min - pad, flip_max + pad)
    ax1.tick_params(axis='y', colors='C2')

    # right axis = stay probability
    ax1b = ax1.twinx()
    ln2, = ax1b.plot(t[1:], stay_prob, color='C3', label='P(stay | pre,sq)')
    ax1b.set_ylabel('P(stay)', color='C3')
    stay_min, stay_max = stay_prob.min(), stay_prob.max()
    pad = 0.05 * (stay_max - stay_min + 1e-9)
    ax1b.set_ylim(stay_min - pad, stay_max + pad)
    ax1b.tick_params(axis='y', colors='C3')

    ax1.set_xlabel(time_column)

    # shade low-pre epochs
    for s, e in low_periods:
        ax1 .axvspan(t[s], t[e-1], color='gray', alpha=0.3)
        ax1b.axvspan(t[s], t[e-1], color='gray', alpha=0.3)

    # combined legend
    ax1.legend([ln1, ln2], ['P(flip)', 'P(stay)'], loc='upper right')

    return flip_prob, stay_prob









def heatmap_binwise_local_te(
        pre, sq,
        pre_bins, sq_bins,
        *, k=1, cmap='viridis', vmin=None, vmax=None,
        te_func=transfer_entropy):
    """
    Compute and plot a bin-wise mean local TE(pre → sq) heat-map.

    Parameters
    ----------
    pre, sq : 1-D array-like, same length
        Time series (index 0 = oldest sample).  Both are REVERSED inside.
    pre_bins, sq_bins : int or 1-D array-like
        If int  -> number of equal-width histogram bins.
        If array -> explicit bin edges as for np.histogram.
    k : int
        History length for TE (default 1).
    cmap, vmin, vmax : passed to imshow.
    te_func : callable
        e.g. pyinform.transfer_entropy.
    """
    # -------- 0) prep -------------------------------------------------------
    pre = np.asarray(pre)[::-1]          # reverse to match your workflow
    sq  = np.asarray(sq )[::-1]
    if pre.shape != sq.shape:
        raise ValueError("pre and sq must have the same length")

    # -------- 1) build bin edges -------------------------------------------
    def _edges(data, spec):
        if np.isscalar(spec):                       # integer → histogram edges
            nb = int(spec)
            if nb < 1:
                raise ValueError("bins integer must be ≥1")
            return np.histogram_bin_edges(data, bins=nb)
        else:                                       # assume iterable of edges
            edges = np.asarray(spec, dtype=float)
            if edges.ndim != 1 or edges.size < 2:
                raise ValueError("bin edges must be 1-D with ≥2 values")
            # ensure monotone
            if not (np.all(np.diff(edges) > 0) or np.all(np.diff(edges) < 0)):
                raise ValueError("bin edges must be strictly monotonic")
            return edges

    pre_edges = _edges(pre, pre_bins)
    sq_edges  = _edges(sq , sq_bins)

    n_pre = len(pre_edges) - 1
    n_sq  = len(sq_edges)  - 1

    # -------- 2) digitise ---------------------------------------------------
    pre_bin = np.digitize(pre, pre_edges) - 1          # 0 … n_pre-1
    sq_bin  = np.digitize(sq , sq_edges ) - 1          # 0 … n_sq-1

    # -------- 3) local TE ---------------------------------------------------
    arr = te_func(pre_bin, sq_bin, k=k, local=True).flatten()
    if   arr.size == pre.size:        # pyinform returned N   → drop the first k
        local_te = arr[k:]
    elif arr.size == pre.size - k:    # pyinform returned N-k → already aligned
        local_te = arr
    else:
        raise ValueError(f"Unexpected local_te length {arr.size}")

    # past-state indices that correspond to each local TE
    idx_pre = pre_bin[:-k]
    idx_sq  = sq_bin [:-k]

    # -------- 4) accumulate by bin -----------------------------------------
    te_sum   = np.zeros((n_pre, n_sq))
    te_count = np.zeros_like(te_sum, dtype=int)

    for i, j, te in zip(idx_pre, idx_sq, local_te):
        if 0 <= i < n_pre and 0 <= j < n_sq:
            te_sum[i, j]   += te
            te_count[i, j] += 1

    te_grid = np.full_like(te_sum, np.nan, dtype=float)
    mask    = te_count > 0
    te_grid[mask] = te_sum[mask] / te_count[mask]

    # -------- 5) plot -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 2))
    im = ax.imshow(
        te_grid.T, origin='lower', aspect='auto',
        cmap=cmap, vmin=vmin, vmax=vmax,
        extent=[pre_edges[0], pre_edges[-1], sq_edges[0], sq_edges[-1]]
    )
    cbar = fig.colorbar(im, ax=ax, label='Mean local TE (bits)')
    ax.set_xlabel('pre bin')
    ax.set_ylabel('sq bin')
    ax.set_title('Bin-wise mean local transfer entropy')
    plt.tight_layout()
    plt.show()

    return te_grid







def local_TE(
    df_pre,
    df_sq,
    forcing_column='pre',
    target_column='sq',
    time_column='age',
    nbins_pre=4,
    nbins_target=2
):

    # --- 1) Extract raw numpy arrays and reverse if needed
    pre_raw = df_pre[forcing_column].values[::-1]
    sq_raw  = df_sq[target_column].values[::-1]
    t        = df_pre[time_column].values[::-1]

    # pre_raw = df_pre[forcing_column].values
    # sq_raw  = df_sq[target_column].values
    # t        = df_pre[time_column].values

    # --- 1.1) Find “low-pre” periods (below 25% quantile)
    q25   = np.quantile(pre_raw, 0.5)
    low   = pre_raw < q25
    # find rising/falling edges of the boolean mask
    edges = np.diff(low.astype(int))
    starts = list(np.where(edges == 1)[0] + 1)
    ends   = list(np.where(edges == -1)[0] + 1)
    # if it starts low right away
    if low[0]:
        starts.insert(0, 0)
    # if it ends low
    if low[-1]:
        ends.append(len(low))
    # pair them
    low_periods = list(zip(starts, ends))

    # --- 2) Discretize
    bins_pre = np.histogram_bin_edges(pre_raw, bins=nbins_pre)
    pre_disc = np.digitize(pre_raw, bins_pre) - 1
    sq_disc  = (sq_raw > 0).astype(int)
    
    # --- 3) Compute local transfer entropy (unchanged)
    local_te = transfer_entropy(pre_disc, sq_disc, k=1, local=True)

    # --- 4) Plot setup with two panels
    fig = plt.figure(figsize=(12, 4))
    gs  = fig.add_gridspec(2, 1, height_ratios=[0.8, 1.6], hspace=0)

    # Top panel: pre & sq
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(t, pre_raw, label=f'{forcing_column} (raw)')
    ax0.set_ylabel(forcing_column)
    ax0.tick_params(axis='x', which='both', labelbottom=False)
    ax0.set_xlim(t[1], t[-1])
    
    ax0b = ax0.twinx()
    ax0b.plot(t, sq_raw, color='C1', label=f'{target_column} (raw)')
    ax0b.set_ylabel(target_column)

    # Shade low-pre on top panel
    for start, end in low_periods:
        ax0.axvspan(t[start], t[end-1], color='gray', alpha=0.3)
        ax0b.axvspan(t[start], t[end-1], color='gray', alpha=0.3)

    # combine legends
    lns = ax0.get_lines() + ax0b.get_lines()
    labs = [l.get_label() for l in lns]
    ax0.legend(lns, labs, loc='upper right')
    
    # Bottom panel: local TE
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax1.plot(t[1:], local_te.T, color='C2', label='Local TE')
    ax1.set_xlabel(time_column)
    ax1.set_ylabel('Local TE (bits)')
    ax1.legend(loc='upper right')

    # Shade low-pre on bottom panel
    for start, end in low_periods:
        ax1.axvspan(t[start], t[end-1], color='gray', alpha=0.3)

    plt.show()

    return local_te













def local_TE_prob(
    df_pre,
    df_sq,
    forcing_column='pre',
    target_column='sq',
    time_column='age',
    nbins_pre=4,
    smooth_win=200
):
    """
    Like local_prob, but with an extra middle subplot showing local TE.
    Returns: flip_prob, stay_prob, local_te
    """
    # 1) raw series (reverse → old→young if needed)
    pre_raw = df_pre[forcing_column].values[::-1]
    sq_raw  = df_sq[target_column].values[::-1]
    t       = df_pre[time_column].values[::-1]

    # 2) mark "low-pre" periods (below median)
    q50   = np.quantile(pre_raw, 0.5)
    low   = pre_raw < q50
    edges = np.diff(low.astype(int))
    starts = list(np.where(edges == 1)[0] + 1)
    ends   = list(np.where(edges == -1)[0] + 1)
    if low[0]:  starts.insert(0, 0)
    if low[-1]: ends.append(len(low))
    low_periods = list(zip(starts, ends))

    # 3) discretise
    bins_pre = np.histogram_bin_edges(pre_raw, bins=nbins_pre)
    pre_disc = np.digitize(pre_raw, bins_pre) - 1
    pre_disc = np.clip(pre_disc, 0, nbins_pre-1)
    sq_disc  = (sq_raw > 0).astype(int)

    # 4) compute flip/stay probabilities
    x_idx = pre_disc[:-1]
    y_idx = sq_disc[:-1]
    z_idx = sq_disc[1:]
    counts = np.zeros((nbins_pre, 2, 2), dtype=int)
    for xi, yi, zi in zip(x_idx, y_idx, z_idx):
        counts[xi, yi, zi] += 1
    with np.errstate(divide='ignore', invalid='ignore'):
        totals = counts.sum(axis=2, keepdims=True)
        cond_prob = np.divide(counts, totals, where=totals>0)
    flip_raw = np.array([cond_prob[x, y, 1-y] for x, y in zip(x_idx, y_idx)])
    stay_raw = np.array([cond_prob[x, y,   y ] for x, y in zip(x_idx, y_idx)])
    # simple centred moving average
    def smooth(arr, win=smooth_win, mode='edge'):
        L = len(arr)
        left = win//2
        right = win - left - 1
        arr_p = np.pad(arr, (left, right), mode=mode)
        kern  = np.ones(win)/win
        return np.convolve(arr_p, kern, mode='valid')
    flip_prob = smooth(flip_raw)
    stay_prob = smooth(stay_raw)

    # 5) compute local TE
    # note: TE is defined on pairs, so its length is len(pre_disc)-1 == len(t)-1
    local_te  = transfer_entropy(pre_disc, sq_disc, k=1, local=True)

    # -------- PLOT all three panels --------
    fig = plt.figure(figsize=(12, 6))
    gs  = fig.add_gridspec(3, 1,
                           height_ratios=[0.8, 0.8, 1.6],
                           hspace=0)

    # Top: raw signals
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(t, pre_raw, label=f'{forcing_column} (raw)')
    ax0.set_ylabel(forcing_column)
    ax0b = ax0.twinx()
    ax0b.plot(t, sq_raw, color='C1', label=f'{target_column} (raw)')
    ax0b.set_ylabel(target_column)
    for s, e in low_periods:
        ax0.axvspan(t[s], t[e-1], color='gray', alpha=0.3)
        ax0b.axvspan(t[s], t[e-1], color='gray', alpha=0.3)
    lns = ax0.get_lines() + ax0b.get_lines()
    ax0.legend(lns, [l.get_label() for l in lns], loc='upper right')
    ax0.tick_params(axis='x', labelbottom=False)
    ax0.set_xlim(t[1], t[-1])

    # Middle: local TE
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax1.plot(t[1:], local_te.T, color='C2', label='Local TE')
    ax1.set_ylabel('Local TE (bits)')
    ax1.legend(loc='upper right')
    ax1.tick_params(axis='x', labelbottom=False)
    for s, e in low_periods:
        ax1.axvspan(t[s], t[e-1], color='gray', alpha=0.3)

    # Bottom: flip/stay probabilities
    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    ln1, = ax2.plot(t[1:], flip_prob, color='C3', label='P(flip)')
    ax2.set_ylabel('P(flip)', color='C3')
    ax2.tick_params(axis='y', colors='C3')
    ax2b = ax2.twinx()
    ln2, = ax2b.plot(t[1:], stay_prob, color='steelblue', label='P(stay)')
    ax2b.set_ylabel('P(stay)', color='C4')
    ax2b.tick_params(axis='y', colors='C4')
    ax2.set_xlabel(time_column)
    for s, e in low_periods:
        ax2.axvspan(t[s], t[e-1], color='gray', alpha=0.3)
        ax2b.axvspan(t[s], t[e-1], color='gray', alpha=0.3)
    ax2.legend([ln1, ln2], ['P(flip)', 'P(stay)'], loc='upper right')

    plt.show()

    return flip_prob, stay_prob, local_te



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

def predict_events_future(df_pre,          # original training dataframe
                          flip_prob,       # smoothed P(flip) from training
                          df_pre_future,   # new dataframe with 'age' & 'pre'
                          window=20_000,   # yrs
                          forcing_column='pre',
                          time_column='age',
                          nbins_pre=4,
                          band_sigma=1.0,  # size of uncertainty band (σ, 1 = ~68 %)
                          cmap_name='tab20c'):
    """
    Predict the expected number of flip events – and an uncertainty range – for a
    future precession series sampled at the same Δt as the training data.

    Returns
    -------
    events_mean  : np.ndarray  – expected # events / `window`
    events_lower : np.ndarray  – mean - σ
    events_upper : np.ndarray  – mean + σ
    """

    # ────────── 1) learn mean & σ of P(flip) in each pre-bin ────────────
    t_train   = df_pre[time_column].values[::-1]
    pre_train = df_pre[forcing_column].values[::-1]

    bins_pre = np.histogram_bin_edges(pre_train, bins=nbins_pre)
    pre_disc_train = np.clip(np.digitize(pre_train, bins_pre) - 1, 0, nbins_pre-1)

    # statistics per bin
    means = np.full(nbins_pre, np.nan)
    stdev = np.full(nbins_pre, np.nan)
    for b in range(nbins_pre):
        fp_bin = flip_prob[pre_disc_train[1:] == b]      # align flip_prob with t[1:]
        if len(fp_bin):
            means[b] = fp_bin.mean()
            stdev[b] = fp_bin.std(ddof=1)
    # fallback: if a bin had no training points, use global stats
    global_mean = flip_prob.mean()
    global_std  = flip_prob.std(ddof=1)
    means = np.where(np.isnan(means), global_mean, means)
    stdev = np.where(np.isnan(stdev), global_std, stdev)

    # ────────── 2) convert future pre → expected events ± σ ─────────────
    # t_future   = df_pre_future[time_column].values[::-1]
    # pre_future = df_pre_future[forcing_column].values[::-1]
    t_future   = df_pre_future[time_column].values
    pre_future = df_pre_future[forcing_column].values

    dt_future  = float(np.abs(np.median(np.diff(t_future))))  # yrs / step
    pre_disc_future = np.clip(np.digitize(pre_future, bins_pre) - 1, 0, nbins_pre-1)

    p_mean = means[pre_disc_future]
    p_std  = stdev[pre_disc_future]

    events_mean  =  window * p_mean / dt_future
    events_lower =  window * (p_mean - band_sigma * p_std) / dt_future
    events_upper =  window * (p_mean + band_sigma * p_std) / dt_future

    # guard against negative lower bound
    events_lower = np.clip(events_lower, 0, None)

    # ────────── 3) PLOT ────────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 5))
    gs  = fig.add_gridspec(2, 1, height_ratios=[0.9, 1.3], hspace=0)

    # (a) top – future pre with coloured bands
    ax0 = fig.add_subplot(gs[0])
    cmap = cm.get_cmap(cmap_name, nbins_pre)
    for i in range(nbins_pre):
        ax0.axhspan(bins_pre[i], bins_pre[i+1],
                    facecolor=cmap(i), alpha=0.25)
        ax0.axhline(bins_pre[i], color='white', lw=0.8)
    ax0.plot(t_future, pre_future, color='black', lw=1.4)
    ax0.set_ylabel(forcing_column)
    ax0.set_xlim(t_future[0], t_future[-1])
    ax0.tick_params(axis='x', labelbottom=False)

    # (b) bottom – expected events with ±σ band
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax1.plot(t_future, events_mean, color='C3', label='Mean prediction')
    ax1.fill_between(t_future, events_lower, events_upper,
                     color='C3', alpha=0.3,
                     label=f'±{band_sigma:.0f} σ')
    ax1.set_xlabel(time_column)
    ax1.set_ylabel('Events / window')
    ax1.legend(loc='upper right')

    plt.show()
    return events_mean, events_lower, events_upper




# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import cm                            # colormap for the strips

# def flip_events(df_pre,
#         flip_prob,
#         window=20_000,              # yrs; make this whatever you like
#         forcing_column='pre',
#         time_column='age',
#         nbins_pre=4):
#     """
#     Convert P(flip) → expected number of flip events in a `window`-yr cycle.

#     Parameters
#     ----------
#     df_pre       : pandas.DataFrame  – must contain `forcing_column`, `time_column`
#     flip_prob    : 1-D array (len = len(df_pre)-1)  – smoothed P(flip) from local_TE_prob
#                    *must correspond to t[1:], i.e. skip the very first sample*
#     window       : float  – cycle length in the same units as the time axis (yrs)
#     forcing_column, time_column, nbins_pre : see previous functions

#     Returns
#     -------
#     events : 1-D numpy array  – expected number of flips per `window`
#              (same length as `flip_prob`, aligned with t[1:])
#     """
#     # --- 1) time axis and resolution -----------------------------------------
#     t  = df_pre[time_column].values[::-1]            # ascending old→young
#     dt = float(np.abs(np.median(np.diff(t))))        # median spacing (yrs)

#     # --- 2) expected number of flips in the chosen window --------------------
#     events = (window * flip_prob) / dt               # Eq.:  N = W * p / dt

#     # --- 3) discretise pre for the coloured strips ---------------------------
#     pre_raw = df_pre[forcing_column].values[::-1]
#     bins_pre = np.histogram_bin_edges(pre_raw, bins=nbins_pre)
#     pre_disc = np.clip(np.digitize(pre_raw, bins_pre) - 1, 0, nbins_pre-1)
#     strip_vals = pre_disc[1:]                       # align with events length

#     # --- 4) figure -----------------------------------------------------------
#     fig = plt.figure(figsize=(12, 5))
#     gs  = fig.add_gridspec(2, 1, height_ratios=[0.9, 1.3], hspace=0)

#     # (a) top panel ─ raw forcing signal
#     ax0 = fig.add_subplot(gs[0])
#     ax0.plot(t, pre_raw, color='C0')
#     ax0.set_ylabel(forcing_column)
#     ax0.set_xlim(t[1], t[-1])
#     ax0.tick_params(axis='x', labelbottom=False)

#     # (b) bottom panel ─ expected number of events
#     ax1 = fig.add_subplot(gs[1], sharex=ax0)
#     ax1.plot(t[1:], events, color='C3', label=f'Expected # flips / {window:,} yr')
#     ax1.set_xlabel(time_column)
#     ax1.set_ylabel('Events (count)')

#     # Determine a thin band above the data range for the coloured strips
#     y_min, y_max = ax1.get_ylim()
#     strip_h = 0.06 * (y_max - y_min + 1e-9)
#     ax1.imshow(strip_vals.reshape(1, -1),
#                extent=[t[1], t[-1], y_max, y_max + strip_h],
#                aspect='auto',
#                cmap=cm.viridis, interpolation='nearest')
#     ax1.set_ylim(y_min, y_max + strip_h)
#     ax1.text(0.005, 0.98, f'{nbins_pre} pre-bins',
#              transform=ax1.transAxes, fontsize='x-small', va='top', ha='left')

#     ax1.legend(loc='upper right')
#     plt.show()

#     return events






# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import cm

# def flip_events(df_pre,
#                 flip_prob,
#                 window=20_000,              # yrs; change at will
#                 forcing_column='pre',
#                 time_column='age',
#                 nbins_pre=4,
#                 cmap_name='spring'):
#     """
#     Convert P(flip) → expected number of flip events in a `window`-yr cycle
#     and plot:
#         • top  : pre signal with horizontal bin strips
#         • lower: expected # flip events / window
#     Returns
#     -------
#     events : 1-D numpy array, length == len(flip_prob)
#     """
#     # ------------------------------------------------------------------ data
#     t  = df_pre[time_column].values[::-1]                 # old → young
#     dt = float(np.abs(np.median(np.diff(t))))             # median spacing

#     events = (window * flip_prob) / dt                    # N = W * p / dt

#     pre_raw  = df_pre[forcing_column].values[::-1]
#     bins_pre = np.histogram_bin_edges(pre_raw, bins=nbins_pre)

#     # ----------------------------------------------------------- figure setup
#     fig = plt.figure(figsize=(12, 5))
#     gs  = fig.add_gridspec(2, 1, height_ratios=[0.9, 1.3], hspace=0)

#     # ── (a) top: pre, with horizontal bands for bins ────────────────────────
#     ax0 = fig.add_subplot(gs[0])

#     # draw horizontal colour bands (z-order 0 → behind everything)
#     cmap = cm.get_cmap(cmap_name, nbins_pre)
#     for i in range(nbins_pre):
#         y0, y1 = bins_pre[i], bins_pre[i + 1]
#         ax0.axhspan(y0, y1,
#                     facecolor=cmap(i),
#                     alpha=0.5,
#                     zorder=0,
#                     edgecolor='none')
#         ax0.axhline(bins_pre[i], color='white', lw=0.8)

#     # pre line
#     ax0.plot(t, pre_raw, color='C0', zorder=2)
#     ax0.set_ylabel(forcing_column)
#     ax0.set_xlim(t[1], t[-1])
#     ax0.tick_params(axis='x', labelbottom=False)

#     # optional: indicate bin edges on y-axis ticks
#     ax0.set_yticks(bins_pre)
#     ax0.set_yticklabels([f'{b:.2g}' for b in bins_pre])

#     # ── (b) bottom: expected events ─────────────────────────────────────────
#     ax1 = fig.add_subplot(gs[1], sharex=ax0)
#     ax1.plot(t[1:], events,
#              color='C3',
#              label=f'Expected # flips / {window:,} yr')
#     ax1.set_xlabel(time_column)
#     ax1.set_ylabel('Events (count)')
#     ax1.legend(loc='upper right')

#     plt.show()
#     return events






















# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import cm

# def flip_events(df_pre,
#                 flip_prob,
#                 window=20_000,              # yrs; change at will
#                 forcing_column='pre',
#                 time_column='age',
#                 nbins_pre=4,
#                 cmap_name='spring',
#                 relative=True):
#     """
#     Convert P(flip) → expected number of flip events in a `window`-yr cycle
#     and plot:
#         • top  : pre signal with horizontal bin strips
#         • lower: expected # flip events / window                (absolute mode)
#                  OR the same series expressed relative to its
#                  mean (=100 %)                                  (relative mode)

#     Parameters
#     ----------
#     df_pre : pandas.DataFrame
#     flip_prob : 1-D array-like, same length as df_pre
#     window : int, optional
#         Length of averaging window in years.
#     forcing_column : str
#         Column in *df_pre* holding the forcing signal (“pre”).
#     time_column : str
#         Column in *df_pre* holding the age/time ordinate (“age”).
#     nbins_pre : int
#         Number of bins for the coloured background strips.
#     cmap_name : str
#         Name of the Matplotlib colormap for the strips.
#     relative : bool, default **True**
#         If *True*, scale the expected-events series so that its
#         mean equals 100 %.  If *False*, plot the absolute counts.

#     Returns
#     -------
#     events : 1-D numpy.ndarray
#         The series that was actually plotted (relative or absolute).
#     """
#     # ------------------------------------------------------------------ data
#     t  = df_pre[time_column].values[::-1]                 # old → young
#     dt = float(np.abs(np.median(np.diff(t))))             # median spacing

#     events_abs = (window * flip_prob) / dt                # N = W * p / dt

#     if relative:
#         mean_val = events_abs.mean()
#         if mean_val == 0:
#             raise ValueError("Mean of events is zero; cannot form relative values.")
#         events = 100.0 * events_abs / mean_val
#     else:
#         events = events_abs

#     pre_raw  = df_pre[forcing_column].values[::-1]
#     bins_pre = np.histogram_bin_edges(pre_raw, bins=nbins_pre)

#     # ----------------------------------------------------------- figure setup
#     fig = plt.figure(figsize=(12, 5))
#     gs  = fig.add_gridspec(2, 1, height_ratios=[0.9, 1.3], hspace=0)

#     # ── (a) top: pre, with horizontal bands for bins ────────────────────────
#     ax0 = fig.add_subplot(gs[0])

#     cmap = cm.get_cmap(cmap_name, nbins_pre)
#     for i in range(nbins_pre):
#         y0, y1 = bins_pre[i], bins_pre[i + 1]
#         ax0.axhspan(y0, y1,
#                     facecolor=cmap(i),
#                     alpha=0.5,
#                     zorder=0,
#                     edgecolor='none')
#         ax0.axhline(bins_pre[i], color='white', lw=0.8)

#     ax0.plot(t, pre_raw, color='C0', zorder=2)
#     ax0.set_ylabel(forcing_column)
#     ax0.set_xlim(t[1], t[-1])
#     ax0.tick_params(axis='x', labelbottom=False)
#     ax0.set_yticks(bins_pre)
#     ax0.set_yticklabels([f'{b:.2g}' for b in bins_pre])

#     # ── (b) bottom: expected events ─────────────────────────────────────────
#     ax1 = fig.add_subplot(gs[1], sharex=ax0)
#     ax1.plot(t[1:], events,
#              color='C3',
#              label=(
#                  f'Expected # flips / {window:,} yr'
#                  if not relative else
#                  'Relative # flips (mean = 100 %)'
#              ))
#     ax1.set_xlabel(time_column)
#     if relative:
#         ax1.set_ylabel('Events (% of mean)')
#         # optional visual guide at 100 %
#         ax1.axhline(100, color='grey', ls='--', lw=0.8, zorder=0)
#     else:
#         ax1.set_ylabel('Events (count)')
#     ax1.legend(loc='upper right')

#     plt.show()
#     return events



















import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def flip_events(df_pre,
                flip_prob,
                window=20_000,               # yrs
                forcing_column='pre',
                time_column='age',
                nbins_pre=4,
                cmap_name='spring',
                relative=True):
    """
    As before, but background bins are drawn as VERTICAL bands
    in both sub-plots instead of horizontal strips.
    """
    # ------------------------------- data & basic series --------------------
    t  = df_pre[time_column].values[::-1]          # old → young
    dt = float(np.abs(np.median(np.diff(t))))      # median spacing (yrs)

    events_abs = (window * flip_prob) / dt         # expected N flips / sample

    if relative:
        mean_val = events_abs.mean()
        if mean_val == 0:
            raise ValueError("Mean of events is zero; cannot normalise.")
        events = 100.0 * events_abs / mean_val
    else:
        events = events_abs

    pre_raw  = df_pre[forcing_column].values[::-1]
    bins_pre = np.histogram_bin_edges(pre_raw, bins=nbins_pre)
    bin_idx  = np.digitize(pre_raw, bins_pre, right=False) - 1   # 0 … nbins_pre-1
    cmap     = cm.get_cmap(cmap_name, nbins_pre)

    # ------------------------------- figure ---------------------------------
    fig = plt.figure(figsize=(12, 5))
    gs  = fig.add_gridspec(2, 1, height_ratios=[0.9, 1.3], hspace=0)

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)

    # ---------- draw VERTICAL colour bands on both axes --------------------
    def _add_vertical_bands(ax):
        start = 0
        for i in range(1, len(t)):
            if bin_idx[i] != bin_idx[i-1]:
                # flush previous segment
                x0, x1 = t[start], t[i]
                ax.axvspan(min(x0, x1), max(x0, x1),
                           facecolor=cmap(bin_idx[i-1]),
                           alpha=0.35,
                           edgecolor='none',
                           zorder=0)
                start = i
        # flush final segment
        x0, x1 = t[start], t[-1]
        ax.axvspan(min(x0, x1), max(x0, x1),
                   facecolor=cmap(bin_idx[-1]),
                   alpha=0.35,
                   edgecolor='none',
                   zorder=0)

    _add_vertical_bands(ax0)
    _add_vertical_bands(ax1)

    # ---------- upper panel: pre -------------------------------------------
    ax0.plot(t, pre_raw, color='C0', zorder=2)
    ax0.set_ylabel(forcing_column)
    ax0.set_xlim(t[1], t[-1])
    ax0.tick_params(axis='x', labelbottom=False)

    # y-ticks at bin edges for reference
    ax0.set_yticks(bins_pre)
    ax0.set_yticklabels([f'{b:.2g}' for b in bins_pre])

    # ---------- lower panel: expected flips --------------------------------
        # ---------- lower panel: expected flips --------------------------------
    # make the two series equal length
    tt = t[-len(events):]            # take the tail of t that matches events

    ax1.plot(tt, events, color='C3',
             label=('Relative # flips (mean = 100 %)'
                    if relative else
                    f'Expected # flips / {window:,} yr'))

    # ax1.plot(t, events[1:], color='C3',
    #          label=('Relative # flips (mean = 100 %)'
    #                 if relative else
    #                 f'Expected # flips / {window:,} yr'))
    ax1.set_xlabel(time_column)
    if relative:
        ax1.set_ylabel('Events (% of mean)')
        ax1.axhline(100, color='grey', ls='--', lw=0.8, zorder=0)
    else:
        ax1.set_ylabel('Events (count)')
    ax1.legend(loc='upper right')

    plt.show()
    return tt,events























# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import cm, colors

# def flip_events(df_pre,
#                 flip_prob,
#                 window=20_000,            # yrs
#                 forcing_column='pre',
#                 time_column='age',
#                 nbins_pre=4,
#                 band_alpha=0.25,          # transparency of the colour bands
#                 show_bin_labels=False):   # annotate 0,1,2… in each band
#     """
#     Expected # flip events per `window`, with an improved background
#     showing pre-bins as horizontal colour bands.

#     Returns
#     -------
#     events : 1-D numpy array, same length as `flip_prob`.
#     """
#     # ─────────────────────────── data prep ────────────────────────────────
#     t  = df_pre[time_column].values[::-1]                # old → young
#     dt = float(np.abs(np.median(np.diff(t))))            # median Δt

#     events = (window * flip_prob) / dt                   # N = W * p / dt

#     pre_raw  = df_pre[forcing_column].values[::-1]
#     bins_pre = np.histogram_bin_edges(pre_raw, bins=nbins_pre)

#     # ─────────────────────────── figure ───────────────────────────────────
#     fig = plt.figure(figsize=(12, 5))
#     gs  = fig.add_gridspec(2, 1, height_ratios=[0.9, 1.3], hspace=0)

#     # ── top panel: pre with colour bands ──────────────────────────────────
#     ax0 = fig.add_subplot(gs[0])

#     # choose a qualitative colormap with clearly separated hues
#     cmap = cm.get_cmap('tab20c', nbins_pre)

#     # draw coloured horizontal bands
#     for i in range(nbins_pre):
#         y0, y1 = bins_pre[i], bins_pre[i + 1]
#         ax0.axhspan(y0, y1,
#                     facecolor=cmap(i),
#                     alpha=band_alpha,
#                     zorder=0)

#     # draw thin boundary lines to emphasise separation
#     for y in bins_pre:
#         ax0.axhline(y, color='white', lw=0.8, zorder=1)

#     # pre signal on top
#     ax0.plot(t, pre_raw, color='black', lw=1.4, zorder=2)
#     ax0.set_ylabel(forcing_column)
#     ax0.set_xlim(t[1], t[-1])
#     ax0.tick_params(axis='x', labelbottom=False)

#     # optional: label each band with its bin index
#     if show_bin_labels:
#         for i in range(nbins_pre):
#             y0, y1 = bins_pre[i], bins_pre[i + 1]
#             ax0.text(0.98, (y0 + y1) / 2, str(i),
#                      transform=ax0.get_yaxis_transform(),
#                      ha='right', va='center',
#                      fontsize='x-small',
#                      color='white', alpha=0.9,
#                      zorder=3)

#     # ── bottom panel: expected events ─────────────────────────────────────
#     ax1 = fig.add_subplot(gs[1], sharex=ax0)
#     ax1.plot(t[1:], events,
#              color='C3',
#              label=f'Expected # flips / {window:,} yr')
#     ax1.set_xlabel(time_column)
#     ax1.set_ylabel('Events (count)')
#     ax1.legend(loc='upper right')

#     plt.show()
#     return events

































import numpy as np
import matplotlib.pyplot as plt
from pyinform import transfer_entropy

def transfer_entropy_surrogate_test_v2(
    forcing, sq, k=1,
    forcing_bins=4, sq_bins=2,
    n_surr=100, p=0.05, use_quantile=False,
    if_plot=True, dpi=100
):
    """
    Test for unidirectional causality using transfer entropy and phase-randomized surrogates of `sq`.
    """
    # Reverse series
    x = np.asarray(forcing)[::-1]
    y = np.asarray(sq)[::-1]

    # Discretize data
    if use_quantile:
        xbins = np.quantile(x, np.linspace(0, 1, forcing_bins + 1))
        ybins = np.quantile(y, np.linspace(0, 1, sq_bins + 1))
    else:
        xbins = np.histogram_bin_edges(x, bins=forcing_bins)
        ybins = np.histogram_bin_edges(y, bins=sq_bins)

    x_disc = np.digitize(x, xbins) - 1
    y_disc = np.digitize(y, ybins) - 1

    # Empirical TE
    te_xy = transfer_entropy(x_disc[:-1], y_disc[1:], k=k)
    te_yx = transfer_entropy(y_disc[:-1], x_disc[1:], k=k)

    # Surrogate nulls via phase randomization of `y`
    null_xy = np.zeros(n_surr)
    null_yx = np.zeros(n_surr)
    N = len(y)
    # compute FFT once for original magnitude
    y_fft_orig = np.fft.rfft(y)
    mag = np.abs(y_fft_orig)
    for i in range(n_surr):
        # randomize phases
        phases = np.angle(y_fft_orig)
        random_phases = np.random.uniform(0, 2*np.pi, size=phases.shape)
        # preserve DC and Nyquist (if present)
        random_phases[0] = phases[0]
        if N % 2 == 0:
            random_phases[-1] = phases[-1]
        # build surrogate frequency domain
        y_surr_fft = mag * np.exp(1j * random_phases)
        # inverse FFT
        y_surr = np.fft.irfft(y_surr_fft, n=N)
        # discretize surrogate
        y_surr_disc = np.digitize(y_surr, ybins) - 1
        # compute TE for surrogate
        null_xy[i] = transfer_entropy(x_disc[:-1], y_surr_disc[1:], k=k)
        null_yx[i] = transfer_entropy(y_surr_disc[:-1], x_disc[1:], k=k)

    # p-values
    p_xy = (np.sum(null_xy >= te_xy) + 1) / (n_surr + 1)
    p_yx = (np.sum(null_yx >= te_yx) + 1) / (n_surr + 1)

    fig = None
    if if_plot:
        fig = plt.figure(figsize=(5, 3.5), dpi=dpi)
        plt.hist(null_xy, bins=25, alpha=0.7, label='Null TE (forcing→sq)', edgecolor='white', color = 'darkred')
        plt.axvline(te_xy, lw=2, label=f'TE (forcing→sq), p={p_xy:.3f}', color='darkred')
        plt.hist(null_yx, bins=25, alpha=0.7, label='Null TE (sq→forcing)', edgecolor='white', color = 'skyblue')
        plt.axvline(te_yx, lw=2, label=f'TE (sq→forcing), p={p_yx:.3f}', color='skyblue')
        plt.xlabel('Transfer Entropy (bits)')
        plt.ylabel('Count')
        plt.legend(loc='upper right', frameon=True)
        plt.tight_layout()
        plt.show()

    # significance
    sig_xy = p_xy < p
    sig_yx = p_yx < p
    return sig_xy and not sig_yx, fig











import numpy as np
import matplotlib.pyplot as plt
from pyinform import transfer_entropy
from sklearn.cluster import KMeans

def transfer_entropy_surrogate_test(
    forcing, sq, k=1,
    forcing_bins=4, sq_bins=2,
    n_surr=100, p=0.05, 
    sq_method='hist',  # options: 'hist', 'quantile', 'kmeans'
    binary=False,
    if_plot=True, dpi=100
):
    """
    Test for unidirectional causality using transfer entropy and surrogates.
    """
    # Reverse series
    x = np.asarray(forcing)[::-1]
    y = np.asarray(sq)[::-1]


    # Discretize sq series
    if sq_method == 'quantile':
        xbins = np.quantile(x, np.linspace(0, 1, forcing_bins+1))
        ybins = np.quantile(y, np.linspace(0, 1, sq_bins+1))
        y_disc = np.digitize(y, ybins) - 1
    elif sq_method == 'hist':
        xbins = np.histogram_bin_edges(x, bins=forcing_bins)
        ybins = np.histogram_bin_edges(y, bins=sq_bins)
        y_disc = np.digitize(y, ybins) - 1

    elif sq_method == 'kmeans':
        xbins = np.histogram_bin_edges(x, bins=forcing_bins)
        # Fit k-means on the values
        km = KMeans(n_clusters=sq_bins, n_init=10, random_state=0)
        y_disc = km.fit_predict(y.reshape(-1, 1))
    else:
        raise ValueError(f"Unknown sq_method '{sq_method}'. Use 'hist', 'quantile', or 'kmeans'.")

    if binary:
        y_disc = y

    # xbins = np.histogram_bin_edges(x, bins=forcing_bins)
    # xbins = np.histogram_bin_edges(x, bins=forcing_bins)
    x_disc = np.digitize(x, xbins) - 1

    
    # Empirical TE (one-step, history k)
    # te_xy = transfer_entropy(x_disc[:-1], y_disc[1:], k=k)
    # te_yx = transfer_entropy(y_disc[:-1], x_disc[1:], k=k)
    te_xy = transfer_entropy(x_disc, y_disc, k=k)
    te_yx = transfer_entropy(y_disc, x_disc, k=k)
    
    # Surrogate nulls
    null_xy = np.zeros(n_surr)
    null_yx = np.zeros(n_surr)
    for i in range(n_surr):
        xs = np.random.permutation(x_disc)
        # null_xy[i] = transfer_entropy(xs[:-1], y_disc[1:], k=k)
        null_xy[i] = transfer_entropy(xs, y_disc, k=k)
        ys = np.random.permutation(y_disc)
        # null_yx[i] = transfer_entropy(ys[:-1], x_disc[1:], k=k)
        null_yx[i] = transfer_entropy(ys, x_disc, k=k)
    
    # p-values
    p_xy = (np.sum(null_xy >= te_xy) + 1) / (n_surr + 1)
    p_yx = (np.sum(null_yx >= te_yx) + 1) / (n_surr + 1)
    
    fig=[]
    # Plot if requested
    if if_plot:
        fig = plt.figure(figsize=(5, 3.5), dpi=dpi)

        # forcing → sq histogram
        plt.hist(
            null_xy, bins=25,
            color='#CC6677',       # light blue
            alpha=0.7,
            label='Null TE (forcing→sq)',
            edgecolor='white'
        )
        # forcing → sq empirical line
        plt.axvline(
            te_xy,
            color='#882255',       # teal
            lw=2,
            label=f'TE (forcing→sq), p={p_xy:.3f}'
        )
        # sq → forcing histogram
        plt.hist(
            null_yx, bins=25,
            color='#88CCEE',       # pink-red
            alpha=0.7,
            label='Null TE (sq→forcing)',
            edgecolor='white'
        )
        # sq → forcing empirical line
        plt.axvline(
            te_yx,
            color='#44AA99',       # dark magenta
            lw=2,
            label=f'TE (sq→forcing), p={p_yx:.3f}'
        )
        # set the line width of spines
        for spine in plt.gca().spines.values():
            spine.set_linewidth(1.5)

        plt.xlabel('Transfer Entropy (bits)')
        plt.ylabel('Count')
        plt.legend(loc='upper right', frameon=True)
        plt.tight_layout()
        plt.show()
    
    # Determine unidirectional significance
    sig_xy = p_xy < p
    sig_yx = p_yx < p
    return sig_xy and not sig_yx, fig, te_xy






import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Helper: shift two arrays by an integer number of samples
# ------------------------------------------------------------------
def _shift_pair(driver, target, lag_steps):
    if lag_steps > 0:          # driver leads
        return driver[:-lag_steps], target[lag_steps:]
    elif lag_steps < 0:        # driver lags
        return driver[-lag_steps:], target[:lag_steps]
    else:                      # no shift
        return driver, target


# ------------------------------------------------------------------
# Compute TE(driver→target) across a vector of lags
# ------------------------------------------------------------------
def _te_vs_lag(driver_series, target_series, lags, step, **te_kwargs):
    lag_steps = (np.asarray(lags) / step).astype(int)
    te_vals, sig = [], []

    for s in lag_steps:
        x, y = _shift_pair(driver_series, target_series, s)

        if len(x) < 20:            # avoid very short overlap
            te_vals.append(np.nan)
            sig.append(False)
            continue

        is_sig, _, te_xy = sa.transfer_entropy_surrogate_test(
            x, y, if_plot=False, **te_kwargs
        )
        te_vals.append(te_xy)
        sig.append(is_sig)

    return np.array(te_vals), np.array(sig)


# ------------------------------------------------------------------
# PUBLIC FUNCTION ---------------------------------------------------
def run_lag_te(
    pre, obl, sq,
    STEP      = 200,       # sampling interval in *years*
    max_lag   = 2000,      # ± window (years)
    plot      = True,
    **te_kw                    # forwarded to transfer_entropy_surrogate_test
):
    """
    Sweep lead/lag for Precession and Obliquity drivers and (optionally) plot.
    Parameters
    ----------
    pre, obl : 1-D numpy arrays
        Orbital driver series (already resampled to STEP).
    sq       : 1-D numpy array
        Target MCV series.
    STEP     : int
        Sampling interval in years.
    max_lag  : int
        Maximum absolute lag to test (years).
    plot     : bool
        Produce the two-panel matplotlib figure?
    te_kw    : keyword args passed straight to `sa.transfer_entropy_surrogate_test`.
    Returns
    -------
    results : dict
        {'Precession': {'lags': ..., 'te': ..., 'sig': ...},
         'Obliquity' : {...}}
    fig : matplotlib Figure or None
    """
    # ---------------- lag grid -------------------------------------
    lags = np.arange(-max_lag, max_lag + STEP, STEP)

    # ---------------- compute TE curves ----------------------------
    te_pre, sig_pre = _te_vs_lag(pre, sq, lags, STEP, **te_kw)
    te_obl, sig_obl = _te_vs_lag(obl, sq, lags, STEP, **te_kw)

    results = {
        'Precession': {'lags': lags, 'te': te_pre, 'sig': sig_pre},
        'Obliquity' : {'lags': lags, 'te': te_obl, 'sig': sig_obl},
    }

    # ---------------- plotting -------------------------------------
    fig = None
    if plot:
        fig, axes = plt.subplots(nrows=2, sharex=True, figsize=(4, 6))

        # --- panel 1: Precession -----------------------------------
        ax = axes[0]
        ax.plot(lags, te_pre, lw=2, color='C0')
        ax.scatter(lags[sig_pre], te_pre[sig_pre],
                   s=55, marker='o', facecolor='C0',
                   edgecolor='k', zorder=3, label='p < α')
        ax.axvline(0, color='k', lw=0.8, ls='--')
        ax.set_title('Precession → MCV')
        ax.set_ylabel('TE (bits)')
        ax.legend(loc='upper right', frameon=True)

        # --- panel 2: Obliquity ------------------------------------
        ax = axes[1]
        ax.plot(lags, te_obl, lw=2, color='C1')
        ax.scatter(lags[sig_obl], te_obl[sig_obl],
                   s=55, marker='s', facecolor='C1',
                   edgecolor='k', zorder=3, label='p < α')
        ax.axvline(0, color='k', lw=0.8, ls='--')
        ax.set_title('Obliquity → MCV')
        ax.set_xlabel('Lead / lag (years)\n(positive = driver leads target)')
        ax.set_ylabel('TE (bits)')
        ax.legend(loc='upper right', frameon=True)

        # tidy up axes
        for ax in axes:
            ax.set_xlim(lags.min(), lags.max())
            ax.spines[['top', 'right']].set_visible(False)

        # NOTE: tight_layout() intentionally omitted (user request)

    return results, fig















import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree            
from matplotlib import colormaps
from matplotlib import cm
import matplotlib.colors as mcolors            # ← 1) bring in `colors`


# ───────────────────────── helper: delay-coordinate embedding ──────────────
def _embed(series, E=3, tau=1):
    """Return (matrix, valid_indices)."""
    series = np.asarray(series)
    N = len(series)
    idx = np.arange((E-1)*tau, N)               # times with full history
    mat = np.column_stack([series[idx - j*tau] for j in range(E)])
    return mat, idx


# ───────────────────────────── main CCM-style predictor ────────────────────
def predict_events_future_ccm(df_pre_train,       # historical pre
                              events_train,       # historical events / window
                              df_pre_future,      # future pre
                              window=20_000,      # yr, just meta for label
                              E=3, tau=1, k=None,
                              band_sigma=1.0,     # width of uncertainty band
                              forcing_column='pre',
                              time_column='age',
                              cmap_name='tab20c',
                              panel_label=None,
                              save_fig=False,
                              proxy_name=None,
                              xmax=20_000,
                              dpi=100,
                              ymin=75,
                              y_max=130,
                              show_cb=False,
                              show_legend=False,
                              
                              ):
    """
    Forecast future event counts using a simplex/CCM-style nearest-neighbour
    mapping in precession state-space.

    Returns: (mean, lower, upper) as NumPy arrays aligned to df_pre_future.
    """
    # -------- 0) housekeeping ------------------------------------------------
    if k is None:
        k = E + 1                                # standard simplex choice

    # -------- 1) embed training forcing -------------------------------------
    pre_train = df_pre_train[forcing_column].values[::-1]
    X_train, idx_train = _embed(pre_train, E, tau)

    # align events to embedding indices
    events_train = np.asarray(events_train)
    events_train_emb = events_train[idx_train]   # same length as X_train

    # build KD-tree for fast neighbour search
    tree = cKDTree(X_train)

    # -------- 2) embed future forcing ---------------------------------------
    # pre_future = df_pre_future[forcing_column].values[::-1]
    # t_future   = df_pre_future[time_column].values[::-1]

    pre_future = df_pre_future[forcing_column].values
    t_future   = df_pre_future[time_column].values

    X_future, idx_future = _embed(pre_future, E, tau)

    # -------- 3) nearest-neighbour predictions ------------------------------
    dists, neigh_idx = tree.query(X_future, k=k)        # shape (M,k)
    # distance of closest neighbour, avoid zero
    d1 = dists[:, 0:1] + 1e-12

    # weights: exp( −d / d1 )
    w = np.exp(-dists / d1)

    # weighted mean
    neigh_events = events_train_emb[neigh_idx]          # shape (M,k)
    events_mean = (w * neigh_events).sum(axis=1) / w.sum(axis=1)

    # weighted variance
    var = (w * (neigh_events - events_mean[:, None])**2).sum(axis=1) / w.sum(axis=1)
    sigma = np.sqrt(var)

    events_lower = np.clip(events_mean - band_sigma * sigma, 0, None)
    events_upper = events_mean + band_sigma * sigma

    # pad the first (E-1)*tau points with NaN so arrays align to df_pre_future
    pad = np.full((pre_future.size,), np.nan)
    pad[idx_future] = events_mean
    events_mean = pad

    pad = np.full_like(events_mean, np.nan)
    pad[idx_future] = events_lower
    events_lower = pad

    pad = np.full_like(events_mean, np.nan)
    pad[idx_future] = events_upper
    events_upper = pad

    # ------------------------------------------------------------------ plot ----
   
    nbins_pre = 6
    bins_pre  = np.histogram_bin_edges(pre_train, bins=nbins_pre)

    fig = plt.figure(figsize=(4, 5),dpi=dpi)
    gs  = fig.add_gridspec(2, 1, height_ratios=[1, 3], hspace=0.0)

    # ── (a) future pre with *vertical* colour bands ────────────────────────────
    ax0 = fig.add_subplot(gs[0])
    cmap = cm.get_cmap(cmap_name, nbins_pre)

    # assign each time step to a bin
    bin_idx = np.digitize(pre_future, bins_pre, right=False) - 1

    # draw contiguous vertical spans with same bin colour
    start = 0
    for i in range(1, len(t_future)):
        if bin_idx[i] != bin_idx[i - 1]:
            ax0.axvspan(t_future[start], t_future[i],
                        facecolor=cmap(bin_idx[i - 1]), alpha=0.35, edgecolor='none')
            start = i
    # flush last segment
    ax0.axvspan(t_future[start], t_future[-1],
                facecolor=cmap(bin_idx[-1]), alpha=0.35, edgecolor='none')



    ax0.plot(t_future, pre_future, color='black', lw=1.4, zorder=2)
    ax0.set_ylabel(forcing_column)
    ax0.set_xlim(t_future[0], t_future[-1])
    ax0.tick_params(axis='x', labelbottom=False)
    # add the panel label if provided
    if panel_label is not None:
        ax0.text(-0.1, 1.1, panel_label, transform=ax0.transAxes,
                 fontsize=12, fontweight='bold', va='top', ha='left')
                 




    # ── (b) forecasted events (unchanged) ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax1.plot(t_future, events_mean, color='C3', label='Forecast')
    ax1.axhline(np.mean(events_train), color='C0', ls='--',
                label='Mean historical events')
    ax1.fill_between(t_future, events_lower, events_upper,
                    color='C3', alpha=0.3,
                    label=f'±{band_sigma:.0f} σ')

    # set y-limits
    ax1.set_ylim(ymin, y_max)
    ax1.set_xlabel(time_column)
    ax1.set_ylabel(f'Events / {window:,} yr, % of mean')
    if show_legend:
        ax1.legend(loc='lower right')

    # set the xlim between 0 to xmax
    ax1.set_xlim(0, xmax)


    # if show_cb, plot a verticle colorbar for the vertical spans
    # ── add colour-bar ---------------------------------------------------------
    # if show_cb:

    #     # discrete mapping: one colour per precession bin
    #     norm = mcolors.BoundaryNorm(bins_pre, cmap.N)
    #     sm   = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    #     sm.set_array([])                # Matplotlib ≤3.8 requirement

    #     # ------------------------------------------------------------------
    #     # 1. Build a *floating* colour-bar axis that does NOT steal space
    #     #    from the main sub-plots.
    #     #    Place it to the right of the figure, vertically centred on
    #     #    the combined height of ax0 & ax1.
    #     # ------------------------------------------------------------------
    #     box0 = ax0.get_position()       # bounding boxes of the two panels
    #     box1 = ax1.get_position()

    #     y0   = box1.y0                  # bottom of lower panel
    #     y1   = box0.y1                  # top of upper panel
    #     cbar_height = y1 - y0           # span both panels
    #     cbar_width  = 0.02              # relative figure width
    #     cbar_xpad   = 0.02              # gap between plots and colour-bar

    #     cax = fig.add_axes([box1.x1 + cbar_xpad,   # left
    #                         y0,                    # bottom
    #                         cbar_width,            # width
    #                         cbar_height])          # height

    #     # 2. Draw *discrete* colour-bar
    #     #    - `boundaries` + `norm` give stepwise colours
    #     #    - ticks at bin mid-points look cleaner
    #     mid = 0.5 * (bins_pre[:-1] + bins_pre[1:])

    #     cbar = fig.colorbar(sm,
    #                         cax=cax,
    #                         boundaries=bins_pre,
    #                         spacing='proportional')   # -> blocky steps
    #     cbar.set_ticks(mid)
    #     cbar.set_ticklabels([f'{m:.0f}' for m in mid])  # optional formatting
    #     cbar.ax.set_ylabel(forcing_column)
    if show_cb:
        # import matplotlib.colors as mcolors

        # ── discrete, *opaque* colours
        norm = mcolors.BoundaryNorm(bins_pre, cmap.N)
        sm   = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        box0 = ax0.get_position()       # bounding boxes of the two panels
        box1 = ax1.get_position()

        y0   = box1.y0                  # bottom of lower panel
        y1   = box0.y1                  # top of upper panel
        cbar_height = y1 - y0           # span both panels
        cbar_width  = 0.02              # relative figure width
        cbar_xpad   = 0.02              # gap between plots and colour-bar

        cax = fig.add_axes([box1.x1 + cbar_xpad,   # left
                            y0,                    # bottom
                            cbar_width,            # width
                            cbar_height])          # height

        # 2. Draw *discrete* colour-bar
        #    - `boundaries` + `norm` give stepwise colours
        #    - ticks at bin mid-points look cleaner
        mid = 0.5 * (bins_pre[:-1] + bins_pre[1:])

        cbar = fig.colorbar(
            sm, cax=cax, boundaries=bins_pre, spacing='proportional'
        )

        # ──────────────────────────────────────────────────────────────
        # 1. Make the colour blocks *match the span-alpha* (e.g. 0.35)
        #    This touches the PolyCollection that fills the bar.
        cbar.solids.set_alpha(0.35)          # or your preferred value
        # ──────────────────────────────────────────────────────────────
        # 2. Relabel ticks 1 … nbins_pre instead of showing data mid-points
        mid = 0.5 * (bins_pre[:-1] + bins_pre[1:])
        cbar.set_ticks(mid)
        cbar.set_ticklabels([str(i + 1) for i in range(nbins_pre)])
        # ──────────────────────────────────────────────────────────────

        cbar.ax.set_ylabel(forcing_column)   # keep or modify as needed


    plt.show()

    # save figure as .pdf vector to /Figure_saved, merge the proxy name into the file name
    if save_fig and proxy_name is not None:
        fig.savefig(f'Figure_saved/{proxy_name}_ccm_forecast.pdf', bbox_inches='tight')
        print(f"Figure saved as 'Figure_saved/{proxy_name}_ccm_forecast.pdf'")

    return events_mean, events_lower, events_upper























import numpy as np
import matplotlib.pyplot as plt
from pyinform import transfer_entropy
from sklearn.cluster import KMeans
from numpy.fft import rfft, irfft

# ─────────────────────────────────────────────────────────────
# Helpers for spectrum-preserving surrogates
# ─────────────────────────────────────────────────────────────
def _phase_randomised(x):
    """Single phase-randomised surrogate that preserves power spectrum."""
    n = len(x)
    # Real-FFT (one-sided)
    Xf = rfft(x)
    # Random phases for positive-freq coefficients except DC & Nyquist
    random_ph = np.exp(1j * 2 * np.pi * np.random.rand(len(Xf)))
    random_ph[0] = 1.0                          # keep DC
    if n % 2 == 0:
        random_ph[-1] = 1.0                    # keep Nyquist if it exists
    Xf_surr = np.abs(Xf) * random_ph
    return irfft(Xf_surr, n)

def _iaaft(x, n_iter=1000, atol=1e-8):
    """
    Iterative Amplitude Adjusted FT surrogate.
    Preserves both amplitude distribution and power spectrum.
    """
    # initial random permutation
    s = np.random.permutation(x)
    amp_orig = np.sort(x)
    spec_orig = np.abs(rfft(x))

    for _ in range(n_iter):
        # impose power spectrum
        S = rfft(s)
        S = spec_orig * np.exp(1j * np.angle(S))
        s = irfft(S, len(x))
        # impose amplitude distribution
        ranks = np.argsort(s)
        s[ranks] = amp_orig
        # optional early stop
        if np.linalg.norm(np.abs(rfft(s)) - spec_orig) < atol:
            break
    return s
# ─────────────────────────────────────────────────────────────


def _digitize_safe(arr, edges, n_bins):
    """Return integer states in 0 … n_bins-1, guaranteed non-negative."""
    return np.clip(np.digitize(arr, edges) - 1, 0, n_bins - 1).astype(np.int32)

def transfer_entropy_specsurrogate_test(
    forcing, sq, k=1,
    forcing_bins=4, sq_bins=2,
    n_surr=100, p=0.05,
    sq_method='hist',
    binary=False,
    surrogate_type='iaaft',
    n_iter_iaaft=1000,
    if_plot=True, dpi=100
):
    # 1  reverse time axis
    x = np.asarray(forcing)[::-1].astype(float)
    y = np.asarray(sq)[::-1].astype(float)

    # 2  choose bin edges *once* from the original series
    xbins = np.histogram_bin_edges(x, bins=forcing_bins)

    if sq_method == 'quantile':
        ybins = np.quantile(y, np.linspace(0, 1, sq_bins + 1))
    elif sq_method == 'hist':
        ybins = np.histogram_bin_edges(y, bins=sq_bins)
    elif sq_method == 'kmeans':
        km = KMeans(n_clusters=sq_bins, n_init=10, random_state=0)
        y_disc = km.fit_predict(y.reshape(-1, 1))
        ybins = None  # not used later
    else:
        raise ValueError("sq_method must be 'hist', 'quantile', or 'kmeans'")

    # 3  discretise
    if sq_method in ('hist', 'quantile'):
        y_disc = _digitize_safe(y, ybins, sq_bins)
    if binary:
        y_disc = y.astype(np.int32)

    x_disc = _digitize_safe(x, xbins, forcing_bins)

    # 4  empirical TE
    te_xy = transfer_entropy(x_disc, y_disc, k=k)
    te_yx = transfer_entropy(y_disc, x_disc, k=k)

    # 5  surrogates
    null_xy = np.empty(n_surr)
    null_yx = np.empty(n_surr)

    for i in range(n_surr):
        if surrogate_type == 'shuffle':
            xs = np.random.permutation(x)
        elif surrogate_type == 'phase':
            xs = _phase_randomised(x)
        elif surrogate_type == 'iaaft':
            xs = _iaaft(x, n_iter=n_iter_iaaft)
        else:
            raise ValueError("surrogate_type must be 'shuffle', 'phase', 'iaaft'")

        xs_disc = _digitize_safe(xs, xbins, forcing_bins)
        null_xy[i] = transfer_entropy(xs_disc, y_disc, k=k)

        ys = np.random.permutation(y_disc)
        null_yx[i] = transfer_entropy(ys, x_disc, k=k)

    # 6  p-values
    p_xy = (null_xy >= te_xy).sum() / n_surr
    p_yx = (null_yx >= te_yx).sum() / n_surr

    fig = None
    if if_plot:
        fig = plt.figure(figsize=(5, 3.5), dpi=dpi)

        plt.hist(null_xy, bins=25, alpha=0.7, color='#CC6677',
                 label=f'Null TE ({surrogate_type} forcing→sq)',
                 edgecolor='white')
        plt.axvline(te_xy, color='#882255', lw=2,
                    label=f'TE (forcing→sq), p={p_xy:.3f}')

        plt.hist(null_yx, bins=25, alpha=0.7, color='#88CCEE',
                 label='Null TE (sq→forcing)', edgecolor='white')
        plt.axvline(te_yx, color='#44AA99', lw=2,
                    label=f'TE (sq→forcing), p={p_yx:.3f}')

        for spine in plt.gca().spines.values():
            spine.set_linewidth(1.5)
        plt.xlabel('Transfer Entropy (bits)')
        plt.ylabel('Count')
        plt.legend(frameon=True)
        plt.tight_layout()
        plt.show()

    sig_xy = p_xy < p
    sig_yx = p_yx < p
    return sig_xy and not sig_yx, fig, te_xy































































def create_shift_forcing(df_sq, interval, if_plot=False,shift=-10000):
    """
    Resample the square‐wave and load & resample
    the precession & obliquity forcing to a common age grid.

    Parameters
    ----------
    df_sq : pandas.DataFrame
        Must have columns ['age', 'sq'].
    interval : float
        Desired age‐step for the resampled grid.
    if_plot : bool, default False
        If True, plots the three resampled series.

    Returns
    -------
    df_sq_resampled, df_pre_resampled, df_obl_resampled : DataFrames
        Each has columns ['age', <variable>] on the same age grid.
    """
    # 1) load raw precession & obliquity
    pre_path = r"D:\VScode\bipolar_seesaw_CCM\inso_data\pre_800_inter100.txt"
    obl_path = r"D:\VScode\bipolar_seesaw_CCM\inso_data\obl_800_inter100.txt"
    df_pre_raw = pd.read_csv(pre_path, sep=r'\s+', header=None, engine='python')
    df_obl_raw = pd.read_csv(obl_path, sep=r'\s+', header=None, engine='python')

    # convert to years & ensure age increasing
    df_pre_raw.iloc[:,0] = df_pre_raw.iloc[:,0].abs() * 1000
    df_obl_raw.iloc[:,0] = df_obl_raw.iloc[:,0].abs() * 1000
    df_pre_raw = df_pre_raw.iloc[::-1].reset_index(drop=True)
    df_obl_raw = df_obl_raw.iloc[::-1].reset_index(drop=True)
    df_pre_raw.columns = ['age','pre']
    df_obl_raw.columns = ['age','obl']

    df_pre_raw['age'] = df_pre_raw['age'].values + shift
    df_obl_raw['age'] = df_obl_raw['age'].values + shift


    # 2) compute overlapping age bounds
    a_min = max(df_sq['age'].min(),
                df_pre_raw['age'].min(),
                df_obl_raw['age'].min())
    a_max = min(df_sq['age'].max(),
                df_pre_raw['age'].max(),
                df_obl_raw['age'].max())

    # 3) create unified age vector
    new_age = np.arange(a_min, a_max + 1, interval)

    # 4) interpolate each series onto new_age
    def interp(df, col):
        f = interp1d(df['age'], df[col],
                     kind='nearest',
                     bounds_error=False,
                     fill_value="extrapolate")
        return f(new_age)
    
    column_names = df_sq.columns[1]

    df_sq_rs  = pd.DataFrame({'age': new_age,
                              column_names: interp(df_sq, column_names)})
    df_pre_rs = pd.DataFrame({'age': new_age,
                              'pre': interp(df_pre_raw, 'pre')})
    df_obl_rs = pd.DataFrame({'age': new_age,
                              'obl': interp(df_obl_raw, 'obl')})

    # 5) optional plotting
    if if_plot:
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, tight_layout=True)
        axes[0].plot(df_sq_rs['age'],  df_sq_rs[column_names],  color='black', label='sq')
        axes[1].plot(df_pre_rs['age'], df_pre_rs['pre'], color='blue',  label='pre')
        axes[2].plot(df_obl_rs['age'], df_obl_rs['obl'], color='green', label='obl')
        for ax in axes:
            ax.legend()
            ax.set_ylabel(ax.get_label())
        axes[-1].set_xlabel('Age (years)')
        plt.show()

    return df_sq_rs, df_pre_rs, df_obl_rs















import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def interpolate_data_lr04(
    df_sq,
    lr04_path=r"D:\VScode\bipolar_seesaw_CCM\other_data\lr04.xlsx",
    sheet_name='Sheet1',
    if_plot=False
):
    """
    Interpolate the LR04 global benthic δ18O stack onto the
    age grid already contained in *df_sq* (e.g., the filtered CH4 grid).

    Parameters
    ----------
    df_sq : pandas.DataFrame
        Must contain an 'age' column (yr BP).  Only the age axis is used.
    lr04_path : str
        Full path to lr04.xlsx (two columns: age [kyr], d18O).
    sheet_name : str or int
        Worksheet index / name inside the Excel file.
    if_plot : bool
        When True, show the raw LR04 series and the interpolated version
        on the df_sq grid.

    Returns
    -------
    df_lr04_interp : pandas.DataFrame
        Columns ['age', 'd18O'] on exactly the same age grid as df_sq.
    """
    # ---------- 1. read & tidy LR04 --------------------
    df_lr04 = pd.read_excel(lr04_path, sheet_name=sheet_name)
    df_lr04.columns = ['age', 'd18O']           # enforce names
    df_lr04['age'] *= 1_000                     # kyr → yr BP

    # ---------- 2. interpolate onto df_sq ages --------
    ages_target = df_sq['age'].values
    d18o_interp = np.interp(ages_target,
                            df_lr04['age'].values,
                            df_lr04['d18O'].values)

    df_lr04_interp = pd.DataFrame({
        'age': ages_target,
        'd18O': d18o_interp
    })

    # ---------- 3. optional plots ---------------------
    if if_plot:
        # (a) raw LR04
        plt.figure(figsize=(10, 3))
        plt.plot(df_lr04['age'], df_lr04['d18O'], lw=0.8, label='LR04 raw')
        plt.gca().invert_xaxis(); plt.gca().invert_yaxis()
        plt.xlabel('Age (yr BP)'); plt.ylabel('δ18O')
        plt.title('LR04 stack (raw)')
        plt.tight_layout(); plt.show()

        # (b) interpolated
        plt.figure(figsize=(10, 3))
        plt.plot(df_lr04_interp['age'], df_lr04_interp['d18O'],
                 lw=1.0, label='LR04 interpolated')
        plt.gca().invert_xaxis(); plt.gca().invert_yaxis()
        plt.xlabel('Age (yr BP)'); plt.ylabel('δ18O')
        plt.title('LR04 interpolated to CH4 grid')
        plt.tight_layout(); plt.show()

    return df_lr04_interp












import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
plt.rcParams.update({
    "figure.dpi": 100,
    "axes.linewidth": 0.8,
    "grid.linestyle": "--",
    "grid.alpha": 0.4,
    "font.size": 9,          # main text
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
})





def age_gap_ana(
        df_ch4: pd.DataFrame,
        age_min: int = 0,
        age_max: int = 640_000,
        pre_path = "pre_800_inter100.txt",
        n_bins: int = 6,
        tolerance: int = 200):
    """
    Plots CH₄ sampling resolution (Δage) against orbital precession.
    Raises ValueError if either record does not cover [age_min, age_max].
    Returns (fig_pair, fig_bar) – two ready-to-use Matplotlib Figure objects.
    """

    # -------------------------------------------------
    # 0. Validate coverage of the requested time range
    # -------------------------------------------------
    df_ch4 = df_ch4.sort_values('age').reset_index(drop=True)
    if not (df_ch4['age'].min() <= age_min and df_ch4['age'].max() >= age_max):
        raise ValueError(
            f"CH₄ data span {df_ch4['age'].min():g}-{df_ch4['age'].max():g} yr BP; "
            f"requested window {age_min}-{age_max} yr BP is outside that range."
        )

    # -------------------------------------------------
    # 1. Crop first, *then* compute Δage            ▼
    # -------------------------------------------------
    df_ch4 = df_ch4.query('age >= @age_min and age <= @age_max').reset_index(drop=True)
    if len(df_ch4) < 2:                               #        ▼
        raise ValueError("Not enough CH₄ points in the selected age window.")  # ▼
    df_ch4['diff_age'] = df_ch4['age'].diff().abs()
    df_ch4 = df_ch4.dropna(subset=['diff_age'])

    # -------------------------------------------------
    # 2. Load & crop the raw precession record
    # -------------------------------------------------
    df_pre = (pd.read_csv(pre_path, sep=r'\s+', header=None, engine='python')
                .rename(columns={0: 'age', 1: 'pre'}))
    df_pre['age'] = df_pre['age'].abs() * 1000
    df_pre = df_pre.iloc[::-1].reset_index(drop=True)

    if not (df_pre['age'].min() <= age_min and df_pre['age'].max() >= age_max):
        raise ValueError(
            f"Precession file span {df_pre['age'].min():g}-{df_pre['age'].max():g} yr BP; "
            f"requested window {age_min}-{age_max} yr BP is outside that range."
        )

    df_pre = df_pre.query('age >= @age_min and age <= @age_max').reset_index(drop=True)

    # -------------------------------------------------
    # 3. Nearest-age merge
    # -------------------------------------------------
    df_merged = (pd.merge_asof(
                    df_ch4, df_pre,
                    on='age', direction='nearest', tolerance=tolerance)
                   .dropna(subset=['pre']))

        
    # 4a. Pair of time-series panels
    fig_pair, ax = plt.subplots(
        2, 1, figsize=(4.3, 4.0), sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1]}
    )

    # -- Precession
    ax[0].plot(df_merged["age"], df_merged["pre"],
            lw=1.1, color="#0072B2")                 # CB-friendly blue
    ax[0].set_ylabel("Precession (arb. u.)")
    ax[0].set_title("(a) Orbital precession", loc="left", fontsize=9, fontweight="bold")
    ax[0].grid(True)

    # -- Δage
    ax[1].plot(df_merged["age"], df_merged["diff_age"],
            lw=1.1, color="#D55E00")                 # CB-friendly orange
    ax[1].set_ylabel("Δage (yr)")
    ax[1].set_xlabel("Age (kyr BP)")
    ax[1].set_title("(b) Methane sampling step", loc="left", fontsize=9, fontweight="bold")
    ax[1].grid(True)

    # -- x-axis: invert and label in kyr

    ax[1].invert_xaxis()
    ax[1].xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, pos: f"{int(x/1_000)}")
    )

    fig_pair.align_ylabels()
    fig_pair.tight_layout()

    # 4b. Bar chart: mean Δage per precession bin
    df_merged["pre_bin"], bin_edges = pd.cut(
        df_merged["pre"], n_bins, labels=False, retbins=True
    )
    avg_by_bin = (df_merged.groupby("pre_bin", observed=True)["diff_age"]
                            .mean()
                            .rename("mean_diff_age")
                            .reset_index())

    # Bar centres (bin mid-points) and widths (bin widths)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_widths  = np.diff(bin_edges)

    fig_bar, axb = plt.subplots(figsize=(3.5, 2.3))
    axb.bar(bin_centres, avg_by_bin["mean_diff_age"],
            width=bin_widths, align="center", edgecolor="k",
            color="#009E73", lw=0.5)
    axb.set_xlabel("Precession amplitude")
    axb.set_ylabel("Mean Δage (yr)")
    axb.set_title("Sampling resolution vs precession", fontsize=9, pad=4)
    axb.grid(axis="y")

    # X-tick labels = bin mid-points in prec units (rounded)
    xb_labels = [f"{v:.0f}" for v in bin_centres]
    axb.set_xticks(bin_centres)
    axb.set_xticklabels(xb_labels, rotation=0)

    fig_bar.tight_layout()
    return fig_pair, fig_bar

















import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def age_gap_interactive(
    df_ch4: pd.DataFrame,
    age_min: int = 0,
    age_max: int = 640_000,
    H_line: int = 100
) -> go.Figure:
    """
    Computes and plots the methane sampling age step (Δage) over time interactively using Plotly,
    and prints the maximum Δage within the specified age window.

    Parameters
    ----------
    df_ch4 : pd.DataFrame
        DataFrame containing at least an 'age' column (years BP).
    age_min : int, optional
        Minimum age (yr BP) to include in the analysis. Default is 0.
    age_max : int, optional
        Maximum age (yr BP) to include in the analysis. Default is 640,000.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive Plotly figure of Δage vs. age.

    Raises
    ------
    ValueError
        If the data does not span the requested age range or has fewer than 2 points.
    """
    # Ensure sorted
    df = df_ch4.sort_values('age').reset_index(drop=True)

    # Validate coverage
    if df['age'].min() > age_min or df['age'].max() < age_max:
        raise ValueError(
            f"Data span {df['age'].min():g}-{df['age'].max():g} yr BP; "
            f"requested window {age_min}-{age_max} yr BP is outside that range."
        )

    # Crop to window
    df = df.query('age >= @age_min and age <= @age_max').reset_index(drop=True)
    if len(df) < 2:
        raise ValueError("Not enough data points in the selected age window.")

    # Compute Δage
    df['diff_age'] = df['age'].diff().abs()
    df = df.dropna(subset=['diff_age'])

    # Find and print maximum Δage
    max_gap = df['diff_age'].max()
    print(f"Maximum Δage within {age_min}-{age_max} yr BP: {max_gap:.0f} years")
    # print 95% quantile
    quantile_95 = df['diff_age'].quantile(0.95)
    print(f"95% quantile of Δage: {quantile_95:.0f} years")

    median_gap = df['diff_age'].median()
    print(f"Median Δage: {median_gap:.0f} years")

    # Interactive plot
    fig = px.line(
        df,
        x='age',
        y='diff_age',
        labels={
            'age': 'Age (yr BP)',
            'diff_age': 'Δage (yr)'
        },
        title=''
    )

    # plot a vertical line at H_line
    fig.add_shape(
        type='line',
        x0=age_min, x1=age_max,
        y0=H_line, y1=H_line,
        line=dict(color='red', width=1, dash='dash'),
        name='H Line'
    )

    # Invert x-axis to show decreasing age
    fig.update_xaxes(autorange='reversed')

    fig.update_layout(
        xaxis_title='Age (kyr BP)',
        yaxis_title='Δage (yr)',
        title=dict(x=0.5)
    )

    return fig

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def age_gap_hist(
    df_ch4: pd.DataFrame,
    age_min: int = 0,
    age_max: int = 640_000,
    nbins: int = 50
) -> go.Figure:
    """
    Computes the methane sampling age step (Δage) within a given age window
    and plots its distribution as an interactive histogram.

    Parameters
    ----------
    df_ch4 : pd.DataFrame
        DataFrame containing at least an 'age' column (years BP).
    age_min : int, optional
        Minimum age (yr BP) to include in the analysis. Default is 0.
    age_max : int, optional
        Maximum age (yr BP) to include in the analysis. Default is 640,000.
    nbins : int, optional
        Number of bins for the histogram. Default is 50.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive Plotly figure of the Δage histogram.

    Raises
    ------
    ValueError
        If the data does not span the requested age range or has fewer than 2 points
        in the selected window.
    """
    # Ensure sorted by age
    df = df_ch4.sort_values('age').reset_index(drop=True)

    # Validate that coverage covers the window
    if df['age'].min() > age_min or df['age'].max() < age_max:
        raise ValueError(
            f"Data span {df['age'].min():g}-{df['age'].max():g} yr BP; "
            f"requested window {age_min}-{age_max} yr BP is outside that range."
        )

    # Crop to the requested window
    df_window = df.query('age >= @age_min and age <= @age_max').reset_index(drop=True)
    if len(df_window) < 2:
        raise ValueError("Not enough data points in the selected age window.")

    # Compute Δage
    df_window['diff_age'] = df_window['age'].diff().abs()
    df_window = df_window.dropna(subset=['diff_age'])

    # Print summary statistics
    max_gap = df_window['diff_age'].max()
    q95 = df_window['diff_age'].quantile(0.95)
    median_gap = df_window['diff_age'].median()
    print(f"Maximum Δage within {age_min}-{age_max} yr BP: {max_gap:.0f} years")
    print(f"95% quantile of Δage: {q95:.0f} years")
    print(f"Median Δage: {median_gap:.0f} years")

    # Build the interactive histogram
    fig = px.histogram(
        df_window,
        x='diff_age',
        nbins=nbins,
        labels={'diff_age': 'Δage (yr)'},
        title=f'Distribution of Δage ({len(df_window)} samples)'
    )

    fig.update_layout(
        xaxis_title='Δage (yr)',
        yaxis_title='Count',
        bargap=0.1,
        title=dict(x=0.5)
    )

    return fig


































import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyinform import transfer_entropy
import matplotlib.ticker as mticker

# ──────────────────────────────────────────────────────────────────────────────
def te_vs_dt_scan(
        df_sq_raw,
        dt_grid,                 # iterable of resampling steps (yr)
        *,                       # keyword-only after here
        forcing='pre',           # 'pre' or 'obl'
        k=1,                     # TE history length
        forcing_bins=6, sq_bins=2,
        n_surr=100, alpha=0.05,
        sq_method='hist',        # 'hist', 'quantile', 'kmeans'
        dpi=150, figsize=(4.2, 3.2),
        return_fig=True):
    """
    Compute & plot TE(forcing → sq) versus resampling interval Δt.

    Parameters
    ----------
    forcing : {'pre', 'obl'}
        Which astronomical forcing to treat as the driver time-series.

    Returns
    -------
    results : pandas.DataFrame
        Columns ['dt','te_xy','p_xy','te_yx','p_yx','sig_uni']
    (fig, ax) : Matplotlib Figure and Axes (only if return_fig=True)
    """
    if forcing not in {'pre', 'obl', 'pre_phase', 'ecc'}:
        raise ValueError("forcing must be 'pre' or 'obl' or 'pre_phase' or 'ecc'")

    results = []

    # ------------------------------------------------------------------ loop over Δt
    for dt in dt_grid:
        df_sq_i, df_pre_i, df_obl_i = sa.interpolate_data_forcing(
            df_sq_raw.copy(), interval=dt, if_plot=False)

        _, df_ecc_i = sa.interpolate_data_ecc(df_sq_raw.copy(), dt, if_plot=False)

        # choose the driver series -------------------------------------
        if forcing == 'pre_phase':
            pre = df_pre_i['pre'].to_numpy()[::-1]     # chronological order
            pre_phase = sa.minmax_phase(df_pre_i['age'], pre, min_separation_years=8000)

            mask = np.isfinite(pre_phase)
            pre_phase = pre_phase[mask]
            y = df_sq_i.iloc[:, 1].to_numpy()[::-1][mask]   # filt_ch4 target
            x = pre_phase
            y_label = 'TE (pre_phase → sq)'
            y_rev   = 'TE (sq → pre_phase)'
        
        if forcing == 'pre':
            x = df_pre_i['pre'].to_numpy()[::-1]     # chronological order
            y = df_sq_i.iloc[:, 1].to_numpy()[::-1]       # filt_ch4 target
            y_label = 'TE (pre → sq)'
            y_rev   = 'TE (sq → pre)'

        if forcing == 'ecc':
            x = df_ecc_i['ecc'].to_numpy()[::-1]      # chronological order
            y = df_sq_i.iloc[:, 1].to_numpy()[::-1]       # filt_ch4 target
            y_label = 'TE (ecc → sq)'
            y_rev   = 'TE (sq → ecc)'

        if forcing == 'obl':
            x = df_obl_i['obl'].to_numpy()[::-1]
            y = df_sq_i.iloc[:, 1].to_numpy()[::-1]       # filt_ch4 target
            y_label = 'TE (obl → sq)'
            y_rev   = 'TE (sq → obl)'

        

        # --------------- discretise -----------------------------------


        if sq_method == 'quantile':
            ybins = np.quantile(y, np.linspace(0, 1, sq_bins + 1))
            y_disc = np.digitize(y, ybins) - 1
            # bin x using quantiles as well
            xbins = np.quantile(x, np.linspace(0, 1, forcing_bins + 1))
            x_disc = np.digitize(x, xbins) - 1
        elif sq_method == 'kmeans':
            from sklearn.cluster import KMeans
            y_disc = KMeans(n_clusters=sq_bins, n_init=10,
                            random_state=0).fit_predict(y.reshape(-1, 1))
            
            xbins = np.quantile(x, np.linspace(0, 1, forcing_bins + 1))
            x_disc = np.digitize(x, xbins) - 1
        else:  # 'hist'
            y_disc = (np.digitize(
                y, np.histogram_bin_edges(y, sq_bins)) - 1)
            x_disc = np.digitize(x, np.histogram_bin_edges(x, forcing_bins)) - 1

        # --------------- empirical TE ---------------------------------
        # te_xy = transfer_entropy(x_disc[:-1], y_disc[1:], k=k)
        # te_yx = transfer_entropy(y_disc[:-1], x_disc[1:], k=k)
        te_xy = transfer_entropy(x_disc, y_disc, k=k)
        te_yx = transfer_entropy(y_disc, x_disc, k=k)

        # --------------- surrogate test -------------------------------
        null_xy = np.empty(n_surr)
        null_yx = np.empty(n_surr)
        for i in range(n_surr):
            null_xy[i] = transfer_entropy(
                # np.random.permutation(x_disc)[:-1], y_disc[1:], k=k)
                np.random.permutation(x_disc), y_disc, k=k)
            null_yx[i] = transfer_entropy(
                np.random.permutation(y_disc), x_disc, k=k)

        p_xy = (null_xy >= te_xy).sum() / n_surr
        p_yx = (null_yx >= te_yx).sum() / n_surr

        results.append(dict(dt=dt, te_xy=te_xy, p_xy=p_xy,
                            te_yx=te_yx, p_yx=p_yx,
                            sig_uni=(p_xy < alpha and p_yx >= alpha)))

    results = pd.DataFrame(results).sort_values('dt').reset_index(drop=True)

    # ------------------------------------------------------------------ plotting
    if not return_fig:
        return results

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot(results['dt'], results['te_xy'],
            color='0.3', lw=1.2, marker='o',
            mfc='none', label=y_label)

    # highlight significant points
    sig = results['sig_uni']
    ax.scatter(results.loc[sig, 'dt'],
               results.loc[sig, 'te_xy'],
               s=60, color='#D55E00', zorder=5,
               label=f'Significant (α={alpha})')

    # show reverse TE
    ax.plot(results['dt'], results['te_yx'],
            ls='--', lw=1, color='0.6', label=y_rev)

    ax.set_xscale('log')
    ax.set_xlabel('Resampling interval Δt (yr)')
    ax.set_ylabel('Transfer Entropy (bits)')
    ax.set_title(f'Scale-dependence of TE  ({forcing} → sq), age_min={df_sq_raw["age"].min():g}, age_max={df_sq_raw["age"].max():g}')
    ax.grid(True, which='both', alpha=0.3)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%g'))
    ax.legend(frameon=True, fontsize=8)
    fig.tight_layout()

    return results, (fig, ax)















def interpolate_data_forcing(df_sq, interval, if_plot=False):
    """
    Resample the square‐wave and load & resample
    the precession & obliquity forcing to a common age grid.

    Parameters
    ----------
    df_sq : pandas.DataFrame
        Must have columns ['age', 'sq'].
    interval : float
        Desired age‐step for the resampled grid.
    if_plot : bool, default False
        If True, plots the three resampled series.

    Returns
    -------
    df_sq_resampled, df_pre_resampled, df_obl_resampled : DataFrames
        Each has columns ['age', <variable>] on the same age grid.
    """
    # 1) load raw precession & obliquity
    # pre_path = r"D:\VScode\bipolar_seesaw_CCM\inso_data\pre_800_inter100.txt"
    # obl_path = r"D:\VScode\bipolar_seesaw_CCM\inso_data\obl_800_inter100.txt"
    pre_path = "inso_data/pre_2000_inter100.txt"
    obl_path = "inso_data/obl_2000_inter100.txt"
    df_pre_raw = pd.read_csv(pre_path, sep=r'\s+', header=None, engine='python')
    df_obl_raw = pd.read_csv(obl_path, sep=r'\s+', header=None, engine='python')

    # convert to years & ensure age increasing
    df_pre_raw.iloc[:,0] = df_pre_raw.iloc[:,0].abs() * 1000
    df_obl_raw.iloc[:,0] = df_obl_raw.iloc[:,0].abs() * 1000
    df_pre_raw = df_pre_raw.iloc[::-1].reset_index(drop=True)
    df_obl_raw = df_obl_raw.iloc[::-1].reset_index(drop=True)
    df_pre_raw.columns = ['age','pre']
    df_obl_raw.columns = ['age','obl']

    # 2) compute overlapping age bounds
    a_min = max(df_sq['age'].min(),
                df_pre_raw['age'].min(),
                df_obl_raw['age'].min())
    a_max = min(df_sq['age'].max(),
                df_pre_raw['age'].max(),
                df_obl_raw['age'].max())

    # 3) create unified age vector
    new_age = np.arange(a_min, a_max + 1, interval)

    # 4) interpolate each series onto new_age
    def interp(df, col):
        f = interp1d(df['age'], df[col],
                     kind='nearest',
                     bounds_error=False,
                     fill_value="extrapolate")
        return f(new_age)
    
    column_names = df_sq.columns[1]

    df_sq_rs  = pd.DataFrame({'age': new_age,
                              column_names: interp(df_sq, column_names)})
    df_pre_rs = pd.DataFrame({'age': new_age,
                              'pre': interp(df_pre_raw, 'pre')})
    df_obl_rs = pd.DataFrame({'age': new_age,
                              'obl': interp(df_obl_raw, 'obl')})

    # 5) optional plotting
    if if_plot:
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, tight_layout=True)
        axes[0].plot(df_sq_rs['age'],  df_sq_rs[column_names],  color='black', label='sq')
        axes[1].plot(df_pre_rs['age'], df_pre_rs['pre'], color='blue',  label='pre')
        axes[2].plot(df_obl_rs['age'], df_obl_rs['obl'], color='green', label='obl')
        for ax in axes:
            ax.legend()
            ax.set_ylabel(ax.get_label())
        axes[-1].set_xlabel('Age (years)')
        plt.show()

    return df_sq_rs, df_pre_rs, df_obl_rs






def interpolate_data_ecc(df_sq, interval, if_plot=False):

    # 1) load raw precession & obliquity
    ecc_path = "inso_data/ecc_1000_inter100.txt"

    df_ecc_raw = pd.read_csv(ecc_path, sep=r'\s+', header=None, engine='python')

    # convert to years & ensure age increasing
    df_ecc_raw.iloc[:,0] = df_ecc_raw.iloc[:,0].abs() * 1000
    df_ecc_raw = df_ecc_raw.iloc[::-1].reset_index(drop=True)
    df_ecc_raw.columns = ['age','ecc']

    # 2) compute overlapping age bounds
    a_min = max(df_sq['age'].min(),
                df_ecc_raw['age'].min(),
               )
    a_max = min(df_sq['age'].max(),
                df_ecc_raw['age'].max(),
                )

    # 3) create unified age vector
    new_age = np.arange(a_min, a_max + 1, interval)

    # 4) interpolate each series onto new_age
    def interp(df, col):
        f = interp1d(df['age'], df[col],
                     kind='nearest',
                     bounds_error=False,
                     fill_value="extrapolate")
        return f(new_age)
    
    column_names = df_sq.columns[1]

    df_sq_rs  = pd.DataFrame({'age': new_age,
                              column_names: interp(df_sq, column_names)})
    df_ecc_rs = pd.DataFrame({'age': new_age,
                              'ecc': interp(df_ecc_raw, 'ecc')})

    # 5) optional plotting
    if if_plot:
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, tight_layout=True)
        axes[0].plot(df_sq_rs['age'],  df_sq_rs[column_names],  color='black', label='sq')
        axes[1].plot(df_ecc_rs['age'], df_ecc_rs['ecc'], color='blue',  label='ecc')
        # axes[2].plot(df_obl_rs['age'], df_obl_rs['obl'], color='green', label='obl')
        for ax in axes:
            ax.legend()
            ax.set_ylabel(ax.get_label())
        axes[-1].set_xlabel('Age (years)')
        plt.show()

    return df_sq_rs, df_ecc_rs













# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import find_peaks
# from scipy.stats import ttest_ind

# def plot_phase_durations(df_pre, df_sq, lags=range(0, 901, 100)):
#     """
#     For each lag in `lags`, shifts df_pre manually (introducing NaNs),
#     truncates df_sq accordingly, computes rising/falling phase durations,
#     runs t-tests, and produces the same two-panel figure per lag.
    
#     Parameters
#     ----------
#     df_pre : pandas.DataFrame
#         Must contain a 'pre' column (precession) and matching index.
#     df_sq : pandas.DataFrame
#         Must contain a 'sq' column (square wave ±1) and matching index.
#     lags : iterable of ints, optional
#         List of integer shifts (in data points) to test.
#     """
#     pre_vals_orig = df_pre['pre'].values
#     sq_orig       = np.where(df_sq['sq'].values > 0, 1, -1)
#     N             = len(pre_vals_orig)

#     def count_transitions(intervals, sq_sign):
#         lt, hl = [], []
#         for i0, i1 in intervals:
#             seg = sq_sign[i0:i1+1]
#             lt.append(np.sum((seg[:-1]==-1) & (seg[1:]==+1)))
#             hl.append(np.sum((seg[:-1]==+1) & (seg[1:]==-1)))
#         return lt, hl

#     for lag in lags:
#         # 1) manual shift with NaNs
#         pre_shifted = np.full(N, np.nan)
#         if lag > 0:
#             pre_shifted[0:N-lag] = pre_vals_orig[lag:N]
#         elif lag < 0:
#             k = -lag
#             pre_shifted[k:N] = pre_vals_orig[0:N-k]
#         else:
#             pre_shifted[:] = pre_vals_orig

#         # 2) truncate to non-NaNs
#         valid    = ~np.isnan(pre_shifted)
#         pre_vals = pre_shifted[valid]
#         sq_sign  = sq_orig[valid]

#         # 3) peak/trough detection
#         peaks,   _ = find_peaks(pre_vals)
#         troughs, _ = find_peaks(-pre_vals)

#         # 4) rising & falling intervals
#         rising, falling = [], []
#         for t in troughs:
#             nxt = peaks[peaks > t]
#             if nxt.size: rising.append((t, nxt[0]))
#         for p in peaks:
#             nxt = troughs[troughs > p]
#             if nxt.size: falling.append((p, nxt[0]))

#         # 5) compute durations for each phase
#         r_cold, r_warm = [], []
#         for i0, i1 in rising:
#             seg = sq_sign[i0:i1+1]
#             r_cold.append(np.sum(seg == -1))
#             r_warm.append(np.sum(seg == +1))

#         d_cold, d_warm = [], []
#         for i0, i1 in falling:
#             seg = sq_sign[i0:i1+1]
#             d_cold.append(np.sum(seg == -1))
#             d_warm.append(np.sum(seg == +1))

#         # 6) t-tests on durations
#         t_r, p_r = ttest_ind(r_cold, r_warm, equal_var=False)
#         sig_r = 'Yes' if p_r < 0.05 else 'No'
#         t_d, p_d = ttest_ind(d_cold, d_warm, equal_var=False)
#         sig_d = 'Yes' if p_d < 0.05 else 'No'

#         # 7) plotting (identical to original)
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4), tight_layout=True)
#         fig.suptitle(f"Lag = {lag} indices", fontsize=14)

#         ax1.hist(r_cold, bins='auto', alpha=0.5, color='green', label='cold length')
#         ax1.axvline(np.mean(r_cold), color='green', linestyle='dashed', linewidth=1)
#         ax1.hist(r_warm, bins='auto', alpha=0.5, color='red',   label='warm length')
#         ax1.axvline(np.mean(r_warm), color='red', linestyle='dashed', linewidth=1)
#         ax1.set_title(f"Rising phases\n t={t_r:.2f}, p={p_r:.2e}\nSig? {sig_r}")
#         ax1.set_xlabel('Duration (data points)')
#         ax1.set_ylabel('Frequency')
#         ax1.legend()

#         ax2.hist(d_cold, bins='auto', alpha=0.5, color='green', label='cold length')
#         ax2.axvline(np.mean(d_cold), color='green', linestyle='dashed', linewidth=1)
#         ax2.hist(d_warm, bins='auto', alpha=0.5, color='red',   label='warm length')
#         ax2.axvline(np.mean(d_warm), color='red', linestyle='dashed', linewidth=1)
#         ax2.set_title(f"Decreasing phases\n t={t_d:.2f}, p={p_d:.2e}\nSig? {sig_d}")
#         ax2.set_xlabel('Duration (data points)')
#         ax2.legend()

#         plt.show()



# import numpy as np
# from scipy.signal import find_peaks
# from scipy.stats import ttest_ind
# import matplotlib.pyplot as plt

# def plot_transition_distribution(df_pre, df_sq, lags=None):
#     """
#     For each lag in `lags`, shifts df_pre manually (introducing NaNs),
#     truncates df_sq accordingly, computes rising/falling phase transitions,
#     runs t-tests, and produces the same two-panel figure per lag.

#     Parameters
#     ----------
#     df_pre : pandas.DataFrame
#         Must contain a 'pre' column.
#     df_sq : pandas.DataFrame
#         Must contain a 'sq' column with values ±1.
#     lags : iterable of ints, optional
#         List of integer shifts (in data points) to test.
#         Default is range(0, 901, 100).
#     """
#     if lags is None:
#         lags = range(0, 901, 100)

#     pre_vals_orig = df_pre['pre'].values
#     sq_orig       = np.where(df_sq['sq'].values > 0, 1, -1)
#     N             = len(pre_vals_orig)

#     def count_transitions(intervals, sq_sign):
#         lt, hl = [], []
#         for i0, i1 in intervals:
#             seg = sq_sign[i0:i1+1]
#             lt.append(np.sum((seg[:-1]==-1) & (seg[1:]==+1)))
#             hl.append(np.sum((seg[:-1]==+1) & (seg[1:]==-1)))
#         return lt, hl

#     for lag in lags:
#         # 1) manual shift with NaNs instead of roll()
#         pre_shifted = np.full(N, np.nan)
#         if lag > 0:
#             pre_shifted[0:N-lag] = pre_vals_orig[lag:N]
#         elif lag < 0:
#             k = -lag
#             pre_shifted[k:N] = pre_vals_orig[0:N-k]
#         else:
#             pre_shifted[:] = pre_vals_orig

#         # 2) mask out NaNs & truncate sq to match
#         valid    = ~np.isnan(pre_shifted)
#         pre_vals = pre_shifted[valid]
#         sq_sign  = sq_orig[valid]

#         # 3) detect peaks/troughs
#         peaks,   _ = find_peaks(pre_vals)
#         troughs, _ = find_peaks(-pre_vals)

#         # 4) rising & falling intervals
#         rising, falling = [], []
#         for t in troughs:
#             nxt = peaks[peaks > t]
#             if nxt.size: rising.append((t, nxt[0]))
#         for p in peaks:
#             nxt = troughs[troughs > p]
#             if nxt.size: falling.append((p, nxt[0]))

#         # 5) count transitions
#         r_lt, r_hl = count_transitions(rising,   sq_sign)
#         f_lt, f_hl = count_transitions(falling,  sq_sign)

#         # 6) t-tests
#         t_r, p_r = ttest_ind(r_lt, r_hl, equal_var=False)
#         sig_r = 'Yes' if p_r < 0.05 else 'No'
#         t_f, p_f = ttest_ind(f_lt, f_hl, equal_var=False)
#         sig_f = 'Yes' if p_f < 0.05 else 'No'

#         # 7) plotting (unchanged)
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), tight_layout=True)
#         fig.suptitle(f"Lag = {lag} indices", fontsize=14)

#         ax1.hist(r_lt, bins='auto', alpha=0.5, color='blue',   label='–1→+1')
#         ax1.axvline(np.mean(r_lt), color='blue', linestyle='--')
#         ax1.hist(r_hl, bins='auto', alpha=0.5, color='orange', label='+1→–1')
#         ax1.axvline(np.mean(r_hl), color='orange', linestyle='--')
#         ax1.set_title(f"Rising phases\n t={t_r:.2f}, p={p_r:.2e}\nSig? {sig_r}")
#         ax1.set_xlabel('Transitions per rising phase')
#         ax1.set_ylabel('Count of phases')
#         ax1.legend()

#         ax2.hist(f_lt, bins='auto', alpha=0.5, color='blue',   label='–1→+1')
#         ax2.axvline(np.mean(f_lt), color='blue', linestyle='--')
#         ax2.hist(f_hl, bins='auto', alpha=0.5, color='orange', label='+1→–1')
#         ax2.axvline(np.mean(f_hl), color='orange', linestyle='--')
#         ax2.set_title(f"Decreasing phases\n t={t_f:.2f}, p={p_f:.2e}\nSig? {sig_f}")
#         ax2.set_xlabel('Transitions per decreasing phase')
#         ax2.legend()

#         plt.show()




