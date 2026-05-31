import os
os.environ["CPL_LOG"] = "/dev/null"

"""
Scheldt Estuary — Tier 1 Dynamics Analysis
===========================================
Input  : folder of Sentinel-2 seasonal GeoTIFFs (from acquire_scheldt.py)
Output : four analysis products saved as GeoTIFF + PNG figures

Analyses
--------
1. Shoreline / waterline migration    — NDWI=0 contour stack + retreat map
2. NDVI linear trend map              — per-pixel slope across all snapshots
3. Mudflat area time series           — area curve over time
4. MNDWI change magnitude map         — per-pixel std dev = instability heatmap

Usage
-----
    pip install rioxarray rasterio numpy matplotlib scipy scikit-image tqdm
    python tier1_dynamics.py ./data/scheldt/sentinel2/

Band order expected (matches acquire_scheldt.py):
    band 1 = B02  band 2 = B03  band 3 = B04  band 4 = B08  band 5 = B11
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import matplotlib.cm as cm
from scipy import stats
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    import rioxarray as rxr
    import rasterio
    from rasterio.transform import from_bounds
    from skimage import measure
except ImportError:
    sys.exit(
        "Missing dependencies. Run:\n"
        "  pip install rioxarray rasterio numpy matplotlib scipy scikit-image tqdm"
    )


# ── CONFIG ────────────────────────────────────────────────────────────────────

NDWI_WATER_THRESHOLD  = 0.0    # NDWI > this → water / wet mudflat
NDVI_VEG_THRESHOLD    = 0.2    # NDVI > this → vegetation
NDVI_DENSE_THRESHOLD  = 0.5    # NDVI > this → dense marsh
MUDFLAT_NDVI_MAX      = 0.1    # NDVI < this AND NDWI > 0 → mudflat


# ── HELPERS ───────────────────────────────────────────────────────────────────

def load_bands(tif_path: Path):
    """Return (b02,b03,b04,b08,b11) as float32 numpy arrays."""
    ds = rxr.open_rasterio(tif_path, masked=True)
    b = [ds.sel(band=i).values.astype("float32") for i in range(1, 6)]
    return tuple(b)   # b02 b03 b04 b08 b11


def index(a, b):
    denom = a + b
    denom = np.where(np.abs(denom) < 1e-6, np.nan, denom)
    return (a - b) / denom


def parse_label(path: Path) -> str:
    """Extract a short date label from filename, e.g. S2_2019_spring → 2019 spr"""
    stem = path.stem.replace("S2_", "")
    parts = stem.split("_")
    if len(parts) >= 2:
        season_map = {"spring": "Spr", "summer": "Sum", "winter": "Win"}
        return f"{parts[0]} {season_map.get(parts[1], parts[1])}"
    return stem


def dark_fig(*args, **kwargs):
    fig = plt.figure(*args, facecolor="#0e0e0e", **kwargs)
    return fig


def styled_ax(ax, title):
    ax.set_facecolor("#0e0e0e")
    ax.set_title(title, color="white", fontsize=10, fontweight="bold", pad=5)
    ax.tick_params(colors="white", labelsize=8)
    ax.spines[:].set_color("#444")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")


def save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0e0e0e")
    print(f"  → saved {path.name}")
    plt.close(fig)


# ── LOAD ALL SNAPSHOTS ────────────────────────────────────────────────────────

def load_all(tif_dir: Path):
    tifs = sorted(tif_dir.glob("S2_*.tif"))
    if len(tifs) < 2:
        sys.exit(f"Need at least 2 S2_*.tif files in {tif_dir}")

    print(f"Found {len(tifs)} snapshots:")
    for t in tifs:
        print(f"  {t.name}")

    # Load first to get shape
    b02, b03, b04, b08, b11 = load_bands(tifs[0])
    H, W = b02.shape

    # Allocate stacks
    ndvi_stack  = np.full((len(tifs), H, W), np.nan, dtype="float32")
    ndwi_stack  = np.full((len(tifs), H, W), np.nan, dtype="float32")
    mndwi_stack = np.full((len(tifs), H, W), np.nan, dtype="float32")
    labels      = []

    for i, tif in enumerate(tqdm(tifs, desc="Loading")):
        b02, b03, b04, b08, b11 = load_bands(tif)
        ndvi_stack[i]  = index(b08, b04)
        ndwi_stack[i]  = index(b03, b08)
        mndwi_stack[i] = index(b03, b11)
        labels.append(parse_label(tif))

    # Also get geotransform from first file for raster outputs
    with rasterio.open(tifs[0]) as src:
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs

    return ndvi_stack, ndwi_stack, mndwi_stack, labels, profile, transform, crs, tifs


# ── ANALYSIS 1 — WATERLINE MIGRATION ─────────────────────────────────────────

def analysis_waterline(ndwi_stack, labels, transform, out_dir):
    """
    Extract NDWI=0 contour from each snapshot and overlay them
    colour-coded by time. Also compute per-pixel waterline frequency
    (how often each pixel is classified as water) as a stability proxy.
    """
    print("\n[1] Waterline migration ...")
    n, H, W = ndwi_stack.shape

    # Water frequency map: 0 = never water, 1 = always water
    water_binary = (ndwi_stack > NDWI_WATER_THRESHOLD).astype("float32")
    freq_map = np.nanmean(water_binary, axis=0)

    # Colour ramp for time progression
    colours = cm.plasma(np.linspace(0.15, 0.95, n))

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), facecolor="#0e0e0e")

    # Left: waterline contour stack
    ax = axes[0]
    ax.set_facecolor("#111")
    # Background: mean NDWI
    bg = np.nanmean(ndwi_stack, axis=0)
    ax.imshow(bg, cmap="Blues", vmin=-0.3, vmax=0.6, aspect="equal",
              interpolation="nearest", alpha=0.5)

    for i in range(n):
        frame = ndwi_stack[i].copy()
        frame[~np.isfinite(frame)] = -1
        try:
            contours = measure.find_contours(frame, NDWI_WATER_THRESHOLD)
            for c in contours:
                if len(c) > 50:   # skip tiny fragments
                    ax.plot(c[:, 1], c[:, 0],
                            color=colours[i], lw=0.8, alpha=0.85)
        except Exception:
            pass

    # Colourbar legend for time
    sm = plt.cm.ScalarMappable(
        cmap="plasma",
        norm=mcolors.Normalize(vmin=0, vmax=n - 1)
    )
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cb.set_ticks(np.linspace(0, n - 1, min(n, 6)))
    cb.set_ticklabels([labels[int(t)] for t in np.linspace(0, n - 1, min(n, 6))])
    cb.ax.tick_params(labelsize=7, colors="white")
    cb.ax.yaxis.label.set_color("white")
    styled_ax(ax, "Waterline positions over time\n(each line = one snapshot, purple→yellow = older→newer)")

    # Right: water frequency map
    ax2 = axes[1]
    im = ax2.imshow(freq_map, cmap="RdYlBu_r", vmin=0, vmax=1,
                    aspect="equal", interpolation="nearest")
    cb2 = plt.colorbar(im, ax=ax2, fraction=0.03, pad=0.02)
    cb2.set_label("Fraction of snapshots classified as water", fontsize=7)
    cb2.ax.tick_params(labelsize=7, colors="white")
    cb2.ax.yaxis.label.set_color("white")
    styled_ax(ax2, "Water frequency map\n0 = never water  |  1 = always water  |  0.3–0.7 = dynamic zone")

    # Annotate dynamic zone percentage
    dynamic = np.sum((freq_map > 0.2) & (freq_map < 0.8) & np.isfinite(freq_map))
    total   = np.sum(np.isfinite(freq_map))
    pct = 100 * dynamic / total
    ax2.text(0.02, 0.02, f"Dynamic fringe: {pct:.1f}% of AOI",
             transform=ax2.transAxes, color="white", fontsize=9,
             bbox=dict(facecolor="#333", alpha=0.7, boxstyle="round"))

    fig.tight_layout()
    save(fig, out_dir / "01_waterline_migration.png")

    # Save water frequency as GeoTIFF
    prof = dict(driver="GTiff", dtype="float32", count=1,
            height=H, width=W, crs=crs, transform=transform)
    out_tif = out_dir / "01_water_frequency.tif"
    with rasterio.open(out_tif, "w", **prof) as dst:
        dst.write(freq_map[np.newaxis, :, :])


# ── ANALYSIS 2 — NDVI LINEAR TREND ───────────────────────────────────────────

def analysis_ndvi_trend(ndvi_stack, labels, out_dir):
    """
    Fit a linear regression (slope) through the NDVI time series at each pixel.
    Positive slope = greening (marsh gaining).
    Negative slope = browning (marsh losing / erosion).
    Mask slopes with p-value > 0.1 as not statistically significant.
    """
    print("\n[2] NDVI linear trend ...")
    n, H, W = ndvi_stack.shape
    x = np.arange(n, dtype="float32")

    slope_map = np.full((H, W), np.nan, dtype="float32")
    pval_map  = np.full((H, W), np.nan, dtype="float32")

    # Flatten spatial dims, run regression per pixel
    flat = ndvi_stack.reshape(n, -1)   # (n_times, n_pixels)

    print("  Fitting per-pixel regressions ...")
    for px in tqdm(range(flat.shape[1]), desc="  Pixels", miniters=10000):
        y = flat[:, px]
        valid = np.isfinite(y)
        if valid.sum() < 4:   # need at least 4 valid obs
            continue
        res = stats.linregress(x[valid], y[valid])
        row, col = divmod(px, W)
        slope_map[row, col] = res.slope
        pval_map[row, col]  = res.pvalue

    # Mask insignificant trends
    sig_mask = pval_map > 0.10
    slope_masked = slope_map.copy()
    slope_masked[sig_mask] = np.nan

    # ── figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(21, 7), facecolor="#0e0e0e")

    # Raw slope (all pixels)
    lim = np.nanpercentile(np.abs(slope_map[np.isfinite(slope_map)]), 95)
    im0 = axes[0].imshow(slope_map, cmap="RdYlGn",
                          vmin=-lim, vmax=lim, aspect="equal",
                          interpolation="nearest")
    cb0 = plt.colorbar(im0, ax=axes[0], fraction=0.035, pad=0.02)
    cb0.set_label("NDVI units / timestep", fontsize=7)
    cb0.ax.tick_params(labelsize=7, colors="white")
    cb0.ax.yaxis.label.set_color("white")
    styled_ax(axes[0], "NDVI trend — raw slope\ngreen = greening  |  red = browning")

    # Significant slope only
    im1 = axes[1].imshow(slope_masked, cmap="RdYlGn",
                          vmin=-lim, vmax=lim, aspect="equal",
                          interpolation="nearest")
    cb1 = plt.colorbar(im1, ax=axes[1], fraction=0.035, pad=0.02)
    cb1.set_label("NDVI units / timestep  (p < 0.10)", fontsize=7)
    cb1.ax.tick_params(labelsize=7, colors="white")
    cb1.ax.yaxis.label.set_color("white")
    styled_ax(axes[1], "NDVI trend — significant only  (p < 0.10)\ngrey = no significant trend detected")

    # Histogram of significant slopes
    ax2 = axes[2]
    ax2.set_facecolor("#1a1a1a")
    sig_slopes = slope_masked[np.isfinite(slope_masked)].ravel()
    greening = sig_slopes[sig_slopes > 0]
    browning = sig_slopes[sig_slopes < 0]
    ax2.hist(browning, bins=80, color="#e53935", alpha=0.8, label=f"Browning ({len(browning):,} px)")
    ax2.hist(greening, bins=80, color="#43a047", alpha=0.8, label=f"Greening ({len(greening):,} px)")
    ax2.axvline(0, color="white", lw=1.2, ls="--")
    ax2.set_xlabel("NDVI slope", color="white")
    ax2.set_ylabel("Pixel count", color="white")
    ax2.tick_params(colors="white", labelsize=8)
    ax2.spines[:].set_color("#444")
    ax2.legend(fontsize=8, facecolor="#1a1a1a", labelcolor="white")
    styled_ax(ax2, "Distribution of significant NDVI trends")

    # Summary stats
    total_sig = len(sig_slopes)
    if total_sig > 0:
        green_pct = 100 * len(greening) / total_sig
        brown_pct = 100 * len(browning) / total_sig
        ax2.text(0.02, 0.95,
                 f"Greening: {green_pct:.1f}%\nBrowning: {brown_pct:.1f}%",
                 transform=ax2.transAxes, color="white", fontsize=9, va="top",
                 bbox=dict(facecolor="#333", alpha=0.7, boxstyle="round"))

    fig.tight_layout()
    save(fig, out_dir / "02_ndvi_trend.png")


# ── ANALYSIS 3 — MUDFLAT AREA TIME SERIES ────────────────────────────────────

def analysis_mudflat_timeseries(ndvi_stack, ndwi_stack, mndwi_stack,
                                 labels, tifs, out_dir):
    """
    For each snapshot compute:
      - Mudflat area    (NDWI > 0  AND  NDVI < 0.1)
      - Open water area (MNDWI > 0.1)
      - Dense marsh     (NDVI > 0.5)
    Plot all three as time series curves.
    """
    print("\n[3] Mudflat area time series ...")
    n = ndvi_stack.shape[0]

    mudflat_pct  = []
    water_pct    = []
    marsh_pct    = []

    for i in range(n):
        ndvi  = ndvi_stack[i]
        ndwi  = ndwi_stack[i]
        mndwi = mndwi_stack[i]
        valid = np.isfinite(ndvi) & np.isfinite(ndwi) & np.isfinite(mndwi)
        total = valid.sum()
        if total == 0:
            mudflat_pct.append(np.nan)
            water_pct.append(np.nan)
            marsh_pct.append(np.nan)
            continue
        mudflat_pct.append(100 * np.sum((ndwi > NDWI_WATER_THRESHOLD) &
                                         (ndvi < MUDFLAT_NDVI_MAX) & valid) / total)
        water_pct.append(  100 * np.sum((mndwi > 0.1) & valid) / total)
        marsh_pct.append(  100 * np.sum((ndvi > NDVI_DENSE_THRESHOLD) & valid) / total)

    mudflat_pct = np.array(mudflat_pct)
    water_pct   = np.array(water_pct)
    marsh_pct   = np.array(marsh_pct)
    x           = np.arange(n)

    fig, axes = plt.subplots(3, 1, figsize=(14, 12),
                              sharex=True, facecolor="#0e0e0e")

    series = [
        (mudflat_pct, "#FF9800", "Mudflat area  (NDWI>0 & NDVI<0.1)",
         "% of valid pixels"),
        (water_pct,   "#2196F3", "Open water area  (MNDWI>0.1)",
         "% of valid pixels"),
        (marsh_pct,   "#4CAF50", "Dense marsh  (NDVI>0.5)",
         "% of valid pixels"),
    ]

    for ax, (vals, col, title, ylabel) in zip(axes, series):
        ax.set_facecolor("#1a1a1a")
        ax.plot(x, vals, color=col, lw=2, marker="o",
                markersize=6, markerfacecolor="white", markeredgecolor=col)
        ax.fill_between(x, vals, alpha=0.15, color=col)

        # Trend line if enough points
        valid_idx = np.where(np.isfinite(vals))[0]
        if len(valid_idx) >= 4:
            res = stats.linregress(valid_idx, vals[valid_idx])
            trend = res.slope * x + res.intercept
            ax.plot(x, trend, color=col, lw=1.2, ls="--", alpha=0.6)
            direction = "↑" if res.slope > 0 else "↓"
            ax.text(0.99, 0.92,
                    f"trend: {direction} {abs(res.slope):.3f}%/step  "
                    f"(p={res.pvalue:.3f})",
                    transform=ax.transAxes, ha="right", color="white",
                    fontsize=8,
                    bbox=dict(facecolor="#333", alpha=0.7, boxstyle="round"))

        ax.set_ylabel(ylabel, color="white", fontsize=8)
        ax.tick_params(colors="white", labelsize=8)
        ax.spines[:].set_color("#444")
        ax.set_title(title, color="white", fontsize=10,
                      fontweight="bold", pad=4)
        ax.grid(axis="y", color="#333", lw=0.5)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=45, ha="right",
                               color="white", fontsize=8)
    axes[-1].set_xlabel("Snapshot", color="white", fontsize=9)

    fig.suptitle("Scheldt — Intertidal habitat dynamics",
                  color="white", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, out_dir / "03_mudflat_timeseries.png")

    # Print table
    print(f"\n  {'Snapshot':<15} {'Mudflat%':>9} {'Water%':>9} {'Marsh%':>9}")
    print("  " + "-" * 45)
    for i, lbl in enumerate(labels):
        print(f"  {lbl:<15} {mudflat_pct[i]:>9.2f} "
              f"{water_pct[i]:>9.2f} {marsh_pct[i]:>9.2f}")


# ── ANALYSIS 4 — MNDWI INSTABILITY MAP ───────────────────────────────────────

def analysis_instability(mndwi_stack, ndvi_stack, labels, out_dir):
    """
    Per-pixel standard deviation of MNDWI across all snapshots.
    High std dev = high temporal variability = unstable / dynamic zone.
    Overlay with mean NDVI to contextualise (marsh vs bare vs water).
    """
    print("\n[4] MNDWI instability (change magnitude) map ...")
    n, H, W = mndwi_stack.shape

    std_map  = np.nanstd(mndwi_stack,  axis=0)
    mean_ndvi = np.nanmean(ndvi_stack, axis=0)

    # Classify stability zones
    #   permanent water  : mean NDWI high, low std
    #   stable marsh     : mean NDVI high, low std
    #   dynamic fringe   : moderate NDWI, HIGH std  ← the instability signal
    #   stable land      : low NDWI, low NDVI, low std

    fig, axes = plt.subplots(1, 3, figsize=(21, 7), facecolor="#0e0e0e")

    # Panel 1: raw std dev map
    p95 = np.nanpercentile(std_map[np.isfinite(std_map)], 95)
    im0 = axes[0].imshow(std_map, cmap="inferno",
                          vmin=0, vmax=p95, aspect="equal",
                          interpolation="nearest")
    cb0 = plt.colorbar(im0, ax=axes[0], fraction=0.035, pad=0.02)
    cb0.set_label("MNDWI std dev", fontsize=7)
    cb0.ax.tick_params(labelsize=7, colors="white")
    cb0.ax.yaxis.label.set_color("white")
    styled_ax(axes[0], "MNDWI change magnitude\ndark = stable  |  bright = highly dynamic")

    # Panel 2: classified instability zones
    # Build a simple 4-class map
    zone_map = np.zeros((H, W), dtype="uint8")
    mndwi_mean = np.nanmean(mndwi_stack, axis=0)

    # 1 = permanent water (always wet, low variability)
    zone_map[(mndwi_mean > 0.15) & (std_map < np.nanpercentile(std_map, 50))] = 1
    # 2 = stable land / marsh (vegetated, low variability)
    zone_map[(mean_ndvi > 0.3) & (std_map < np.nanpercentile(std_map, 50))]   = 2
    # 3 = dynamic fringe (high variability — the instability zone)
    zone_map[std_map > np.nanpercentile(std_map, 75)] = 3
    # 4 = highly dynamic (top 10% variability)
    zone_map[std_map > np.nanpercentile(std_map, 90)] = 4

    zone_cmap = mcolors.ListedColormap(
        ["#1a1a1a", "#1565C0", "#2E7D32", "#F57F17", "#C62828"]
    )
    im1 = axes[1].imshow(zone_map, cmap=zone_cmap, vmin=0, vmax=4,
                          aspect="equal", interpolation="nearest")
    legend_elements = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#1a1a1a",
               markersize=10, label="No data"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#1565C0",
               markersize=10, label="Permanent water"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#2E7D32",
               markersize=10, label="Stable marsh/land"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#F57F17",
               markersize=10, label="Dynamic fringe"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#C62828",
               markersize=10, label="Highly dynamic (risk)"),
    ]
    axes[1].legend(handles=legend_elements, loc="lower left",
                   fontsize=7, facecolor="#222", labelcolor="white",
                   framealpha=0.9)
    styled_ax(axes[1], "Instability zone classification\nbased on MNDWI temporal variability")

    # Panel 3: std dev histogram with zone thresholds
    ax2 = axes[2]
    ax2.set_facecolor("#1a1a1a")
    vals = std_map[np.isfinite(std_map)].ravel()
    ax2.hist(vals, bins=150, color="#FF9800", alpha=0.85)
    p50 = np.nanpercentile(std_map, 50)
    p75 = np.nanpercentile(std_map, 75)
    p90 = np.nanpercentile(std_map, 90)
    ax2.axvline(p50, color="#2E7D32", lw=1.5, ls="--",
                label=f"p50 = {p50:.3f}  (stable threshold)")
    ax2.axvline(p75, color="#F57F17", lw=1.5, ls="--",
                label=f"p75 = {p75:.3f}  (dynamic fringe)")
    ax2.axvline(p90, color="#C62828", lw=1.5, ls="--",
                label=f"p90 = {p90:.3f}  (high risk)")
    ax2.set_xlabel("MNDWI std dev", color="white", fontsize=9)
    ax2.set_ylabel("Pixel count", color="white", fontsize=9)
    ax2.tick_params(colors="white", labelsize=8)
    ax2.spines[:].set_color("#444")
    ax2.legend(fontsize=8, facecolor="#1a1a1a", labelcolor="white",
               framealpha=0.9)
    styled_ax(ax2, "MNDWI variability distribution\nwith instability thresholds")

    fig.tight_layout()
    save(fig, out_dir / "04_instability_map.png")

    # Print zone summary
    total = np.sum(zone_map > 0)
    if total > 0:
        print("\n  Zone breakdown:")
        zone_names = ["", "Permanent water", "Stable marsh/land",
                      "Dynamic fringe", "High risk"]
        for z in range(1, 5):
            pct = 100 * np.sum(zone_map == z) / total
            print(f"    {zone_names[z]:<22}: {pct:5.1f}%")


# ── SUMMARY FIGURE ────────────────────────────────────────────────────────────

def summary_figure(ndvi_stack, ndwi_stack, mndwi_stack, labels, out_dir):
    """
    Single-page portfolio summary: one representative panel from each analysis.
    """
    print("\n[5] Building summary figure ...")
    n = ndvi_stack.shape[0]

    std_map   = np.nanstd(mndwi_stack, axis=0)
    ndvi_mean = np.nanmean(ndvi_stack,  axis=0)

    # NDVI slope (quick version for summary)
    x = np.arange(n, dtype="float32")
    flat = ndvi_stack.reshape(n, -1)
    H, W = ndvi_stack.shape[1], ndvi_stack.shape[2]
    slope_map = np.full(H * W, np.nan, dtype="float32")
    for px in range(flat.shape[1]):
        y = flat[:, px]
        valid = np.isfinite(y)
        if valid.sum() >= 4:
            res = stats.linregress(x[valid], y[valid])
            if res.pvalue < 0.10:
                slope_map[px] = res.slope
    slope_map = slope_map.reshape(H, W)

    water_pcts = []
    for i in range(n):
        valid = np.isfinite(mndwi_stack[i])
        total = valid.sum()
        if total > 0:
            water_pcts.append(
                100 * np.sum((mndwi_stack[i] > 0.1) & valid) / total
            )
        else:
            water_pcts.append(np.nan)

    fig = dark_fig(figsize=(20, 14))
    fig.suptitle(
        "Scheldt Estuary — Coastal dynamics  |  Tier 1 Analysis Summary",
        color="white", fontsize=14, fontweight="bold", y=0.99
    )
    gs = GridSpec(2, 3, figure=fig,
                  hspace=0.12, wspace=0.08,
                  left=0.04, right=0.97, top=0.94, bottom=0.08)

    # 1. NDVI trend (significant)
    ax1 = fig.add_subplot(gs[0, 0])
    lim = np.nanpercentile(np.abs(slope_map[np.isfinite(slope_map)]), 95) \
          if np.any(np.isfinite(slope_map)) else 0.01
    im1 = ax1.imshow(slope_map, cmap="RdYlGn", vmin=-lim, vmax=lim,
                      aspect="equal", interpolation="nearest")
    plt.colorbar(im1, ax=ax1, fraction=0.035, pad=0.02).ax.tick_params(
        colors="white", labelsize=6)
    styled_ax(ax1, "NDVI trend  (significant p<0.10)\ngreen=gaining  red=losing")

    # 2. Instability map
    ax2 = fig.add_subplot(gs[0, 1])
    p95 = np.nanpercentile(std_map[np.isfinite(std_map)], 95)
    im2 = ax2.imshow(std_map, cmap="inferno", vmin=0, vmax=p95,
                      aspect="equal", interpolation="nearest")
    plt.colorbar(im2, ax=ax2, fraction=0.035, pad=0.02).ax.tick_params(
        colors="white", labelsize=6)
    styled_ax(ax2, "MNDWI change magnitude\nbright = dynamic / unstable")

    # 3. Water frequency
    water_binary = (ndwi_stack > NDWI_WATER_THRESHOLD).astype("float32")
    freq_map = np.nanmean(water_binary, axis=0)
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(freq_map, cmap="RdYlBu_r", vmin=0, vmax=1,
                      aspect="equal", interpolation="nearest")
    plt.colorbar(im3, ax=ax3, fraction=0.035, pad=0.02).ax.tick_params(
        colors="white", labelsize=6)
    styled_ax(ax3, "Water frequency  (0→1)\n0.3–0.7 = dynamic fringe")

    # 4. Open water time series
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_facecolor("#1a1a1a")
    x_ts = np.arange(len(labels))
    ax4.plot(x_ts, water_pcts, color="#2196F3", lw=2,
             marker="o", markersize=6,
             markerfacecolor="white", markeredgecolor="#2196F3")
    ax4.fill_between(x_ts, water_pcts, alpha=0.15, color="#2196F3")
    ax4.set_xticks(x_ts)
    ax4.set_xticklabels(labels, rotation=45, ha="right",
                         color="white", fontsize=7)
    ax4.set_ylabel("% open water", color="white", fontsize=8)
    ax4.grid(axis="y", color="#333", lw=0.5)
    ax4.tick_params(colors="white", labelsize=7)
    ax4.spines[:].set_color("#444")
    styled_ax(ax4, "Open water area over time")

    # 5. Mean NDVI map
    ax5 = fig.add_subplot(gs[1, 1])
    im5 = ax5.imshow(ndvi_mean, cmap="RdYlGn", vmin=-0.2, vmax=0.8,
                      aspect="equal", interpolation="nearest")
    plt.colorbar(im5, ax=ax5, fraction=0.035, pad=0.02).ax.tick_params(
        colors="white", labelsize=6)
    styled_ax(ax5, "Mean NDVI across all snapshots\nbaseline vegetation map")

    # 6. NDVI std dev (variability in vegetation)
    ndvi_std = np.nanstd(ndvi_stack, axis=0)
    ax6 = fig.add_subplot(gs[1, 2])
    p95n = np.nanpercentile(ndvi_std[np.isfinite(ndvi_std)], 95)
    im6 = ax6.imshow(ndvi_std, cmap="YlOrRd", vmin=0, vmax=p95n,
                      aspect="equal", interpolation="nearest")
    plt.colorbar(im6, ax=ax6, fraction=0.035, pad=0.02).ax.tick_params(
        colors="white", labelsize=6)
    styled_ax(ax6, "NDVI variability (std dev)\nhigh = seasonally or inter-annually dynamic")

    save(fig, out_dir / "00_summary.png")


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python tier1_dynamics.py ./data/scheldt/sentinel2/")

    tif_dir = Path(sys.argv[1])
    out_dir = tif_dir / "analysis"
    out_dir.mkdir(exist_ok=True)
    print(f"Output folder: {out_dir}\n")

    (ndvi_stack, ndwi_stack, mndwi_stack,
     labels, profile, transform, crs, tifs) = load_all(tif_dir)

    analysis_waterline(ndwi_stack, labels, transform, out_dir)
    analysis_ndvi_trend(ndvi_stack, labels, out_dir)
    analysis_mudflat_timeseries(ndvi_stack, ndwi_stack, mndwi_stack,
                                 labels, tifs, out_dir)
    analysis_instability(mndwi_stack, ndvi_stack, labels, out_dir)
    summary_figure(ndvi_stack, ndwi_stack, mndwi_stack, labels, out_dir)

    print(f"\n✓ All outputs saved to {out_dir}")
    print("  01_waterline_migration.png")
    print("  02_ndvi_trend.png")
    print("  03_mudflat_timeseries.png")
    print("  04_instability_map.png")
    print("  00_summary.png   ← portfolio summary page")
