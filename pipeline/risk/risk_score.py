"""
Scheldt Estuary — Coastal Instability Risk Score
=================================================
Combines five evidence layers into a single composite
instability index per pixel.

Risk = w1*MNDWI_instability + w2*NDVI_loss + w3*slope
     + w4*plan_curvature    + w5*EAD_exposure

All layers are normalised to [0,1] before weighting.
Output is a risk score in [0,1] where 1 = highest instability.

Input layers (all at 10m — S2 grid)
-------------------------------------
    MNDWI std dev          tier1 analysis output
    NDVI trend (negative)  tier1 analysis output
    Slope                  AHN4 derivatives resampled to 10m
    Plan curvature (neg.)  AHN4 derivatives resampled to 10m
    EAD (inverted)         compute_ead output resampled to 10m

Outputs
-------
    risk/risk_score.tif          composite index [0,1]
    risk/risk_classified.tif     4-class map (Low/Med/High/Critical)
    risk/risk_overview.png       portfolio summary figure
    risk/risk_weights.json       weights used

Usage
-----
    # Default equal weights
    python risk_score.py

    # Custom weights (must sum to 1.0)
    python risk_score.py --w1 0.30 --w2 0.20 --w3 0.15 --w4 0.15 --w5 0.20

    # Sensitivity analysis (run with multiple weight sets)
    python risk_score.py --sensitivity
"""

import json
import argparse
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import rioxarray as rxr
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.ndimage import gaussian_filter

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ── ARGUMENT PARSING ─────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--w1", type=float, default=0.25,
                    help="Weight: MNDWI instability (default 0.25)")
parser.add_argument("--w2", type=float, default=0.20,
                    help="Weight: NDVI loss (default 0.20)")
parser.add_argument("--w3", type=float, default=0.20,
                    help="Weight: Slope (default 0.20)")
parser.add_argument("--w4", type=float, default=0.15,
                    help="Weight: Plan curvature (default 0.15)")
parser.add_argument("--w5", type=float, default=0.20,
                    help="Weight: EAD exposure (default 0.20)")
parser.add_argument("--sensitivity", action="store_true",
                    help="Run sensitivity analysis with multiple weight sets")
args = parser.parse_args()

# Normalise weights to sum to 1
raw_weights = np.array([args.w1, args.w2, args.w3, args.w4, args.w5])
weights = raw_weights / raw_weights.sum()

# ── PATHS ─────────────────────────────────────────────────────────────────────

# Tier-1 outputs (S2 analysis — already at 10m)
TIER1_DIR = Path("./data/scheldt/sentinel2/analysis")

# AHN4 derivatives resampled to 10m
AHN4_10M  = Path("./data/scheldt/dem_ahn4/derivatives/resampled_10m")

# EAD resampled to 10m
EAD_10M   = AHN4_10M / "EAD.tif"

# Output
OUT_DIR   = Path("./data/scheldt/risk")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# S2 reference for grid
S2_REF    = next(Path("./data/scheldt/sentinel2").glob("S2_*.tif"), None)


# ── LAYER PATHS ───────────────────────────────────────────────────────────────

LAYER_DEFS = {
    "mndwi_instability": {
        "path":    TIER1_DIR / "04_instability_map_mndwi_std.tif",
        "fallback": TIER1_DIR / "mndwi_std.tif",
        "desc":    "MNDWI temporal std dev",
        "invert":  False,   # high std = high instability = high risk
    },
    "ndvi_loss": {
        "path":    TIER1_DIR / "02_ndvi_slope_negative.tif",
        "fallback": TIER1_DIR / "ndvi_trend.tif",
        "desc":    "NDVI negative trend (browning)",
        "invert":  False,   # already encoded as loss
    },
    "slope": {
        "path":    AHN4_10M / "slope_degrees.tif",
        "fallback": Path("./data/scheldt/dem/derivatives/resampled_10m/slope_degrees.tif"),
        "desc":    "Terrain slope (°)",
        "invert":  False,   # steeper = more exposed bank face
    },
    "plan_curvature": {
        "path":    AHN4_10M / "plan_curvature.tif",
        "fallback": Path("./data/scheldt/dem/derivatives/resampled_10m/plan_curvature.tif"),
        "desc":    "Plan curvature (convergent = negative)",
        "invert":  True,    # convergent (negative) = erosion-prone
    },
    "ead_exposure": {
        "path":    EAD_10M,
        "fallback": Path("./data/scheldt/dem_ahn4/derivatives/EAD.tif"),
        "desc":    "Elevation above datum (inverted)",
        "invert":  True,    # low EAD = high exposure
    },
}


# ── HELPERS ───────────────────────────────────────────────────────────────────

def load_layer(name: str, ref_shape: tuple,
               ref_transform, ref_crs) -> np.ndarray | None:
    """
    Load a layer, reproject to reference grid if needed.
    Returns float32 array or None if not found.
    """
    defn = LAYER_DEFS[name]
    path = defn["path"]

    if not path.exists():
        path = defn["fallback"]
    if not path.exists():
        print(f"  ✗ {name}: not found at {defn['path']}")
        print(f"           or {defn['fallback']}")
        return None

    try:
        da = rxr.open_rasterio(path, masked=True)
        arr = da.values[0].astype("float32")
        H, W = ref_shape

        # Reproject if shape doesn't match
        if arr.shape != (H, W):
            try:
                da_repr = da.rio.reproject(
                    ref_crs,
                    shape=(H, W),
                    resampling=Resampling.bilinear,
                )
                arr = da_repr.values[0].astype("float32")
            except Exception:
                # Fallback: numpy resize (approximate)
                from scipy.ndimage import zoom as nd_zoom
                arr = nd_zoom(arr,
                              (H / arr.shape[0], W / arr.shape[1]),
                              order=1).astype("float32")

        print(f"  ✓ {name:20s}: {arr.shape}  "
              f"range [{np.nanmin(arr):.4f}, {np.nanmax(arr):.4f}]")
        return arr

    except Exception as e:
        print(f"  ✗ {name}: {e}")
        return None


def normalise(arr: np.ndarray,
              p_low: float = 5.0,
              p_high: float = 95.0) -> np.ndarray:
    """
    Percentile stretch to [0, 1].
    Handles zero-heavy distributions (e.g. NDVI loss, MNDWI std)
    by using the NON-ZERO distribution for calibration.
    NaN preserved.
    """
    valid = arr[np.isfinite(arr)]
    if len(valid) == 0:
        return arr

    # Check if distribution is zero-heavy (>50% zeros)
    zero_frac = np.mean(valid == 0.0)
    if zero_frac > 0.50:
        # Use non-zero pixels only for percentile calibration
        nonzero = valid[valid > 0]
        if len(nonzero) == 0:
            return np.zeros_like(arr)
        lo = 0.0
        hi = np.percentile(nonzero, 90)   # p90 of non-zero values
    else:
        lo = np.percentile(valid, p_low)
        hi = np.percentile(valid, p_high)

    if hi <= lo:
        return np.zeros_like(arr)
    normed = (arr - lo) / (hi - lo)
    return np.clip(normed, 0.0, 1.0)


def build_ndvi_loss(tier1_dir: Path, shape: tuple,
                    ref_crs, ref_transform) -> np.ndarray | None:
    """
    Build NDVI loss layer from tier1 ndvi_trend output.
    Negative slopes = browning = risk signal.
    Positive slopes = greening = low risk.
    """
    # Try loading the raw ndvi slope tif
    candidates = [
        tier1_dir / "ndvi_trend.tif",
        tier1_dir / "02_ndvi_trend.tif",
        tier1_dir / "ndvi_slope.tif",
    ]
    for p in candidates:
        if p.exists():
            da  = rxr.open_rasterio(p, masked=True)
            arr = da.values[0].astype("float32")
            # Invert: make negative slopes positive (loss = risk)
            loss = np.where(arr < 0, -arr, 0.0)
            loss = np.where(~np.isfinite(arr), np.nan, loss)
            return loss

    # If no precomputed tif, try to load ndvi stack and compute slope
    print(f"    NDVI trend tif not found — attempting to compute from stack ...")
    tifs = sorted(Path("./data/scheldt/sentinel2").glob("S2_*.tif"))
    if len(tifs) < 3:
        return None

    from scipy import stats
    ds0   = rxr.open_rasterio(tifs[0], masked=True)
    H0, W0 = ds0.values[0].shape
    n = len(tifs)
    stack = np.full((n, H0, W0), np.nan, dtype="float32")

    for i, tif in enumerate(tifs):
        ds = rxr.open_rasterio(tif, masked=True)
        b8 = ds.sel(band=4).values.astype("float32")
        b4 = ds.sel(band=3).values.astype("float32")
        d  = b8 + b4
        stack[i] = np.where(np.abs(d) < 1e-6, np.nan,
                             (b8 - b4) / d)

    x    = np.arange(n, dtype="float32")
    flat = stack.reshape(n, -1)
    slopes = np.full(H0 * W0, np.nan, dtype="float32")
    for px in range(flat.shape[1]):
        y     = flat[:, px]
        valid = np.isfinite(y)
        if valid.sum() >= 4:
            res = stats.linregress(x[valid], y[valid])
            if res.pvalue < 0.10:
                slopes[px] = res.slope

    slope_map = slopes.reshape(H0, W0)
    loss = np.where(slope_map < 0, -slope_map, 0.0)
    return loss.astype("float32")


def build_mndwi_std(tier1_dir: Path) -> np.ndarray | None:
    """Build MNDWI std dev layer from S2 stack if not already computed."""
    candidates = [
        tier1_dir / "mndwi_std.tif",
        tier1_dir / "04_instability_map_mndwi_std.tif",
        tier1_dir / "instability.tif",
    ]
    for p in candidates:
        if p.exists():
            return rxr.open_rasterio(p, masked=True).values[0].astype("float32")

    print("    MNDWI std tif not found — computing from S2 stack ...")
    tifs = sorted(Path("./data/scheldt/sentinel2").glob("S2_*.tif"))
    if len(tifs) < 2:
        return None

    ds0 = rxr.open_rasterio(tifs[0], masked=True)
    H0, W0 = ds0.values[0].shape
    n = len(tifs)
    stack = np.full((n, H0, W0), np.nan, dtype="float32")

    for i, tif in enumerate(tifs):
        ds  = rxr.open_rasterio(tif, masked=True)
        g   = ds.sel(band=2).values.astype("float32")
        sw  = ds.sel(band=5).values.astype("float32")
        d   = g + sw
        stack[i] = np.where(np.abs(d) < 1e-6, np.nan, (g - sw) / d)

    return np.nanstd(stack, axis=0).astype("float32")


# ── LOAD REFERENCE GRID FROM S2 ───────────────────────────────────────────────

print("=== Coastal Instability Risk Score ===\n")
print(f"Weights: MNDWI={weights[0]:.2f}  NDVI={weights[1]:.2f}  "
      f"Slope={weights[2]:.2f}  Curv={weights[3]:.2f}  EAD={weights[4]:.2f}\n")

if S2_REF is None:
    raise FileNotFoundError("No S2 reference found in sentinel2/")

with rasterio.open(S2_REF) as ref:
    ref_shape     = (ref.height, ref.width)
    ref_transform = ref.transform
    ref_crs       = ref.crs.to_string()

print(f"Reference grid: {ref_shape[0]}×{ref_shape[1]} px "
      f"({ref_crs})\n")


# ── LOAD ALL LAYERS ───────────────────────────────────────────────────────────

print("Loading layers ...")

# MNDWI instability — try precomputed first, then compute
mndwi_raw = build_mndwi_std(TIER1_DIR)
if mndwi_raw is None:
    mndwi_raw = load_layer("mndwi_instability", ref_shape,
                            ref_transform, ref_crs)
else:
    print(f"  ✓ {'mndwi_instability':20s}: {mndwi_raw.shape}  "
          f"range [{np.nanmin(mndwi_raw):.4f}, {np.nanmax(mndwi_raw):.4f}]")

# NDVI loss — try precomputed first, then compute
ndvi_raw = build_ndvi_loss(TIER1_DIR, ref_shape, ref_crs, ref_transform)
if ndvi_raw is None:
    ndvi_raw = load_layer("ndvi_loss", ref_shape, ref_transform, ref_crs)
else:
    print(f"  ✓ {'ndvi_loss':20s}: {ndvi_raw.shape}  "
          f"range [{np.nanmin(ndvi_raw):.5f}, {np.nanmax(ndvi_raw):.5f}]")

# Slope, curvature, EAD
slope_raw = load_layer("slope",         ref_shape, ref_transform, ref_crs)
curv_raw  = load_layer("plan_curvature",ref_shape, ref_transform, ref_crs)
ead_raw   = load_layer("ead_exposure",  ref_shape, ref_transform, ref_crs)


# ── NORMALISE LAYERS ─────────────────────────────────────────────────────────

print("\nNormalising layers to [0, 1] ...")

layers     = {}
layer_names = ["MNDWI instability", "NDVI loss",
                "Slope", "Plan curvature", "EAD exposure"]
raw_arrays  = [mndwi_raw, ndvi_raw, slope_raw, curv_raw, ead_raw]
invert_flags = [False, False, False, True, True]

for name, raw, inv in zip(layer_names, raw_arrays, invert_flags):
    if raw is None:
        print(f"  ✗ {name}: missing — skipping")
        layers[name] = None
        continue

    # Resize to reference grid if needed
    if raw.shape != ref_shape:
        from scipy.ndimage import zoom as nd_zoom
        raw = nd_zoom(raw,
                      (ref_shape[0]/raw.shape[0],
                       ref_shape[1]/raw.shape[1]),
                      order=1).astype("float32")

    normed = normalise(raw)

    # Invert if needed (low value = high risk)
    if inv:
        normed = 1.0 - normed

    layers[name] = normed
    print(f"  ✓ {name:25s}: "
          f"[{np.nanmin(normed):.3f}, {np.nanmax(normed):.3f}]"
          + (" (inverted)" if inv else ""))


# ── COMPOSITE RISK SCORE ──────────────────────────────────────────────────────

print("\nComputing composite risk score ...")

available   = [(w, layers[n]) for w, n in zip(weights, layer_names)
               if layers[n] is not None]
actual_w    = np.array([w for w, _ in available])
actual_w    = actual_w / actual_w.sum()   # re-normalise for missing layers
arrs        = [a for _, a in available]

# Build common valid mask
valid_mask = np.ones(ref_shape, dtype=bool)
for arr in arrs:
    valid_mask &= np.isfinite(arr)

# Weighted sum
risk = np.zeros(ref_shape, dtype="float32")
for w, arr in zip(actual_w, arrs):
    risk += w * np.where(np.isfinite(arr), arr, 0.0)

risk = np.where(valid_mask, risk, np.nan)

# Light spatial smoothing — removes salt-and-pepper noise
# sigma=1 px at 10m = 10m effective smoothing
risk_smooth = gaussian_filter(
    np.where(np.isfinite(risk), risk, 0.0), sigma=1.0
)
weight_smooth = gaussian_filter(
    np.isfinite(risk).astype("float32"), sigma=1.0
)
risk_smooth = np.where(
    weight_smooth > 0.1,
    risk_smooth / weight_smooth,
    np.nan
).astype("float32")
risk_smooth = np.where(valid_mask, risk_smooth, np.nan)

valid_risk = risk_smooth[np.isfinite(risk_smooth)]
print(f"  Risk range : [{valid_risk.min():.4f}, {valid_risk.max():.4f}]")
print(f"  Risk mean  : {valid_risk.mean():.4f}")
print(f"  Risk std   : {valid_risk.std():.4f}")


# ── CLASSIFY RISK ─────────────────────────────────────────────────────────────

# Absolute thresholds based on physical meaning
# Calibrated to the Scheldt macrotidal context
# Thresholds calibrated to Scheldt macrotidal context
# After fixing layer normalisation, scores should span 0.15–0.70
# Adjust these if the histogram shows a different range
T_LOW  = 0.25   # below this = genuinely stable (bottom tercile)
T_MED  = 0.35   # below this = moderate exposure
T_HIGH = 0.45   # below this = high instability risk

# Also compute quartiles for reference
p25 = np.nanpercentile(risk_smooth, 25)
p50 = np.nanpercentile(risk_smooth, 50)
p75 = np.nanpercentile(risk_smooth, 75)

classified = np.full(ref_shape, np.nan, dtype="float32")
classified[np.isfinite(risk_smooth) & (risk_smooth <= T_LOW)]  = 1  # Low
classified[np.isfinite(risk_smooth) & (risk_smooth >  T_LOW) &
           (risk_smooth <= T_MED)]  = 2  # Medium
classified[np.isfinite(risk_smooth) & (risk_smooth >  T_MED) &
           (risk_smooth <= T_HIGH)] = 3  # High
classified[np.isfinite(risk_smooth) & (risk_smooth >  T_HIGH)] = 4  # Critical

print(f"\n  Class thresholds (absolute):")
print(f"    Low      (1): score ≤ {T_LOW}")
print(f"    Medium   (2): {T_LOW} < score ≤ {T_MED}")
print(f"    High     (3): {T_MED} < score ≤ {T_HIGH}")
print(f"    Critical (4): score > {T_HIGH}")
print(f"\n  Quartiles for reference: "
      f"p25={p25:.3f}  p50={p50:.3f}  p75={p75:.3f}")

for cls, label in [(1,"Low"), (2,"Medium"), (3,"High"), (4,"Critical")]:
    pct = 100 * np.nanmean(classified == cls)
    print(f"    {label:8s}: {pct:.1f}%")


# ── SAVE OUTPUTS ─────────────────────────────────────────────────────────────

def save_tif(arr, path):
    with rasterio.open(S2_REF) as ref:
        prof = ref.profile.copy()
    prof.update(count=1, dtype="float32",
                nodata=float("nan"), compress="lzw")
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr[np.newaxis, :, :].astype("float32"))
    print(f"  ✓ {path.name}")

print("\nSaving outputs ...")
save_tif(risk_smooth,  OUT_DIR / "risk_score.tif")
save_tif(classified,   OUT_DIR / "risk_classified.tif")

# Save each normalised layer too
for name, arr in layers.items():
    if arr is not None:
        out = OUT_DIR / f"layer_{name.lower().replace(' ','_')}.tif"
        save_tif(arr, out)

# Save weights
weights_out = {
    "weights": {
        name: float(w)
        for name, w in zip(layer_names, actual_w)
    },
    "thresholds": {
        "Low":      T_LOW,
        "Medium":   T_MED,
        "High":     T_HIGH,
        "Critical": float(valid_risk.max()),
        "type":     "absolute",
        "p25":      float(p25),
        "p50":      float(p50),
        "p75":      float(p75),
    },
    "n_layers_used": len(available),
}
with open(OUT_DIR / "risk_weights.json", "w") as f:
    json.dump(weights_out, f, indent=2)
print(f"  ✓ risk_weights.json")


# ── VISUALISE ────────────────────────────────────────────────────────────────

print("\nGenerating figure ...")

fig = plt.figure(figsize=(24, 16), facecolor="#0e0e0e")
gs  = fig.add_gridspec(3, 4, hspace=0.10, wspace=0.08,
                        left=0.03, right=0.97,
                        top=0.92, bottom=0.04)

# ── Row 1: input layers ───────────────────────────────────────────────────────
# Map display config to exact layer_names keys
layer_panels = [
    ("MNDWI instability\n(temporal variability)",
     "MNDWI instability", "inferno",  False),
    ("NDVI loss\n(browning trend)",
     "NDVI loss",          "RdYlGn_r", False),
    ("Slope\n(bank steepness)",
     "Slope",              "YlOrRd",   False),
    ("EAD exposure\n(low elevation = risk)",
     "EAD exposure",       "RdYlBu",   True),
]

for col, (title, layer_key, cmap, invert_display) in enumerate(layer_panels):
    ax = fig.add_subplot(gs[0, col])
    arr = layers.get(layer_key)
    if arr is None:
        ax.text(0.5, 0.5, f"Missing:\n{layer_key}",
                transform=ax.transAxes, color="white",
                ha="center", va="center", fontsize=8)
        ax.set_facecolor("#1a1a1a")
        ax.axis("off")
        continue
    # For inverted layers, display the pre-inversion form for readability
    display = 1.0 - arr if invert_display else arr
    im = ax.imshow(display, cmap=cmap, vmin=0, vmax=1,
                    aspect="equal", interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02
                 ).ax.tick_params(colors="white", labelsize=6)
    ax.set_title(title, color="white", fontsize=8,
                  fontweight="bold", pad=3)
    ax.axis("off")
    ax.set_facecolor("#0e0e0e")

# ── Row 2: risk score + classified ───────────────────────────────────────────
ax_risk = fig.add_subplot(gs[1, :2])
risk_cmap = plt.cm.RdYlGn_r
im_r = ax_risk.imshow(risk_smooth, cmap=risk_cmap, vmin=0, vmax=1,
                       aspect="equal", interpolation="bilinear")
cb_r = plt.colorbar(im_r, ax=ax_risk, fraction=0.025, pad=0.02)
cb_r.set_label("Risk score [0–1]", color="white", fontsize=8)
cb_r.ax.tick_params(colors="white", labelsize=7)
ax_risk.set_title("Composite instability risk score\n"
                   f"(weighted sum of {len(available)} layers)",
                   color="white", fontsize=9, fontweight="bold")
ax_risk.axis("off")
ax_risk.set_facecolor("#0e0e0e")

ax_cls = fig.add_subplot(gs[1, 2:])
cls_colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]
cls_cmap   = mcolors.ListedColormap(cls_colors)
im_c = ax_cls.imshow(classified, cmap=cls_cmap,
                      vmin=0.5, vmax=4.5,
                      aspect="equal", interpolation="nearest")
legend_patches = [
    mpatches.Patch(color=cls_colors[0], label="Low"),
    mpatches.Patch(color=cls_colors[1], label="Medium"),
    mpatches.Patch(color=cls_colors[2], label="High"),
    mpatches.Patch(color=cls_colors[3], label="Critical"),
]
ax_cls.legend(handles=legend_patches, loc="lower left",
               fontsize=8, facecolor="#222",
               labelcolor="white", framealpha=0.9)
ax_cls.set_title("Classified instability zones\n"
                  "(Low / Medium / High / Critical)",
                  color="white", fontsize=9, fontweight="bold")
ax_cls.axis("off")
ax_cls.set_facecolor("#0e0e0e")

# ── Row 3: histogram + weight bar + summary ───────────────────────────────────
ax_hist = fig.add_subplot(gs[2, :2])
ax_hist.set_facecolor("#1a1a1a")
ax_hist.hist(valid_risk, bins=150, color="#FF9800", alpha=0.85)
for thresh, lbl, col in [
    (T_LOW,  f"Low→Med ({T_LOW})",   "#4CAF50"),
    (T_MED,  f"Med→High ({T_MED})",  "#FF9800"),
    (T_HIGH, f"High→Crit ({T_HIGH})","#F44336"),
]:
    ax_hist.axvline(thresh, color=col, lw=1.5, ls="--", label=lbl)
ax_hist.set_xlabel("Risk score", color="white", fontsize=9)
ax_hist.set_ylabel("Pixel count", color="white", fontsize=9)
ax_hist.tick_params(colors="white", labelsize=8)
ax_hist.spines[:].set_color("#444")
ax_hist.legend(fontsize=8, facecolor="#1a1a1a",
               labelcolor="white", framealpha=0.9)
ax_hist.set_title("Risk score distribution",
                   color="white", fontsize=9, fontweight="bold")

ax_wt = fig.add_subplot(gs[2, 2])
ax_wt.set_facecolor("#1a1a1a")
short_names = ["MNDWI\ninstab.", "NDVI\nloss",
               "Slope", "Plan\ncurv.", "EAD\nexpos."]
bar_cols = ["#42A5F5", "#66BB6A", "#FF7043", "#AB47BC", "#EC407A"]
bars = ax_wt.bar(short_names[:len(actual_w)],
                  actual_w * 100,
                  color=bar_cols[:len(actual_w)],
                  alpha=0.9, edgecolor="#333")
for bar, val in zip(bars, actual_w * 100):
    ax_wt.text(bar.get_x() + bar.get_width()/2,
               bar.get_height() + 0.5,
               f"{val:.0f}%", ha="center", va="bottom",
               color="white", fontsize=8)
ax_wt.set_ylabel("Weight (%)", color="white", fontsize=9)
ax_wt.tick_params(colors="white", labelsize=8)
ax_wt.spines[:].set_color("#444")
ax_wt.set_title("Layer weights", color="white",
                 fontsize=9, fontweight="bold")

ax_sum = fig.add_subplot(gs[2, 3])
ax_sum.set_facecolor("#1a1a1a")
ax_sum.axis("off")
summary = (
    f"RISK SUMMARY — Scheldt AOI\n"
    f"{'─'*32}\n"
    f"Layers used     : {len(available)}/5\n"
    f"Valid pixels    : {np.sum(np.isfinite(risk_smooth)):,}\n"
    f"\n"
    f"Mean risk score : {valid_risk.mean():.3f}\n"
    f"Std dev         : {valid_risk.std():.3f}\n"
    f"\n"
    f"Low (≤0.30)     : {100*np.nanmean(classified==1):.1f}%\n"
    f"Medium (0.30-0.45): {100*np.nanmean(classified==2):.1f}%\n"
    f"High (0.45-0.60): {100*np.nanmean(classified==3):.1f}%\n"
    f"Critical (>0.60): {100*np.nanmean(classified==4):.1f}%\n"
    f"\n"
    f"MHW (Vlissingen): +1.979 m NAP\n"
    f"Tidal range     : 3.676 m\n"
    f"AOI coverage    : NL bank only\n"
    f"DEM source      : AHN4 5m lidar\n"
    f"S2 snapshots    : 14 (2017–2025)"
)
ax_sum.text(0.05, 0.95, summary,
             transform=ax_sum.transAxes,
             color="white", fontsize=7.5,
             va="top", fontfamily="monospace",
             bbox=dict(facecolor="#111", alpha=0.9,
                       boxstyle="round", edgecolor="#444"))

fig.suptitle(
    "Scheldt Estuary — Coastal Instability Risk Index  |  "
    "Sentinel-2 + AHN4 Lidar + Vlissingen Tide Gauge",
    color="white", fontsize=12, fontweight="bold"
)

out_png = OUT_DIR / "risk_overview.png"
fig.savefig(str(out_png), dpi=150,
            bbox_inches="tight", facecolor="#0e0e0e")
print(f"\n✓ Saved → {out_png}")
plt.show()

print("\n✓ Risk scoring complete.")
print(f"\nOutputs in {OUT_DIR}:")
for f in sorted(OUT_DIR.glob("*.tif")):
    print(f"  {f.name}")
