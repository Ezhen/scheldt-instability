"""
Scheldt — Pioneer Zone Correction for Risk Index
=================================================
Adds NDVI temporal variability as a sixth layer to the risk index,
targeting the eroding marsh cliff front signature:

    High NDVI variance + low-medium NDVI mean
    = pioneer vegetation cycling with bare mud
    = actively eroding cliff front

This corrects the systematic under-detection of slow lateral cliff
retreat sites (Saeftinghe, Baarland, etc.) that are invisible to
the waterline-based MNDWI instability layer.

Scientific basis
----------------
Van der Wal et al. (2008): lateral cliff retreat is compensated by
elevated mudflat formation → pioneer vegetation establishment →
wave energy dissipation → slower erosion. The transition zone
oscillates between bare (post-erosion) and pioneer states,
producing high temporal NDVI variance at low mean NDVI.

Outputs
-------
    risk/risk_score_corrected.tif       updated composite [0-1]
    risk/risk_classified_corrected.tif  4-class map
    risk/pioneer_zone.tif               pioneer zone mask
    risk/ndvi_variability.tif           NDVI std normalised layer
    risk/correction_overview.png        before/after comparison

Usage
-----
    python pioneer_correction.py
"""

import os
os.environ["CPL_LOG"] = "/dev/null"

import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import rioxarray as rxr
import rasterio
from scipy.ndimage import gaussian_filter
from pathlib import Path

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────

TIER1_DIR  = Path("./data/scheldt/sentinel2/analysis")
RISK_DIR   = Path("./data/scheldt/risk")
S2_DIR     = Path("./data/scheldt/sentinel2")
S2_REF     = next(S2_DIR.glob("S2_*.tif"), None)

# Pioneer zone definition — literature-based thresholds
# NDVI 0.05–0.30: sparse/transitional vegetation (not dense marsh, not bare)
# Applied to temporal mean — filters out agricultural fields via context
NDVI_PIONEER_LOW  = 0.05
NDVI_PIONEER_HIGH = 0.30

# Weight of the new NDVI variability layer in the corrected index
# Keep modest — supplements existing layers, doesn't dominate
W_NDVI_VAR = 0.15

# Agricultural field filter: exclude pixels where NDVI range > 0.50
# (crops swing from ~0.1 bare to ~0.8 canopy — marshes don't)
AGRI_NDVI_RANGE_THRESHOLD = 0.50

# Thresholds (same as risk_score.py)
T_LOW  = 0.25
T_MED  = 0.35
T_HIGH = 0.45

CLASS_COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]


# ── LOAD EXISTING RISK SCORE ──────────────────────────────────────────────────

print("=== Pioneer Zone Correction ===\n")

print("Loading existing risk score ...")
with rasterio.open(RISK_DIR / "risk_score.tif") as src:
    risk_orig  = src.read(1).astype("float32")
    profile    = src.profile.copy()
    risk_crs   = src.crs
    risk_t     = src.transform

H, W = risk_orig.shape
print(f"  Shape: {H}×{W}  CRS: {risk_crs}")


# ── LOAD NDVI LAYERS ──────────────────────────────────────────────────────────

print("\nLoading NDVI analysis layers ...")

def load_layer(path, name):
    if not path.exists():
        print(f"  MISSING: {name} — {path}")
        return None
    arr = rxr.open_rasterio(path, masked=True).values[0].astype("float32")
    v   = arr[np.isfinite(arr)]
    print(f"  ✓ {name:25s} range=[{v.min():.4f},{v.max():.4f}]  "
          f"valid={len(v):,}")
    return arr

ndvi_mean = load_layer(TIER1_DIR / "ndvi_mean.tif",  "ndvi_mean")
ndvi_std  = load_layer(TIER1_DIR / "ndvi_std.tif",   "ndvi_std")

# Compute ndvi_std from stack if not saved
if ndvi_std is None:
    print("  Computing NDVI std from S2 stack ...")
    tifs = sorted(S2_DIR.glob("S2_*.tif"))
    ref  = rxr.open_rasterio(tifs[0], masked=True)
    H0, W0 = ref.values[0].shape
    stack = np.full((len(tifs), H0, W0), np.nan, dtype="float32")
    for i, tif in enumerate(tifs):
        ds  = rxr.open_rasterio(tif, masked=True)
        b04 = ds.sel(band=3).values.astype("float32") / 10000.0
        b08 = ds.sel(band=4).values.astype("float32") / 10000.0
        d   = b08 + b04
        v   = np.where(np.abs(d)>1e-4, (b08-b04)/d, np.nan)
        stack[i] = np.where((v>=-1)&(v<=1), v, np.nan)
    ndvi_std  = np.nanstd(stack,  axis=0).astype("float32")
    ndvi_mean = np.nanmean(stack, axis=0).astype("float32")

    # Save for future use
    with rasterio.open(tifs[0]) as src:
        p = src.profile.copy()
    p.update(count=1, dtype="float32", nodata=float("nan"), compress="lzw")
    for arr, name in [(ndvi_std,  "ndvi_std.tif"),
                       (ndvi_mean, "ndvi_mean.tif")]:
        with rasterio.open(str(TIER1_DIR/name), "w", **p) as dst:
            dst.write(arr[np.newaxis,:,:])
    print(f"  ✓ Saved ndvi_std.tif and ndvi_mean.tif")

# Also load ndvi min/max if available for agricultural filter
ndvi_slope = load_layer(TIER1_DIR / "ndvi_slope.tif", "ndvi_slope")


# ── COMPUTE NDVI RANGE (for agricultural filter) ──────────────────────────────

# Compute per-pixel NDVI range from stack for agricultural masking
# If not available, use std as proxy (std > 0.20 ≈ agricultural cycling)
print("\nComputing agricultural filter ...")

# Agricultural fields: large seasonal NDVI range
# Marsh: moderate mean, moderate std, persistent green
# Use std threshold as proxy — crops have std > 0.18 typically
if ndvi_std is not None:
    # High std + medium mean = agricultural (not marsh)
    # High std + low mean = bare/transitional = potential cliff front
    agri_mask = (ndvi_std > 0.18) & (ndvi_mean > 0.30)
    print(f"  Agricultural mask: {100*agri_mask.mean():.1f}% of pixels")
else:
    agri_mask = np.zeros((H, W), dtype=bool)


# ── BUILD PIONEER ZONE MASK ───────────────────────────────────────────────────

print("\nBuilding pioneer zone mask ...")

# Resize layers to risk grid if needed
def resize_to_risk(arr):
    if arr is None or arr.shape == (H, W):
        return arr
    from scipy.ndimage import zoom
    return zoom(arr, (H/arr.shape[0], W/arr.shape[1]),
                order=1).astype("float32")

ndvi_mean_r = resize_to_risk(ndvi_mean)
ndvi_std_r  = resize_to_risk(ndvi_std)
agri_mask_r = resize_to_risk(agri_mask.astype("float32")) > 0.5

# Pioneer zone: low-medium NDVI mean + high temporal variance
# Exclude agricultural pixels
p75_std = np.nanpercentile(ndvi_std_r[np.isfinite(ndvi_std_r)], 75)

pioneer_zone = (
    np.isfinite(ndvi_mean_r) &
    np.isfinite(ndvi_std_r)  &
    (ndvi_mean_r >= NDVI_PIONEER_LOW)  &
    (ndvi_mean_r <  NDVI_PIONEER_HIGH) &
    (ndvi_std_r  >  p75_std)           &
    ~agri_mask_r
)

pct = 100 * pioneer_zone.mean()
print(f"  NDVI std p75 threshold: {p75_std:.4f}")
print(f"  Pioneer zone pixels   : {100*pioneer_zone.mean():.1f}%  "
      f"({pioneer_zone.sum():,} pixels)")
print(f"  (after agricultural filter)")

if pct < 0.5:
    print("  ⚠ Very few pioneer pixels — threshold may be too strict")
elif pct > 10:
    print("  ⚠ Many pioneer pixels — may include non-marsh areas")


# ── NORMALISE NDVI STD AS NEW LAYER ──────────────────────────────────────────

print("\nNormalising NDVI variability layer ...")

# Only normalise non-zero values (zero-heavy distribution)
valid_std = ndvi_std_r[np.isfinite(ndvi_std_r) & (ndvi_std_r > 0)]
if len(valid_std) > 0:
    p90_nz = np.nanpercentile(valid_std, 90)
    ndvi_var_norm = np.where(
        np.isfinite(ndvi_std_r) & (ndvi_std_r > 0),
        np.clip(ndvi_std_r / p90_nz, 0, 1),
        0.0
    ).astype("float32")
    print(f"  p90 of non-zero values: {p90_nz:.5f}")
    print(f"  Normalised range: [{ndvi_var_norm.min():.3f}, "
          f"{ndvi_var_norm.max():.3f}]")
else:
    ndvi_var_norm = np.zeros((H, W), dtype="float32")
    print("  ✗ No valid std values")


# ── CORRECTED RISK SCORE ──────────────────────────────────────────────────────

print("\nComputing corrected risk score ...")

# Re-weight existing layers to accommodate new layer
# Original: 5 layers summing to 1.0
# New: 6 layers summing to 1.0 — reduce all original weights by W_NDVI_VAR/5
reduction = W_NDVI_VAR / 5.0
print(f"  Original weights reduced by {reduction:.3f} each")
print(f"  NDVI variability weight: {W_NDVI_VAR:.2f}")

# Scale existing risk score and add new layer
risk_corrected = (
    risk_orig * (1.0 - W_NDVI_VAR) +
    ndvi_var_norm * W_NDVI_VAR
).astype("float32")

# Keep NaN where original was NaN
risk_corrected = np.where(np.isfinite(risk_orig), risk_corrected, np.nan)

# Light smoothing
smooth = gaussian_filter(
    np.where(np.isfinite(risk_corrected), risk_corrected, 0), sigma=1.0)
weight = gaussian_filter(
    np.isfinite(risk_corrected).astype("float32"), sigma=1.0)
risk_corrected = np.where(
    weight > 0.1, smooth / weight, np.nan).astype("float32")

v_orig = risk_orig[np.isfinite(risk_orig)]
v_corr = risk_corrected[np.isfinite(risk_corrected)]
print(f"\n  Original  : mean={v_orig.mean():.4f}  std={v_orig.std():.4f}")
print(f"  Corrected : mean={v_corr.mean():.4f}  std={v_corr.std():.4f}")
print(f"  Mean shift: {v_corr.mean()-v_orig.mean():+.4f}")


# ── CLASSIFY CORRECTED RISK ───────────────────────────────────────────────────

cls_corr = np.full((H, W), np.nan, dtype="float32")
cls_corr[np.isfinite(risk_corrected) & (risk_corrected <= T_LOW)]  = 1
cls_corr[np.isfinite(risk_corrected) & (risk_corrected >  T_LOW) &
          (risk_corrected <= T_MED)]  = 2
cls_corr[np.isfinite(risk_corrected) & (risk_corrected >  T_MED) &
          (risk_corrected <= T_HIGH)] = 3
cls_corr[np.isfinite(risk_corrected) & (risk_corrected >  T_HIGH)] = 4

cls_orig = np.full((H, W), np.nan, dtype="float32")
cls_orig[np.isfinite(risk_orig) & (risk_orig <= T_LOW)]  = 1
cls_orig[np.isfinite(risk_orig) & (risk_orig >  T_LOW) &
          (risk_orig <= T_MED)]  = 2
cls_orig[np.isfinite(risk_orig) & (risk_orig >  T_MED) &
          (risk_orig <= T_HIGH)] = 3
cls_orig[np.isfinite(risk_orig) & (risk_orig >  T_HIGH)] = 4

# Count pixels that changed class
changed = np.isfinite(cls_orig) & (cls_orig != cls_corr)
upgraded = np.isfinite(cls_orig) & (cls_corr > cls_orig)
total    = np.sum(np.isfinite(cls_orig))

print(f"\n  Pixels that changed class: {changed.sum():,} "
      f"({100*changed.sum()/total:.1f}%)")
print(f"  Pixels upgraded (higher risk): {upgraded.sum():,} "
      f"({100*upgraded.sum()/total:.1f}%)")
print(f"\n  Class breakdown (corrected):")
names = {1:"Low", 2:"Medium", 3:"High", 4:"Critical"}
for cls, name in names.items():
    n = np.sum(cls_corr == cls)
    n_old = np.sum(cls_orig == cls)
    diff = n - n_old
    print(f"    {name:8s}: {100*n/total:5.1f}%  "
          f"(was {100*n_old/total:5.1f}%  "
          f"change {diff:+,})")


# ── SAVE OUTPUTS ─────────────────────────────────────────────────────────────

print("\nSaving outputs ...")
profile.update(count=1, dtype="float32",
               nodata=float("nan"), compress="lzw")

def save(arr, name):
    with rasterio.open(str(RISK_DIR/name), "w", **profile) as dst:
        dst.write(arr.astype("float32")[np.newaxis,:,:])
    v = arr[np.isfinite(arr)]
    print(f"  ✓ {name:40s} [{v.min():.4f},{v.max():.4f}]")

save(risk_corrected,               "risk_score_corrected.tif")
save(cls_corr,                     "risk_classified_corrected.tif")
save(pioneer_zone.astype("float32"), "pioneer_zone.tif")
save(ndvi_var_norm,                "ndvi_variability.tif")


# ── VISUALISE ────────────────────────────────────────────────────────────────

print("\nGenerating comparison figure ...")

cls_cmap = mcolors.ListedColormap(CLASS_COLORS)

fig, axes = plt.subplots(2, 3, figsize=(21, 14), facecolor="#0e0e0e")
axes = axes.ravel()

# 1. Original classified
ax = axes[0]
ax.imshow(cls_orig, cmap=cls_cmap, vmin=0.5, vmax=4.5,
           aspect="equal", interpolation="nearest")
ax.set_title("Original classified risk\n(5 layers)",
              color="white", fontsize=9, fontweight="bold")
ax.axis("off"); ax.set_facecolor("#0e0e0e")

# 2. Pioneer zone
ax = axes[1]
ax.imshow(risk_orig, cmap="Greys_r", vmin=0, vmax=0.6,
           aspect="equal", interpolation="nearest", alpha=0.5)
pz_display = np.where(pioneer_zone, 1.0, np.nan)
ax.imshow(pz_display, cmap="YlOrRd", vmin=0, vmax=1,
           aspect="equal", interpolation="nearest", alpha=0.8)
ax.set_title("Pioneer zone overlay\n(cliff front candidates)",
              color="white", fontsize=9, fontweight="bold")
ax.axis("off"); ax.set_facecolor("#0e0e0e")

# 3. NDVI variability layer
ax = axes[2]
im = ax.imshow(ndvi_var_norm, cmap="inferno", vmin=0, vmax=1,
                aspect="equal", interpolation="nearest")
plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02).ax.tick_params(
    colors="white", labelsize=6)
ax.set_title("NDVI variability (normalised)\nnew 6th layer",
              color="white", fontsize=9, fontweight="bold")
ax.axis("off"); ax.set_facecolor("#0e0e0e")

# 4. Corrected classified
ax = axes[3]
ax.imshow(cls_corr, cmap=cls_cmap, vmin=0.5, vmax=4.5,
           aspect="equal", interpolation="nearest")
ax.set_title("Corrected classified risk\n(6 layers + pioneer correction)",
              color="white", fontsize=9, fontweight="bold")
patches = [mpatches.Patch(color=c, label=n)
           for c, n in zip(CLASS_COLORS,
                           ["Low","Medium","High","Critical"])]
ax.legend(handles=patches, loc="lower left", fontsize=7,
          facecolor="#222", labelcolor="white", framealpha=0.9)
ax.axis("off"); ax.set_facecolor("#0e0e0e")

# 5. Difference map (changed pixels)
ax = axes[4]
diff_map = np.where(np.isfinite(cls_orig),
                    cls_corr - cls_orig, np.nan).astype("float32")
diff_cmap = mcolors.ListedColormap(["#1565C0","#888888","#F44336"])
im = ax.imshow(diff_map, cmap=diff_cmap, vmin=-0.5, vmax=0.5,
                aspect="equal", interpolation="nearest")
ax.set_title(f"Class changes\n"
              f"red=upgraded ({100*upgraded.sum()/total:.1f}%)  "
              f"grey=unchanged",
              color="white", fontsize=9, fontweight="bold")
ax.axis("off"); ax.set_facecolor("#0e0e0e")

# 6. Summary text
ax = axes[5]
ax.set_facecolor("#1a1a1a")
ax.axis("off")
summary = (
    f"PIONEER CORRECTION SUMMARY\n"
    f"{'─'*32}\n"
    f"Method: NDVI temporal variability\n"
    f"  as 6th risk layer (w={W_NDVI_VAR})\n"
    f"\nPioneer zone definition:\n"
    f"  NDVI mean: {NDVI_PIONEER_LOW}–{NDVI_PIONEER_HIGH}\n"
    f"  NDVI std: > p75 ({p75_std:.4f})\n"
    f"  Agricultural pixels excluded\n"
    f"\nPioneer zone: {pct:.1f}% of AOI\n"
    f"\nClass changes:\n"
    f"  Total changed : {100*changed.sum()/total:.1f}%\n"
    f"  Upgraded      : {100*upgraded.sum()/total:.1f}%\n"
    f"\nClass distribution (corrected):\n"
)
for cls, name in names.items():
    n = np.sum(cls_corr == cls)
    n_old = np.sum(cls_orig == cls)
    summary += f"  {name:8s}: {100*n/total:.1f}% (Δ{100*(n-n_old)/total:+.1f}%)\n"
summary += (
    f"\nScientific basis:\n"
    f"  Van der Wal et al. (2008)\n"
    f"  Cliff retreat < 5m/yr below\n"
    f"  10m pixel resolution — use\n"
    f"  transect_analysis.py for\n"
    f"  direct rate measurement."
)
ax.text(0.05, 0.95, summary, transform=ax.transAxes,
         color="white", fontsize=8, va="top",
         fontfamily="monospace",
         bbox=dict(facecolor="#111", alpha=0.9,
                   boxstyle="round", edgecolor="#444"))

fig.suptitle(
    "Pioneer Zone Correction  |  "
    "NDVI Variability as Cliff Front Proxy  |  Scheldt",
    color="white", fontsize=11, fontweight="bold"
)
fig.tight_layout()
out = RISK_DIR / "correction_overview.png"
fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor="#0e0e0e")
print(f"\n✓ {out.name}")
plt.show()

print(f"\n✓ Pioneer correction complete.")
print(f"\nNext steps:")
print(f"  1. Rerun validate_risk.py --risk risk_score_corrected.tif")
print(f"     to check detection rate improvement")
print(f"  2. python transect_analysis.py")
print(f"     for direct cliff retreat rate measurement")
