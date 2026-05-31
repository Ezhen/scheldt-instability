import os
os.environ["CPL_LOG"]   = "/dev/null"
os.environ["PROJ_DATA"] = "/home/ulg/mast/eivanov/Yoda/lib/python3.10/site-packages/rasterio/proj_data"
os.environ["PROJ_LIB"]  = "/home/ulg/mast/eivanov/Yoda/lib/python3.10/site-packages/rasterio/proj_data"

"""
Scheldt Estuary — Tier 2 SAR Analysis
======================================
Computes SAR backscatter statistics and coherence proxies
from Sentinel-1 GRD seasonal composites.

What this adds over Tier 1
---------------------------
Tier 1 (optical) answers: where did the waterline and vegetation change?
Tier 2 (SAR)     answers: where is the surface physically unstable?
                          (deformation, slumping, sediment reworking)

SAR is cloud-independent — captures dynamics invisible to optical sensors:
  - Wet/dry sediment transitions (backscatter change)
  - Surface roughness changes (erosion/deposition texture)
  - Temporal coherence loss (disturbed or moving surfaces)

Analyses
---------
1. VV/VH backscatter time series statistics
   - Mean backscatter per pixel across seasons
   - Temporal std dev (instability proxy — same logic as MNDWI std)
   - Seasonal amplitude (summer/winter contrast)

2. Backscatter ratio VV/VH
   - High ratio = smooth water/mudflat surface
   - Low ratio  = rough/vegetated surface
   - Change in ratio = surface state transition

3. Pseudo-coherence (temporal correlation proxy)
   - True InSAR coherence needs SLC data (not available from GRD)
   - GRD proxy: normalised cross-correlation between consecutive
     same-orbit acquisitions
   - Low correlation = surface changed between acquisitions
   - Decorrelation hotspots = active erosion/deposition zones

4. Change point detection
   - Per-pixel: detect season where backscatter shifts significantly
   - Identifies when (not just where) instability occurred

Inputs
------
    data/scheldt/sentinel1/S1_YYYY_season.tif
    Band 1 = VV (sigma0, dB)
    Band 2 = VH (sigma0, dB)

Outputs
-------
    sentinel1/analysis/
        vv_mean.tif              mean VV backscatter
        vv_std.tif               VV temporal std dev (instability)
        vh_mean.tif              mean VH backscatter
        vv_vh_ratio_mean.tif     mean VV/VH ratio
        vv_vh_ratio_std.tif      ratio variability (surface state change)
        pseudo_coherence.tif     temporal correlation proxy
        backscatter_trend.tif    VV linear trend (drying/wetting)
        sar_instability.tif      composite SAR instability score
        sar_overview.png         figure

Note on GRD vs SLC
------------------
True interferometric coherence requires Single Look Complex (SLC) products.
CDSE provides SLC data but processing requires SNAP or a full InSAR pipeline.
The GRD-based proxy here is a valid approximation for change detection
at seasonal timescales and is defensible for a portfolio-level analysis.
For rigorous coherence: use SNAP Graph Builder with S1 SLC pairs,
same orbit, 6-12 day interval, terrain-corrected.

Usage
-----
    python tier2_sar.py

Dependencies
------------
    pip install rioxarray rasterio numpy scipy matplotlib tqdm
"""

import os
os.environ["CPL_LOG"] = "/dev/null"

import warnings
import numpy as np
import matplotlib.pyplot as plt
import rioxarray as rxr
import rasterio
from scipy import stats
from scipy.ndimage import gaussian_filter
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────

S1_DIR  = Path("./data/scheldt/sentinel1")
OUT_DIR = Path("./data/scheldt/sentinel1/analysis")
OUT_DIR.mkdir(exist_ok=True)

S2_REF  = next(Path("./data/scheldt/sentinel2").glob("S2_*.tif"), None)

# Valid backscatter range in dB for Sentinel-1 over land/water
# Values outside this are noise or fill values
VV_VALID = (-35.0, 5.0)
VH_VALID = (-40.0, 0.0)


# ── STEP 1: LOAD S1 STACK ────────────────────────────────────────────────────

tifs = sorted(S1_DIR.glob("S1_*.tif"))
if not tifs:
    raise FileNotFoundError(
        f"No S1_*.tif files in {S1_DIR}\n"
        f"Run acquire_scheldt.py first to download Sentinel-1 data."
    )

print(f"=== Tier 2 SAR Analysis ===")
print(f"Found {len(tifs)} S1 snapshots\n")

ref_ds = rxr.open_rasterio(tifs[0], masked=True)
H, W   = ref_ds.values[0].shape
n      = len(tifs)
labels = [t.stem for t in tifs]

vv_stack    = np.full((n, H, W), np.nan, dtype="float32")
vh_stack    = np.full((n, H, W), np.nan, dtype="float32")
ratio_stack = np.full((n, H, W), np.nan, dtype="float32")

print("Loading S1 backscatter stack ...")
for i, tif in enumerate(tqdm(tifs, desc="  Loading")):
    ds = rxr.open_rasterio(tif, masked=True)

    # Band 1 = VV, Band 2 = VH
    # Values are sigma0 in dB (already terrain-corrected by openEO)
    vv = ds.sel(band=1).values.astype("float32")
    vh = ds.sel(band=2).values.astype("float32") \
         if ds.shape[0] >= 2 else np.full_like(vv, np.nan)

    # Mask invalid values
    vv = np.where((vv >= VV_VALID[0]) & (vv <= VV_VALID[1]), vv, np.nan)
    vh = np.where((vh >= VH_VALID[0]) & (vh <= VH_VALID[1]), vh, np.nan)

    vv_stack[i] = vv
    vh_stack[i] = vh

    # VV/VH ratio (dB difference = linear ratio in log space)
    ratio_stack[i] = np.where(
        np.isfinite(vv) & np.isfinite(vh),
        vv - vh, np.nan
    )

    vv_v = vv[np.isfinite(vv)]
    print(f"  [{i+1:02d}/{n}] {tif.stem:25s}  "
          f"VV=[{vv_v.min():.1f},{vv_v.max():.1f}] dB  "
          f"valid={100*len(vv_v)/(H*W):.1f}%")


# ── STEP 2: BACKSCATTER STATISTICS ───────────────────────────────────────────

print("\n[1] Computing backscatter statistics ...")

vv_mean  = np.nanmean(vv_stack,    axis=0).astype("float32")
vv_std   = np.nanstd(vv_stack,     axis=0).astype("float32")
vh_mean  = np.nanmean(vh_stack,    axis=0).astype("float32")
ratio_mean = np.nanmean(ratio_stack, axis=0).astype("float32")
ratio_std  = np.nanstd(ratio_stack,  axis=0).astype("float32")

# Require at least 4 valid observations
n_valid = np.sum(np.isfinite(vv_stack), axis=0)
vv_std  = np.where(n_valid >= 4, vv_std, np.nan)

for name, arr in [("VV mean", vv_mean), ("VV std", vv_std),
                   ("VH mean", vh_mean), ("Ratio mean", ratio_mean)]:
    v = arr[np.isfinite(arr)]
    if len(v) == 0:
        print(f"  {name:15s}: all NaN — single-band S1 (VV only)")
    else:
        print(f"  {name:15s}: [{v.min():.3f}, {v.max():.3f}]  "
              f"mean={v.mean():.3f}")


# ── STEP 3: VV LINEAR TREND ───────────────────────────────────────────────────

print("\n[2] VV backscatter linear trend ...")

x    = np.arange(n, dtype="float64")
xm   = x.mean()
xvar = ((x - xm)**2).sum()
flat = vv_stack.reshape(n, -1).astype("float64")
ym   = np.nanmean(flat, axis=0)
nv   = np.sum(np.isfinite(flat), axis=0)
num  = ((x[:,None]-xm) * np.where(np.isfinite(flat), flat-ym, 0)).sum(axis=0)
trend = np.where(nv >= 4, num/xvar, np.nan).reshape(H, W).astype("float32")

tv = trend[np.isfinite(trend)]
print(f"  Trend range: [{tv.min():.4f}, {tv.max():.4f}] dB/timestep")
print(f"  Drying  (VV increasing): {100*(tv>0).mean():.1f}%")
print(f"  Wetting (VV decreasing): {100*(tv<0).mean():.1f}%")


# ── STEP 4: PSEUDO-COHERENCE (temporal correlation proxy) ────────────────────

print("\n[3] Computing pseudo-coherence proxy ...")

# Normalised cross-correlation between consecutive acquisitions
# For each pixel: mean correlation across all consecutive pairs
# Low correlation = surface changed = instability signal
correlations = []

for i in range(n - 1):
    a = vv_stack[i].ravel()
    b = vv_stack[i+1].ravel()
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 100:
        continue

    # Compute local normalised correlation in 5×5 windows
    # using sliding approach via uniform_filter
    from scipy.ndimage import uniform_filter

    a2d = np.where(np.isfinite(vv_stack[i]),   vv_stack[i],   0.0)
    b2d = np.where(np.isfinite(vv_stack[i+1]), vv_stack[i+1], 0.0)
    w   = np.isfinite(vv_stack[i]) & np.isfinite(vv_stack[i+1])
    w   = w.astype("float32")

    # Local means
    k    = 5
    am   = uniform_filter(a2d * w, k) / (uniform_filter(w, k) + 1e-9)
    bm   = uniform_filter(b2d * w, k) / (uniform_filter(w, k) + 1e-9)

    # Local cross-correlation
    ab   = uniform_filter((a2d - am) * (b2d - bm) * w, k)
    aa   = uniform_filter((a2d - am)**2 * w, k)
    bb   = uniform_filter((b2d - bm)**2 * w, k)
    denom = np.sqrt(aa * bb) + 1e-9
    corr  = np.clip(ab / denom, -1, 1)
    corr  = np.where(w > 0, corr, np.nan)
    correlations.append(corr.astype("float32"))

if correlations:
    pseudo_coh = np.nanmean(np.stack(correlations, axis=0),
                             axis=0).astype("float32")
    v = pseudo_coh[np.isfinite(pseudo_coh)]
    print(f"  Pseudo-coherence range: [{v.min():.3f}, {v.max():.3f}]")
    print(f"  Mean coherence: {v.mean():.3f}")
    print(f"  Low coherence (<0.5): {100*(v<0.5).mean():.1f}% of pixels")
else:
    print("  Not enough valid pairs for coherence computation")
    pseudo_coh = np.full((H, W), np.nan, dtype="float32")


# ── STEP 5: COMPOSITE SAR INSTABILITY SCORE ──────────────────────────────────

print("\n[4] Building SAR instability composite ...")

def norm(arr, p_lo=5, p_hi=95):
    v = arr[np.isfinite(arr)]
    if len(v) == 0:
        return arr
    lo = np.percentile(v, p_lo)
    hi = np.percentile(v, p_hi)
    if hi <= lo:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0, 1).astype("float32")

# Components — use only layers with valid data
# VV std: always available
n_vv_std = norm(vv_std)

# Ratio std: only if VH available
n_ratio_std = norm(ratio_std) if np.any(np.isfinite(ratio_std)) else None

# Pseudo-coherence: only if computed
n_incoh = norm(1.0 - np.where(np.isfinite(pseudo_coh),
                               pseudo_coh, np.nan))           if np.any(np.isfinite(pseudo_coh)) else None

# Build composite from available layers only
layers_avail = [(0.40, n_vv_std, "VV std")]
if n_ratio_std is not None and np.any(np.isfinite(n_ratio_std)):
    layers_avail.append((0.30, n_ratio_std, "Ratio std"))
if n_incoh is not None and np.any(np.isfinite(n_incoh)):
    layers_avail.append((0.30, n_incoh, "Incoherence"))

# Re-normalise weights to sum to 1
total_w = sum(w for w, _, _ in layers_avail)
print(f"  SAR composite from {len(layers_avail)} layers: "
      f"{[n for _,_,n in layers_avail]}")

valid = np.isfinite(n_vv_std)   # minimum: VV std must be finite
sar_instability = np.zeros(vv_std.shape, dtype="float32")
for w, arr, name in layers_avail:
    sar_instability += (w / total_w) * np.where(np.isfinite(arr), arr, 0.0)
sar_instability = np.where(valid, sar_instability, np.nan).astype("float32")

# Light smoothing
sar_smooth = gaussian_filter(
    np.where(np.isfinite(sar_instability), sar_instability, 0), sigma=1.0
)
w_smooth = gaussian_filter(np.isfinite(sar_instability).astype("float32"),
                            sigma=1.0)
sar_instability = np.where(
    w_smooth > 0.1, sar_smooth / w_smooth, np.nan
).astype("float32")

sv = sar_instability[np.isfinite(sar_instability)]
if len(sv) == 0:
    print("  SAR instability: all NaN — insufficient input layers")
else:
    print(f"  SAR instability range: [{sv.min():.4f}, {sv.max():.4f}]")
    print(f"  Mean: {sv.mean():.4f}  Std: {sv.std():.4f}")


# ── STEP 6: SAVE GEOTIFFS ─────────────────────────────────────────────────────

print("\nSaving outputs ...")

with rasterio.open(tifs[0]) as src:
    profile = src.profile.copy()
profile.update(count=1, dtype="float32",
               nodata=float("nan"), compress="lzw")

def save(arr, name, label):
    out = OUT_DIR / name
    arr_f = arr.astype("float32")
    with rasterio.open(str(out), "w", **profile) as dst:
        dst.write(arr_f[np.newaxis, :, :])
    v = arr_f[np.isfinite(arr_f)]
    print(f"  ✓ {label:35s} [{v.min():.4f}, {v.max():.4f}]")

# Only save layers with valid data
def save_if_valid(arr, name, label):
    v = arr[np.isfinite(arr)]
    if len(v) > 0:
        save(arr, name, label)
    else:
        print(f"  SKIP {label} — all NaN (VH not available?)")

save_if_valid(vv_mean,        "vv_mean.tif",            "VV mean backscatter (dB)")
save_if_valid(vv_std,         "vv_std.tif",             "VV temporal std dev")
save_if_valid(vh_mean,        "vh_mean.tif",            "VH mean backscatter (dB)")
save_if_valid(ratio_mean,     "vv_vh_ratio_mean.tif",   "VV/VH ratio mean (dB)")
save_if_valid(ratio_std,      "vv_vh_ratio_std.tif",    "VV/VH ratio std dev")
save_if_valid(pseudo_coh,     "pseudo_coherence.tif",   "Pseudo-coherence proxy")
save_if_valid(trend,          "backscatter_trend.tif",  "VV linear trend (dB/step)")
save_if_valid(sar_instability,"sar_instability.tif",    "SAR composite instability")


# ── STEP 7: VISUALISE ────────────────────────────────────────────────────────

print("\nGenerating figure ...")

fig, axes = plt.subplots(2, 4, figsize=(24, 12), facecolor="#0e0e0e")
axes = axes.ravel()

panels = [
    ("VV mean backscatter (dB)\nbrighter = rougher / drier",
     vv_mean,        "RdYlGn",   "elev"),
    ("VV temporal std dev\nhigh = unstable surface",
     vv_std,         "inferno",  "std"),
    ("VV/VH ratio mean\nhigh = smooth (water/mud)",
     ratio_mean,     "PuBu",     "std"),
    ("VV/VH ratio variability\nhigh = surface state changing",
     ratio_std,      "YlOrRd",   "std"),
    ("Pseudo-coherence\nlow = surface disturbed between acquisitions",
     pseudo_coh,     "RdYlGn",   "coh"),
    ("VV backscatter trend\nred=drying  blue=wetting",
     trend,          "RdBu_r",   "div"),
    ("SAR composite instability\n(VV std + ratio change + incoherence)",
     sar_instability,"inferno",  "std"),
    ("VV backscatter time series\nsample pixel at channel margin",
     None,           None,       "ts"),
]

for ax, (title, data, cmap, kind) in zip(axes, panels):
    ax.set_facecolor("#0e0e0e")

    if kind == "ts":
        # Time series for a sample pixel near channel margin
        ax.set_facecolor("#1a1a1a")
        cy, cx = H//2, W//2
        ts_vv = vv_stack[:, cy, cx]
        ts_vh = vh_stack[:, cy, cx]
        x_ts  = np.arange(n)
        valid_vv = np.isfinite(ts_vv)
        valid_vh = np.isfinite(ts_vh)
        if valid_vv.sum() > 2:
            ax.plot(x_ts[valid_vv], ts_vv[valid_vv],
                    color="#42A5F5", lw=1.5, marker="o",
                    markersize=4, label="VV")
        if valid_vh.sum() > 2:
            ax.plot(x_ts[valid_vh], ts_vh[valid_vh],
                    color="#EF5350", lw=1.5, marker="s",
                    markersize=4, label="VH")
        ax.set_xticks(x_ts[::3])
        ax.set_xticklabels([labels[j] for j in x_ts[::3]],
                            rotation=45, ha="right",
                            color="white", fontsize=6)
        ax.set_ylabel("Backscatter (dB)", color="white", fontsize=8)
        ax.tick_params(colors="white", labelsize=7)
        ax.spines[:].set_color("#444")
        ax.legend(fontsize=8, facecolor="#1a1a1a",
                  labelcolor="white", framealpha=0.9)
        ax.set_title("VV/VH time series — sample pixel\n"
                      f"(row={cy}, col={cx})",
                      color="white", fontsize=8, fontweight="bold")
        continue

    if data is None:
        ax.set_visible(False)
        continue

    valid = data[np.isfinite(data)]
    if len(valid) == 0:
        ax.set_visible(False)
        continue

    if kind == "elev":
        vmin, vmax = np.percentile(valid, 2), np.percentile(valid, 98)
    elif kind == "std":
        vmin, vmax = 0, np.percentile(valid, 98)
    elif kind == "coh":
        vmin, vmax = 0, 1
    elif kind == "div":
        p = np.percentile(np.abs(valid), 95)
        vmin, vmax = -p, p
    else:
        vmin, vmax = np.percentile(valid, 2), np.percentile(valid, 98)

    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax,
                    aspect="equal", interpolation="bilinear")
    cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.ax.tick_params(colors="white", labelsize=6)
    ax.set_title(title, color="white", fontsize=8,
                  fontweight="bold", pad=3)
    ax.axis("off")

fig.suptitle(
    "Scheldt — Tier 2 SAR Instability Analysis  |  Sentinel-1 GRD",
    color="white", fontsize=12, fontweight="bold"
)
fig.tight_layout()
out_png = OUT_DIR / "sar_overview.png"
fig.savefig(str(out_png), dpi=150,
            bbox_inches="tight", facecolor="#0e0e0e")
print(f"\n✓ Saved → {out_png}")
plt.show()

print(f"\n✓ Tier 2 SAR complete.")
print(f"\nTo add SAR instability to the risk index:")
print(f"  Add to risk_score.py LAYER_DEFS:")
print(f'    "sar_instability": {{')
print(f'        "path": Path("./data/scheldt/sentinel1/analysis/sar_instability.tif"),')
print(f'        "desc": "SAR surface instability proxy",')
print(f'        "invert": False,')
print(f'    }}')
print(f"  Then add --w6 argument and rerun risk_score.py")
