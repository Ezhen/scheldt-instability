"""
Scheldt — Tier 1 GeoTIFF Export
================================
Computes and saves analysis layers as GeoTIFFs for use in risk_score.py.
Handles DN→reflectance scaling, clips artefacts to valid index range,
and stores corrupted/extreme pixels separately for QA inspection.

Outputs (data/scheldt/sentinel2/analysis/)
------------------------------------------
    mndwi_std.tif            MNDWI temporal std dev (instability proxy)
    ndvi_slope.tif           NDVI linear trend slope
    ndvi_slope_significant.tif   slope where p < 0.10
    ndvi_slope_loss.tif      browning signal (negative slopes only)
    water_frequency.tif      fraction of snapshots classified as water
    ndvi_mean.tif            mean NDVI across all snapshots
    mndwi_mean.tif           mean MNDWI across all snapshots

QA outputs (data/scheldt/sentinel2/analysis/qa/)
-------------------------------------------------
    snapshot_stats.csv       per-snapshot valid pixel %, index ranges
    extreme_pixels_mndwi.tif pixels where any snapshot had |MNDWI| > 1
    extreme_pixels_ndvi.tif  pixels where any snapshot had |NDVI| > 1
    corruption_fraction.tif  fraction of snapshots that were corrupted
                             per pixel (useful for identifying bad scenes)

Usage
-----
    python tier1_save_tifs.py
"""

import os
os.environ["CPL_LOG"] = "/dev/null"

import warnings
import json
import numpy as np
import pandas as pd
import rioxarray as rxr
from pathlib import Path
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────

S2_DIR   = Path("./data/scheldt/sentinel2")
OUT_DIR  = Path("./data/scheldt/sentinel2/analysis")
QA_DIR   = OUT_DIR / "qa"
OUT_DIR.mkdir(exist_ok=True)
QA_DIR.mkdir(exist_ok=True)

# Valid index range — anything outside is cloud, fill value, or artefact
NDVI_VALID  = (-1.0, 1.0)
MNDWI_VALID = (-1.0, 1.0)

# Minimum valid pixel fraction per snapshot to include in statistics
MIN_VALID_FRAC = 0.30   # skip snapshots with < 30% valid pixels


# ── STEP 1: LOAD ALL SNAPSHOTS ────────────────────────────────────────────────

tifs = sorted(S2_DIR.glob("S2_*.tif"))
if not tifs:
    raise FileNotFoundError(f"No S2_*.tif files in {S2_DIR}")

print(f"=== Tier 1 GeoTIFF export ===")
print(f"Found {len(tifs)} snapshots\n")

ref_ds = rxr.open_rasterio(tifs[0], masked=True)
H, W   = ref_ds.values[0].shape
n      = len(tifs)

# Allocate stacks — NaN for missing/masked
ndvi_stack    = np.full((n, H, W), np.nan, dtype="float32")
mndwi_stack   = np.full((n, H, W), np.nan, dtype="float32")
# Artefact tracking — True where a pixel was clipped (corrupted) in snapshot i
ndvi_corrupt  = np.zeros((n, H, W), dtype=bool)
mndwi_corrupt = np.zeros((n, H, W), dtype=bool)

snapshot_stats = []

print("Loading and cleaning snapshots ...")
for i, tif in enumerate(tifs):
    ds  = rxr.open_rasterio(tif, masked=True)

    # Band order: 1=B02 2=B03(Green) 3=B04(Red) 4=B08(NIR) 5=B11(SWIR)
    b03 = ds.sel(band=2).values.astype("float32")
    b04 = ds.sel(band=3).values.astype("float32")
    b08 = ds.sel(band=4).values.astype("float32")
    b11 = ds.sel(band=5).values.astype("float32")

    # DN → reflectance (always /10000 for Sentinel-2 L2A)
    b03 /= 10000.0
    b04 /= 10000.0
    b08 /= 10000.0
    b11 /= 10000.0

    # Compute indices
    d_ndvi  = b08 + b04
    d_mndwi = b03 + b11

    ndvi_raw  = np.where(np.abs(d_ndvi)  > 1e-4, (b08-b04)/d_ndvi,  np.nan)
    mndwi_raw = np.where(np.abs(d_mndwi) > 1e-4, (b03-b11)/d_mndwi, np.nan)

    # ── Track extreme/corrupted pixels BEFORE clipping ────────────────────────
    ndvi_corrupt[i]  = np.isfinite(ndvi_raw)  & \
                       ((ndvi_raw  < NDVI_VALID[0])  | (ndvi_raw  > NDVI_VALID[1]))
    mndwi_corrupt[i] = np.isfinite(mndwi_raw) & \
                       ((mndwi_raw < MNDWI_VALID[0]) | (mndwi_raw > MNDWI_VALID[1]))

    # ── Clip to valid range ────────────────────────────────────────────────────
    ndvi_clean  = np.where(
        (ndvi_raw  >= NDVI_VALID[0])  & (ndvi_raw  <= NDVI_VALID[1]),
        ndvi_raw,  np.nan
    )
    mndwi_clean = np.where(
        (mndwi_raw >= MNDWI_VALID[0]) & (mndwi_raw <= MNDWI_VALID[1]),
        mndwi_raw, np.nan
    )

    ndvi_stack[i]  = ndvi_clean
    mndwi_stack[i] = mndwi_clean

    # ── Per-snapshot stats ─────────────────────────────────────────────────────
    n_total   = H * W
    n_valid_n = int(np.sum(np.isfinite(ndvi_clean)))
    n_valid_m = int(np.sum(np.isfinite(mndwi_clean)))
    n_corr_n  = int(np.sum(ndvi_corrupt[i]))
    n_corr_m  = int(np.sum(mndwi_corrupt[i]))
    valid_frac = n_valid_n / n_total

    stats = {
        "snapshot":        tif.name,
        "valid_pct":       round(100 * valid_frac, 1),
        "corrupted_ndvi":  n_corr_n,
        "corrupted_mndwi": n_corr_m,
        "ndvi_min":        round(float(np.nanmin(ndvi_raw)),  3),
        "ndvi_max":        round(float(np.nanmax(ndvi_raw)),  3),
        "ndvi_mean":       round(float(np.nanmean(ndvi_clean)), 4),
        "mndwi_min":       round(float(np.nanmin(mndwi_raw)), 3),
        "mndwi_max":       round(float(np.nanmax(mndwi_raw)), 3),
        "mndwi_mean":      round(float(np.nanmean(mndwi_clean)), 4),
        "flagged":         valid_frac < MIN_VALID_FRAC or
                           n_corr_n > n_total * 0.05,
    }
    snapshot_stats.append(stats)

    flag = " ⚠ FLAGGED" if stats["flagged"] else ""
    print(f"  [{i+1:02d}/{n}] {tif.stem:25s}  "
          f"valid={stats['valid_pct']:5.1f}%  "
          f"corr_n={n_corr_n:6,}  corr_m={n_corr_m:6,}{flag}")


# ── STEP 2: QA — SAVE EXTREME PIXEL MAPS ─────────────────────────────────────

print("\n[QA] Saving extreme pixel diagnostics ...")

# Fraction of snapshots corrupted per pixel
corrupt_frac_ndvi  = ndvi_corrupt.mean(axis=0).astype("float32")
corrupt_frac_mndwi = mndwi_corrupt.mean(axis=0).astype("float32")

# Any-snapshot corruption mask
any_corrupt_ndvi  = ndvi_corrupt.any(axis=0).astype("float32")
any_corrupt_mndwi = mndwi_corrupt.any(axis=0).astype("float32")

def save_qa(arr, name):
    da = ref_ds.sel(band=1).copy(data=arr)
    for a in ["_FillValue","long_name","AREA_OR_POINT",
              "scale_factor","add_offset"]:
        da.attrs.pop(a, None); da.encoding.pop(a, None)
    da = da.rio.write_nodata(float("nan"))
    da.expand_dims("band").rio.to_raster(
        str(QA_DIR / name), compress="lzw", dtype="float32")

save_qa(any_corrupt_ndvi,   "extreme_pixels_ndvi.tif")
save_qa(any_corrupt_mndwi,  "extreme_pixels_mndwi.tif")
save_qa(corrupt_frac_ndvi,  "corruption_fraction_ndvi.tif")
save_qa(corrupt_frac_mndwi, "corruption_fraction_mndwi.tif")

# Save snapshot stats CSV
df_stats = pd.DataFrame(snapshot_stats)
df_stats.to_csv(str(QA_DIR / "snapshot_stats.csv"), index=False)

flagged = df_stats[df_stats["flagged"]]
print(f"  ✓ Extreme pixel maps saved to {QA_DIR}")
print(f"  ✓ snapshot_stats.csv  ({len(df_stats)} rows)")
print(f"  Flagged snapshots: {len(flagged)}")
if len(flagged) > 0:
    for _, row in flagged.iterrows():
        print(f"    ⚠  {row['snapshot']}  "
              f"valid={row['valid_pct']}%  "
              f"corrupted_ndvi={row['corrupted_ndvi']:,}")


# ── STEP 3: MNDWI TEMPORAL STD DEV ───────────────────────────────────────────

print("[1] MNDWI temporal std dev ...")

# Mask any pixel that was corrupted in ANY snapshot before computing std
# Prevents ±1 oscillating artefacts from dominating the std map
mndwi_clean = mndwi_stack.copy()
mndwi_clean[mndwi_corrupt] = np.nan

# Require at least 4 clean observations per pixel
n_clean = np.sum(np.isfinite(mndwi_clean), axis=0)
mndwi_std = np.nanstd(mndwi_clean, axis=0).astype("float32")
mndwi_std = np.where(n_clean >= 4, mndwi_std, np.nan)

# Final p99 clip — removes any remaining edge artefacts
p99 = np.nanpercentile(mndwi_std[np.isfinite(mndwi_std)], 99.0)
mndwi_std = np.where(mndwi_std > p99, np.nan, mndwi_std).astype("float32")

v = mndwi_std[np.isfinite(mndwi_std)]
print(f"  Pixels masked (<4 clean obs): {int(np.sum(n_clean < 4)):,}")
print(f"  p99 clip at: {p99:.5f}")
print(f"  range=[{v.min():.5f},{v.max():.5f}]  "
      f"mean={v.mean():.5f}  p50={np.percentile(v,50):.5f}  "
      f"p95={np.percentile(v,95):.5f}")


# ── STEP 4: NDVI LINEAR TREND (vectorised) ───────────────────────────────────

print("\n[2] NDVI linear trend (vectorised) ...")
x    = np.arange(n, dtype="float64")
xm   = x.mean()
xvar = ((x - xm)**2).sum()
flat = ndvi_stack.reshape(n, -1).astype("float64")
ym   = np.nanmean(flat, axis=0)
nv   = np.sum(np.isfinite(flat), axis=0)
num  = ((x[:,None]-xm) * np.where(np.isfinite(flat), flat-ym, 0)).sum(axis=0)
slope_flat = np.where(nv >= 4, num/xvar, np.nan).astype("float32")
slope_map  = slope_flat.reshape(H, W)

# Significance: t-statistic approximation
y_pred_dev = slope_flat[None,:] * (x[:,None] - xm)
ss_resid   = np.nansum(
    (np.where(np.isfinite(flat), flat-ym, 0) - y_pred_dev)**2, axis=0)
se = np.sqrt(
    np.where(nv > 2, ss_resid / (nv-2), np.nan) /
    np.where(xvar > 0, xvar, np.nan)
)
t_stat  = np.abs(slope_flat / (se + 1e-12))
sig_map = np.where(
    (t_stat.reshape(H, W) > 1.65) & (nv.reshape(H, W) >= 4),
    slope_map, np.nan
).astype("float32")

# Loss: negative slopes only (browning = risk)
loss_map = np.where(
    np.isfinite(slope_map) & (slope_map < 0),
    -slope_map, 0.0
).astype("float32")

sv = slope_map[np.isfinite(slope_map)]
print(f"  slope range=[{sv.min():.5f},{sv.max():.5f}]  "
      f"greening={100*(sv>0).mean():.1f}%  browning={100*(sv<0).mean():.1f}%")
lv = loss_map[loss_map > 0]
if len(lv):
    print(f"  loss  non-zero={len(lv):,}  "
          f"p50={np.percentile(lv,50):.5f}  p95={np.percentile(lv,95):.5f}")


# ── STEP 5: ADDITIONAL LAYERS ─────────────────────────────────────────────────

print("\n[3] Water frequency, mean NDVI/MNDWI ...")
water_freq = np.nanmean(
    (mndwi_stack > 0.0).astype("float32"), axis=0
).astype("float32")
ndvi_mean  = np.nanmean(ndvi_stack,  axis=0).astype("float32")
mndwi_mean = np.nanmean(mndwi_stack, axis=0).astype("float32")


# ── STEP 6: SAVE ALL OUTPUTS ──────────────────────────────────────────────────

print("\nSaving GeoTIFFs ...")

def save_tif(arr, path, label):
    arr = arr.astype("float32")
    da  = ref_ds.sel(band=1).copy(data=arr)
    for a in ["_FillValue","long_name","AREA_OR_POINT",
              "scale_factor","add_offset"]:
        da.attrs.pop(a, None); da.encoding.pop(a, None)
    da  = da.rio.write_nodata(float("nan"))
    da.expand_dims("band").rio.to_raster(
        str(path), compress="lzw", dtype="float32")
    v = arr[np.isfinite(arr)]
    print(f"  ✓ {label:35s} range=[{v.min():.5f},{v.max():.5f}]")

save_tif(mndwi_std,  OUT_DIR/"mndwi_std.tif",               "MNDWI std dev")
save_tif(slope_map,  OUT_DIR/"ndvi_slope.tif",               "NDVI slope (all)")
save_tif(sig_map,    OUT_DIR/"ndvi_slope_significant.tif",   "NDVI slope (p<0.10)")
save_tif(loss_map,   OUT_DIR/"ndvi_slope_loss.tif",          "NDVI loss (browning)")
save_tif(water_freq, OUT_DIR/"water_frequency.tif",          "Water frequency")
save_tif(ndvi_mean,  OUT_DIR/"ndvi_mean.tif",                "Mean NDVI")
save_tif(mndwi_mean, OUT_DIR/"mndwi_mean.tif",               "Mean MNDWI")


# ── STEP 7: PRINT QA SUMMARY ──────────────────────────────────────────────────

print(f"\n{'='*55}")
print(f"QA SUMMARY")
print(f"{'='*55}")
print(f"Total snapshots    : {n}")
print(f"Flagged snapshots  : {len(flagged)}")
print(f"AOI shape          : {H} × {W} px")
print(f"\nCorruption by snapshot (NDVI):")
print(df_stats[["snapshot","valid_pct","corrupted_ndvi","flagged"]]
      .sort_values("corrupted_ndvi", ascending=False)
      .head(10).to_string(index=False))

print(f"\n✓ All outputs saved to {OUT_DIR}")
print(f"  QA files in {QA_DIR}")
print(f"\nNext: python risk_score.py")
