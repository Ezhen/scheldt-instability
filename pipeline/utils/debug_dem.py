"""
DEM Derivatives Diagnostic Script
===================================
Run this in the same directory as acquire_dem.py to trace
exactly what's in each derivative file.

Usage:
    python debug_dem.py
"""

import numpy as np
import rioxarray as rxr
from pathlib import Path

DEM_DIR = Path("./data/scheldt/dem")

files = {
    "DEM (elevation)":    DEM_DIR / "DEM_scheldt_clipped.tif",
    "Slope":              DEM_DIR / "derivatives/slope_degrees.tif",
    "Plan curvature":     DEM_DIR / "derivatives/plan_curvature.tif",
    "Profile curvature":  DEM_DIR / "derivatives/profile_curvature.tif",
    "TWI":                DEM_DIR / "derivatives/TWI.tif",
}

print("=" * 65)
print("DEM DERIVATIVES DIAGNOSTIC")
print("=" * 65)

for name, path in files.items():
    if not path.exists():
        print(f"\n{name}: FILE NOT FOUND — {path}")
        continue

    ds   = rxr.open_rasterio(path, masked=True)
    data = ds.values[0].astype("float32")
    flat = data.ravel()

    print(f"\n── {name}")
    print(f"   File    : {path.name}")
    print(f"   Shape   : {data.shape}")
    print(f"   CRS     : {ds.rio.crs}")
    print(f"   nodata  : {ds.rio.nodata}")
    print(f"   dtype   : {data.dtype}")
    print(f"   --- Raw array (including nodata) ---")
    print(f"   min     : {np.nanmin(flat):.6f}")
    print(f"   max     : {np.nanmax(flat):.6f}")
    print(f"   mean    : {np.nanmean(flat):.6f}")
    print(f"   NaN px  : {np.sum(~np.isfinite(flat)):,}  "
          f"({100*np.mean(~np.isfinite(flat)):.1f}%)")
    print(f"   --- Valid pixels only ---")
    valid = flat[np.isfinite(flat)]
    if len(valid) == 0:
        print("   *** ALL PIXELS ARE NaN — file is empty ***")
        continue
    print(f"   count   : {len(valid):,}")
    print(f"   min     : {np.min(valid):.6f}")
    print(f"   max     : {np.max(valid):.6f}")
    print(f"   mean    : {np.mean(valid):.6f}")
    print(f"   std     : {np.std(valid):.6f}")
    print(f"   p02     : {np.percentile(valid, 2):.6f}")
    print(f"   p10     : {np.percentile(valid, 10):.6f}")
    print(f"   p50     : {np.percentile(valid, 50):.6f}")
    print(f"   p90     : {np.percentile(valid, 90):.6f}")
    print(f"   p98     : {np.percentile(valid, 98):.6f}")
    print(f"   --- Zero check ---")
    zero_pct = 100 * np.mean(valid == 0.0)
    near_zero_pct = 100 * np.mean(np.abs(valid) < 1e-6)
    print(f"   exact 0 : {zero_pct:.1f}% of valid pixels")
    print(f"   ~0 <1e-6: {near_zero_pct:.1f}% of valid pixels")
    if near_zero_pct > 90:
        print(f"   *** WARNING: >90% near-zero — derivative likely not computed ***")

print("\n" + "=" * 65)
print("TRANSFORM CHECK")
print("=" * 65)
dem_path = DEM_DIR / "DEM_scheldt_clipped.tif"
if dem_path.exists():
    ds = rxr.open_rasterio(dem_path, masked=True)
    t  = ds.rio.transform()
    print(f"  Transform : {t}")
    print(f"  res_x deg : {abs(t.a):.6f}°")
    print(f"  res_y deg : {abs(t.e):.6f}°")
    import numpy as np
    res_x_m = abs(t.a) * 111_320 * np.cos(np.radians(51.4))
    res_y_m = abs(t.e) * 111_320
    print(f"  res_x m   : {res_x_m:.2f} m")
    print(f"  res_y m   : {res_y_m:.2f} m")
    elev = ds.values[0].astype("float32")
    print(f"\nELEVATION RAW (before masking):")
    print(f"  unique nodata candidates: "
          f"{sorted(set([elev.min(), elev.max()]))}")
    print(f"  values < -100 : {np.sum(elev < -100):,} pixels")
    print(f"  values = 0    : {np.sum(elev == 0):,} pixels")
    print(f"  values > 100  : {np.sum(elev > 100):,} pixels")
    # Sample centre pixel
    cy, cx = elev.shape[0]//2, elev.shape[1]//2
    print(f"  centre pixel  : {elev[cy,cx]:.2f} m  (row={cy}, col={cx})")
    print(f"  corner pixels : TL={elev[0,0]:.2f}  TR={elev[0,-1]:.2f}  "
          f"BL={elev[-1,0]:.2f}  BR={elev[-1,-1]:.2f}")
