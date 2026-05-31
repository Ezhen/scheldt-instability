"""
Curvature diagnostic — run in same directory as acquire_dem.py
Prints intermediate values at each computation step.
Usage: python debug_curv.py
"""
import numpy as np
from scipy.ndimage import gaussian_filter
import rioxarray as rxr
from pathlib import Path

DEM_PATH = Path("./data/scheldt/dem/DEM_scheldt_clipped.tif")

ds   = rxr.open_rasterio(DEM_PATH, masked=True)
elev = ds.values[0].astype("float32")
t    = ds.rio.transform()

res_x_m = abs(t.a) * 111_320 * np.cos(np.radians(51.4))
res_y_m = abs(t.e) * 111_320

print(f"Pixel size: {res_x_m:.2f} x {res_y_m:.2f} m")
print(f"Elevation  min={elev.min():.3f}  max={elev.max():.3f}  "
      f"unique values: {np.unique(elev)}")

# Step 1: smooth
smooth = gaussian_filter(elev, sigma=2.5)
print(f"\nAfter Gaussian sigma=2.5:")
print(f"  min={smooth.min():.4f}  max={smooth.max():.4f}  "
      f"std={smooth.std():.4f}")

# Step 2: first derivatives
dz_dy, dz_dx = np.gradient(smooth, res_y_m, res_x_m)
print(f"\nFirst derivatives:")
print(f"  dz_dx: min={dz_dx.min():.6f}  max={dz_dx.max():.6f}  "
      f"std={dz_dx.std():.6f}")
print(f"  dz_dy: min={dz_dy.min():.6f}  max={dz_dy.max():.6f}  "
      f"std={dz_dy.std():.6f}")

# Step 3: second derivatives
dxx = np.gradient(dz_dx, res_x_m, axis=1)
dyy = np.gradient(dz_dy, res_y_m, axis=0)
dxy = np.gradient(dz_dx, res_y_m, axis=0)
print(f"\nSecond derivatives:")
print(f"  dxx: min={dxx.min():.8f}  max={dxx.max():.8f}  std={dxx.std():.8f}")
print(f"  dyy: min={dyy.min():.8f}  max={dyy.max():.8f}  std={dyy.std():.8f}")
print(f"  dxy: min={dxy.min():.8f}  max={dxy.max():.8f}  std={dxy.std():.8f}")

# Step 4: denominator
px, py = dz_dx**2, dz_dy**2
denom_planc = (px + py) * np.sqrt(1 + px + py)
print(f"\nPlan curv denominator:")
print(f"  min={denom_planc.min():.2e}  max={denom_planc.max():.2e}")
print(f"  % < 1e-20: {100*np.mean(np.abs(denom_planc) < 1e-20):.1f}%")
print(f"  % < 1e-10: {100*np.mean(np.abs(denom_planc) < 1e-10):.1f}%")
print(f"  % < 1e-6 : {100*np.mean(np.abs(denom_planc) < 1e-6):.1f}%")

# Step 5: raw curvature before masking
denom_safe = np.where(np.abs(denom_planc) < 1e-20, np.nan, denom_planc)
planc_raw = (dxx*py - 2*dxy*dz_dx*dz_dy + dyy*px) / denom_safe
valid = planc_raw[np.isfinite(planc_raw)]
print(f"\nRaw plan curvature (valid pixels):")
print(f"  count={len(valid):,}  min={valid.min():.4f}  max={valid.max():.4f}")
print(f"  p01={np.percentile(valid,1):.4f}  p10={np.percentile(valid,10):.4f}")
print(f"  p50={np.percentile(valid,50):.6f}")
print(f"  p90={np.percentile(valid,90):.4f}  p99={np.percentile(valid,99):.4f}")
print(f"  std={valid.std():.4f}")
print(f"  % > 10  : {100*np.mean(np.abs(valid) > 10):.1f}%")
print(f"  % > 1   : {100*np.mean(np.abs(valid) > 1):.1f}%")
print(f"  % > 0.1 : {100*np.mean(np.abs(valid) > 0.1):.1f}%")
print(f"  % < 0.01: {100*np.mean(np.abs(valid) < 0.01):.1f}%")
