"""
Western Scheldt — Spatially Varying Manning Roughness
======================================================
Derives Manning n at every bedlevel.xyz sample point from:
  1. Bed elevation  (AHN4 5m lidar — intertidal zone)
  2. NDVI           (Sentinel-2 — vegetation proxy for marsh)
  3. EMODnet depth  (bedlevel.xyz — subtidal zone)

Manning classification:
  Navigation channel  (z < -10m NAP)          n = 0.019  smooth sand
  Deep subtidal       (-10 < z < -2m NAP)      n = 0.022  sandy bed
  Low intertidal      (-2 < z < 0m NAP)        n = 0.026  mixed sand/mud
  High intertidal     (0 < z < +1m NAP)        n = 0.032  mudflat/pioneer
  Salt marsh          (z > +1m AND NDVI > 0.3) n = 0.045  dense vegetation
  High marsh / polder (z > +1m AND NDVI < 0.3) n = 0.035  sparse/bare

Scientific basis:
  Values from Temmerman et al. (2005), van der Wal et al. (2008),
  and Meire et al. (2005) for Westerschelde/Zeeschelde.
  Depth-dependent n follows Chow (1959) for estuarine channels.

Outputs:
  dfm_clean/dflowfm/roughness_spatial.xyz     Manning n per sample point
  dfm_clean/dflowfm/roughness_spatial.png     Diagnostic map
  fieldFile.ini updated with roughness block

Usage:
    python generate_roughness.py
    python generate_roughness.py --preview   # plot only, don't write files
"""

import os
_P = "/home/ulg/mast/eivanov/.conda/envs/Yoda/lib/python3.10/site-packages/pyproj/proj_dir/share/proj"
_R = "/home/ulg/mast/eivanov/.conda/envs/Yoda/lib/python3.10/site-packages/rasterio/proj_data"
os.environ["CPL_LOG"]   = "/dev/null"
os.environ["PROJ_DATA"] = _R
os.environ["PROJ_LIB"]  = _R
import pyproj; pyproj.datadir.set_data_dir(_P)

import sys
import argparse
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")

try:
    import rasterio
    from rasterio.transform import rowcol
except ImportError:
    sys.exit("pip install rasterio")

# ── CONFIG ────────────────────────────────────────────────────────────────────

ap = argparse.ArgumentParser()
ap.add_argument("--xyz",     default="dfm_model/dflowfm/bedlevel.xyz",
                # Note: ensure this is the 3.96M point 29m EMODnet version
                help="Input bedlevel sample points (RD New)")
ap.add_argument("--ahn4",    default="data/scheldt/dem_ahn4/AHN4_DTM_scheldt_5m_rd.tif",
                help="AHN4 5m lidar DTM (RD New)")
ap.add_argument("--ndvi",    default="data/scheldt/sentinel2/analysis/ndvi_slope.tif",
                help="NDVI raster (RD New, any resolution)")
ap.add_argument("--out",     default="dfm_clean/dflowfm",
                help="Output directory")
ap.add_argument("--preview", action="store_true",
                help="Plot diagnostic only, do not write roughness files")
args = ap.parse_args()

OUT_DIR = Path(args.out)

# ── MANNING CLASSIFICATION PARAMETERS ────────────────────────────────────────

# Depth thresholds (m NAP)
Z_DEEP_CHANNEL = -10.0
Z_SUBTIDAL     =  -2.0
Z_LOW_INTERTIDAL = 0.0
Z_HIGH_INTERTIDAL = 1.0

# NDVI threshold for vegetation (marsh vs bare flat)
NDVI_MARSH = 0.25

# Manning n values
N_DEEP_CHANNEL    = 0.019   # navigation channel — smooth dredged sand
N_SUBTIDAL        = 0.022   # subtidal flats — sandy bed with ripples
N_LOW_INTERTIDAL  = 0.026   # low intertidal — mixed sand/mud
N_HIGH_INTERTIDAL = 0.032   # high intertidal bare — mudflat, pioneer zone
N_MARSH_DENSE     = 0.045   # dense salt marsh — Spartina, Puccinellia
N_MARSH_SPARSE    = 0.035   # sparse marsh / polder grassland

# AHN4 valid range (mask fill values)
AHN4_NODATA_THRESHOLD = 100.0

print("=== Spatially Varying Manning Roughness — Western Scheldt ===\n")
print(f"Classification:")
print(f"  z < {Z_DEEP_CHANNEL}m:                     n = {N_DEEP_CHANNEL}")
print(f"  {Z_DEEP_CHANNEL} ≤ z < {Z_SUBTIDAL}m:        n = {N_SUBTIDAL}")
print(f"  {Z_SUBTIDAL} ≤ z < {Z_LOW_INTERTIDAL}m:           n = {N_LOW_INTERTIDAL}")
print(f"  {Z_LOW_INTERTIDAL} ≤ z < {Z_HIGH_INTERTIDAL}m, bare:   n = {N_HIGH_INTERTIDAL}")
print(f"  z ≥ {Z_HIGH_INTERTIDAL}m, NDVI ≥ {NDVI_MARSH}: n = {N_MARSH_DENSE}  (dense marsh)")
print(f"  z ≥ {Z_HIGH_INTERTIDAL}m, NDVI < {NDVI_MARSH}: n = {N_MARSH_SPARSE}  (sparse/bare)")

# ── STEP 1: LOAD BEDLEVEL SAMPLE POINTS ──────────────────────────────────────

print("\n[1] Loading bedlevel sample points ...")
xyz = np.loadtxt(args.xyz, comments="#")
bx, by, bz = xyz[:,0], xyz[:,1], xyz[:,2]
print(f"  {len(bx):,} points, z=[{bz.min():.2f},{bz.max():.2f}]m")

# ── STEP 2: LOAD AHN4 AND SAMPLE AT XYZ LOCATIONS ────────────────────────────

print("\n[2] Loading AHN4 DTM ...")
ahn4_z = np.full(len(bx), np.nan)

ahn4_path = Path(args.ahn4)
if ahn4_path.exists():
    with rasterio.open(str(ahn4_path)) as src:
        print(f"  Shape: {src.height}×{src.width}, res={src.res[0]:.1f}m")
        print(f"  Bounds: {src.bounds}")

        # Sample AHN4 at each xyz point
        # rasterio.sample expects (x, y) pairs
        coords = list(zip(bx, by))
        try:
            vals = list(src.sample(coords, indexes=1))
            ahn4_z = np.array([v[0] for v in vals], dtype=np.float32)
            # Mask nodata
            ahn4_z = np.where(
                (ahn4_z > AHN4_NODATA_THRESHOLD) | (ahn4_z < -100),
                np.nan, ahn4_z
            )
            valid = np.isfinite(ahn4_z).sum()
            print(f"  AHN4 valid: {valid:,}/{len(bx):,} points")
            print(f"  AHN4 range: [{np.nanmin(ahn4_z):.2f},{np.nanmax(ahn4_z):.2f}]m")
        except Exception as e:
            print(f"  AHN4 sampling failed: {e}")
else:
    print(f"  AHN4 not found at {ahn4_path} — using EMODnet depth only")

# ── STEP 3: LOAD NDVI AND SAMPLE AT XYZ LOCATIONS ────────────────────────────
print("\n[3] Computing median NDVI from Sentinel-2 scenes ...")
ndvi_z = np.full(len(bx), np.nan)

s2_dir = Path("data/scheldt/sentinel2")
scenes = sorted(s2_dir.glob("S2_*.tif"))
print(f"  Found {len(scenes)} S2 scenes")

if len(scenes) > 0:
    RED_BAND = 3   # B04
    NIR_BAND = 4   # B08

    with rasterio.open(str(scenes[0])) as src:
        b      = src.bounds
        s2_crs = src.crs.to_epsg()
        s2_H, s2_W = src.height, src.width
        s2_T = src.transform
    print(f"  S2 CRS: EPSG:{s2_crs}, size={s2_H}x{s2_W}")
    print(f"  S2 bounds: x=[{b.left:.0f},{b.right:.0f}] y=[{b.bottom:.0f},{b.top:.0f}]")

    from pyproj import Transformer
    if s2_crs != 28992:
        tr = Transformer.from_crs(28992, s2_crs, always_xy=True)
        bx_crs, by_crs = tr.transform(bx, by)
    else:
        bx_crs, by_crs = bx, by

    in_s2 = ((bx_crs >= b.left) & (bx_crs <= b.right) &
             (by_crs >= b.bottom) & (by_crs <= b.top))
    print(f"  Points in S2 AOI: {in_s2.sum():,}/{len(bx):,}")

    # Vectorized pixel index — much faster than rasterio.sample
    bx_s2 = bx_crs[in_s2]
    by_s2 = by_crs[in_s2]
    cols  = np.clip(np.floor((bx_s2 - s2_T.c) / s2_T.a).astype(np.int32), 0, s2_W-1)
    rows  = np.clip(np.floor((by_s2 - s2_T.f) / s2_T.e).astype(np.int32), 0, s2_H-1)

    use_scenes = [s for s in scenes if "summer" in s.name] or scenes[:5]
    print(f"  Processing {len(use_scenes)} scenes via vectorized lookup ...")

    ndvi_stack = []
    for scene in use_scenes:
        try:
            with rasterio.open(str(scene)) as src:
                red_full = src.read(RED_BAND).astype(np.float32)
                nir_full = src.read(NIR_BAND).astype(np.float32)
            red_r = red_full[rows, cols] / 10000.0
            nir_r = nir_full[rows, cols] / 10000.0
            denom = nir_r + red_r
            ndvi_sc = np.where(denom > 0.01, (nir_r - red_r) / denom, np.nan)
            ndvi_stack.append(ndvi_sc)
            print(f"    {scene.name}: NDVI=[{np.nanmin(ndvi_sc):.2f},{np.nanmax(ndvi_sc):.2f}]")
        except Exception as e:
            print(f"  {scene.name} failed: {e}")

    if ndvi_stack:
        ndvi_s2 = np.nanmedian(np.stack(ndvi_stack), axis=0)
        ndvi_s2 = np.where(ndvi_s2 < -0.1, np.nan, ndvi_s2)
        ndvi_z[in_s2] = ndvi_s2
        valid = np.isfinite(ndvi_z).sum()
        print(f"  NDVI valid: {valid:,} points")
        if valid > 0:
            print(f"  NDVI range: [{np.nanmin(ndvi_z):.3f},{np.nanmax(ndvi_z):.3f}]")
            print(f"  NDVI > {NDVI_MARSH}: {(ndvi_z[np.isfinite(ndvi_z)]>NDVI_MARSH).sum():,} marsh pts")
else:
    print("  No S2 scenes — depth-only classification")

# ── STEP 4: CLASSIFY MANNING N ────────────────────────────────────────────────

print("\n[4] Classifying Manning roughness ...")

z_best   = np.where(np.isfinite(ahn4_z), ahn4_z, bz)
ndvi_safe = np.where(np.isfinite(ndvi_z), ndvi_z, 0.0)

n = np.full(len(bx), N_SUBTIDAL)
n = np.where(z_best < Z_DEEP_CHANNEL, N_DEEP_CHANNEL, n)
n = np.where((z_best >= Z_SUBTIDAL) & (z_best < Z_LOW_INTERTIDAL), N_LOW_INTERTIDAL, n)
n = np.where((z_best >= Z_LOW_INTERTIDAL) & (z_best < Z_HIGH_INTERTIDAL), N_HIGH_INTERTIDAL, n)
n = np.where((z_best >= Z_HIGH_INTERTIDAL) & (ndvi_safe >= NDVI_MARSH), N_MARSH_DENSE, n)
n = np.where((z_best >= Z_HIGH_INTERTIDAL) & (ndvi_safe < NDVI_MARSH), N_MARSH_SPARSE, n)

for nval, label in [
    (N_DEEP_CHANNEL, "Deep channel"), (N_SUBTIDAL, "Subtidal"),
    (N_LOW_INTERTIDAL, "Low intertidal"), (N_HIGH_INTERTIDAL, "High intertidal bare"),
    (N_MARSH_DENSE, "Dense marsh"), (N_MARSH_SPARSE, "Sparse marsh"),
]:
    cnt = (n == nval).sum()
    print(f"  {label:22s} (n={nval}): {cnt:,} points ({cnt/len(n)*100:.1f}%)")
print(f"  n range: [{n.min():.3f}, {n.max():.3f}]")

# ── STEP 5: DIAGNOSTIC FIGURE ────────────────────────────────────────────────

print("\n[5] Generating diagnostic figure ...")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class_colors = {
    N_DEEP_CHANNEL:    ("#1565C0", "Deep channel"),
    N_SUBTIDAL:        ("#42A5F5", "Subtidal"),
    N_LOW_INTERTIDAL:  ("#A5D6A7", "Low intertidal"),
    N_HIGH_INTERTIDAL: ("#FFF176", "High intertidal bare"),
    N_MARSH_SPARSE:    ("#FF8F00", "Sparse marsh"),
    N_MARSH_DENSE:     ("#2E7D32", "Dense marsh"),
}
DARK_BG = "#0e0e0e"
AOI_X, AOI_Y = (14000, 65000), (372000, 397000)

fig, ax = plt.subplots(figsize=(14, 7), facecolor=DARK_BG)
ax.set_facecolor(DARK_BG)
mask_aoi = ((bx>=AOI_X[0])&(bx<=AOI_X[1])&(by>=AOI_Y[0])&(by<=AOI_Y[1]))
idx_plot = np.where(mask_aoi)[0][::max(1,mask_aoi.sum()//200000)]
for nval, (color, label) in class_colors.items():
    m = n[idx_plot] == nval
    if m.sum() > 0:
        ax.scatter(bx[idx_plot][m], by[idx_plot][m], c=color, s=0.5, alpha=0.6,
                   label=f"n={nval} {label}", rasterized=True)
ax.set_xlim(*AOI_X); ax.set_ylim(*AOI_Y)
ax.set_xlabel("RD New X (m)", color="white", fontsize=9)
ax.set_ylabel("RD New Y (m)", color="white", fontsize=9)
ax.tick_params(colors="white", labelsize=8); ax.spines[:].set_color("#333")
ax.legend(fontsize=7.5, facecolor="#1a1a1a", labelcolor="white",
          loc="upper right", markerscale=8, framealpha=0.9)
ax.set_title("Spatially Varying Manning Roughness — Western Scheldt\n"
             "Derived from AHN4 DTM + Sentinel-2 NDVI",
             color="white", fontsize=10, fontweight="bold")
out_png = OUT_DIR / "roughness_spatial.png"
fig.savefig(str(out_png), dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close(fig)
print(f"  ✓ {out_png}")

if args.preview:
    print("\n--preview mode: skipping file writing")
    import sys; sys.exit(0)

# ── STEP 6: WRITE ROUGHNESS XYZ ──────────────────────────────────────────────

print("\n[6] Writing roughness_spatial.xyz ...")
out_xyz = OUT_DIR / "roughness_spatial.xyz"
with open(str(out_xyz), "w") as f:
    f.write("# Manning roughness derived from AHN4 + NDVI\n")
    f.write("# x(RD New m)  y(RD New m)  n(Manning s/m^(1/3))\n")
    for xi, yi, ni in zip(bx, by, n):
        f.write(f"{xi:.2f}  {yi:.2f}  {ni:.4f}\n")
print(f"  ✓ {out_xyz.name}  ({len(bx):,} points)")

# ── STEP 7: UPDATE fieldFile.ini ─────────────────────────────────────────────

print("\n[7] Updating fieldFile.ini ...")
ini_path = OUT_DIR / "fieldFile.ini"
roughness_block = """
[Parameter]
quantity              = frictioncoefficient
dataFile              = roughness_spatial.xyz
dataFileType          = sample
interpolationMethod   = triangulation
operand               = O
"""
if ini_path.exists():
    with open(ini_path) as f:
        ini_content = f.read()
    if "frictioncoefficient" not in ini_content:
        ini_content += roughness_block
        with open(ini_path, "w") as f:
            f.write(ini_content)
        print(f"  ✓ Added roughness block to {ini_path.name}")
    else:
        print(f"  frictioncoefficient already in {ini_path.name}")
else:
    with open(ini_path, "w") as f:
        f.write("[General]\nfileVersion           = 2.00\nfileType              = iniField\n")
        f.write(roughness_block)
    print(f"  ✓ Created {ini_path.name}")

# ── STEP 8: UPDATE MDU ────────────────────────────────────────────────────────

print("\n[8] Checking MDU ...")
mdu_path = OUT_DIR / "WesternScheldt.mdu"
if mdu_path.exists():
    with open(mdu_path) as f:
        mdu = f.read()
    if "IniFieldFile" not in mdu:
        mdu = mdu.replace("WaterLevIni           = 0.0",
                          "WaterLevIni           = 0.0\n        IniFieldFile          = fieldFile.ini")
        with open(mdu_path, "w") as f:
            f.write(mdu)
        print(f"  ✓ IniFieldFile added to MDU")
    else:
        print(f"  IniFieldFile already in MDU")

print(f"""
{'='*55}
✓ Roughness complete. Files in {OUT_DIR}:
  roughness_spatial.xyz  ({len(bx):,} points)
  roughness_spatial.png
  fieldFile.ini updated

Tuning: N_DEEP_CHANNEL={N_DEEP_CHANNEL}, N_SUBTIDAL={N_SUBTIDAL},
        NDVI_MARSH={NDVI_MARSH}
""")
