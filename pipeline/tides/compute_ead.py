import os
os.environ["CPL_LOG"]   = "/dev/null"
os.environ["PROJ_DATA"] = "/home/ulg/mast/eivanov/.conda/envs/Yoda/lib/python3.10/site-packages/rasterio/proj_data"
os.environ["PROJ_LIB"]  = "/home/ulg/mast/eivanov/.conda/envs/Yoda/lib/python3.10/site-packages/rasterio/proj_data"

import os

"""
Scheldt Estuary — Elevation Above Datum (EAD)
=============================================
Computes EAD = AHN4 elevation - MHW (from tidal_datums.json)

EAD interpretation:
    EAD > +1m   above normal high water       → low tidal exposure
    0 < EAD < +1m  vulnerable — floods on spring tides  → medium risk
    -2m < EAD < 0  regular intertidal zone     → high exposure
    EAD < -2m   permanently submerged          → channel / deep tidal flat

Also produces:
    intertidal_zone.tif    binary mask (1 = between MLW and MHW)
    tidal_flat.tif         binary mask (1 = between MLW and MHWS, unvegetated)

Outputs
-------
    dem_ahn4/derivatives/EAD.tif
    dem_ahn4/derivatives/intertidal_zone.tif
    dem_ahn4/derivatives/tidal_flat.tif
    dem_ahn4/derivatives/resampled_10m/EAD.tif
    tides/EAD_overview.png

Dependencies
------------
    pip install rioxarray rasterio numpy matplotlib

Usage
-----
    python compute_ead.py

    # Override paths
    python compute_ead.py --dem path/to/dem.tif --datums path/to/datums.json
"""

import json
import argparse
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import rioxarray as rxr
import rasterio
from rasterio.warp import reproject, Resampling

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ── ARGUMENT PARSING ─────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--dem",    default="./data/scheldt/dem_ahn4/AHN4_DTM_scheldt_5m_rd.tif")
parser.add_argument("--datums", default="./data/scheldt/tides/tidal_datums.json")
parser.add_argument("--out",    default="./data/scheldt/dem_ahn4/derivatives")
args = parser.parse_args()

DEM_PATH   = Path(args.dem)
DATUMS_PATH = Path(args.datums)
DERIV_DIR  = Path(args.out)
S2_DIR     = Path("./data/scheldt/sentinel2")
PNG_OUT    = Path("./data/scheldt/tides/EAD_overview.png")
DERIV_DIR.mkdir(parents=True, exist_ok=True)
PNG_OUT.parent.mkdir(parents=True, exist_ok=True)


# ── LOAD DATUMS ───────────────────────────────────────────────────────────────

print("=== Elevation Above Datum (EAD) ===\n")
print(f"Loading tidal datums from {DATUMS_PATH.name} ...")

with open(DATUMS_PATH) as f:
    datums = json.load(f)

MHW  = datums["MHW"]
MLW  = datums["MLW"]
MHWS = datums["MHWS"]
MLWS = datums["MLWS"]
MTL  = datums["MTL"]
TR   = datums["TR"]

print(f"  MHW  = {MHW:+.3f} m NAP")
print(f"  MLW  = {MLW:+.3f} m NAP")
print(f"  MHWS = {MHWS:+.3f} m NAP")
print(f"  MLWS = {MLWS:+.3f} m NAP")
print(f"  TR   = {TR:.3f} m")


# ── LOAD DEM ──────────────────────────────────────────────────────────────────

print(f"\nLoading DEM from {DEM_PATH.name} ...")
ds   = rxr.open_rasterio(DEM_PATH, masked=True)
elev = ds.values[0].astype("float32")

# Mask implausible values
elev = np.where((elev > 1000) | (elev < -100), np.nan, elev)

valid = elev[np.isfinite(elev)]
print(f"  Shape : {elev.shape}")
print(f"  Range : {valid.min():.2f} – {valid.max():.2f} m NAP")


# ── COMPUTE EAD ───────────────────────────────────────────────────────────────

print("\nComputing EAD = elevation - MHW ...")

ead = (elev - MHW).astype("float32")

# Mask water pixels (below MLWS = permanently submerged channel)
# Keep these as NaN — they're not bankline pixels
water_mask   = elev < (MLWS - 0.5)
ead_masked   = np.where(water_mask, np.nan, ead)

valid_ead = ead_masked[np.isfinite(ead_masked)]
print(f"  EAD range : {valid_ead.min():.2f} – {valid_ead.max():.2f} m")

# Risk zone breakdown
total = len(valid_ead)
z1 = np.sum(valid_ead >  1.0)               # safe — above spring tide
z2 = np.sum((valid_ead > 0) & (valid_ead <= 1.0))   # spring flood risk
z3 = np.sum((valid_ead > -TR) & (valid_ead <= 0))   # regular intertidal
z4 = np.sum(valid_ead <= -TR)               # deep / permanent water

print(f"\n  EAD zone breakdown:")
print(f"    > +1m     (safe, above MHWS)    : {100*z1/total:5.1f}%")
print(f"    0 to +1m  (spring flood risk)   : {100*z2/total:5.1f}%")
print(f"    -{TR:.1f}m to 0  (intertidal)         : {100*z3/total:5.1f}%")
print(f"    < -{TR:.1f}m  (permanent water)      : {100*z4/total:5.1f}%")


# ── COMPUTE ZONE MASKS ────────────────────────────────────────────────────────

print("\nComputing zone masks ...")

# Intertidal zone: between MLW and MHW
intertidal = np.where(
    np.isfinite(elev) & (elev >= MLW) & (elev <= MHW),
    1.0, np.nan
).astype("float32")

# Tidal flat: between MLWS and MHW (broader intertidal)
tidal_flat = np.where(
    np.isfinite(elev) & (elev >= MLWS) & (elev <= MHW),
    1.0, np.nan
).astype("float32")

# Vulnerable fringe: 0 to +1m above MHW (spring flood zone)
vulnerable = np.where(
    np.isfinite(elev) & (elev > MHW) & (elev <= MHW + 1.0),
    1.0, np.nan
).astype("float32")

n_inter = int(np.nansum(intertidal))
n_flat  = int(np.nansum(tidal_flat))
n_vuln  = int(np.nansum(vulnerable))
print(f"  Intertidal (MLW–MHW)    : {n_inter:,} pixels  "
      f"({100*n_inter/total:.1f}%)")
print(f"  Tidal flat (MLWS–MHW)   : {n_flat:,} pixels  "
      f"({100*n_flat/total:.1f}%)")
print(f"  Vulnerable (MHW–MHW+1m) : {n_vuln:,} pixels  "
      f"({100*n_vuln/total:.1f}%)")


# ── SAVE GEOTIFFS ─────────────────────────────────────────────────────────────

def save_tif(arr, path, label):
    """Save using rasterio directly — avoids rioxarray lazy eval zeros."""
    arr = arr.astype("float32")
    with rasterio.open(DEM_PATH) as src:
        prof = src.profile.copy()
    prof.update(count=1, dtype="float32",
                nodata=float("nan"), compress="lzw")
    with rasterio.open(str(path), "w", **prof) as dst:
        dst.write(arr[np.newaxis, :, :])
    valid = arr[np.isfinite(arr)]
    print(f"  ✓ {label:30s} → {path.name}  "
          f"({'range=['+f'{valid.min():.3f},{valid.max():.3f}]' if len(valid)>0 else 'all NaN'})")


print("\nSaving ...")
save_tif(ead_masked,  DERIV_DIR / "EAD.tif",             "EAD (elev - MHW)")
save_tif(intertidal,  DERIV_DIR / "intertidal_zone.tif",  "Intertidal zone mask")
save_tif(tidal_flat,  DERIV_DIR / "tidal_flat.tif",       "Tidal flat mask")
save_tif(vulnerable,  DERIV_DIR / "vulnerable_fringe.tif","Vulnerable fringe mask")


# ── RESAMPLE TO S2 GRID ───────────────────────────────────────────────────────

s2_ref = next(S2_DIR.glob("S2_*.tif"), None)
if s2_ref is not None:
    print(f"\nResampling to S2 10m grid ...")
    rs_dir = DERIV_DIR / "resampled_10m"
    rs_dir.mkdir(exist_ok=True)

    with rasterio.open(s2_ref) as ref:
        dst_t   = ref.transform
        dst_crs = ref.crs.to_wkt()
        dst_h   = ref.height
        dst_w   = ref.width
        out_prof = ref.profile.copy()
    out_prof.update(count=1, dtype="float32",
                    nodata=float("nan"), compress="lzw")

    for tif_name in ["EAD.tif", "intertidal_zone.tif",
                      "tidal_flat.tif", "vulnerable_fringe.tif"]:
        tif = DERIV_DIR / tif_name
        out = rs_dir / tif_name
        if not tif.exists():
            continue
        try:
            # Use rasterio.warp directly with explicit src_crs string
            # Bypasses EngineeringCRS PROJ lookup failure on Yoda
            with rasterio.open(tif) as src:
                src_data = src.read(1).astype("float32")
                src_t    = src.transform
            dst_data = np.empty((dst_h, dst_w), dtype=np.float32)
            reproject(
                source=src_data,
                destination=dst_data,
                src_transform=src_t,
                src_crs="EPSG:28992",   # force RD New — bypass EngineeringCRS
                dst_transform=dst_t,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
                src_nodata=float("nan"),
                dst_nodata=float("nan"),
            )
            with rasterio.open(str(out), "w", **out_prof) as d:
                d.write(dst_data[np.newaxis, :, :])
            v = dst_data[np.isfinite(dst_data)]
            print(f"  ✓ {tif_name:35s} "
                  f"finite={len(v):,}  "
                  f"{'range=['+f'{v.min():.3f},{v.max():.3f}]' if len(v)>0 else 'ALL NaN'}")
        except Exception as e:
            print(f"  ✗ {tif_name}: {e.__class__.__name__}: {e}")


# ── VISUALISE ────────────────────────────────────────────────────────────────

print("\nGenerating figure ...")

# Custom risk colormap: green (safe) → yellow → orange → red (flooded)
risk_colors = [
    (0.00, "#1565C0"),   # deep blue:  permanent water
    (0.20, "#42A5F5"),   # light blue: regular intertidal
    (0.40, "#FFF176"),   # yellow:     spring flood zone
    (0.60, "#FFB300"),   # amber:      just above MHW
    (0.80, "#66BB6A"),   # green:      safe polder
    (1.00, "#2E7D32"),   # dark green: high polder
]
risk_cmap = mcolors.LinearSegmentedColormap.from_list(
    "ead_risk", risk_colors
)

fig, axes = plt.subplots(1, 3, figsize=(21, 8), facecolor="#0e0e0e")

# ── Panel 1: EAD continuous ───────────────────────────────────────────────────
ax = axes[0]
vmin = max(valid_ead.min(), -TR - 0.5)
vmax = min(valid_ead.max(),  TR + 1.0)
im = ax.imshow(ead_masked, cmap="RdYlGn", vmin=vmin, vmax=vmax,
                aspect="equal", interpolation="bilinear")
cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
cb.set_label("m above MHW", color="white", fontsize=8)
cb.ax.tick_params(colors="white", labelsize=7)
# Add datum lines to colourbar
for val, label, col in [
    (0.0,  "MHW",  "red"),
    (-TR,  "MLW",  "blue"),
    (1.0,  "+1m",  "orange"),
]:
    cb.ax.axhline((val - vmin)/(vmax - vmin), color=col,
                   lw=1.5, ls="--")
ax.set_title("Elevation Above Datum\n(EAD = elev − MHW)",
              color="white", fontsize=9, fontweight="bold")
ax.axis("off")
ax.set_facecolor("#0e0e0e")

# ── Panel 2: classified risk zones ───────────────────────────────────────────
ax2 = axes[1]

# Build 5-class zone map
zone = np.full(elev.shape, np.nan, dtype="float32")
zone[np.isfinite(elev) & (elev < MLW)]                        = 0  # permanent water
zone[np.isfinite(elev) & (elev >= MLW)  & (elev < MHW)]       = 1  # intertidal
zone[np.isfinite(elev) & (elev >= MHW)  & (elev < MHW + 1.0)] = 2  # vulnerable
zone[np.isfinite(elev) & (elev >= MHW + 1.0)]                 = 3  # safe

zone_cmap = mcolors.ListedColormap([
    "#1565C0",   # 0 permanent water
    "#42A5F5",   # 1 intertidal
    "#FFB300",   # 2 vulnerable fringe
    "#43A047",   # 3 safe polder
])
im2 = ax2.imshow(zone, cmap=zone_cmap, vmin=-0.5, vmax=3.5,
                  aspect="equal", interpolation="nearest")
cb2 = plt.colorbar(im2, ax=ax2, fraction=0.035, pad=0.02,
                    ticks=[0, 1, 2, 3])
cb2.set_ticklabels([
    f"Perm. water\n(<MLW {MLW:.2f}m)",
    f"Intertidal\n(MLW–MHW)",
    f"Vuln. fringe\n(MHW–MHW+1m)",
    f"Safe polder\n(>{MHW+1:.2f}m NAP)",
])
cb2.ax.tick_params(colors="white", labelsize=6)
ax2.set_title("Tidal risk zones\nbased on EAD",
               color="white", fontsize=9, fontweight="bold")
ax2.axis("off")
ax2.set_facecolor("#0e0e0e")

# Add zone stats
stats_txt = (
    f"Intertidal : {100*n_inter/total:.1f}%\n"
    f"Vulnerable : {100*n_vuln/total:.1f}%\n"
    f"MHW = {MHW:+.3f} m NAP\n"
    f"TR  = {TR:.3f} m"
)
ax2.text(0.02, 0.04, stats_txt, transform=ax2.transAxes,
          color="white", fontsize=8, va="bottom",
          bbox=dict(facecolor="#222", alpha=0.85, boxstyle="round"))

# ── Panel 3: EAD histogram ───────────────────────────────────────────────────
ax3 = axes[2]
ax3.set_facecolor("#1a1a1a")
ax3.hist(valid_ead, bins=200, color="#42A5F5", alpha=0.85,
          range=(vmin, vmax))
for val, lbl, col in [
    (0.0,  "MHW (0m)",        "#EF5350"),
    (-TR,  f"MLW (−{TR:.1f}m)", "#42A5F5"),
    (1.0,  "+1m above MHW",   "#FFB300"),
    (-TR/2, "Mid-intertidal", "#FFFFFF"),
]:
    ax3.axvline(val, color=col, lw=1.5, ls="--", label=lbl)

ax3.set_xlabel("EAD (m above MHW)", color="white", fontsize=9)
ax3.set_ylabel("Pixel count", color="white", fontsize=9)
ax3.tick_params(colors="white", labelsize=8)
ax3.spines[:].set_color("#444")
ax3.legend(fontsize=8, facecolor="#1a1a1a", labelcolor="white",
            framealpha=0.9)
ax3.set_title("EAD distribution\n(risk profile of the AOI)",
               color="white", fontsize=9, fontweight="bold")

fig.suptitle(
    f"Scheldt — Elevation Above Datum  |  AHN4 5m lidar + Vlissingen MHW ({MHW:+.3f}m NAP)",
    color="white", fontsize=11, fontweight="bold"
)
fig.tight_layout()
fig.savefig(str(PNG_OUT), dpi=150, bbox_inches="tight", facecolor="#0e0e0e")
print(f"\n✓ Saved → {PNG_OUT}")
plt.show()

print(f"\n✓ EAD complete.")
print(f"\nKey outputs for risk scoring:")
print(f"  {DERIV_DIR}/EAD.tif")
print(f"  {DERIV_DIR}/intertidal_zone.tif")
print(f"  {DERIV_DIR}/resampled_10m/EAD.tif")
print(f"\nNext: python risk_score.py")
