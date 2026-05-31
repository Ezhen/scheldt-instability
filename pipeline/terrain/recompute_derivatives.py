import os
os.environ["CPL_LOG"]   = "/dev/null"
os.environ["PROJ_DATA"] = "/home/ulg/mast/eivanov/Yoda/lib/python3.10/site-packages/rasterio/proj_data"
os.environ["PROJ_LIB"]  = "/home/ulg/mast/eivanov/Yoda/lib/python3.10/site-packages/rasterio/proj_data"

import os

"""
Scheldt — Terrain Derivatives (Evans-Young method)
===================================================
Computes slope, plan curvature, profile curvature, TWI
from any clipped DEM GeoTIFF.

Usage
-----
    # GLO-30 30m (default)
    python recompute_derivatives.py

    # AHN4 5m lidar
    python recompute_derivatives.py --source ahn4

    # Any custom path
    python recompute_derivatives.py --dem path/to/dem.tif --out path/to/derivs/

Available --source shortcuts:
    glo30   ./data/scheldt/dem/DEM_scheldt_clipped.tif
    ahn4    ./data/scheldt/dem_ahn4/AHN4_DTM_scheldt_5m_wgs84.tif
"""

import argparse
import warnings
import numpy as np
import matplotlib.pyplot as plt
import rioxarray as rxr
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.ndimage import gaussian_filter, uniform_filter
from pathlib import Path

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ── ARGUMENT PARSING ─────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute terrain derivatives from a DEM GeoTIFF."
    )
    parser.add_argument(
        "--source", choices=["glo30", "ahn4"], default="glo30",
        help="Shortcut: glo30 (30m default) or ahn4 (5m lidar)"
    )
    parser.add_argument(
        "--dem", type=str, default=None,
        help="Override: path to any DEM GeoTIFF"
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Override: output derivatives directory"
    )
    return parser.parse_args()


SOURCES = {
    "glo30": {
        "dem":  "./data/scheldt/dem/DEM_scheldt_clipped.tif",
        "out":  "./data/scheldt/dem/derivatives",
        "png":  "./data/scheldt/dem/terrain_derivatives.png",
        "sigma": 1.0,
        "label": "GLO-30 30m",
    },
    "ahn4": {
        "dem":  "./data/scheldt/dem_ahn4/AHN4_DTM_scheldt_5m_wgs84.tif",
        "out":  "./data/scheldt/dem_ahn4/derivatives",
        "png":  "./data/scheldt/dem_ahn4/terrain_derivatives.png",
        "sigma": 2.0,   # AHN4 5m — sigma=2.0 stabilises curvature
        "label": "AHN4 5m lidar",
    },
}


# ── LOAD CONFIG ───────────────────────────────────────────────────────────────

args      = parse_args()
cfg       = SOURCES[args.source]
DEM_PATH  = Path(args.dem) if args.dem else Path(cfg["dem"])
DERIV_DIR = Path(args.out) if args.out else Path(cfg["out"])
OUT_PNG   = Path(cfg["png"])
SIGMA     = cfg["sigma"]
LABEL     = cfg["label"]
S2_DIR    = Path("./data/scheldt/sentinel2")
DERIV_DIR.mkdir(parents=True, exist_ok=True)

print(f"=== Terrain derivatives — {LABEL} ===")
print(f"  DEM      : {DEM_PATH}")
print(f"  Output   : {DERIV_DIR}")

if not DEM_PATH.exists():
    raise FileNotFoundError(f"DEM not found: {DEM_PATH}")


# ── LOAD DEM ──────────────────────────────────────────────────────────────────

print("\nLoading DEM ...")
ds   = rxr.open_rasterio(DEM_PATH, masked=True)
elev = ds.values[0].astype("float32")
t    = ds.rio.transform()

# CRS-aware pixel size
if ds.rio.crs.is_geographic:
    res_x = abs(t.a) * 111_320 * np.cos(np.radians(51.4))
    res_y = abs(t.e) * 111_320
    print(f"  CRS: geographic → metres via lat correction")
else:
    res_x = abs(t.a)
    res_y = abs(t.e)
    print(f"  CRS: projected → metres direct")

print(f"  Shape      : {elev.shape}")
print(f"  Pixel size : {res_x:.1f} × {res_y:.1f} m")
print(f"  Elev range : {np.nanmin(elev):.2f} – {np.nanmax(elev):.2f} m")


# ── WATER MASK FROM S2 MNDWI ─────────────────────────────────────────────────

print("\nBuilding water mask ...")
s2_ref     = next(S2_DIR.glob("S2_*.tif"), None)
water_mask = np.zeros(elev.shape, dtype=bool)
H, W       = elev.shape

if s2_ref is not None and ds.rio.crs.is_geographic:
    # Only reproject S2→DEM when both are geographic (same CRS family)
    # Projected CRS (RD New) triggers PROJ DB lookup which fails on Yoda
    try:
        with rasterio.open(s2_ref) as s2_src:
            green    = s2_src.read(2).astype("float32")
            swir     = s2_src.read(5).astype("float32")
            mndwi_s2 = (green - swir) / (green + swir + 1e-9)
            s2_crs   = s2_src.crs
            s2_t     = s2_src.transform
        mndwi_dem = np.empty((H, W), dtype=np.float32)
        reproject(
            source=mndwi_s2, destination=mndwi_dem,
            src_transform=s2_t, src_crs=s2_crs,
            dst_transform=t, dst_crs=ds.rio.crs,
            resampling=Resampling.bilinear,
        )
        water_mask = mndwi_dem > 0.0
        print(f"  ✓ S2 MNDWI water mask ({100*water_mask.mean():.1f}% water)")
    except Exception as e:
        print(f"  S2 reproject failed ({e.__class__.__name__}) — using elevation mask")
        water_mask = (elev < -5.0) | ~np.isfinite(elev)
        print(f"  ✓ Elevation mask < -5m NAP ({100*water_mask.mean():.1f}% masked)")
else:
    # DEM is projected (RD New) — skip S2 reproject, use NAP threshold
    # AHN4: tidal channel sits well below -2m NAP
    print("  Projected CRS → elevation-based water mask")
    water_mask = (elev < -2.0) | ~np.isfinite(elev)
    print(f"  ✓ Elevation mask < -2m NAP ({100*water_mask.mean():.1f}% masked)")


# ── FILL WATER PIXELS THEN SMOOTH ─────────────────────────────────────────────

print("\nFilling water pixels and smoothing ...")
elev_filled = elev.copy()

if water_mask.any():
    fill = np.where(water_mask, np.nan, elev_filled)
    for _ in range(5):
        kernel = uniform_filter(np.where(np.isfinite(fill), fill, 0.0), size=5)
        weight = uniform_filter(np.isfinite(fill).astype("float32"),    size=5)
        weight = np.where(weight < 1e-6, 1.0, weight)
        fill   = np.where(water_mask, kernel / weight, fill)
    elev_filled = fill

# Auto-adjust sigma based on pixel size
# Fine resolution (≤6m): less smoothing needed
# Coarse resolution (>6m): more smoothing to handle integer quantisation
if res_x <= 6.0:
    sigma_used = 2.0
else:
    sigma_used = SIGMA
elev_smooth = gaussian_filter(elev_filled.astype("float64"), sigma=sigma_used)
_sv = elev_smooth[np.isfinite(elev_smooth)]
print(f"  ✓ Gaussian smoothing sigma={sigma_used} (px size={res_x:.1f}m)")
print(f"    Post-smooth: finite={len(_sv):,}  "
      f"range=[{_sv.min():.2f},{_sv.max():.2f}]" if len(_sv)>0
      else "    Post-smooth: ALL NaN — check fill step")


# ── EVANS-YOUNG DERIVATIVES ───────────────────────────────────────────────────

def evans_young(z, rx, ry):
    L  = (rx + ry) / 2.0
    z  = np.array(z, dtype=np.float64)
    # Replace any remaining NaN with local mean before padding
    # to prevent NaN propagation through convolution
    z_mean = np.nanmean(z)
    z  = np.where(np.isfinite(z), z, z_mean)
    H, W = z.shape
    p  = np.pad(z, 1, mode="reflect")

    z_n  = p[0:H,   1:W+1];  z_s  = p[2:H+2, 1:W+1]
    z_e  = p[1:H+1, 2:W+2];  z_w  = p[1:H+1, 0:W  ]
    z_c  = p[1:H+1, 1:W+1]
    z_ne = p[0:H,   2:W+2];  z_nw = p[0:H,   0:W  ]
    z_se = p[2:H+2, 2:W+2];  z_sw = p[2:H+2, 0:W  ]

    D  = (z_e  - z_w ) / (2 * L)
    E  = (z_n  - z_s ) / (2 * L)
    A  = (z_e  - 2*z_c + z_w ) / L**2
    B  = (z_n  - 2*z_c + z_s ) / L**2
    C  = (z_ne - z_nw - z_se + z_sw) / (4 * L**2)

    D2E2  = D**2 + E**2
    slope = np.degrees(np.arctan(np.sqrt(D2E2)))

    with np.errstate(invalid="ignore", divide="ignore"):
        denom = D2E2 ** 1.5
        plan  = np.where(D2E2 > 1e-12,
                         -2*(A*E**2 + B*D**2 - C*D*E) / denom, np.nan)
        prof  = np.where(D2E2 > 1e-12,
                         -2*(A*D**2 + B*E**2 + C*D*E) / denom, np.nan)

    slope = np.where(water_mask, np.nan, slope)
    plan  = np.where(water_mask, np.nan, plan)
    prof  = np.where(water_mask, np.nan, prof)

    return slope.astype("float32"), plan.astype("float32"), prof.astype("float32")


print("\nComputing Evans-Young derivatives ...")
slope, plan_curv, prof_curv = evans_young(elev_smooth, res_x, res_y)

for name, arr in [("slope", slope), ("plan_curv", plan_curv),
                   ("prof_curv", prof_curv)]:
    v = arr[np.isfinite(arr)]
    total = arr.size
    if len(v) == 0:
        print(f"  {name:12s}: all NaN (water masked) — "
              f"NaN={np.sum(~np.isfinite(arr)):,}/{total:,}")
    else:
        print(f"  {name:12s}: [{v.min():.5f}, {v.max():.5f}]  "
              f"std={v.std():.5f}  "
              f"valid={len(v):,}  NaN={np.sum(~np.isfinite(arr)):,}")


# ── TWI ───────────────────────────────────────────────────────────────────────

twi_path  = DERIV_DIR / "TWI.tif"
twi_label = "TWI (D8)"

existing_twi = Path("./data/scheldt/dem/derivatives/TWI.tif")

if twi_path.exists():
    # Verify shape matches current DEM before reusing
    _twi_check = rxr.open_rasterio(twi_path, masked=True)
    if _twi_check.values[0].shape == elev.shape:
        print(f"\nTWI found at {twi_path.name} — reusing (shape matches).")
        twi = _twi_check.values[0].astype("float32")
        twi = np.where(water_mask, np.nan, twi)
    else:
        print(f"\nTWI shape mismatch "
              f"({_twi_check.values[0].shape} vs {elev.shape}) — recomputing.")
        twi_path = Path("__force_recompute__")  # trigger recompute block
else:
    print("\nComputing TWI ...")
    try:
        import richdem as rd
        dem_rd = rd.rdarray(elev_smooth, no_data=-9999)
        dem_rd.geotransform = (t.c, t.a, t.b, t.f, t.d, t.e)
        rd.FillDepressions(dem_rd, epsilon=True, in_place=True)
        accum    = rd.FlowAccumulation(dem_rd, method="D8")
        accum_m2 = np.array(accum).astype("float32") * res_x * res_y
        sr       = np.arctan(np.sqrt(
            np.gradient(elev_smooth, res_y, axis=0)**2 +
            np.gradient(elev_smooth, res_x, axis=1)**2
        ))
        sr   = np.where(sr < 0.001, 0.001, sr)
        twi  = np.log(accum_m2 / np.tan(sr)).astype("float32")
        twi  = np.where(water_mask, np.nan, twi)
        print("  ✓ richdem D8 TWI")
    except Exception as e:
        print(f"  richdem failed ({e}) — simplified TWI")
        sr      = np.arctan(np.sqrt(
            np.gradient(elev_smooth, res_y, axis=0)**2 +
            np.gradient(elev_smooth, res_x, axis=1)**2
        ))
        sr      = np.where(sr < 0.001, 0.001, sr)
        contrib = uniform_filter(
            np.ones_like(elev_smooth), size=7) * (res_x * res_y * 49)
        twi     = np.log(contrib / np.tan(sr)).astype("float32")
        twi     = np.where(water_mask, np.nan, twi)
        twi_label = "TWI proxy (flatness)"


# ── SAVE ─────────────────────────────────────────────────────────────────────

def save_tif(arr, path):
    arr = arr.astype("float32")
    with rasterio.open(DEM_PATH) as _src:
        _prof = _src.profile.copy()
    _prof.update(count=1, dtype="float32",
                 nodata=float("nan"), compress="lzw")
    with rasterio.open(str(path), "w", **_prof) as _dst:
        _dst.write(arr[np.newaxis, :, :])
    v = arr[np.isfinite(arr)]
    if len(v) > 0:
        print(f"  ✓ {path.name:35s} [{v.min():.4f}, {v.max():.4f}]")
    else:
        print(f"  ✓ {path.name:35s} [all NaN — water masked]")

print("\nSaving ...")
save_tif(slope,     DERIV_DIR / "slope_degrees.tif")
save_tif(plan_curv, DERIV_DIR / "plan_curvature.tif")
save_tif(prof_curv, DERIV_DIR / "profile_curvature.tif")
save_tif(twi,       twi_path)


# ── RESAMPLE TO S2 GRID ───────────────────────────────────────────────────────

if s2_ref is not None:
    print(f"\nResampling to S2 10m grid ...")
    rs_dir = DERIV_DIR / "resampled_10m"
    rs_dir.mkdir(exist_ok=True)

    with rasterio.open(s2_ref) as ref:
        dst_crs = ref.crs
        dst_t   = ref.transform
        dst_w   = ref.width
        dst_h   = ref.height

    # Get S2 CRS as string to avoid CRS.from_epsg() issues on Yoda
    with rasterio.open(s2_ref) as ref:
        dst_crs_str = ref.crs.to_string()

    for tif in sorted(DERIV_DIR.glob("*.tif")):
        out = rs_dir / tif.name
        try:
            # Try rioxarray reproject — handles CRS lookup differently
            da = rxr.open_rasterio(tif, masked=True)
            da_repr = da.rio.reproject(
                dst_crs_str,
                shape=(dst_h, dst_w),
                resampling=Resampling.bilinear,
            )
            da_repr.attrs.pop("_FillValue", None)
            da_repr.encoding.pop("_FillValue", None)
            da_repr = da_repr.rio.write_nodata(float("nan"))
            arr_r = da_repr.values[0].astype("float32")
            with rasterio.open(s2_ref) as _ref:
                _rp = _ref.profile.copy()
            _rp.update(count=1, dtype="float32",
                       nodata=float("nan"), compress="lzw")
            with rasterio.open(str(out), "w", **_rp) as _dst:
                _dst.write(arr_r[np.newaxis, :, :])
            print(f"  ✓ {tif.name} → 10m")
        except Exception as e:
            print(f"  ✗ {tif.name} reproject failed: {e.__class__.__name__}: {e}")
            print(f"    Derivative saved in native CRS — skipping 10m resample")


# ── VISUALISE ────────────────────────────────────────────────────────────────

print("\nGenerating figure ...")
elev_display = np.where(water_mask, np.nan, elev)

panels = [
    ("Elevation (m)\nwater masked",           elev_display, "terrain", "elev"),
    ("Slope (°)\nreliable",                   slope,        "YlOrRd",  "slope"),
    ("Plan curvature\ngeomorphic proxy",       plan_curv,    "RdBu_r",  "curv"),
    ("Profile curvature\ngeomorphic proxy",    prof_curv,    "RdBu_r",  "curv"),
    (twi_label,                                twi,          "Blues",   "twi"),
]

fig, axes = plt.subplots(1, 5, figsize=(25, 7), facecolor="#0e0e0e")

for ax, (title, data, cmap, kind) in zip(axes, panels):
    valid = data[np.isfinite(data)]
    if len(valid) == 0:
        ax.set_visible(False)
        continue

    if kind == "elev":
        vmin, vmax = np.percentile(valid, 2), np.percentile(valid, 98)
    elif kind == "slope":
        vmin, vmax = 0, np.percentile(valid, 98)
    elif kind == "curv":
        p = max(np.percentile(np.abs(valid), 90), 1e-6)
        vmin, vmax = -p, p
    elif kind == "twi":
        vmin, vmax = np.percentile(valid, 5), np.percentile(valid, 95)
    else:
        vmin, vmax = np.percentile(valid, 2), np.percentile(valid, 98)

    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax,
                    aspect="equal", interpolation="bilinear")
    cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.ax.tick_params(colors="white", labelsize=7)
    ax.set_title(title, color="white", fontsize=9,
                  fontweight="bold", pad=4)
    ax.axis("off")
    ax.set_facecolor("#0e0e0e")

fig.suptitle(
    f"Scheldt — Terrain derivatives  |  {LABEL}  |  Evans-Young",
    color="white", fontsize=12, fontweight="bold"
)
fig.tight_layout()
fig.savefig(str(OUT_PNG), dpi=150, bbox_inches="tight", facecolor="#0e0e0e")
print(f"\n✓ Saved → {OUT_PNG}")
plt.show()
