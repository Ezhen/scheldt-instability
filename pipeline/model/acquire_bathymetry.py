"""
Scheldt Estuary — Bathymetry Acquisition via EMODnet WCS
=========================================================
Downloads the EMODnet Digital Terrain Model (DTM 2024) for the
Western Scheldt domain via the confirmed-working OGC WCS endpoint.

Resolution  : ~115m (1/16 arc-minute)
Coverage    : Full Western + Sea Scheldt including Belgian side
CRS         : EPSG:4326 (WGS84) — reprojected to RD New on output
Depth datum : m below MSL (negated → m NAP for consistency)
Source      : EMODnet Bathymetry consortium, 2024 release
WCS URL     : https://ows.emodnet-bathymetry.eu/wcs

Why not vaklodingen 20m?
------------------------
The Deltares OPeNDAP vaklodingen server covers the North Sea coast
only — not the Zeeuwse Delta / Western Scheldt. The Waterinfo-Extra
download returns zero-byte ZIPs. The EMODnet WCS is the only
confirmed-working automated source for this region.

Outputs
-------
  dem/bathymetry/emodnet_bathy_wgs84.tif      raw WCS output
  dem/bathymetry/emodnet_bathy_rd.tif         reprojected to RD New
  dem/bathymetry/emodnet_bathy_overview.png   figure
  dem/bathymetry/bathymetry_metadata.json     provenance

Usage
-----
    pip install owslib rasterio numpy matplotlib
    python acquire_bathymetry.py

    # Custom AOI (WGS84 degrees)
    python acquire_bathymetry.py --bbox 3.3 51.2 4.5 51.6

    # Higher resolution request (slower but sharper)
    python acquire_bathymetry.py --res 0.001
"""

import os
# PROJ database paths for Yoda HPC:
# - rasterio uses PROJ_DATA/PROJ_LIB env vars → point to rasterio's proj_data
# - pyproj 3.7+ has its own bundled proj.db and ignores env vars;
#   must be set via pyproj.datadir.set_data_dir() before any EPSG lookup.
_RASTERIO_PROJ = "/home/ulg/mast/eivanov/.conda/envs/Yoda/lib/python3.10/site-packages/rasterio/proj_data"
_PYPROJ_DATA   = "/home/ulg/mast/eivanov/.conda/envs/Yoda/lib/python3.10/site-packages/pyproj/proj_dir/share/proj"
os.environ["CPL_LOG"]   = "/dev/null"
os.environ["PROJ_DATA"] = _RASTERIO_PROJ
os.environ["PROJ_LIB"]  = _RASTERIO_PROJ
import pyproj
pyproj.datadir.set_data_dir(_PYPROJ_DATA)

import sys
import json
import argparse
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

try:
    from owslib.wcs import WebCoverageService
except ImportError:
    sys.exit("pip install owslib")

try:
    import rasterio
    from rasterio.warp import reproject, Resampling, calculate_default_transform
    from rasterio.transform import from_bounds
    from rasterio.io import MemoryFile
except ImportError:
    sys.exit("pip install rasterio")


# ── CONFIG ────────────────────────────────────────────────────────────────────

WCS_URL    = "https://ows.emodnet-bathymetry.eu/wcs"
COVERAGE   = "emodnet:mean"          # mean depth layer — most complete
WCS_VER    = "1.0.0"

# Model domain AOI — wider than satellite AOI for DFM buffer
# WGS84 degrees: includes sea approach west of Vlissingen + Belgian reach
BBOX_WGS84 = (3.15, 51.18, 4.38, 51.52)  # covers MODEL_RD x=[-5000,80000]

# Output resolution — 1/16 arcmin (~115m) is native EMODnet resolution
# Use 0.001° (~111m) for slightly higher detail
RES_DEG    = 0.00208333   # native 1/16 arcmin

# Satellite AOI for annotation
SAT_AOI    = (3.80, 51.28, 4.25, 51.45)

OUT_DIR    = Path("./data/scheldt/dem/bathymetry")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── ARGUMENT PARSING ─────────────────────────────────────────────────────────

ap = argparse.ArgumentParser()
ap.add_argument("--bbox", nargs=4, type=float,
                default=list(BBOX_WGS84),
                metavar=("W","S","E","N"),
                help="Bounding box WGS84 (default: full Scheldt domain)")
ap.add_argument("--res",  type=float, default=RES_DEG,
                help=f"Resolution degrees (default: {RES_DEG})")
ap.add_argument("--coverage", default=COVERAGE,
                help="WCS coverage name (default: emodnet:mean)")
args = ap.parse_args()

bbox = tuple(args.bbox)
res  = args.res


# ── WCS DOWNLOAD ─────────────────────────────────────────────────────────────

print("=== EMODnet Bathymetry — Western Scheldt ===\n")
print(f"WCS endpoint : {WCS_URL}")
print(f"Coverage     : {args.coverage}")
print(f"BBOX (WGS84) : {bbox}")
print(f"Resolution   : {res}° (~{res*111000:.0f}m)")

print("\n[1] Connecting to WCS ...")
try:
    wcs = WebCoverageService(WCS_URL, version=WCS_VER, timeout=60)
    print(f"  ✓ Connected — {len(wcs.contents)} coverages available")
    if args.coverage in wcs.contents:
        cov = wcs.contents[args.coverage]
        print(f"  ✓ Coverage '{args.coverage}' found")
        try:
            print(f"    Title   : {cov.title}")
            bb = cov.boundingBoxWGS84
            print(f"    Extent  : {bb}")
        except Exception:
            pass
    else:
        available = list(wcs.contents.keys())[:10]
        print(f"  ✗ Coverage not found. Available: {available}")
        sys.exit(1)
except Exception as e:
    print(f"  ✗ Connection failed: {e}")
    print(f"  Check network: curl '{WCS_URL}?SERVICE=WCS&REQUEST=GetCapabilities'")
    sys.exit(1)

print("\n[2] Requesting coverage ...")
try:
    response = wcs.getCoverage(
        identifier=args.coverage,
        bbox=bbox,
        crs="EPSG:4326",
        format="image/tiff",
        resx=res,
        resy=res,
        timeout=120,
    )
    data = response.read()
    print(f"  ✓ Received {len(data)/1024:.0f} KB")
    if len(data) < 1000:
        print(f"  ✗ Response too small — may be empty")
        print(f"  Raw: {data[:200]}")
        sys.exit(1)
except Exception as e:
    print(f"  ✗ GetCoverage failed: {e}")
    sys.exit(1)

print("\n[3] Parsing and saving GeoTIFF ...")

# Write to file via MemoryFile
out_wgs = OUT_DIR / "emodnet_bathy_wgs84.tif"
try:
    with MemoryFile(data) as mem:
        with mem.open() as src:
            arr    = src.read(1).astype("float32")
            t_in   = src.transform
            crs_in = src.crs
            prof   = src.profile.copy()

    print(f"  Shape  : {arr.shape}")
    print(f"  CRS    : {crs_in}")
    print(f"  Transform: {t_in}")

    # EMODnet depth convention: positive = depth below sea surface
    # Convert to m NAP convention: negative below surface, positive above
    # EMODnet values: negative = depth (already below surface)
    # No sign flip needed — negative values are already below MSL
    v = arr[np.isfinite(arr) & (arr > -1000) & (arr < 1000)]
    if len(v) == 0:
        print("  ✗ No valid depth values")
        sys.exit(1)
    print(f"  Range  : [{v.min():.2f}, {v.max():.2f}] m")
    print(f"  Valid  : {len(v):,} pixels")

    # Mask nodata
    nodata_val = prof.get("nodata", None)
    if nodata_val is not None:
        arr = np.where(arr == nodata_val, np.nan, arr)
    arr = np.where((arr < -1000) | (arr > 500), np.nan, arr)

    # Save WGS84 version
    prof.update(dtype="float32", nodata=float("nan"), compress="lzw")
    with rasterio.open(str(out_wgs), "w", **prof) as dst:
        dst.write(arr[np.newaxis, :, :])
    print(f"  ✓ {out_wgs.name}")

except Exception as e:
    # Fallback: write raw bytes and try to open
    out_raw = OUT_DIR / "emodnet_raw.tif"
    with open(out_raw, "wb") as f:
        f.write(data)
    print(f"  Saved raw to {out_raw.name}")
    print(f"  Error parsing: {e}")
    with rasterio.open(str(out_raw)) as src:
        arr    = src.read(1).astype("float32")
        t_in   = src.transform
        crs_in = src.crs
        prof   = src.profile.copy()
    arr = np.where((arr < -1000) | (arr > 500), np.nan, arr)
    prof.update(dtype="float32", nodata=float("nan"), compress="lzw")
    with rasterio.open(str(out_wgs), "w", **prof) as dst:
        dst.write(arr[np.newaxis, :, :])
    print(f"  ✓ Recovered: {out_wgs.name}")

print("\n[4] Reprojecting to RD New (EPSG:28992) ...")
out_rd = OUT_DIR / "emodnet_bathy_rd.tif"

try:
    # EMODnet WCS always delivers EPSG:4326 — force plain string to avoid
    # the rasterio/PROJ EngineeringCRS OGC-WKT object parse failure that
    # occurs when crs_in is passed as a CRS object on some PROJ builds.
    src_crs_str = "EPSG:4326"
    dst_crs     = "EPSG:28992"

    transform_rd, width_rd, height_rd = calculate_default_transform(
        src_crs_str, dst_crs,
        arr.shape[1], arr.shape[0],          # width, height from array
        left=bbox[0], bottom=bbox[1],
        right=bbox[2], top=bbox[3],
    )
    arr_rd = np.full((height_rd, width_rd), np.nan, dtype="float32")
    reproject(
        source=arr,
        destination=arr_rd,
        src_transform=t_in,
        src_crs=src_crs_str,                 # plain string — not crs_in object
        dst_transform=transform_rd,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
        src_nodata=float("nan"),
        dst_nodata=float("nan"),
    )
    prof_rd = {
        "driver":    "GTiff",
        "dtype":     "float32",
        "width":     width_rd,
        "height":    height_rd,
        "count":     1,
        "crs":       dst_crs,
        "transform": transform_rd,
        "nodata":    float("nan"),
        "compress":  "lzw",
    }
    with rasterio.open(str(out_rd), "w", **prof_rd) as dst:
        dst.write(arr_rd[np.newaxis, :, :])
    v_rd = arr_rd[np.isfinite(arr_rd)]
    if len(v_rd) > 0:
        print(f"  ✓ {out_rd.name}  shape={arr_rd.shape}  "
              f"range=[{v_rd.min():.2f},{v_rd.max():.2f}]m")
    else:
        print(f"  ✓ {out_rd.name}  (all NaN — check bbox/PROJ config)")
except Exception as e:
    print(f"  ✗ Reproject failed: {e}")

print("\n[5] Generating overview figure ...")
try:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor="#0e0e0e")

    # WGS84 version
    ax = axes[0]
    ax.set_facecolor("#0e0e0e")
    v = arr[np.isfinite(arr)]
    if len(v) > 0:
        vmin = max(np.percentile(v, 2), -40)
        vmax = min(np.percentile(v, 98),  5)
        im = ax.imshow(np.flipud(arr), cmap="RdBu_r",
                        vmin=vmin, vmax=vmax,
                        extent=[bbox[0],bbox[2],bbox[1],bbox[3]],
                        aspect="equal", interpolation="bilinear")
        plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02,
                     label="m").ax.tick_params(colors="white", labelsize=7)
        # Mark satellite AOI
        from matplotlib.patches import Rectangle
        rect = Rectangle((SAT_AOI[0], SAT_AOI[1]),
                           SAT_AOI[2]-SAT_AOI[0],
                           SAT_AOI[3]-SAT_AOI[1],
                           linewidth=2, edgecolor="yellow",
                           facecolor="none", linestyle="--")
        ax.add_patch(rect)
        ax.text(SAT_AOI[0], SAT_AOI[3]+0.01, "Satellite AOI",
                color="yellow", fontsize=8)
        ax.set_xlabel("Longitude (°E)", color="white")
        ax.set_ylabel("Latitude (°N)", color="white")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#444")
    ax.set_title("EMODnet Bathymetry 2024\nWestern Scheldt domain",
                  color="white", fontsize=9, fontweight="bold")

    # Histogram
    ax2 = axes[1]
    ax2.set_facecolor("#1a1a1a")
    if len(v) > 0:
        ax2.hist(v[v < 2], bins=100, color="#42A5F5", alpha=0.85)
        ax2.axvline(0,    color="white",   lw=1.5, ls="--",
                    label="MSL (0m)")
        ax2.axvline(-2,   color="#FF9800", lw=1.2, ls=":",
                    label="MLW approx")
        ax2.axvline(-15,  color="#EF5350", lw=1.2, ls=":",
                    label="Navigation channel depth")
        ax2.set_xlabel("Depth (m, negative = below MSL)",
                        color="white", fontsize=9)
        ax2.set_ylabel("Pixel count", color="white", fontsize=9)
        ax2.tick_params(colors="white", labelsize=8)
        ax2.spines[:].set_color("#444")
        ax2.legend(fontsize=8, facecolor="#1a1a1a", labelcolor="white")
    ax2.set_title("Depth distribution\nmodel domain",
                   color="white", fontsize=9, fontweight="bold")

    fig.suptitle(
        "EMODnet Bathymetry DTM 2024  |  Western Scheldt  |  ~115m resolution",
        color="white", fontsize=11, fontweight="bold"
    )
    fig.subplots_adjust(top=0.90)
    out_png = OUT_DIR / "emodnet_bathy_overview.png"
    fig.savefig(str(out_png), dpi=150, bbox_inches=None, facecolor="#0e0e0e")
    print(f"  ✓ {out_png.name}")
    plt.close(fig)
except Exception as e:
    print(f"  Figure failed: {e}")

# Metadata
meta = {
    "source":       "EMODnet Bathymetry DTM 2024",
    "wcs_url":      WCS_URL,
    "coverage":     args.coverage,
    "bbox_wgs84":   list(bbox),
    "resolution_deg": res,
    "resolution_m_approx": round(res * 111000),
    "crs_output":   "EPSG:28992 (RD New)",
    "depth_convention": "m, negative below MSL (= approx m NAP)",
    "note": ("Vaklodingen 20m not available via API for Western Scheldt. "
             "EMODnet 115m used for DFM model domain bathymetry. "
             "AHN4 5m lidar covers intertidal zone."),
    "generated": datetime.now().isoformat(),
}
with open(OUT_DIR/"bathymetry_metadata.json","w") as f:
    json.dump(meta, f, indent=2)
print(f"  ✓ bathymetry_metadata.json")

print(f"\n✓ Complete. Outputs in {OUT_DIR}/")
print(f"\nFor DFM model (coarse domain bathymetry):")
print(f"  python prepare_dfm_model.py \\")
print(f"    --bathy {out_rd}")
print(f"\nFor channel detail (intertidal + subtidal near Saeftinghe):")
print(f"  Merge with AHN4: AHN4 takes priority where it has data,")
print(f"  EMODnet fills the subtidal zone below AHN4 coverage.")
print(f"\nFor the 20m vaklodingen data specifically:")
print(f"  Manual download from waterinfo-extra.rws.nl")
print(f"  → Morfologie → Westerschelde → draw bbox → download GeoTIFF")
