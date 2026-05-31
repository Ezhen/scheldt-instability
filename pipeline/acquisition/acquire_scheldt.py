import os
os.environ["CPL_LOG"] = "/dev/null"

"""
Scheldt Estuary — Satellite Data Acquisition Script
====================================================
Study area : Verdronken Land van Saeftinghe + adjacent Sea Scheldt banks
Platform   : Copernicus Data Space Ecosystem (CDSE) via openEO
Sensors    : Sentinel-2 L2A (optical)  +  Sentinel-1 GRD (SAR)
Strategy   : Seasonal composites 2017–2025, clipped to AOI in the cloud
             → only ~MB of final GeoTIFFs downloaded locally

Dependencies
------------
    pip install openeo

Registration (free)
-------------------
    https://dataspace.copernicus.eu  → create an account

Author : your-name-here
"""

import openeo
from pathlib import Path
from datetime import date

# ── 0. CONFIG ────────────────────────────────────────────────────────────────

# Saeftinghe + Sea Scheldt banks bounding box  (WGS84 / EPSG:4326)
# Covers ~30 × 20 km: Verdronken Land, tidal channel, Belgian bank upstream
AOI = {
    "west":  3.80,
    "east":  4.25,
    "south": 51.28,
    "north": 51.45,
    "crs":   "EPSG:4326",
}

# Output directory — one sub-folder per sensor
OUT_DIR = Path("./data/scheldt")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Seasonal windows — dry season (May–Jun), late summer (Aug–Sep), winter (Dec–Jan)
# Structured as list of (label, start, end) tuples
# Adjust the year range to taste; 2017 = Sentinel-2 launch
SEASONS = []
for year in range(2025, 2026):
    SEASONS += [
        (f"{year}_spring",  f"{year}-05-01",   f"{year}-06-30"),
        (f"{year}_summer",  f"{year}-08-01",   f"{year}-09-30"),
        (f"{year}_winter",  f"{year}-12-01",   f"{year+1}-01-31"),
    ]

# Cloud cover threshold for Sentinel-2 scene selection (%)
MAX_CLOUD_COVER = 20

# Sentinel-2 bands needed for this workflow
# B02 Blue, B03 Green, B04 Red, B08 NIR  → NDWI = (B03-B08)/(B03+B08)
#                                         → NDVI = (B08-B04)/(B08+B04)
# SCL = Scene Classification Layer (used for cloud masking)
S2_BANDS = ["B02", "B03", "B04", "B08", "B11", "SCL"]

# Sentinel-1 polarisations (VV + VH for dual-pol GRD over land/water edges)
S1_BANDS = ["VV", "VH"]


# ── 1. AUTHENTICATION ────────────────────────────────────────────────────────

def connect() -> openeo.Connection:
    """
    Authenticate against the CDSE openEO back-end.
    On first run this opens a browser tab for OAuth login.
    Credentials are cached locally after that.
    """
    conn = openeo.connect("https://openeo.dataspace.copernicus.eu")
    conn.authenticate_oidc()          # OAuth2 — no password stored in script
    return conn


# ── 2. CLOUD MASKING HELPER (Sentinel-2) ─────────────────────────────────────

def mask_clouds(cube: openeo.DataCube) -> openeo.DataCube:
    """
    Use the SCL band to mask clouds, cloud shadows, and saturated pixels.
    SCL classes kept (valid):  4 = vegetation, 5 = bare soil, 6 = water,
                                7 = unclassified, 11 = snow (kept for winter)
    Everything else → masked out (set to nodata).
    """
    scl = cube.band("SCL")
    # Build a boolean mask: True where pixel is INVALID
    invalid = (
        (scl == 0)  |   # no data
        (scl == 1)  |   # saturated / defective
        (scl == 2)  |   # dark area (shadow-like)
        (scl == 3)  |   # cloud shadow
        (scl == 8)  |   # cloud medium prob
        (scl == 9)  |   # cloud high prob
        (scl == 10)     # thin cirrus
    )
    return cube.mask(invalid)


# ── 3. SENTINEL-2 ACQUISITION ────────────────────────────────────────────────

def acquire_s2(conn: openeo.Connection) -> None:
    """
    For each seasonal window:
      1. Load SENTINEL2_L2A clipped to AOI (server-side)
      2. Filter by cloud cover threshold
      3. Apply SCL cloud mask
      4. Compute median composite (temporal reduction → single image per season)
      5. Submit as batch job → download GeoTIFF when done
    """
    s2_dir = OUT_DIR / "sentinel2"
    s2_dir.mkdir(exist_ok=True)

    for label, start, end in SEASONS:
        out_path = s2_dir / f"S2_{label}.tif"
        if out_path.exists():
            print(f"  [S2] {label} already downloaded, skipping.")
            continue

        print(f"  [S2] Submitting job: {label}  ({start} → {end})")

        cube = conn.load_collection(
            "SENTINEL2_L2A",
            spatial_extent=AOI,
            temporal_extent=[start, end],
            bands=S2_BANDS,
            max_cloud_cover=MAX_CLOUD_COVER,   # pre-filter at scene level
        )

        # Pixel-level cloud masking
        cube = mask_clouds(cube)

        # Temporal median composite → collapses time dimension
        # median is more robust to remaining cloud edges than mean
        composite = cube.reduce_dimension(dimension="t", reducer="median")

        # Drop SCL band from output (only needed for masking)
        composite = composite.filter_bands(["B02", "B03", "B04", "B08", "B11"])

        # Save as cloud-optimised GeoTIFF
        result = composite.save_result(format="GTiff")

        job = result.create_job(title=f"Scheldt_S2_{label}")
        job.start_and_wait()                   # blocks; use start_job() for async
        job.get_results().download_files(str(s2_dir))

        # Rename to our convention
        downloaded = sorted(s2_dir.glob("*.tif"))[-1]
        downloaded.rename(out_path)
        print(f"  [S2] Saved → {out_path}")


# ── 4. SENTINEL-1 ACQUISITION ────────────────────────────────────────────────

def acquire_s1(conn: openeo.Connection) -> None:
    """
    For each seasonal window:
      1. Load SENTINEL1_GRD clipped to AOI, IW mode, ascending orbit
      2. Apply radiometric terrain correction (sar_backscatter)
         → converts DN to gamma0 in dB, normalised to local terrain slope
      3. Temporal mean composite
      4. Submit batch job → download GeoTIFF

    Note on orbit selection:
      Ascending pass (~18:00 UTC) and descending (~06:00 UTC) give different
      look angles over the Scheldt. Pick ONE consistently to avoid mixing
      backscatter geometries in your time series.
      Ascending relative orbit 37 gives good coverage over this AOI.
    """
    s1_dir = OUT_DIR / "sentinel1"
    s1_dir.mkdir(exist_ok=True)

    for label, start, end in SEASONS:
        out_path = s1_dir / f"S1_{label}.tif"
        if out_path.exists():
            print(f"  [S1] {label} already downloaded, skipping.")
            continue

        print(f"  [S1] Submitting job: {label}  ({start} → {end})")

        cube = conn.load_collection(
            "SENTINEL1_GRD",
            spatial_extent=AOI,
            temporal_extent=[start, end],
            bands=S1_BANDS,
            # Filter to ascending orbit for consistent look geometry
            properties={
                "sat:orbit_state": lambda x: x == "ascending",
            },
        )

        # Radiometric terrain correction → gamma0 dB (CARD4L compliant)
        # to this:
        cube = cube.sar_backscatter(
            coefficient="sigma0-ellipsoid",
            local_incidence_angle=False,
        )

        # Temporal mean composite (SAR has no cloud issue; mean is fine)
        composite = cube.reduce_dimension(dimension="t", reducer="mean")

        result = composite.save_result(format="GTiff")

        job = result.create_job(title=f"Scheldt_S1_{label}")
        job.start_and_wait()
        job.get_results().download_files(str(s1_dir))

        downloaded = sorted(s1_dir.glob("*.tif"))[-1]
        downloaded.rename(out_path)
        print(f"  [S1] Saved → {out_path}")


# ── 5. COMPUTE INDICES ───────────────────────────────────────────────────────

def compute_indices_locally() -> None:
    """
    After download, compute NDWI and NDVI from the Sentinel-2 GeoTIFFs.
    Runs locally with rioxarray — lightweight, no cloud credits needed.

    Install:  pip install rioxarray
    """
    try:
        import rioxarray as rxr
        import numpy as np
    except ImportError:
        print("  [IDX] rioxarray not installed — skipping index computation.")
        print("        Run:  pip install rioxarray")
        return

    idx_dir = OUT_DIR / "indices"
    idx_dir.mkdir(exist_ok=True)

    for s2_tif in sorted((OUT_DIR / "sentinel2").glob("S2_*.tif")):
        label = s2_tif.stem.replace("S2_", "")

        ndwi_path = idx_dir / f"NDWI_{label}.tif"
        ndvi_path  = idx_dir / f"NDVI_{label}.tif"

        if ndwi_path.exists() and ndvi_path.exists():
            print(f"  [IDX] {label} indices already exist, skipping.")
            continue

        ds = rxr.open_rasterio(s2_tif, masked=True)
        # Band order matches S2_BANDS after dropping SCL: B02 B03 B04 B08 B11
        # Index:                                           0    1    2    3    4
        green = ds.sel(band=2).astype("float32")   # B03
        red   = ds.sel(band=3).astype("float32")   # B04
        nir   = ds.sel(band=4).astype("float32")   # B08

        # NDWI (McFeeters) — positive = water / exposed mudflat
        ndwi = (green - nir) / (green + nir + 1e-9)
        ndwi = ndwi.clip(-1, 1)
        ndwi.rio.to_raster(str(ndwi_path))

        # NDVI — positive = vegetation (marsh, reed bed)
        ndvi = (nir - red) / (nir + red + 1e-9)
        ndvi = ndvi.clip(-1, 1)
        ndvi.rio.to_raster(str(ndvi_path))

        print(f"  [IDX] {label} → NDWI + NDVI saved.")


# ── 6. MAIN ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Scheldt acquisition pipeline ===")
    print(f"AOI     : {AOI}")
    print(f"Seasons : {len(SEASONS)} windows (2017–2025)")
    print(f"Output  : {OUT_DIR.resolve()}\n")

    conn = connect()

    print("\n── Sentinel-2 ──────────────────────────")
    acquire_s2(conn)

    print("\n── Sentinel-1 ──────────────────────────")
    acquire_s1(conn)

    print("\n── Computing indices locally ───────────")
    compute_indices_locally()

    print("\nDone. Outputs:")
    for f in sorted(OUT_DIR.rglob("*.tif")):
        print(f"  {f}")
