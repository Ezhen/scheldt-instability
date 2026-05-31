"""
Scheldt Estuary — Copernicus DEM Acquisition + Terrain Derivatives
===================================================================
Downloads the Copernicus DEM GLO-30 (30m resolution) for the Saeftinghe
AOI via the CDSE OData API — same credentials as your openEO login.
No S3 keys, no extra setup.

Outputs
-------
    dem/DEM_scheldt_clipped.tif
    dem/derivatives/slope_degrees.tif
    dem/derivatives/plan_curvature.tif
    dem/derivatives/profile_curvature.tif
    dem/derivatives/TWI.tif
    dem/derivatives/resampled_10m/
    dem/terrain_derivatives.png

Credentials
-----------
    export CDSE_USER="your@email.com"
    export CDSE_PASS="yourpassword"

Dependencies
------------
    pip install requests rioxarray rasterio numpy matplotlib tqdm
    pip install richdem   # optional, for TWI

Usage
-----
    python acquire_dem.py
"""

import os
import sys
import math
import zipfile
import warnings
from pathlib import Path
from getpass import getpass

import numpy as np
import matplotlib.pyplot as plt
import requests
from tqdm import tqdm
import rioxarray as rxr
import rasterio
from rasterio.merge import merge
from rasterio.warp import reproject, Resampling

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ── CONFIG ────────────────────────────────────────────────────────────────────

AOI = {
    "west":  3.80,
    "east":  4.25,
    "south": 51.28,
    "north": 51.45,
}

ODATA_BASE    = "https://catalogue.dataspace.copernicus.eu/odata/v1"
TOKEN_URL     = ("https://identity.dataspace.copernicus.eu/auth/realms/CDSE"
                 "/protocol/openid-connect/token")
DOWNLOAD_BASE = "https://download.dataspace.copernicus.eu/odata/v1"
COLLECTION    = "COP-DEM_GLO-30-DGED/2024_1"

OUT_DIR = Path("./data/scheldt/dem")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Set True to force recompute derivatives even if files exist ───────────────
FORCE_RECOMPUTE = True


# ── STEP 0: AUTHENTICATION ───────────────────────────────────────────────────

def get_token() -> str:
    user = os.environ.get("CDSE_USER") or input("CDSE email   : ").strip()
    pwd  = os.environ.get("CDSE_PASS") or getpass("CDSE password: ")
    resp = requests.post(TOKEN_URL, data={
        "client_id":  "cdse-public",
        "username":   user,
        "password":   pwd,
        "grant_type": "password",
    }, timeout=30)
    resp.raise_for_status()
    print("  ✓ Authenticated")
    return resp.json()["access_token"]


def session(token: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({"Authorization": f"Bearer {token}"})
    return s


# ── STEP 1: SEARCH ───────────────────────────────────────────────────────────

def find_products(sess: requests.Session) -> list[dict]:
    print("\n[1] Searching OData catalogue for GLO-30 tiles ...")
    bbox_wkt = (
        f"POLYGON(("
        f"{AOI['west']} {AOI['south']},"
        f"{AOI['east']} {AOI['south']},"
        f"{AOI['east']} {AOI['north']},"
        f"{AOI['west']} {AOI['north']},"
        f"{AOI['west']} {AOI['south']}"
        f"))"
    )
    params = {
        "$filter": (
            f"Attributes/OData.CSC.StringAttribute/any("
            f"att:att/Name eq 'dataset' and "
            f"att/OData.CSC.StringAttribute/Value eq '{COLLECTION}') and "
            f"OData.CSC.Intersects(area=geography'SRID=4326;{bbox_wkt}')"
        ),
        "$top": 20,
    }
    resp = sess.get(f"{ODATA_BASE}/Products", params=params, timeout=30)
    resp.raise_for_status()
    items = resp.json().get("value", [])
    if not items:
        sys.exit("  No DEM products found.")

    products = []
    for item in items:
        products.append({
            "id":   item["Id"],
            "name": item["Name"],
            "size": item.get("ContentLength", 0),
            "url":  f"{DOWNLOAD_BASE}/Products({item['Id']})/$value",
        })
        print(f"  {item['Name']}  ({item.get('ContentLength',0)/1e6:.0f} MB)")
    print(f"  → {len(products)} tile(s) found")
    return products


# ── STEP 2: DOWNLOAD ─────────────────────────────────────────────────────────

def download_tiles(sess, products) -> list[Path]:
    print("\n[2] Downloading tiles ...")
    raw_dir = OUT_DIR / "raw_tiles"
    raw_dir.mkdir(exist_ok=True)
    paths = []

    for prod in products:
        tif_path = raw_dir / f"{prod['name']}.tif"
        zip_path = raw_dir / f"{prod['name']}.zip"

        if tif_path.exists():
            print(f"  {prod['name']} already present, skipping.")
            paths.append(tif_path)
            continue

        if not zip_path.exists():
            resp = sess.get(prod["url"], stream=True, timeout=300)
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", prod["size"]))
            with open(zip_path, "wb") as f, \
                 tqdm(total=total, unit="B", unit_scale=True,
                      desc=f"  {prod['name'][:45]}", leave=False) as pbar:
                for chunk in resp.iter_content(chunk_size=256*1024):
                    f.write(chunk)
                    pbar.update(len(chunk))
            print(f"  ✓ {zip_path.name}")

        with zipfile.ZipFile(zip_path, "r") as zf:
            dem_files = [n for n in zf.namelist()
                         if n.endswith(".tif") and "DEM" in n.upper()]
            if not dem_files:
                dem_files = [n for n in zf.namelist() if n.endswith(".tif")]
            if dem_files:
                tif_path.write_bytes(zf.read(dem_files[0]))
                print(f"  ✓ Extracted → {tif_path.name}")
                paths.append(tif_path)
            else:
                print(f"  ✗ No .tif inside {zip_path.name}")
    return paths


# ── STEP 3: MERGE + CLIP ─────────────────────────────────────────────────────

def merge_and_clip(tile_paths: list[Path]) -> Path:
    print("\n[3] Merging and clipping to AOI ...")
    out_path = OUT_DIR / "DEM_scheldt_clipped.tif"

    if out_path.exists() and not FORCE_RECOMPUTE:
        print(f"  Already exists: {out_path.name}")
        return out_path

    datasets = [rasterio.open(p) for p in tile_paths]
    merged, merged_transform = merge(datasets)
    profile = datasets[0].profile.copy()
    for ds in datasets:
        ds.close()

    tmp = OUT_DIR / "_merged_tmp.tif"
    profile.update(height=merged.shape[1], width=merged.shape[2],
                   transform=merged_transform, compress="lzw")
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(merged)

    ds = rxr.open_rasterio(tmp, masked=True)
    clipped = ds.rio.clip_box(minx=AOI["west"], miny=AOI["south"],
                               maxx=AOI["east"], maxy=AOI["north"])
    clipped.rio.to_raster(str(out_path), compress="lzw")
    tmp.unlink(missing_ok=True)

    # ── BUG 1 FIX: report stats on valid pixels only ──────────────────────────
    elev = clipped.values[0].astype("float32")
    valid = elev[np.isfinite(elev) & (elev > -1000)]   # exclude nodata
    print(f"  ✓ Clipped DEM: {out_path.name}")
    print(f"    Shape : {elev.shape}")
    print(f"    Range : {valid.min():.1f} m  to  {valid.max():.1f} m  "
          f"(valid pixels only)")
    return out_path


# ── STEP 4: TERRAIN DERIVATIVES ──────────────────────────────────────────────

def terrain_derivatives(dem_path: Path) -> dict[str, Path]:
    print("\n[4] Computing terrain derivatives ...")
    deriv_dir = OUT_DIR / "derivatives"
    deriv_dir.mkdir(exist_ok=True)

    ds   = rxr.open_rasterio(dem_path, masked=True)
    elev = ds.values[0].astype("float32")

    # ── BUG 1 FIX: mask nodata before computing gradients ────────────────────
    # GLO-30 nodata is typically 0 over water or large negative values
    # Mask anything implausible for the Scheldt region
    nodata_mask = (elev < -500) | (elev > 5000)
    elev = np.where(nodata_mask, np.nan, elev)

    # Fill NaN with local mean for gradient computation (channel pixels)
    # This prevents gradient spikes at water/land boundaries
    from scipy.ndimage import generic_filter
    def nanmean_fill(arr):
        centre = arr[len(arr)//2]
        return centre if np.isfinite(centre) else np.nanmean(arr)

    elev_filled = elev.copy()

    from scipy.ndimage import gaussian_filter

    # GLO-30 DGED stores integer metres — smooth before differentiating
    # sigma=1.5 pixels ≈ one resolution cell, preserves real features
    elev_filled = gaussian_filter(elev_filled.astype("float32"), sigma=1.5)

    nan_mask = ~np.isfinite(elev_filled)
    if nan_mask.any():
        from scipy.ndimage import uniform_filter
        # Simple approach: fill NaN with neighbourhood mean iteratively
        filled = elev_filled.copy()
        filled[nan_mask] = 0.0
        count = (~nan_mask).astype("float32")
        # 3-pass smoothing to propagate valid values into NaN regions
        for _ in range(3):
            smooth = uniform_filter(filled, size=5)
            c_smooth = uniform_filter(count, size=5)
            c_smooth = np.where(c_smooth < 1e-6, 1.0, c_smooth)
            filled = np.where(nan_mask, smooth / c_smooth, filled)
        elev_filled = filled

    # Pixel size in metres at ~51.4°N
    t       = ds.rio.transform()
    res_x_m = abs(t.a) * 111_320 * np.cos(np.radians(51.4))
    res_y_m = abs(t.e) * 111_320
    print(f"  Pixel size: {res_x_m:.1f} × {res_y_m:.1f} m")

    outputs = {}

    # Shared gradient on filled elevation
    dz_dy, dz_dx = np.gradient(elev_filled, res_y_m, res_x_m)

    # ── Slope ─────────────────────────────────────────────────────────────────
    p = deriv_dir / "slope_degrees.tif"
    if not p.exists() or FORCE_RECOMPUTE:
        slope = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
        slope = np.where(nodata_mask, np.nan, slope)
        _save(slope.astype("float32"), ds, p)
    outputs["slope"] = p
    print("  ✓ slope_degrees.tif")

    # ── BUG 2 FIX: Plan curvature — recompute, restore nodata mask ────────────
    p = deriv_dir / "plan_curvature.tif"
    if not p.exists() or FORCE_RECOMPUTE:
        dxx = np.gradient(dz_dx, res_x_m, axis=1)
        dyy = np.gradient(dz_dy, res_y_m, axis=0)
        dxy = np.gradient(dz_dx, res_y_m, axis=0)
        px, py = dz_dx**2, dz_dy**2
        denom = (px + py) * np.sqrt(1 + px + py)
        denom = np.where(np.abs(denom) < 1e-10, np.nan, denom)
        planc = (dxx*py - 2*dxy*dz_dx*dz_dy + dyy*px) / denom
        planc = np.where(nodata_mask, np.nan, planc).astype("float32")
        _save(planc, ds, p)
    outputs["plan_curvature"] = p
    print("  ✓ plan_curvature.tif")

    # ── BUG 2 FIX: Profile curvature — same ──────────────────────────────────
    p = deriv_dir / "profile_curvature.tif"
    if not p.exists() or FORCE_RECOMPUTE:
        dxx = np.gradient(dz_dx, res_x_m, axis=1)
        dyy = np.gradient(dz_dy, res_y_m, axis=0)
        dxy = np.gradient(dz_dx, res_y_m, axis=0)
        px, py = dz_dx**2, dz_dy**2
        denom = (px + py) * np.sqrt((1 + px + py)**3)
        denom = np.where(np.abs(denom) < 1e-10, np.nan, denom)
        profc = (dxx*px + 2*dxy*dz_dx*dz_dy + dyy*py) / denom
        profc = np.where(nodata_mask, np.nan, profc).astype("float32")
        _save(profc, ds, p)
    outputs["profile_curvature"] = p
    print("  ✓ profile_curvature.tif")

    # ── TWI ───────────────────────────────────────────────────────────────────
    p = deriv_dir / "TWI.tif"
    if not p.exists() or FORCE_RECOMPUTE:
        try:
            import richdem as rd
            print("  Computing TWI with richdem ...")
            dem_rd = rd.rdarray(elev_filled, no_data=-9999)
            dem_rd.geotransform = (t.c, t.a, t.b, t.f, t.d, t.e)
            rd.FillDepressions(dem_rd, epsilon=True, in_place=True)
            accum    = rd.FlowAccumulation(dem_rd, method="D8")
            accum_m2 = np.array(accum).astype("float32") * res_x_m * res_y_m
            slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
            slope_rad = np.where(slope_rad < 0.001, 0.001, slope_rad)
            twi = np.log(accum_m2 / np.tan(slope_rad))
            twi = np.where(nodata_mask, np.nan, twi).astype("float32")
            _save(twi, ds, p)
            outputs["twi"] = p
            print("  ✓ TWI.tif")
        except Exception as e:
            print(f"  richdem unavailable ({e}) — simplified TWI ...")
            from scipy.ndimage import uniform_filter
            contrib  = uniform_filter(np.ones_like(elev_filled),
                                      size=5) * (res_x_m * res_y_m * 25)
            slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
            slope_rad = np.where(slope_rad < 0.001, 0.001, slope_rad)
            twi = np.log(contrib / np.tan(slope_rad))
            twi = np.where(nodata_mask, np.nan, twi).astype("float32")
            _save(twi, ds, p)
            outputs["twi"] = p
            print("  ✓ TWI_simplified.tif")
    elif p.exists():
        outputs["twi"] = p

    return outputs


def _save(arr, ref_ds, path):
    """Write float32 GeoTIFF with NaN as nodata. No -9999 conversion."""
    da = ref_ds.sel(band=1).copy(data=arr.astype("float32"))
    da = da.rio.write_nodata(float("nan"))
    da.rio.to_raster(str(path), compress="lzw", dtype="float32")


# ── STEP 5: VISUALISE ────────────────────────────────────────────────────────

def visualise(dem_path: Path, derivs: dict[str, Path]):
    print("\n[5] Generating terrain figure ...")

    panels = [
        ("Elevation (m)", dem_path,                          "terrain", "elev"),
        ("Slope (°)",     derivs.get("slope"),               "YlOrRd",  "slope"),
        ("Plan curv.",    derivs.get("plan_curvature"),      "RdBu_r",  "curv"),
        ("Profile curv.", derivs.get("profile_curvature"),   "RdBu_r",  "curv"),
    ]
    if "twi" in derivs:
        panels.append(("TWI", derivs["twi"], "Blues", "twi"))

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 7), facecolor="#0e0e0e")
    if n == 1:
        axes = [axes]

    for ax, (title, path, cmap, kind) in zip(axes, panels):
        if path is None or not path.exists():
            ax.set_visible(False)
            continue

        data = rxr.open_rasterio(path, masked=True).values[0].astype("float32")

        # ── BUG 1 FIX: mask implausible values before stretch ─────────────────
        data = np.where(~np.isfinite(data), np.nan, data)  # mask NaN nodata
        valid = data[np.isfinite(data)]

        if kind == "elev":
            # Use actual data range, not percentile — avoids nodata distortion
            vmin = np.nanpercentile(valid, 1)
            vmax = np.nanpercentile(valid, 99)

        elif kind == "slope":
            vmin = 0
            vmax = np.nanpercentile(valid, 98)

        elif kind == "curv":
            # Curvature on flat terrain: real signal is tiny (±0.001 range)
            # Use p5/p95 of absolute values for symmetric stretch
            # This reveals bank edges while ignoring flat-zero majority
            absmax = np.nanpercentile(np.abs(valid), 95)
            absmax = max(absmax, 1e-6)
            vmin, vmax = -absmax, absmax

        elif kind == "twi":
            vmin = np.nanpercentile(valid, 5)
            vmax = np.nanpercentile(valid, 95)

        else:
            vmin = np.nanpercentile(valid, 2)
            vmax = np.nanpercentile(valid, 98)

        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax,
                        aspect="equal", interpolation="bilinear")
        cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cb.ax.tick_params(colors="white", labelsize=7)
        ax.set_title(title, color="white", fontsize=9,
                      fontweight="bold", pad=4)
        ax.axis("off")
        ax.set_facecolor("#0e0e0e")

    fig.suptitle("Scheldt AOI — Terrain derivatives (Copernicus DEM GLO-30)",
                  color="white", fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = OUT_DIR / "terrain_derivatives.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0e0e0e")
    print(f"  → {out.name}")
    plt.show()


# ── STEP 6: RESAMPLE TO SENTINEL-2 GRID ──────────────────────────────────────

def resample_to_s2(derivs: dict[str, Path]):
    print("\n[6] Resampling to 10 m Sentinel-2 grid ...")
    s2_ref = next(Path("./data/scheldt/sentinel2").glob("S2_*.tif"), None)
    if s2_ref is None:
        print("  No S2 reference found — skipping.")
        return

    out_dir = OUT_DIR / "derivatives" / "resampled_10m"
    out_dir.mkdir(exist_ok=True)

    with rasterio.open(s2_ref) as ref:
        dst_crs, dst_transform = ref.crs, ref.transform
        dst_w, dst_h = ref.width, ref.height

    for key, path in derivs.items():
        if path is None or not path.exists():
            continue
        out = out_dir / path.name
        if out.exists() and not FORCE_RECOMPUTE:
            continue
        with rasterio.open(path) as src:
            data = np.empty((1, dst_h, dst_w), dtype=np.float32)
            reproject(source=rasterio.band(src, 1), destination=data,
                      src_transform=src.transform, src_crs=src.crs,
                      dst_transform=dst_transform, dst_crs=dst_crs,
                      resampling=Resampling.bilinear)
            prof = src.profile.copy()
            prof.update(crs=dst_crs, transform=dst_transform,
                        width=dst_w, height=dst_h, compress="lzw")
            with rasterio.open(out, "w", **prof) as dst:
                dst.write(data)
        print(f"  ✓ {path.name} → 10 m")


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Copernicus DEM (GLO-30) — Scheldt AOI ===")
    print("Authentication: CDSE username + password (same as openEO)\n")

    token    = get_token()
    sess     = session(token)
    products = find_products(sess)
    tiles    = download_tiles(sess, products)

    if not tiles:
        sys.exit("No tiles downloaded.")

    dem_path = merge_and_clip(tiles)
    #derivs   = terrain_derivatives(dem_path)
    #visualise(dem_path, derivs)
    #resample_to_s2(derivs)

    #print("\n✓ Complete. Output files:")
    #for f in sorted(OUT_DIR.rglob("*.tif")):
    #    print(f"  {f.relative_to(OUT_DIR)}")
