"""
AHN4 Post-processing — merge, clean, clip, reproject
=====================================================
Run this AFTER acquire_dem_ahn4.py has downloaded the raw tiles.
Bypasses rasterio CRS/PROJ issues by using rioxarray for reprojection.

Usage
-----
    python process_ahn4.py
"""

import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import rioxarray as rxr
import rasterio
from rasterio.merge import merge as rio_merge
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Fix PROJ conflict — must be set before any rasterio/pyproj import ─────────
# Find rasterio's bundled proj data and use that
try:
    import rasterio as _ra
    _ra_dir = os.path.dirname(_ra.__file__)
    for _candidate in [
        os.path.join(_ra_dir, "proj_data"),
        os.path.join(_ra_dir, "..", "share", "proj"),
        os.path.join(_ra_dir, "..", "proj"),
    ]:
        if os.path.exists(os.path.join(_candidate, "proj.db")):
            os.environ["PROJ_DATA"] = _candidate
            os.environ["PROJ_LIB"]  = _candidate
            print(f"  PROJ_DATA set to: {_candidate}")
            break
except Exception:
    pass


AOI_RD    = None   # will be read from tile filenames
TILE_DIR  = Path("./data/scheldt/dem_ahn4/raw_tiles")
OUT_DIR   = Path("./data/scheldt/dem_ahn4")
S2_REF    = next(Path("./data/scheldt/sentinel2").glob("S2_*.tif"), None)

OUT_RD    = OUT_DIR / "AHN4_DTM_scheldt_5m_rd.tif"
OUT_WGS84 = OUT_DIR / "AHN4_DTM_scheldt_5m_wgs84.tif"
OUT_PNG   = OUT_DIR / "AHN4_overview.png"


# ── STEP 1: CLEAN + MERGE TILES ──────────────────────────────────────────────

def clean_and_merge():
    print("\n[1] Cleaning and merging raw tiles ...")

    tiles = sorted(TILE_DIR.glob("ahn4_r*.tif"))
    if not tiles:
        raise FileNotFoundError(f"No tiles in {TILE_DIR}")
    print(f"  Found {len(tiles)} tiles")

    clean_dir = TILE_DIR / "cleaned"
    clean_dir.mkdir(exist_ok=True)
    cleaned = []

    for tile in tiles:
        out = clean_dir / tile.name
        if out.exists():
            cleaned.append(out)
            continue

        with rasterio.open(tile) as src:
            data = src.read(1).astype("float32")
            prof = src.profile.copy()

        # Replace all nodata candidates with NaN
        # AHN4 nodata = 3.4028235e+38 (float32 max)
        # Also mask -9999 and any physically impossible values
        data[data >  1000] = np.nan
        data[data < -100]  = np.nan

        prof.update(dtype="float32", nodata=float("nan"), compress="lzw")
        with rasterio.open(out, "w", **prof) as dst:
            dst.write(data[np.newaxis, :, :])
        cleaned.append(out)

    print(f"  ✓ Cleaned {len(cleaned)} tiles")

    # Merge
    datasets = [rasterio.open(p) for p in cleaned]
    merged, merged_t = rio_merge(datasets, nodata=np.nan)
    profile = datasets[0].profile.copy()
    for ds in datasets:
        ds.close()

    merged_f = merged[0].astype("float32")
    valid = merged_f[np.isfinite(merged_f)]
    print(f"  Merged shape : {merged_f.shape}")
    print(f"  Value range  : {valid.min():.2f} – {valid.max():.2f} m NAP")

    if valid.max() - valid.min() < 0.01:
        print("  ⚠  WARNING: nearly all values identical — nodata mask may be wrong")
        print("     Checking raw tile directly ...")
        with rasterio.open(tiles[0]) as src:
            raw = src.read(1)
            print(f"     Raw tile: min={raw.min():.4f}  max={raw.max():.4f}  "
                  f"nodata={src.nodata}")

    profile.update(
        height=merged_f.shape[0],
        width=merged_f.shape[1],
        transform=merged_t,
        dtype="float32",
        nodata=float("nan"),
        compress="lzw",
        count=1,
    )

    tmp = OUT_DIR / "_merged.tif"
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(merged_f[np.newaxis, :, :])
    print(f"  ✓ Merged → {tmp.name}")
    return tmp


# ── STEP 2: CLIP TO AOI ───────────────────────────────────────────────────────

def clip(merged_path: Path) -> Path:
    print("\n[2] Clipping to AOI ...")

    ds = rxr.open_rasterio(merged_path, masked=True)
    print(f"  Input CRS  : {ds.rio.crs}")
    print(f"  Input shape: {ds.shape}")

    # Clip in native RD New
    # AOI in RD New — use pyproj to convert from WGS84
    try:
        from pyproj import Transformer
        tr = Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=True)
        corners = [(3.80, 51.28), (4.25, 51.28),
                   (4.25, 51.45), (3.80, 51.45)]
        xs = [tr.transform(lon, lat)[0] for lon, lat in corners]
        ys = [tr.transform(lon, lat)[1] for lon, lat in corners]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
    except Exception:
        xmin, ymin, xmax, ymax = 30_000, 364_000, 68_000, 390_000

    clipped = ds.rio.clip_box(minx=xmin, miny=ymin,
                               maxx=xmax, maxy=ymax)

    elev = clipped.values[0].astype("float32")
    valid = elev[np.isfinite(elev)]
    print(f"  Clipped shape: {elev.shape}")
    print(f"  Range : {valid.min():.2f} – {valid.max():.2f} m NAP")
    print(f"  Below 0m NAP : {100*np.mean(valid < 0):.1f}%")
    print(f"  0 – 2m NAP   : {100*np.mean((valid>=0) & (valid<2)):.1f}%")
    print(f"  > 2m NAP     : {100*np.mean(valid >= 2):.1f}%")

    da = clipped.sel(band=1).copy(data=elev)
    # Clear any existing _FillValue from attrs before writing nodata
    da.attrs.pop("_FillValue", None)
    da.encoding.pop("_FillValue", None)
    da = da.rio.write_nodata(float("nan"))
    da.expand_dims("band").rio.to_raster(
        str(OUT_RD), compress="lzw", dtype="float32"
    )
    merged_path.unlink(missing_ok=True)
    print(f"  ✓ Saved → {OUT_RD.name}")
    return OUT_RD


# ── STEP 3: REPROJECT TO WGS84 via rioxarray ─────────────────────────────────

def reproject_wgs84(rd_path: Path) -> Path:
    """
    Use rioxarray's reproject — avoids rasterio's CRS.from_epsg() call
    which crashes when PROJ_DATA is misconfigured on Yoda.
    """
    print("\n[3] Reprojecting to WGS84 via rioxarray ...")

    ds = rxr.open_rasterio(rd_path, masked=True)

    try:
        reprojected = ds.rio.reproject("EPSG:4326")
        reprojected = reprojected.rio.write_nodata(float("nan"))
        reprojected.rio.to_raster(
            str(OUT_WGS84), compress="lzw", dtype="float32"
        )
        print(f"  ✓ WGS84 → {OUT_WGS84.name}")
        print(f"    Shape : {reprojected.shape}")
        return OUT_WGS84

    except Exception as e:
        print(f"  ✗ Reprojection failed: {e}")
        print(f"  Using RD New file for derivatives instead.")
        print(f"  Update recompute_derivatives.py to use:")
        print(f"    DEM_PATH = Path('{rd_path}')")
        print(f"  And set CRS manually:")
        print(f"    ds.rio.set_crs('EPSG:28992')")
        return rd_path


# ── STEP 4: VISUALISE ────────────────────────────────────────────────────────

def visualise(paths: list[Path], labels: list[str]):
    print("\n[4] Generating figure ...")

    valid_panels = [(p, l) for p, l in zip(paths, labels) if p.exists()]
    if not valid_panels:
        print("  No files to visualise.")
        return

    fig, axes = plt.subplots(1, len(valid_panels),
                              figsize=(8*len(valid_panels), 7),
                              facecolor="#0e0e0e")
    if len(valid_panels) == 1:
        axes = [axes]

    for ax, (path, title) in zip(axes, valid_panels):
        ds   = rxr.open_rasterio(path, masked=True)
        data = ds.values[0].astype("float32")
        data = np.where((data > 1e10) | (data < -100), np.nan, data)
        valid = data[np.isfinite(data)]

        if len(valid) == 0:
            ax.set_visible(False)
            continue

        ext = np.percentile(np.abs(valid), 98)
        ext = max(ext, 0.5)
        im = ax.imshow(data, cmap="RdYlGn_r",
                        vmin=-ext, vmax=ext,
                        aspect="equal", interpolation="bilinear")
        cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cb.set_label("m NAP", color="white", fontsize=8)
        cb.ax.tick_params(colors="white", labelsize=7)
        ax.text(0.02, 0.05,
                f"< 0m : {100*np.mean(valid < 0):.1f}%\n"
                f"0–2m : {100*np.mean((valid>=0)&(valid<2)):.1f}%\n"
                f">2m  : {100*np.mean(valid >= 2):.1f}%",
                transform=ax.transAxes, color="white", fontsize=9,
                bbox=dict(facecolor="#222", alpha=0.85, boxstyle="round"))
        ax.set_title(title, color="white", fontsize=9,
                      fontweight="bold", pad=4)
        ax.axis("off")
        ax.set_facecolor("#0e0e0e")

    fig.suptitle("AHN4 Lidar DTM  |  Saeftinghe + Western Scheldt",
                  color="white", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight", facecolor="#0e0e0e")
    print(f"  → {OUT_PNG.name}")
    plt.show()


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== AHN4 Post-processing ===\n")

    merged  = clean_and_merge()
    rd_path = clip(merged)
    wgs84   = reproject_wgs84(rd_path)

    visualise(
        [rd_path, wgs84],
        ["AHN4 DTM 5m — RD New (m NAP)",
         "AHN4 DTM 5m — WGS84"]
    )

    print(f"\n✓ Done. Files:")
    for f in sorted(OUT_DIR.glob("*.tif")):
        print(f"  {f.name}")

    print(f"\nNext — run recompute_derivatives.py with:")
    wgs84_exists = OUT_WGS84.exists()
    dem_for_derivs = OUT_WGS84 if wgs84_exists else OUT_RD
    print(f"  DEM_PATH  = Path('{dem_for_derivs}')")
    print(f"  DERIV_DIR = Path('./data/scheldt/dem_ahn4/derivatives')")
    print(f"  OUT_PNG   = Path('./data/scheldt/dem_ahn4/terrain_derivatives.png')")
