import os
os.environ["CPL_LOG"] = "/dev/null"

"""
Scheldt Snapshot Explorer
=========================
Visualise a single Sentinel-2 seasonal composite GeoTIFF across
multiple representations to familiarise yourself with the data
before any time-series comparison.

Panels produced
---------------
1. True colour RGB        (B04 / B03 / B02)
2. False colour NIR       (B08 / B04 / B03)  — vegetation in red
3. NDVI                   vegetation health / marsh extent
4. NDWI (McFeeters)       open water + exposed mudflat
5. MNDWI                  (B03 - B11) / (B03 + B11)  — better for turbid water
6. Band 11 SWIR           bare soil / sediment texture
7. NDVI × NDWI composite  custom overlay: green = marsh, blue = water

Usage
-----
    pip install rasterio rioxarray matplotlib numpy
    python explore_snapshot.py path/to/S2_2017_spring.tif

Band order expected (matches acquire_scheldt.py output):
    band 1 = B02 (Blue)
    band 2 = B03 (Green)
    band 3 = B04 (Red)
    band 4 = B08 (NIR)
    band 5 = B11 (SWIR)
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import rioxarray as rxr
from pathlib import Path


# ── helpers ──────────────────────────────────────────────────────────────────

def norm(arr, p_low=2, p_high=98):
    """Percentile stretch to [0, 1] — handles nodata gracefully."""
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return np.zeros_like(arr)
    lo, hi = np.nanpercentile(valid, [p_low, p_high])
    if hi == lo:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0, 1)


def index(a, b):
    """Normalised difference index (a - b) / (a + b), safe division."""
    denom = a + b
    denom = np.where(np.abs(denom) < 1e-6, np.nan, denom)
    return (a - b) / denom


def show(ax, data, title, cmap="viridis", vmin=None, vmax=None,
         cbar=True, cbar_label=""):
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest", aspect="equal")
    ax.set_title(title, fontsize=10, fontweight="bold", pad=4)
    ax.axis("off")
    if cbar:
        cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cb.ax.tick_params(labelsize=7)
        if cbar_label:
            cb.set_label(cbar_label, fontsize=7)
    return im


# ── main ─────────────────────────────────────────────────────────────────────

def explore(tif_path: str):
    path = Path(tif_path)
    if not path.exists():
        sys.exit(f"File not found: {tif_path}")

    print(f"Loading {path.name} ...")
    ds = rxr.open_rasterio(path, masked=True).squeeze()

    # ── extract bands ────────────────────────────────────────────────────────
    # rioxarray bands are 1-indexed; .values gives numpy array
    b02 = ds.sel(band=1).values.astype("float32")   # Blue
    b03 = ds.sel(band=2).values.astype("float32")   # Green
    b04 = ds.sel(band=3).values.astype("float32")   # Red
    b08 = ds.sel(band=4).values.astype("float32")   # NIR
    b11 = ds.sel(band=5).values.astype("float32")   # SWIR

    print(f"  Shape : {b02.shape}  (rows × cols)")
    print(f"  CRS   : {ds.rio.crs}")
    print(f"  Bounds: {ds.rio.bounds()}")

    # ── compute indices ──────────────────────────────────────────────────────
    ndvi  = index(b08, b04)                  # vegetation
    ndwi  = index(b03, b08)                  # open water (McFeeters)
    mndwi = index(b03, b11)                  # modified NDWI — turbid water

    # ── build RGB composites ─────────────────────────────────────────────────
    rgb = np.dstack([norm(b04), norm(b03), norm(b02)])
    nir_fc = np.dstack([norm(b08), norm(b04), norm(b03)])

    # Custom 7th panel: NDVI×NDWI composite
    # green channel = clamped NDVI  (marsh / vegetation)
    # blue channel  = clamped NDWI  (water / mudflat)
    # red channel   = SWIR norm     (bare sediment)
    composite = np.dstack([
        norm(b11, 5, 95),
        np.clip(ndvi, 0, 1),
        np.clip(ndwi, 0, 1),
    ])

    # ── figure layout ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 13), facecolor="#0e0e0e")
    fig.suptitle(
        f"Scheldt snapshot explorer  —  {path.stem}",
        color="white", fontsize=14, fontweight="bold", y=0.98
    )

    gs = GridSpec(2, 4, figure=fig,
                  hspace=0.08, wspace=0.06,
                  left=0.02, right=0.98, top=0.94, bottom=0.04)

    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(4)]
    for ax in axes:
        ax.set_facecolor("#0e0e0e")

    # ── panel 1: True colour ─────────────────────────────────────────────────
    axes[0].imshow(rgb, aspect="equal", interpolation="nearest")
    axes[0].set_title("True colour  (R·G·B)", fontsize=10,
                       fontweight="bold", color="white", pad=4)
    axes[0].axis("off")

    # ── panel 2: False colour NIR ────────────────────────────────────────────
    axes[1].imshow(nir_fc, aspect="equal", interpolation="nearest")
    axes[1].set_title("False colour NIR  (NIR·R·G)\nveg = magenta/red",
                       fontsize=10, fontweight="bold", color="white", pad=4)
    axes[1].axis("off")

    # ── panel 3: NDVI ────────────────────────────────────────────────────────
    im3 = axes[2].imshow(ndvi, cmap="RdYlGn", vmin=-0.4, vmax=0.8,
                          aspect="equal", interpolation="nearest")
    axes[2].set_title("NDVI\n−0.4 (bare/water) → 0.8 (dense marsh)",
                       fontsize=10, fontweight="bold", color="white", pad=4)
    axes[2].axis("off")
    cb3 = plt.colorbar(im3, ax=axes[2], fraction=0.035, pad=0.02)
    cb3.ax.tick_params(labelsize=7, colors="white")
    cb3.ax.yaxis.label.set_color("white")

    # ── panel 4: NDWI ────────────────────────────────────────────────────────
    im4 = axes[3].imshow(ndwi, cmap="Blues", vmin=-0.3, vmax=0.5,
                          aspect="equal", interpolation="nearest")
    axes[3].set_title("NDWI (McFeeters)\n>0 = water / exposed mudflat",
                       fontsize=10, fontweight="bold", color="white", pad=4)
    axes[3].axis("off")
    cb4 = plt.colorbar(im4, ax=axes[3], fraction=0.035, pad=0.02)
    cb4.ax.tick_params(labelsize=7, colors="white")

    # ── panel 5: MNDWI ───────────────────────────────────────────────────────
    im5 = axes[4].imshow(mndwi, cmap="PuBu", vmin=-0.2, vmax=0.6,
                          aspect="equal", interpolation="nearest")
    axes[4].set_title("MNDWI  (Green − SWIR)\nbetter for turbid Scheldt water",
                       fontsize=10, fontweight="bold", color="white", pad=4)
    axes[4].axis("off")
    cb5 = plt.colorbar(im5, ax=axes[4], fraction=0.035, pad=0.02)
    cb5.ax.tick_params(labelsize=7, colors="white")

    # ── panel 6: SWIR B11 ────────────────────────────────────────────────────
    im6 = axes[5].imshow(norm(b11), cmap="copper", vmin=0, vmax=1,
                          aspect="equal", interpolation="nearest")
    axes[5].set_title("Band 11 SWIR\nbare sediment / intertidal texture",
                       fontsize=10, fontweight="bold", color="white", pad=4)
    axes[5].axis("off")
    cb6 = plt.colorbar(im6, ax=axes[5], fraction=0.035, pad=0.02)
    cb6.ax.tick_params(labelsize=7, colors="white")

    # ── panel 7: custom composite ────────────────────────────────────────────
    axes[6].imshow(composite, aspect="equal", interpolation="nearest")
    axes[6].set_title("Custom composite\nR=SWIR  G=NDVI  B=NDWI",
                       fontsize=10, fontweight="bold", color="white", pad=4)
    axes[6].axis("off")

    # ── panel 8: NDVI histogram ──────────────────────────────────────────────
    ax8 = axes[7]
    ax8.set_facecolor("#1a1a1a")
    valid_ndvi = ndvi[np.isfinite(ndvi)].ravel()
    ax8.hist(valid_ndvi, bins=200, color="#4caf50", alpha=0.85,
             range=(-0.5, 1.0))
    ax8.axvline(0.0,  color="#ff5252", lw=1.2, ls="--", label="0 (water/bare)")
    ax8.axvline(0.2,  color="#ffeb3b", lw=1.2, ls="--", label="0.2 (sparse veg)")
    ax8.axvline(0.5,  color="#8bc34a", lw=1.2, ls="--", label="0.5 (dense marsh)")
    ax8.set_title("NDVI pixel distribution",
                   fontsize=10, fontweight="bold", color="white", pad=4)
    ax8.set_xlabel("NDVI value", color="white", fontsize=8)
    ax8.set_ylabel("Pixel count", color="white", fontsize=8)
    ax8.tick_params(colors="white", labelsize=7)
    ax8.spines[:].set_color("#444")
    ax8.legend(fontsize=7, facecolor="#1a1a1a", labelcolor="white",
               framealpha=0.8)

    # ── print quick stats ────────────────────────────────────────────────────
    total_px = np.sum(np.isfinite(ndvi))
    water_pct = 100 * np.sum(ndwi  > 0.0) / total_px
    veg_pct   = 100 * np.sum(ndvi  > 0.2) / total_px
    marsh_pct = 100 * np.sum(ndvi  > 0.5) / total_px
    bare_pct  = 100 * np.sum((ndvi < 0.1) & (ndwi < 0.0)) / total_px

    print("\n── Quick stats ─────────────────────────────────")
    print(f"  Open water / mudflat  (NDWI > 0)   : {water_pct:5.1f}%")
    print(f"  Sparse vegetation     (NDVI > 0.2)  : {veg_pct:5.1f}%")
    print(f"  Dense marsh           (NDVI > 0.5)  : {marsh_pct:5.1f}%")
    print(f"  Bare / dry sediment                 : {bare_pct:5.1f}%")
    print("─────────────────────────────────────────────────\n")

    # ── save & show ──────────────────────────────────────────────────────────
    out_png = path.parent / f"{path.stem}_explorer.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight",
                facecolor="#0e0e0e")
    print(f"Saved → {out_png}")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python explore_snapshot.py path/to/S2_YYYY_season.tif")
    explore(sys.argv[1])
