"""
Scheldt — Saltmarsh Cliff Retreat Rate Analysis
================================================
Measures lateral cliff retreat rates at known monitoring sites
using waterline position time series from Sentinel-2.

Method
------
For each validation site:
  1. Extract a cross-shore transect perpendicular to the marsh edge
  2. Sample MNDWI along the transect for each snapshot
  3. Find MNDWI=0 crossing position per snapshot (= waterline)
  4. Fit linear trend to waterline position vs time
  5. Retreat rate in m/year (negative = retreat, positive = advance)

This directly replicates the field transect monitoring in the
IMDC report — but from satellite rather than GPS surveys.

Sensitivity
-----------
Detection limit: ~1–2 m/year at 10m pixel resolution over 8 years
(8 years × 1 m/yr = 8m = ~1 pixel shift — marginal but detectable
 with subpixel waterline interpolation)

Outputs
-------
    risk/transect_retreat_rates.csv    retreat rate per site
    risk/transect_analysis.png         per-site waterline time series
    risk/transect_profiles/            individual site figures

Usage
-----
    python transect_analysis.py
    python transect_analysis.py --sites reports/extracted/imdc_rapport_2016_2017_validation.csv
"""

import os
os.environ["CPL_LOG"] = "/dev/null"

import warnings
import csv
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import rioxarray as rxr
import rasterio
from scipy import stats as sp_stats
from scipy.ndimage import gaussian_filter1d
from pathlib import Path

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────

S2_DIR    = Path("./data/scheldt/sentinel2")
RISK_DIR  = Path("./data/scheldt/risk")
VAL_CSV   = Path("./reports/extracted/"
                  "imdc_rapport_2016_2017_validation.csv")
OUT_DIR   = RISK_DIR
PROF_DIR  = RISK_DIR / "transect_profiles"
PROF_DIR.mkdir(exist_ok=True)

# Transect parameters
TRANSECT_LENGTH_PX = 30    # pixels each side of waterline (~300m)
TRANSECT_WIDTH_PX  = 5     # average over N pixels perpendicular
MNDWI_THRESHOLD    = 0.0   # water/land boundary

# Minimum valid snapshots to compute a trend
MIN_SNAPSHOTS = 6


# ── LOAD SENTINEL-2 STACK ────────────────────────────────────────────────────

def build_mndwi_stack():
    """Load all S2 snapshots and compute MNDWI stack + timestamps."""
    tifs = sorted(S2_DIR.glob("S2_*.tif"))
    print(f"  {len(tifs)} S2 snapshots")

    ref   = rxr.open_rasterio(tifs[0], masked=True)
    H, W  = ref.values[0].shape
    n     = len(tifs)
    stack = np.full((n, H, W), np.nan, dtype="float32")
    years = []

    for i, tif in enumerate(tifs):
        ds  = rxr.open_rasterio(tif, masked=True)
        b03 = ds.sel(band=2).values.astype("float32") / 10000.0
        b11 = ds.sel(band=5).values.astype("float32") / 10000.0
        d   = b03 + b11
        m   = np.where(np.abs(d)>1e-4, (b03-b11)/d, np.nan)
        stack[i] = np.where((m>=-1)&(m<=1), m, np.nan)

        # Parse year from filename S2_YYYY_season.tif
        parts = tif.stem.split("_")
        year  = int(parts[1]) if len(parts) > 1 else 2020
        season_offset = {"spring": 0.4, "summer": 0.7, "winter": 0.1}
        offset = season_offset.get(parts[2] if len(parts)>2 else "", 0.5)
        years.append(year + offset)

    return stack, np.array(years), ref


# ── WATERLINE POSITION ALONG TRANSECT ────────────────────────────────────────

def find_waterline_position(mndwi_profile, threshold=0.0):
    """
    Find the subpixel position of the MNDWI=threshold crossing
    along a 1D profile using linear interpolation.
    Returns fractional pixel position, or NaN if not found.
    """
    # Smooth profile slightly
    profile = gaussian_filter1d(
        np.where(np.isfinite(mndwi_profile), mndwi_profile, 0),
        sigma=1.0
    )

    # Find zero crossing
    crossings = []
    for j in range(len(profile) - 1):
        if not (np.isfinite(profile[j]) and np.isfinite(profile[j+1])):
            continue
        v0, v1 = profile[j] - threshold, profile[j+1] - threshold
        if v0 * v1 < 0:   # sign change
            # Linear interpolation
            frac = -v0 / (v1 - v0)
            crossings.append(j + frac)

    if not crossings:
        return np.nan

    # Return median crossing (most stable)
    return float(np.median(crossings))


# ── LOAD VALIDATION SITES ─────────────────────────────────────────────────────

def load_sites(csv_path):
    """Load validation/problem sites from CSV."""
    sites = []
    if not csv_path.exists():
        # Fall back to JSON
        json_path = Path("./reports/extracted/"
                         "imdc_rapport_2016_2017_extracted.json")
        if json_path.exists():
            with open(json_path) as f:
                d = json.load(f)
            for s in d.get("problem_sites", []) + \
                     d.get("validation_sites", []):
                if s.get("lon_approx") and s.get("lat_approx"):
                    sites.append(s)
            return sites
        print(f"  ✗ No validation file found")
        return []

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                lon = float(row.get("lon_approx", "") or 0)
                lat = float(row.get("lat_approx", "") or 0)
                if lon and lat:
                    row["lon_approx"] = lon
                    row["lat_approx"] = lat
                    sites.append(row)
            except ValueError:
                pass
    return sites


# ── PIXEL COORDINATE CONVERSION ──────────────────────────────────────────────

def wgs84_to_pixel(lon, lat, transform, crs):
    try:
        from pyproj import Transformer
        tr = Transformer.from_crs("EPSG:4326", crs.to_string(),
                                   always_xy=True)
        x, y = tr.transform(lon, lat)
    except Exception:
        x, y = lon, lat
    col = int((x - transform.c) / transform.a)
    row = int((y - transform.f) / transform.e)
    return row, col


# ── TRANSECT EXTRACTION ───────────────────────────────────────────────────────

def extract_transect(mndwi_stack: np.ndarray,
                     center_row: int, center_col: int,
                     direction: str = "cross_shore",
                     H: int = None, W: int = None) -> np.ndarray:
    """
    Extract a 1D transect through (center_row, center_col).
    Direction: 'EW' (east-west) or 'NS' (north-south).
    Returns array of shape (n_snapshots, transect_length).
    """
    n = mndwi_stack.shape[0]
    L = TRANSECT_LENGTH_PX
    profiles = np.full((n, 2*L+1), np.nan, dtype="float32")

    for i in range(n):
        snap = mndwi_stack[i]
        if direction == "EW":
            c0 = max(0, center_col - L)
            c1 = min(W or snap.shape[1], center_col + L + 1)
            strip = snap[max(0, center_row-TRANSECT_WIDTH_PX):
                         min(snap.shape[0], center_row+TRANSECT_WIDTH_PX+1),
                         c0:c1]
        else:  # NS
            r0 = max(0, center_row - L)
            r1 = min(H or snap.shape[0], center_row + L + 1)
            strip = snap[r0:r1,
                         max(0, center_col-TRANSECT_WIDTH_PX):
                         min(snap.shape[1], center_col+TRANSECT_WIDTH_PX+1)]
            strip = strip.T

        if strip.size > 0:
            profile = np.nanmean(strip, axis=0)
            length  = min(len(profile), 2*L+1)
            profiles[i, :length] = profile[:length]

    return profiles


# ── MAIN ANALYSIS ─────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--sites", type=Path, default=VAL_CSV)
args = parser.parse_args()

print("=== Saltmarsh Cliff Retreat Rate Analysis ===\n")

print("[1] Loading Sentinel-2 MNDWI stack ...")
mndwi_stack, years, ref_ds = build_mndwi_stack()
n_snaps = len(years)
H_s2, W_s2 = mndwi_stack.shape[1], mndwi_stack.shape[2]
s2_t   = ref_ds.rio.transform()
s2_crs = ref_ds.rio.crs
print(f"  Stack: {n_snaps} snapshots  {H_s2}×{W_s2}  "
      f"year range {years.min():.1f}–{years.max():.1f}")

print("\n[2] Loading validation sites ...")
sites = load_sites(args.sites)
print(f"  {len(sites)} sites loaded")

# ── Process each site ─────────────────────────────────────────────────────────

results = []
print("\n[3] Measuring retreat rates ...")
print(f"  {'Site':35s} {'Rate(m/yr)':>10} {'R²':>6} {'p':>6} "
      f"{'n_valid':>8} {'status'}")
print(f"  {'─'*35} {'─'*10} {'─'*6} {'─'*6} {'─'*8} {'─'*10}")

for site in sites:
    name = site.get("name", "?")[:34]
    lon  = float(site.get("lon_approx", 0))
    lat  = float(site.get("lat_approx", 0))

    row, col = wgs84_to_pixel(lon, lat, s2_t, s2_crs)

    if not (TRANSECT_LENGTH_PX <= row < H_s2 - TRANSECT_LENGTH_PX and
            TRANSECT_LENGTH_PX <= col < W_s2 - TRANSECT_LENGTH_PX):
        print(f"  {'':35s} {'':>10} {'':>6} {'':>6} {'':>8} outside AOI")
        results.append({
            "name": site.get("name",""),
            "lon": lon, "lat": lat,
            "retreat_rate_m_per_yr": None,
            "r2": None, "p_value": None,
            "n_valid": 0,
            "status": "outside_AOI",
            "issue_type": site.get("issue_type",""),
            "severity": site.get("severity",""),
        })
        continue

    # Try both EW and NS transects, use the one with more valid crossings
    best_positions = None
    best_n_valid   = 0

    for direction in ["EW", "NS"]:
        profiles = extract_transect(mndwi_stack, row, col,
                                    direction, H_s2, W_s2)
        positions = np.array([
            find_waterline_position(profiles[i]) for i in range(n_snaps)
        ])
        n_valid = np.sum(np.isfinite(positions))
        if n_valid > best_n_valid:
            best_n_valid   = n_valid
            best_positions = positions
            best_direction = direction

    positions = best_positions
    n_valid   = best_n_valid

    if n_valid < MIN_SNAPSHOTS:
        print(f"  {name:35s} {'N/A':>10} {'—':>6} {'—':>6} "
              f"{n_valid:>8} insufficient_data")
        results.append({
            "name": site.get("name",""),
            "lon": lon, "lat": lat,
            "retreat_rate_m_per_yr": None,
            "r2": None, "p_value": None,
            "n_valid": int(n_valid),
            "status": "insufficient_data",
            "issue_type": site.get("issue_type",""),
            "severity": site.get("severity",""),
        })
        continue

    # Fit linear trend to waterline position
    valid_mask = np.isfinite(positions)
    x = years[valid_mask]
    y = positions[valid_mask] * 10.0   # pixels → metres (10m pixel size)

    res = sp_stats.linregress(x, y)
    rate  = res.slope          # m/year (negative = retreat)
    r2    = res.rvalue**2
    pval  = res.pvalue
    sig   = "✓" if pval < 0.10 else "~"

    direction_str = "retreat" if rate < 0 else "advance"
    status = f"{sig} {direction_str}"

    print(f"  {name:35s} {rate:>+10.2f} {r2:>6.3f} {pval:>6.3f} "
          f"{n_valid:>8} {status}")

    results.append({
        "name":                  site.get("name",""),
        "lon":                   lon,
        "lat":                   lat,
        "retreat_rate_m_per_yr": round(rate, 3),
        "r2":                    round(r2, 4),
        "p_value":               round(pval, 4),
        "n_valid":               int(n_valid),
        "transect_direction":    best_direction,
        "significant":           pval < 0.10,
        "status":                status,
        "issue_type":            site.get("issue_type",""),
        "severity":              site.get("severity",""),
        "positions_px":          positions.tolist(),
        "years":                 years.tolist(),
    })

# ── SAVE CSV ──────────────────────────────────────────────────────────────────

csv_out = OUT_DIR / "transect_retreat_rates.csv"
fields  = ["name","lon","lat","retreat_rate_m_per_yr","r2","p_value",
           "n_valid","significant","status","issue_type","severity"]
with open(csv_out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    w.writeheader(); w.writerows(results)
print(f"\n✓ {csv_out.name}")


# ── VISUALISE ────────────────────────────────────────────────────────────────

in_bounds = [r for r in results
             if r.get("retreat_rate_m_per_yr") is not None]

if in_bounds:
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), facecolor="#0e0e0e")

    # Panel 1: retreat rates bar chart
    ax = axes[0]
    ax.set_facecolor("#1a1a1a")
    names_plot = [r["name"].split("-")[0].strip()[:25] for r in in_bounds]
    rates_plot = [r["retreat_rate_m_per_yr"] for r in in_bounds]
    sigs_plot  = [r.get("significant", False) for r in in_bounds]
    colors_plot= ["#EF5350" if r < 0 else "#4CAF50" for r in rates_plot]
    alphas_plot= [0.9 if s else 0.4 for s in sigs_plot]

    y = np.arange(len(names_plot))
    for i, (rate, color, alpha) in enumerate(
            zip(rates_plot, colors_plot, alphas_plot)):
        ax.barh(y[i], rate, color=color, alpha=alpha, edgecolor="#333")

    ax.axvline(0, color="white", lw=1.0, ls="--")
    ax.axvline(-5, color="#FF9800", lw=1.0, ls=":",
               label="5m/yr detection limit")
    ax.set_yticks(y)
    ax.set_yticklabels(names_plot, color="white", fontsize=8)
    ax.set_xlabel("Retreat rate (m/year)\nnegative = retreat",
                   color="white", fontsize=9)
    ax.tick_params(colors="white", labelsize=8)
    ax.spines[:].set_color("#444")
    ax.legend(fontsize=8, facecolor="#1a1a1a",
              labelcolor="white", framealpha=0.9)
    ax.set_title("Waterline retreat rates from S2 transects\n"
                  "faded bars = p > 0.10 (not significant)",
                  color="white", fontsize=9, fontweight="bold")

    # Panel 2: summary text
    ax2 = axes[1]
    ax2.set_facecolor("#1a1a1a")
    ax2.axis("off")

    retreating = [r for r in in_bounds if r.get("retreat_rate_m_per_yr", 0) < 0]
    advancing  = [r for r in in_bounds if r.get("retreat_rate_m_per_yr", 0) >= 0]
    sig_retreat = [r for r in retreating if r.get("significant")]

    txt = (
        f"TRANSECT ANALYSIS SUMMARY\n"
        f"{'─'*32}\n"
        f"Sites analysed   : {len(in_bounds)}\n"
        f"Retreating       : {len(retreating)}\n"
        f"Advancing        : {len(advancing)}\n"
        f"Significant (p<0.10): {len(sig_retreat)}\n"
        f"\nDetection limit:\n"
        f"  ~1–2 m/yr over 8 years\n"
        f"  (subpixel interpolation)\n"
        f"\nMethod:\n"
        f"  MNDWI=0 waterline position\n"
        f"  per snapshot along transect\n"
        f"  Linear OLS trend vs time\n"
        f"\nSites with significant retreat:\n"
    )
    for r in sig_retreat:
        txt += (f"  {r['name'][:25]}: "
                f"{r['retreat_rate_m_per_yr']:+.1f} m/yr\n")
    if not sig_retreat:
        txt += "  None detected at p<0.10\n"
    txt += (
        f"\nNote: rates <5 m/yr are near\n"
        f"the detection threshold of\n"
        f"10m Sentinel-2 pixels.\n"
        f"Field measurements (IMDC)\n"
        f"typically show 1–3 m/yr\n"
        f"for Western Scheldt cliffs."
    )
    ax2.text(0.05, 0.95, txt, transform=ax2.transAxes,
              color="white", fontsize=8, va="top",
              fontfamily="monospace",
              bbox=dict(facecolor="#111", alpha=0.9,
                        boxstyle="round", edgecolor="#444"))
    ax2.set_title("Summary statistics",
                   color="white", fontsize=9, fontweight="bold")

    fig.suptitle(
        "Scheldt — Saltmarsh Cliff Retreat Rates  |  "
        "Sentinel-2 Waterline Transects",
        color="white", fontsize=11, fontweight="bold"
    )
    fig.tight_layout()
    out_png = OUT_DIR / "transect_analysis.png"
    fig.savefig(str(out_png), dpi=150,
                bbox_inches="tight", facecolor="#0e0e0e")
    print(f"✓ {out_png.name}")
    plt.show()
else:
    print("No in-bounds sites to visualise")

print(f"\n✓ Transect analysis complete.")
print(f"\nInterpretation note:")
print(f"  Rates < 5 m/yr are near the Sentinel-2 detection threshold.")
print(f"  Significant results indicate detectable waterline migration.")
print(f"  Non-significant results do not mean no erosion — the process")
print(f"  may simply be too slow for 10m optical detection.")
print(f"  Field data from IMDC (1–3 m/yr typical) confirm this limit.")


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — SPEED CORRELATION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
"""
Tests whether faster-retreating sites have higher fetch / MNDWI exposure.

Predictors:
    max_fetch_m      maximum open water distance in 8 directions (m)
    mndwi_std        temporal MNDWI std dev at site (hydrodynamic proxy)
    ead_m            elevation above MHW (inundation proxy)
    pioneer_present  1/0 whether site is in pioneer zone

Response:
    retreat_rate_m_per_yr  from transect analysis above

Reference: Van der Wal & Pye (2004), van der Wal et al. (2008)
"""

from scipy.stats import spearmanr, pearsonr

print("\n" + "="*55)
print("PART 2 — Speed Correlation Analysis")
print("="*55)

# ── Load predictor layers ─────────────────────────────────────────────────────

TIER1_DIR2  = Path("./data/scheldt/sentinel2/analysis")
AHN4_RS_DIR = Path("./data/scheldt/dem_ahn4/derivatives/resampled_10m")
RISK_DIR2   = Path("./data/scheldt/risk")

def load_layer_corr(path, name):
    if not path.exists():
        print(f"  MISSING: {name}")
        return None
    arr = rxr.open_rasterio(path, masked=True).values[0].astype("float32")
    print(f"  ✓ {name}")
    return arr

print("\nLoading predictor layers ...")
water_freq   = load_layer_corr(TIER1_DIR2 / "water_frequency.tif",
                                "water_frequency")
mndwi_std_l  = load_layer_corr(TIER1_DIR2 / "mndwi_std.tif",
                                "mndwi_std")
ead_layer    = load_layer_corr(AHN4_RS_DIR / "EAD.tif",       "EAD")
pioneer_zone_l = load_layer_corr(RISK_DIR2 / "pioneer_zone.tif",
                                  "pioneer_zone")


# ── Fetch computation ─────────────────────────────────────────────────────────

def compute_fetch(water_mask: np.ndarray,
                  row: int, col: int,
                  pixel_size_m: float = 10.0,
                  n_directions: int = 8) -> tuple[float, float]:
    """
    Compute fetch (m) in n compass directions from pixel (row, col).
    Fetch = distance to nearest land pixel along each direction.
    Returns (max_fetch_m, mean_fetch_m).
    """
    H_w, W_w = water_mask.shape
    angles    = np.linspace(0, 2*np.pi, n_directions, endpoint=False)
    fetches   = []

    for angle in angles:
        dr = -np.sin(angle)    # row direction (N = negative row)
        dc =  np.cos(angle)    # col direction (E = positive col)
        r, c  = float(row), float(col)
        steps = 0
        while steps < 500:     # cap at 5km
            r += dr; c += dc; steps += 1
            ri, ci = int(round(r)), int(round(c))
            if not (0 <= ri < H_w and 0 <= ci < W_w):
                break
            if not water_mask[ri, ci]:
                break
        fetches.append(steps * pixel_size_m)

    return float(max(fetches)), float(np.mean(fetches))


# ── Sample predictors at each site ───────────────────────────────────────────

print("\nSampling predictors at each site ...")

# Build water mask from water frequency
if water_freq is not None:
    water_mask_corr = water_freq > 0.30   # water > 30% of snapshots
else:
    water_mask_corr = None

BUF = 3   # pixel buffer for sampling

corr_data = []
for r in results:
    if r.get("retreat_rate_m_per_yr") is None:
        continue
    if r.get("status") in ["outside_AOI", "insufficient_data"]:
        continue

    lon = r["lon"]; lat = r["lat"]
    row, col = wgs84_to_pixel(lon, lat, s2_t, s2_crs)

    if not (BUF <= row < H_s2-BUF and BUF <= col < W_s2-BUF):
        continue

    # Buffer sample helper
    def sample(layer):
        if layer is None:
            return np.nan
        h, w = layer.shape
        if row >= h or col >= w:
            return np.nan
        patch = layer[max(0,row-BUF):min(h,row+BUF+1),
                      max(0,col-BUF):min(w,col+BUF+1)]
        v = patch[np.isfinite(patch)]
        return float(np.nanmean(v)) if len(v) > 0 else np.nan

    # Fetch
    if water_mask_corr is not None:
        max_fetch, mean_fetch = compute_fetch(
            water_mask_corr, row, col, pixel_size_m=10.0)
    else:
        max_fetch = mean_fetch = np.nan

    site_data = {
        "name":               r["name"],
        "retreat_rate":       r["retreat_rate_m_per_yr"],
        "significant":        r.get("significant", False),
        "max_fetch_m":        round(max_fetch, 1),
        "mean_fetch_m":       round(mean_fetch, 1),
        "mndwi_std":          round(sample(mndwi_std_l), 5),
        "ead_m":              round(sample(ead_layer), 3),
        "pioneer_present":    int(sample(pioneer_zone_l) > 0.5)
                              if pioneer_zone_l is not None else np.nan,
        "issue_type":         r.get("issue_type",""),
        "severity":           r.get("severity",""),
    }
    corr_data.append(site_data)

    print(f"  {r['name'][:30]:30s}  "
          f"rate={r['retreat_rate_m_per_yr']:+6.2f}  "
          f"fetch={max_fetch:6.0f}m  "
          f"mndwi_std={site_data['mndwi_std']:.4f}  "
          f"ead={site_data['ead_m']:.2f}m")


# ── Spearman correlation ──────────────────────────────────────────────────────

print(f"\n── Correlation results (Spearman ρ) ─────────────────────")
print(f"  n = {len(corr_data)} sites with complete data\n")

if len(corr_data) >= 4:
    rates   = np.array([d["retreat_rate"] for d in corr_data])
    # Absolute retreat (positive = faster regardless of direction)
    abs_rates = np.abs(rates)

    predictors = {
        "Max fetch (m)":       np.array([d["max_fetch_m"]   for d in corr_data]),
        "MNDWI std":           np.array([d["mndwi_std"]     for d in corr_data]),
        "EAD (m above MHW)":   np.array([d["ead_m"]         for d in corr_data]),
        "Pioneer present":     np.array([d["pioneer_present"]for d in corr_data]),
    }

    corr_results = []
    for pred_name, pred_vals in predictors.items():
        valid = (np.isfinite(pred_vals) & np.isfinite(rates))
        if valid.sum() < 4:
            print(f"  {pred_name:25s}: insufficient data ({valid.sum()} pts)")
            continue

        # Spearman (non-parametric — better for small n)
        rho_dir,  p_dir  = spearmanr(pred_vals[valid], rates[valid])
        rho_abs,  p_abs  = spearmanr(pred_vals[valid], abs_rates[valid])

        sig_dir = "*" if p_dir < 0.10 else " "
        sig_abs = "*" if p_abs < 0.10 else " "

        print(f"  {pred_name:25s}: "
              f"ρ(directional)={rho_dir:+.3f}{sig_dir}  "
              f"p={p_dir:.3f}   "
              f"ρ(magnitude)={rho_abs:+.3f}{sig_abs}  "
              f"p={p_abs:.3f}")

        corr_results.append({
            "predictor":        pred_name,
            "rho_directional":  round(rho_dir, 4),
            "p_directional":    round(p_dir, 4),
            "rho_magnitude":    round(rho_abs, 4),
            "p_magnitude":      round(p_abs, 4),
            "n":                int(valid.sum()),
            "significant_0.10": p_abs < 0.10 or p_dir < 0.10,
        })

    print(f"\n  * = p < 0.10")
    print(f"\n  Expected from literature:")
    print(f"    Fetch: positive correlation (exposed = faster retreat)")
    print(f"    MNDWI std: positive (dynamic = exposed)")
    print(f"    EAD: negative (lower elevation = longer inundation)")
    print(f"    Pioneer: negative (protection reduces retreat)")

else:
    print(f"  Too few sites ({len(corr_data)}) for correlation — need ≥ 4")
    corr_results = []


# ── Save correlation results ──────────────────────────────────────────────────

corr_csv = OUT_DIR / "speed_correlation.csv"
if corr_data:
    with open(corr_csv, "w", newline="", encoding="utf-8") as f:
        fields = ["name","retreat_rate","significant","max_fetch_m",
                  "mean_fetch_m","mndwi_std","ead_m","pioneer_present",
                  "issue_type","severity"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(corr_data)
    print(f"\n✓ {corr_csv.name}")

rho_csv = OUT_DIR / "correlation_summary.csv"
if corr_results:
    with open(rho_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(corr_results[0].keys()))
        w.writeheader(); w.writerows(corr_results)
    print(f"✓ {rho_csv.name}")


# ── Visualise correlation ─────────────────────────────────────────────────────

if len(corr_data) >= 4:
    print("\nGenerating correlation figure ...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), facecolor="#0e0e0e")
    axes = axes.ravel()

    pred_list = [
        ("Max fetch (m)",     "max_fetch_m",    "#42A5F5",
         "Faster retreat at exposed sites\n(Van der Wal & Pye 2004)"),
        ("MNDWI std dev",     "mndwi_std",      "#FF9800",
         "Dynamic waterline = exposed site"),
        ("EAD (m above MHW)", "ead_m",          "#EF5350",
         "Lower elevation = more inundation"),
        ("Pioneer present",   "pioneer_present","#66BB6A",
         "Pioneer veg reduces retreat rate"),
    ]

    for ax, (xlabel, key, color, note) in zip(axes, pred_list):
        ax.set_facecolor("#1a1a1a")

        xs = np.array([d[key]          for d in corr_data])
        ys = np.array([d["retreat_rate"] for d in corr_data])
        sigs = [d["significant"] for d in corr_data]
        names_c = [d["name"].split("-")[0].strip()[:12] for d in corr_data]

        valid = np.isfinite(xs) & np.isfinite(ys)
        if valid.sum() < 2:
            ax.set_visible(False)
            continue

        # Scatter — significant retreats filled, non-significant hollow
        for i, (x, y, sig, name) in enumerate(
                zip(xs, ys, sigs, names_c)):
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            marker = "o" if sig else "^"
            alpha  = 0.9 if sig else 0.5
            ax.scatter(x, y, color=color, s=80, marker=marker,
                       alpha=alpha, edgecolors="white", linewidths=0.5,
                       zorder=5)
            ax.annotate(name, (x, y),
                        textcoords="offset points", xytext=(4,4),
                        color="white", fontsize=6, alpha=0.8)

        # Trend line if enough valid points
        xv, yv = xs[valid], ys[valid]
        if len(xv) >= 3 and xv.std() > 1e-6 and yv.std() > 1e-6:
            try:
                m, b, r, p, _ = sp_stats.linregress(xv, yv)
                xfit = np.linspace(xv.min(), xv.max(), 50)
                ls   = "-" if p < 0.10 else "--"
                ax.plot(xfit, m*xfit+b, color=color,
                        lw=1.5, ls=ls, alpha=0.7)
                rho, p_s = spearmanr(xv, yv)
                sig_str = "p<0.10 *" if p_s < 0.10 else f"p={p_s:.2f}"
                ax.set_title(f"{xlabel}\nρ={rho:+.3f}  {sig_str}\n{note}",
                              color="white", fontsize=8, fontweight="bold")
            except Exception as _e:
                ax.set_title(f"{xlabel}\nno variance ({_e})\n{note}",
                              color="white", fontsize=8)
        elif len(xv) >= 3 and xv.std() <= 1e-6:
            rho, p_s = spearmanr(xv, yv) if yv.std() > 1e-6 else (0, 1)
            ax.set_title(f"{xlabel}\nall identical — no fetch variance\n{note}",
                          color="white", fontsize=8)
        else:
            ax.set_title(f"{xlabel}\ninsufficient data\n{note}",
                          color="white", fontsize=8)

        ax.axhline(0, color="#888", lw=0.8, ls=":")
        ax.set_xlabel(xlabel,           color="white", fontsize=8)
        ax.set_ylabel("Retreat rate (m/yr)\nneg = retreat",
                       color="white", fontsize=8)
        ax.tick_params(colors="white", labelsize=7)
        ax.spines[:].set_color("#444")

        # Annotation: ● = significant transect  ▲ = non-significant
        ax.text(0.98, 0.02,
                "● sig. transect (p<0.10)\n▲ non-sig. transect",
                transform=ax.transAxes, color="white", fontsize=6,
                ha="right", va="bottom",
                bbox=dict(facecolor="#222", alpha=0.7,
                          boxstyle="round"))

    fig.suptitle(
        "Retreat Rate vs Physical Predictors  |  Spearman Correlation\n"
        "Western Scheldt saltmarsh cliff sites",
        color="white", fontsize=11, fontweight="bold"
    )
    fig.tight_layout()
    corr_png = OUT_DIR / "speed_correlation.png"
    fig.savefig(str(corr_png), dpi=150,
                bbox_inches="tight", facecolor="#0e0e0e")
    print(f"✓ {corr_png.name}")
    plt.show()

print(f"\n✓ Speed correlation analysis complete.")
print(f"\nInterpretation guide:")
print(f"  Fetch (+):      exposed sites retreat faster — expected")
print(f"  MNDWI std (+):  dynamic sites retreat faster — expected")
print(f"  EAD (-):        low-elevation sites retreat faster — expected")
print(f"  Pioneer (-):    vegetation buffer slows retreat — expected")
print(f"\n  Non-significant results (n={len(corr_data)}) are expected")
print(f"  given small sample size — direction of correlation matters")
print(f"  as much as significance at this n.")
