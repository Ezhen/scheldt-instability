"""
Scheldt — Risk Index Validation
================================
Cross-references the composite risk score against known problem sites
extracted from the IMDC monitoring report.

For each problem site:
  1. Sample the risk score at the site location
  2. Check which risk class it falls in
  3. Compare against expected severity from the report
  4. Produce a validation figure and summary table

Outputs
-------
    risk/validation_results.csv     per-site risk score vs expected severity
    risk/validation_figure.png      map + scatter + confusion matrix
    risk/validation_summary.json    overall statistics

Usage
-----
    python validate_risk.py
    python validate_risk.py --json reports/extracted/other_report_extracted.json
"""

import os
os.environ["CPL_LOG"] = "/dev/null"

import json
import argparse
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import rioxarray as rxr
import rasterio
from rasterio.transform import rowcol
from pathlib import Path

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────

RISK_TIF     = Path("./data/scheldt/risk/risk_score.tif")
RISK_CLS_TIF = Path("./data/scheldt/risk/risk_classified.tif")
EXTRACTED    = Path("./reports/extracted/imdc_rapport_2016_2017_extracted.json")
OUT_DIR      = Path("./data/scheldt/risk")

# Severity → expected minimum risk class
# Low severity → expect Medium or above
# Medium severity → expect High or above
# High severity → expect Critical
SEVERITY_EXPECTED_CLASS = {
    "low":    2,   # Medium
    "medium": 3,   # High
    "high":   4,   # Critical
}

CLASS_NAMES  = {1: "Low", 2: "Medium", 3: "High", 4: "Critical"}
CLASS_COLORS = {1: "#2196F3", 2: "#4CAF50", 3: "#FF9800", 4: "#F44336"}

# ── ARGUMENT PARSING ─────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--json", type=Path, default=EXTRACTED,
                    help="Extracted report JSON")
parser.add_argument("--buffer", type=int, default=5,
                    help="Buffer radius in pixels around each site (default 5)")
parser.add_argument("--risk", type=Path, default=RISK_TIF,
                    help="Risk score GeoTIFF (default: risk_score.tif)")
parser.add_argument("--classified", type=Path, default=RISK_CLS_TIF,
                    help="Classified risk GeoTIFF (default: risk_classified.tif)")
args = parser.parse_args()


# ── LOAD DATA ─────────────────────────────────────────────────────────────────

print("=== Risk Index Validation ===\n")

# Load extracted report
print(f"Loading problem sites from {args.json.name} ...")
with open(args.json) as f:
    report = json.load(f)

sites = report.get("problem_sites", []) + report.get("validation_sites", [])
sites = [s for s in sites
         if s.get("lon_approx") is not None
         and s.get("lat_approx") is not None]
print(f"  {len(sites)} sites with coordinates")

# Load risk score
print(f"\nLoading risk score ...")
with rasterio.open(args.risk) as src:
    risk_arr = src.read(1).astype("float32")
    risk_crs = src.crs
    risk_t   = src.transform
    risk_bounds = src.bounds

with rasterio.open(args.classified) as src:
    cls_arr = src.read(1).astype("float32")

print(f"  Shape: {risk_arr.shape}")
print(f"  Bounds: {risk_bounds.left:.3f} {risk_bounds.bottom:.3f} "
      f"{risk_bounds.right:.3f} {risk_bounds.top:.3f}")
print(f"  CRS: {risk_crs}")

# If risk raster is in WGS84, bounds are in degrees
# If projected, need to transform site coords before bounds check
_crs_is_geographic = risk_crs.is_geographic if hasattr(risk_crs, "is_geographic") else True

# ── REPROJECT SITE COORDS IF NEEDED ──────────────────────────────────────────

def lon_lat_to_pixel(lon, lat, transform, crs):
    """Convert WGS84 lon/lat to pixel row/col in the raster."""
    try:
        from pyproj import Transformer
        crs_str = crs.to_string() if hasattr(crs, "to_string") else str(crs)
        tr  = Transformer.from_crs("EPSG:4326", crs_str, always_xy=True)
        x, y = tr.transform(lon, lat)
    except Exception:
        x, y = lon, lat  # fall back: assume already in raster CRS

    col = int((x - transform.c) / transform.a)
    row = int((y - transform.f) / transform.e)
    return row, col


def site_in_bounds(lon, lat, transform, crs, shape):
    """Check whether a WGS84 coordinate falls within the raster."""
    row, col = lon_lat_to_pixel(lon, lat, transform, crs)
    H, W = shape
    return (0 <= row < H) and (0 <= col < W), row, col


# ── SAMPLE RISK AT EACH SITE ─────────────────────────────────────────────────

print("\nSampling risk score at each site ...")
results = []

H, W = risk_arr.shape
buf  = args.buffer

for site in sites:
    lon = site["lon_approx"]
    lat = site["lat_approx"]
    in_bounds, row, col = site_in_bounds(lon, lat, risk_t, risk_crs,
                                          (H, W))

    if not in_bounds:
        print(f"  ✗ {site['name'][:40]}  "
              f"lon={lon:.3f} lat={lat:.3f} → outside raster bounds "
              f"(row={row} col={col} vs {H}×{W})")
        results.append({**site,
                        "in_bounds": False,
                        "risk_score": None,
                        "risk_class": None,
                        "risk_class_name": "Outside AOI",
                        "expected_class": None,
                        "validated": None})
        continue

    # Sample with buffer — take mean of surrounding pixels
    r0, r1 = max(0, row-buf), min(H, row+buf+1)
    c0, c1 = max(0, col-buf), min(W, col+buf+1)
    patch_risk = risk_arr[r0:r1, c0:c1]
    patch_cls  = cls_arr[r0:r1, c0:c1]

    valid_risk = patch_risk[np.isfinite(patch_risk)]
    valid_cls  = patch_cls[np.isfinite(patch_cls)]

    if len(valid_risk) == 0:
        risk_val  = None
        cls_val   = None
        cls_name  = "No data"
    else:
        risk_val = float(np.nanmean(valid_risk))
        # Most common class in buffer
        cls_val  = int(np.bincount(
            valid_cls[np.isfinite(valid_cls)].astype(int)
        ).argmax()) if len(valid_cls) > 0 else None
        cls_name = CLASS_NAMES.get(cls_val, "Unknown")

    # Validation: does detected class meet expected minimum?
    severity   = site.get("severity", "medium") or "medium"
    exp_class  = SEVERITY_EXPECTED_CLASS.get(severity, 2)
    validated  = (cls_val is not None) and (cls_val >= exp_class) \
                 if cls_val else None

    result = {
        **site,
        "pixel_row":         row,
        "pixel_col":         col,
        "in_bounds":         True,
        "risk_score":        round(risk_val, 4) if risk_val else None,
        "risk_class":        cls_val,
        "risk_class_name":   cls_name,
        "expected_min_class":exp_class,
        "expected_class_name": CLASS_NAMES.get(exp_class, "?"),
        "validated":         validated,
    }
    results.append(result)

    status = "✓" if validated else ("?" if validated is None else "✗")
    risk_str = f"{risk_val:.3f}" if risk_val is not None else "?"
    print(f"  {status} {site['name'][:38]:38s} "
          f"risk={risk_str:>6}  "
          f"class={cls_name:8s}  "
          f"expected>={CLASS_NAMES.get(exp_class,'?')}")


# ── STATISTICS ────────────────────────────────────────────────────────────────

in_bounds = [r for r in results if r["in_bounds"] and r["validated"] is not None]
n_total   = len(in_bounds)
n_pass    = sum(1 for r in in_bounds if r["validated"])
n_fail    = n_total - n_pass

print(f"\n── Validation summary ──────────────────────────────────")
print(f"  Sites in AOI        : {n_total}")
print(f"  Correctly detected  : {n_pass}  ({100*n_pass/n_total:.0f}%)" if n_total else "  No sites in AOI")
print(f"  Not detected        : {n_fail}  ({100*n_fail/n_total:.0f}%)" if n_total else "")


# ── SAVE CSV ──────────────────────────────────────────────────────────────────

import csv
csv_path = OUT_DIR / "validation_results.csv"
fields = ["name", "macrocel", "issue_type", "severity",
          "lon_approx", "lat_approx", "in_bounds",
          "risk_score", "risk_class", "risk_class_name",
          "expected_min_class", "expected_class_name", "validated"]
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    w.writeheader(); w.writerows(results)
print(f"\n✓ {csv_path.name}")

# Save summary JSON
summary = {
    "n_sites_total":     len(results),
    "n_in_bounds":       len(in_bounds),
    "n_validated":       n_pass,
    "n_not_detected":    n_fail,
    "detection_rate":    round(n_pass/n_total, 3) if n_total else None,
    "per_site":          [{
        "name":         r["name"],
        "risk_score":   r.get("risk_score"),
        "risk_class":   r.get("risk_class_name"),
        "validated":    r.get("validated"),
    } for r in results if r["in_bounds"]],
}
with open(OUT_DIR / "validation_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"✓ validation_summary.json")


# ── VISUALISE ─────────────────────────────────────────────────────────────────

print("\nGenerating validation figure ...")

fig, axes = plt.subplots(1, 3, figsize=(21, 8), facecolor="#0e0e0e")

# ── Panel 1: Risk map with site overlay ──────────────────────────────────────
ax = axes[0]
ax.set_facecolor("#0e0e0e")

# Background: classified risk map
cls_cmap = mcolors.ListedColormap(
    ["#2196F3", "#4CAF50", "#FF9800", "#F44336"])
ax.imshow(cls_arr, cmap=cls_cmap, vmin=0.5, vmax=4.5,
           aspect="equal", interpolation="nearest", alpha=0.7)

# Overlay problem sites
for r in results:
    if not r.get("in_bounds"): continue
    row, col = r["pixel_row"], r["pixel_col"]
    validated = r.get("validated")
    color  = "#00FF00" if validated else "#FF0000" if validated is False else "#FFFF00"
    marker = "o" if validated else "x" if validated is False else "^"
    ax.plot(col, row, marker, color=color, ms=10, mew=2,
            label=f"{r['name'][:20]}")
    ax.annotate(r["name"].split("-")[0].strip()[:15],
                (col, row), textcoords="offset points",
                xytext=(5, 5), color="white", fontsize=6)

# Legend
handles = [
    mpatches.Patch(color="#00FF00", label="✓ Validated (risk ≥ expected)"),
    mpatches.Patch(color="#FF0000", label="✗ Not detected (risk < expected)"),
    mpatches.Patch(color="#FFFF00", label="? Outside AOI / no data"),
]
handles += [
    mpatches.Patch(color=c, label=f"Risk: {n}")
    for n, c in [("Low","#2196F3"),("Medium","#4CAF50"),
                 ("High","#FF9800"),("Critical","#F44336")]
]
ax.legend(handles=handles, loc="lower left", fontsize=6,
          facecolor="#222", labelcolor="white", framealpha=0.9)
ax.set_title("Problem sites vs classified risk zones\n"
              "◯=validated  ✗=not detected",
              color="white", fontsize=9, fontweight="bold")
ax.axis("off")

# ── Panel 2: Risk score per site ─────────────────────────────────────────────
ax2 = axes[1]
ax2.set_facecolor("#1a1a1a")

in_b = [r for r in results if r.get("in_bounds") and r.get("risk_score")]
names   = [r["name"].split("-")[0].strip()[:20] for r in in_b]
scores  = [r["risk_score"] for r in in_b]
exp_cls = [r.get("expected_min_class", 2) for r in in_b]
colors  = [CLASS_COLORS.get(r.get("risk_class"), "#888") for r in in_b]

y = np.arange(len(names))
bars = ax2.barh(y, scores, color=colors, alpha=0.85, edgecolor="#333")

# Mark expected minimum threshold
thresholds = {1: 0.25, 2: 0.35, 3: 0.45, 4: 0.60}
for i, (ec, score) in enumerate(zip(exp_cls, scores)):
    thresh = thresholds.get(ec, 0.35)
    ax2.plot([thresh, thresh], [i-0.4, i+0.4],
             color="white", lw=1.5, ls="--", alpha=0.7)

ax2.set_yticks(y); ax2.set_yticklabels(names, color="white", fontsize=7)
ax2.set_xlabel("Risk score [0–1]", color="white", fontsize=9)
ax2.tick_params(colors="white", labelsize=8)
ax2.spines[:].set_color("#444")
ax2.set_title("Risk score per problem site\n"
              "dashed line = expected minimum threshold",
              color="white", fontsize=9, fontweight="bold")
ax2.set_xlim(0, 1)

# ── Panel 3: Detection summary ───────────────────────────────────────────────
ax3 = axes[2]
ax3.set_facecolor("#1a1a1a")
ax3.axis("off")

if n_total > 0:
    # Pie chart
    vals   = [n_pass, n_fail]
    labels = [f"Detected\n({n_pass}/{n_total})",
              f"Missed\n({n_fail}/{n_total})"]
    cols   = ["#4CAF50", "#F44336"]
    wedges, texts, pcts = ax3.pie(
        vals, labels=labels, colors=cols,
        autopct="%1.0f%%", startangle=90,
        textprops={"color":"white","fontsize":10},
        wedgeprops={"edgecolor":"#333","linewidth":1.5},
        pctdistance=0.75
    )
    for pct in pcts:
        pct.set_color("white")
        pct.set_fontsize(12)
        pct.set_fontweight("bold")

summary_txt = (
    f"VALIDATION SUMMARY\n"
    f"{'─'*28}\n"
    f"Report: IMDC 2016–2017\n"
    f"Sites extracted   : {len(results)}\n"
    f"Sites in AOI      : {n_total}\n"
    f"Detected          : {n_pass}\n"
    f"Missed            : {n_fail}\n"
    f"Detection rate    : {100*n_pass/n_total:.0f}%\n"
    if n_total else
    f"VALIDATION SUMMARY\n{'─'*28}\nNo sites in AOI\n"
)
ax3.text(0.5, 0.05, summary_txt, transform=ax3.transAxes,
          color="white", fontsize=8, va="bottom", ha="center",
          fontfamily="monospace",
          bbox=dict(facecolor="#111", alpha=0.9,
                    boxstyle="round", edgecolor="#444"))
ax3.set_title("Detection rate\n(risk class ≥ expected severity)",
               color="white", fontsize=9, fontweight="bold")

fig.suptitle(
    "Risk Index Validation  |  IMDC Monitoring Report 2016–2017  |  "
    "Scheldt Estuary",
    color="white", fontsize=11, fontweight="bold"
)
fig.tight_layout()
out_png = OUT_DIR / "validation_figure.png"
fig.savefig(str(out_png), dpi=150,
            bbox_inches="tight", facecolor="#0e0e0e")
print(f"✓ {out_png.name}")
plt.show()

print(f"\n✓ Validation complete.")
print(f"  Key question: does your risk index correctly flag the")
print(f"  sites the government monitoring program identified as problematic?")
