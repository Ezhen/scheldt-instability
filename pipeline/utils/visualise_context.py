"""
Scheldt Observatory — spatial + longitudinal context visualisation
=================================================================
Purpose
-------
Build a Python-only overview figure before duplicating/refining it in QGIS.
The figure connects: tide-gauge stations, Saeftinghe/Bath risk sites,
ground-truth report anchors, and the longitudinal tidal amplification signal.

Inputs expected in the project root or supplied via CLI:
    analysis_summary.json
    structured_scheldt_ground_truth.geojson

Default output:
    data/scheldt/figures/scheldt_context_overview.png
    data/scheldt/figures/scheldt_context_overview.pdf

Usage
-----
    python visualize_scheldt_context.py

or
    python visualize_scheldt_context.py \
        --summary data/scheldt/analysis_summary.json \
        --ground-truth data/scheldt/structured_scheldt_ground_truth.geojson \
        --outdir data/scheldt/figures

Dependencies
------------
    pip install numpy pandas matplotlib

Notes
-----
- This is intentionally lightweight: no GeoPandas/Cartopy required.
- It is not a replacement for QGIS cartography. It is a scientific overview:
  where are stations/sites, what is the longitudinal tide signal, and where
  does report-based ground truth exist?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# -----------------------------------------------------------------------------
# Defaults / project geometry
# -----------------------------------------------------------------------------

AOI = {
    "west": 3.80,
    "east": 4.25,
    "south": 51.28,
    "north": 51.45,
}

STATION_COORDS = {
    "VLISSGN": {"name": "Vlissingen", "lon": 3.596},
    "TERNZN": {"name": "Terneuzen", "lon": 3.830},
    "HANSWT": {"name": "Hansweert", "lon": 3.996},
    "BATH": {"name": "Bath", "lon": 4.212},
    "KALLSLZS": {"name": "Kallosluis", "lon": 4.291},
}

# Approximate coordinates for visual context only.
# Replace with precise geometries later in QGIS / GeoPackage.
RISK_SITES = [
    {
        "site": "Geulwandverdediging",
        "lon": 4.17,
        "lat": 51.375,
        "status": "stable control",
        "note": "engineered bank / negative control",
    },
    {
        "site": "Plaat van Saeftinghe",
        "lon": 4.05,
        "lat": 51.382,
        "status": "eroding / dynamic",
        "note": "plate margin / steepening",
    },
    {
        "site": "Land van Saeftinghe east edge",
        "lon": 4.16,
        "lat": 51.350,
        "status": "eroding hotspot",
        "note": "marsh-edge erosion",
    },
    {
        "site": "Nauw van Bath",
        "lon": 4.20,
        "lat": 51.395,
        "status": "dynamic channel",
        "note": "navigation-channel forcing",
    },
    {
        "site": "Bath",
        "lon": 4.212,
        "lat": 51.405,
        "status": "tide station / forcing",
        "note": "near upstream AOI boundary",
    },
]


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def load_summary(path: Path) -> pd.DataFrame:
    """Load station tidal datums/asymmetry from analysis_summary.json.

    Robustness note:
    older/intermediate summary files may contain only tidal values and omit
    coordinates. In that case we inject known approximate station longitudes
    from STATION_COORDS so plotting does not fail with KeyError: 'lon'.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for code, rec in data.get("tidal_asymmetry", {}).items():
        row = {"station_code": code}
        row.update(rec)

        # Fallback station metadata if missing from JSON.
        fallback = STATION_COORDS.get(code, {})
        row.setdefault("name", fallback.get("name", code))
        row.setdefault("lon", fallback.get("lon", np.nan))

        rows.append(row)

    if not rows:
        raise ValueError(f"No tidal_asymmetry block found in {path}")

    df = pd.DataFrame(rows)
    df["lon"] = pd.to_numeric(df.get("lon"), errors="coerce")
    missing = df[df["lon"].isna()]["station_code"].tolist()
    if missing:
        raise ValueError(f"Missing station longitude for: {missing}. Add them to STATION_COORDS.")

    df = df.sort_values("lon").reset_index(drop=True)
    return df


def _geometry_centroid(geometry: dict) -> Tuple[float, float]:
    """Very small GeoJSON centroid helper for Point/Polygon/MultiPolygon.

    This avoids a GeoPandas dependency. It is only for overview plotting,
    not legal/engineering-grade geodesic geometry.
    """
    if not geometry:
        return np.nan, np.nan
    gtype = geometry.get("type")
    coords = geometry.get("coordinates", [])

    if gtype == "Point" and len(coords) >= 2:
        return float(coords[0]), float(coords[1])

    pts = []
    if gtype == "Polygon":
        rings = coords
        for ring in rings:
            pts.extend(ring)
    elif gtype == "MultiPolygon":
        for poly in coords:
            for ring in poly:
                pts.extend(ring)

    if not pts:
        return np.nan, np.nan
    arr = np.asarray(pts, dtype="float64")
    return float(np.nanmean(arr[:, 0])), float(np.nanmean(arr[:, 1]))


def load_ground_truth(path: Path) -> pd.DataFrame:
    """Load report-derived ground-truth anchors from GeoJSON properties.

    Accepts either explicit anchor_lon/anchor_lat properties or, if absent,
    computes a simple centroid from GeoJSON geometry.
    """
    with open(path, "r", encoding="utf-8") as f:
        gj = json.load(f)

    rows = []
    for i, feat in enumerate(gj.get("features", []), start=1):
        p = feat.get("properties", {})
        lon = p.get("anchor_lon", p.get("lon", p.get("longitude")))
        lat = p.get("anchor_lat", p.get("lat", p.get("latitude")))

        if lon is None or lat is None:
            lon, lat = _geometry_centroid(feat.get("geometry", {}))

        rows.append(
            {
                "obs_id": i,
                "class": p.get("class", p.get("ground_truth_class",
                         p.get("risk_class_name",
                         p.get("issue_type", "Unknown")))),
                "risk_rank": p.get("risk_rank", np.nan),
                "page": p.get("page", np.nan),
                "lon": lon,
                "lat": lat,
                "context": p.get("raw_context", p.get("context", "")),
                "source": p.get("doc_source", p.get("source", "")),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["obs_id", "class", "risk_rank", "page", "lon", "lat", "context", "source"])

    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    return df.dropna(subset=["lon", "lat"]).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def nearest_station(site_lon: float, stations: pd.DataFrame) -> pd.Series:
    idx = np.abs(stations["lon"].values - site_lon).argmin()
    return stations.iloc[idx]


def build_site_table(stations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for site in RISK_SITES:
        st = nearest_station(site["lon"], stations)
        rows.append(
            {
                **site,
                "nearest_station": st["name"],
                "nearest_station_code": st["station_code"],
                "MHW": st["MHW"],
                "MLW": st["MLW"],
                "TR": st["TR"],
                "MTL": st["MTL"],
                "asymmetry": st["asymmetry"],
            }
        )
    return pd.DataFrame(rows)


def style_dark(ax):
    ax.set_facecolor("#151515")
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#555")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.grid(True, color="#333", lw=0.5, alpha=0.8)


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def make_figure(stations: pd.DataFrame, gt: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    sites = build_site_table(stations)

    fig = plt.figure(figsize=(18, 12), facecolor="#0e0e0e")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0], hspace=0.26, wspace=0.16)

    # ------------------------------------------------------------------
    # Panel A: spatial context map
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    style_dark(ax)
    ax.set_title("A. Scheldt AOI — stations, risk sites, report anchors", fontweight="bold", pad=10)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")

    # Wider frame to include Vlissingen -> Kallosluis station chain.
    xmin = min(stations["lon"].min() - 0.05, AOI["west"] - 0.05)
    xmax = max(stations["lon"].max() + 0.05, AOI["east"] + 0.08)
    ymin = AOI["south"] - 0.03
    ymax = AOI["north"] + 0.03
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # AOI rectangle.
    rect = Rectangle(
        (AOI["west"], AOI["south"]),
        AOI["east"] - AOI["west"],
        AOI["north"] - AOI["south"],
        fill=False,
        edgecolor="#ffcc66",
        linewidth=2.0,
        linestyle="--",
        label="Satellite AOI",
    )
    ax.add_patch(rect)

    # Schematic estuary axis following station longitudes.
    # This is not coastline geometry, just a geographic spine.
    estuary_lons = stations["lon"].values
    estuary_lats = np.interp(estuary_lons, [estuary_lons.min(), estuary_lons.max()], [51.43, 51.25])
    ax.plot(estuary_lons, estuary_lats, color="#3f8fd2", lw=5, alpha=0.35, label="Estuary axis")

    # Ground-truth anchors. Many may share the same anchor; jitter very slightly for visibility.
    if not gt.empty:
        rng = np.random.default_rng(7)
        jitter_lon = gt["lon"].values + rng.normal(0, 0.004, size=len(gt))
        jitter_lat = gt["lat"].values + rng.normal(0, 0.002, size=len(gt))
        colors = gt["class"].map({"Stable": "#4aa3df", "Dynamic": "#ff9f1c"}).fillna("#aaaaaa")
        ax.scatter(
            jitter_lon,
            jitter_lat,
            c=colors,
            s=32,
            alpha=0.75,
            edgecolors="black",
            linewidths=0.25,
            label="Report anchors",
        )

    # Tide stations.
    ax.scatter(stations["lon"], estuary_lats, s=90, c="#4aa3df", edgecolors="white", linewidths=0.8, zorder=5)
    for _, r in stations.iterrows():
        lat = np.interp(r["lon"], [estuary_lons.min(), estuary_lons.max()], [51.43, 51.25])
        ax.text(r["lon"], lat + 0.015, r["name"], color="white", fontsize=8, ha="center")

    # Risk sites.
    status_color = {
        "stable control": "#4aa3df",
        "eroding / dynamic": "#f4a261",
        "eroding hotspot": "#e63946",
        "dynamic channel": "#ff9f1c",
        "tide station / forcing": "#80ed99",
    }
    for _, r in sites.iterrows():
        ax.scatter(r["lon"], r["lat"], marker="*", s=220,
                   c=status_color.get(r["status"], "white"), edgecolors="white", linewidths=0.6, zorder=6)
        ax.text(r["lon"] + 0.006, r["lat"] + 0.002, r["site"], color="white", fontsize=8, va="center")

    ax.legend(loc="lower left", facecolor="#202020", labelcolor="white", edgecolor="#777", fontsize=8)

    # ------------------------------------------------------------------
    # Panel B: longitudinal tidal amplification
    # ------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    style_dark(ax2)
    ax2.set_title("B. Longitudinal tidal amplification", fontweight="bold", pad=10)
    ax2.set_xlabel("Longitude (°E)  → upstream")
    ax2.set_ylabel("Water level (m NAP)")

    x = stations["lon"].values
    ax2.fill_between(x, stations["MLW"], stations["MHW"], color="#2b6f9e", alpha=0.28, label="Mean tidal envelope")
    ax2.plot(x, stations["MHW"], color="#ff595e", marker="o", lw=2.2, label="MHW")
    ax2.plot(x, stations["MLW"], color="#4aa3df", marker="o", lw=2.2, label="MLW")
    ax2.axhline(0, color="white", lw=0.8, ls=":", alpha=0.7)
    for _, r in stations.iterrows():
        ax2.text(r["lon"], r["MHW"] + 0.13, r["name"], color="white", fontsize=8, ha="center", rotation=25)

    ax2b = ax2.twinx()
    ax2b.set_ylabel("Tidal range (m)", color="#ffb703")
    ax2b.tick_params(colors="#ffb703", labelsize=8)
    for spine in ax2b.spines.values():
        spine.set_color("#555")
    ax2b.plot(x, stations["TR"], color="#ffb703", marker="s", lw=2.0, ls="--", label="TR")
    ax2b.set_ylim(max(0, stations["TR"].min() - 0.5), stations["TR"].max() + 0.5)

    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="lower right", facecolor="#202020", labelcolor="white", edgecolor="#777", fontsize=8)

    # ------------------------------------------------------------------
    # Panel C: evidence distribution
    # ------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    style_dark(ax3)
    ax3.set_title("C. Report-derived ground-truth evidence", fontweight="bold", pad=10)
    # Show issue type distribution from validation sites
    cls_field = None
    for f in ["issue_type", "class", "ground_truth_class",
               "risk_class_name", "status"]:
        if f in gt.columns and gt[f].notna().any():
            cls_field = f; break
    if cls_field:
        counts = gt[cls_field].value_counts().head(6)
        colors_c = ["#EF5350","#FF9800","#4CAF50","#42A5F5","#AB47BC","#EC407A"]
        bars = ax3.bar(counts.index, counts.values,
                        color=colors_c[:len(counts)],
                        edgecolor="white", linewidth=0.5)
        ax3.set_ylabel("Number of sites")
        ax3.tick_params(axis="x", labelrotation=25)
        for b in bars:
            ax3.text(b.get_x()+b.get_width()/2, b.get_height()+0.1,
                     str(int(b.get_height())), color="white",
                     ha="center", va="bottom", fontsize=10, fontweight="bold")
    else:
        ax3.text(0.5, 0.5, "No class data", transform=ax3.transAxes,
                  color="white", ha="center", va="center")
    ax3.text(
        0.02, 0.95,
        "Use these as validation anchors, not as training labels.\n"
        "Next: sample raster risk inside buffers around each site.",
        transform=ax3.transAxes,
        va="top",
        color="white",
        fontsize=9,
        bbox=dict(facecolor="#222", edgecolor="#555", boxstyle="round", alpha=0.9),
    )

    # ------------------------------------------------------------------
    # Panel D: risk site forcing profile
    # ------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    style_dark(ax4)
    ax4.set_title("D. Risk-site hydrodynamic context from nearest tide station", fontweight="bold", pad=10)
    y = np.arange(len(sites))
    sites_sorted = sites.sort_values("lon").reset_index(drop=True)
    ax4.barh(y, sites_sorted["TR"], color="#ffb703", edgecolor="white", linewidth=0.4, alpha=0.85)
    ax4.set_yticks(y)
    ax4.set_yticklabels(sites_sorted["site"], color="white")
    ax4.set_xlabel("Assigned tidal range from nearest station (m)")
    ax4.invert_yaxis()
    for i, r in sites_sorted.iterrows():
        ax4.text(r["TR"] + 0.03, i, f"{r['TR']:.2f} m  ({r['nearest_station']})",
                 color="white", va="center", fontsize=8)
    ax4.set_xlim(0, max(sites_sorted["TR"].max() + 0.8, 5.8))

    fig.suptitle(
        "Scheldt Observatory — spatial evidence and hydrodynamic forcing context",
        color="white", fontsize=13, fontweight="bold",
    )
    fig.subplots_adjust(top=0.93, bottom=0.06, left=0.07,
                         right=0.97, hspace=0.30, wspace=0.20)

    png = outdir / "scheldt_context_overview.png"
    pdf = outdir / "scheldt_context_overview.pdf"
    csv = outdir / "scheldt_context_site_table.csv"
    sites.to_csv(csv, index=False)
    fig.savefig(png, dpi=150, bbox_inches=None, facecolor="#0e0e0e")
    fig.savefig(pdf, bbox_inches=None, facecolor="#0e0e0e")
    plt.close(fig)

    print(f"Saved: {png}")
    print(f"Saved: {pdf}")
    print(f"Saved: {csv}")



# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Visualise Scheldt stations, risk sites and ground-truth context.")
    ap.add_argument("--summary", default="data/scheldt/monitoring/analysis_summary.json", help="Path to analysis_summary.json")
    ap.add_argument("--ground-truth", default="results/jsons/config_spatial_anchors.json", help="Path to structured ground-truth GeoJSON")
    ap.add_argument("--outdir", default="results/maps", help="Output folder")
    args = ap.parse_args()

    summary_path = Path(args.summary)
    gt_path = Path(args.ground_truth)
    outdir = Path(args.outdir)

    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"Missing ground-truth file: {gt_path}")

    stations = load_summary(summary_path)
    gt = load_ground_truth(gt_path)
    make_figure(stations, gt, outdir)


if __name__ == "__main__":
    main()
