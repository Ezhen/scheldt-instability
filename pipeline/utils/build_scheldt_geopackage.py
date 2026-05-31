#!/usr/bin/env python3
"""
Build Scheldt Observatory GeoPackage from current project files
==============================================================

This version assumes you have:
  - config_spatial_anchors.json     (report/evidence registry; NOT geometry)
  - analysis_summary.json           (tidal station metrics, optional)
  - GeoTIFF outputs                 (risk_score.tif, layer_slope.tif, etc.)

It creates one GeoPackage with vector layers + metadata tables:
  - aoi
  - tide_stations
  - risk_sites_manual
  - ground_truth_registry           from config_spatial_anchors.json
  - raster_catalog                  references to external GeoTIFFs
  - raster_samples_at_risk_sites    optional, if --sample-rasters is used

Important
---------
GeoTIFF rasters are NOT embedded. They remain external files and are linked in
raster_catalog. Load the GeoPackage + GeoTIFFs together in QGIS.

Usage from ~/Satellite_Project:
    python build_scheldt_geopackage_from_config.py --sample-rasters

Optional:
    python build_scheldt_geopackage_from_config.py \
      --config config_spatial_anchors.json \
      --summary analysis_summary.json \
      --raster-root data/scheldt/risk \
      --out scheldt_observatory.gpkg \
      --sample-rasters

Dependencies:
    pip install geopandas shapely pandas rasterio pyogrio
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box

try:
    import rasterio
    from rasterio.mask import mask as rio_mask
except Exception:  # allows vector-only build
    rasterio = None
    rio_mask = None

PROJECT_ROOT = Path('./') #Path(__file__).resolve().parent
CRS_WGS84 = "EPSG:4326"

AOI = {
    "west": 3.80,
    "east": 4.25,
    "south": 51.28,
    "north": 51.45,
}

# Approximate starting points for QGIS refinement.
RISK_SITES = [
    {
        "site_id": "geulwandverdediging_control",
        "name": "Geulwandverdediging bij Saeftinghe",
        "status": "stable_control",
        "lon": 4.170,
        "lat": 51.375,
        "interpretation": "engineered channel-wall protection; negative-control site",
    },
    {
        "site_id": "plaat_saeftinghe_margin",
        "name": "Plaat van Saeftinghe margin",
        "status": "eroding_dynamic",
        "lon": 4.050,
        "lat": 51.382,
        "interpretation": "documented eroding / steepening intertidal plate margin",
    },
    {
        "site_id": "saeftinghe_east_edge",
        "name": "Land van Saeftinghe eastern edge",
        "status": "eroding_hotspot",
        "lon": 4.160,
        "lat": 51.350,
        "interpretation": "marsh-edge erosion hotspot",
    },
    {
        "site_id": "nauw_van_bath",
        "name": "Nauw van Bath",
        "status": "dynamic_channel",
        "lon": 4.200,
        "lat": 51.395,
        "interpretation": "dynamic navigation-channel forcing zone",
    },
    {
        "site_id": "bath_station_context",
        "name": "Bath",
        "status": "tide_station_context",
        "lon": 4.212,
        "lat": 51.405,
        "interpretation": "upstream tide-gauge / forcing reference near AOI",
    },
]

STATION_FALLBACK = {
    "VLISSGN": {"name": "Vlissingen", "lon": 3.596, "lat": 51.442},
    "TERNZN": {"name": "Terneuzen", "lon": 3.830, "lat": 51.333},
    "HANSWT": {"name": "Hansweert", "lon": 3.996, "lat": 51.450},
    "BATH": {"name": "Bath", "lon": 4.212, "lat": 51.405},
    "KALLSLZS": {"name": "Kallosluis", "lon": 4.291, "lat": 51.252},
}

KNOWN_RASTERS = [
    "layer_ead_exposure.tif",
    "layer_mndwi_instability.tif",
    "layer_ndvi_loss.tif",
    "layer_plan_curvature.tif",
    "layer_slope.tif",
    "risk_classified.tif",
    "risk_score.tif",
]


def resolve(root: Path, p: str | Path) -> Path:
    path = Path(p).expanduser()
    return path if path.is_absolute() else root / path


def safe_json(x: Any) -> Any:
    if isinstance(x, (dict, list, tuple)):
        return json.dumps(x, ensure_ascii=False)
    return x


def write_layer(gdf: gpd.GeoDataFrame, gpkg: Path, layer: str) -> None:
    if gdf.empty:
        print(f"  - skipped empty layer: {layer}")
        return
    out = gdf.copy()
    for col in out.columns:
        if col != "geometry":
            out[col] = out[col].map(safe_json)
    out.to_file(gpkg, layer=layer, driver="GPKG")
    print(f"  + layer {layer:<28s} {len(out):>5d} features")


def write_table(df: pd.DataFrame, gpkg: Path, name: str) -> None:
    if df.empty:
        print(f"  - skipped empty table: {name}")
        return
    out = df.copy()
    for col in out.columns:
        out[col] = out[col].map(safe_json)
    with sqlite3.connect(gpkg) as con:
        out.to_sql(name, con, if_exists="replace", index=False)
    print(f"  + table {name:<28s} {len(out):>5d} rows")


def make_aoi() -> gpd.GeoDataFrame:
    geom = box(AOI["west"], AOI["south"], AOI["east"], AOI["north"])
    return gpd.GeoDataFrame(
        [{"name": "Scheldt_Saeftinghe_AOI", **AOI}], geometry=[geom], crs=CRS_WGS84
    )


def make_risk_sites() -> gpd.GeoDataFrame:
    df = pd.DataFrame(RISK_SITES)
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs=CRS_WGS84)


def load_tide_stations(summary_path: Path) -> gpd.GeoDataFrame:
    rows: list[dict[str, Any]] = []

    if summary_path.exists():
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        for code, rec in data.get("tidal_asymmetry", {}).items():
            base = STATION_FALLBACK.get(code, {})
            row = {"station_code": code, **base, **rec}
            row["lon"] = row.get("lon", base.get("lon"))
            row["lat"] = row.get("lat", base.get("lat"))
            row["name"] = row.get("name", base.get("name", code))
            rows.append(row)

    if not rows:
        print(f"  ! no usable summary found at {summary_path}; using station fallback only")
        rows = [{"station_code": code, **rec} for code, rec in STATION_FALLBACK.items()]

    df = pd.DataFrame(rows)
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df = df.dropna(subset=["lon", "lat"])
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs=CRS_WGS84)


def load_config_registry(config_path: Path) -> pd.DataFrame:
    """Read evidence registry — accepts both config_spatial_anchors.json
    (dict of anchors) and GeoJSON FeatureCollection formats."""
    if not config_path.exists():
        print(f"  ! missing config registry: {config_path}")
        return pd.DataFrame()

    data = json.loads(config_path.read_text(encoding="utf-8"))
    rows = []

    # GeoJSON FeatureCollection format
    if isinstance(data, dict) and data.get("type") == "FeatureCollection":
        for feat in data.get("features", []):
            p = feat.get("properties", {})
            coords = feat.get("geometry", {}).get("coordinates", [None, None])
            contexts = p.get("environmental_context", [])
            if isinstance(contexts, str):
                contexts = [contexts]
            rows.append({
                "anchor_id":            p.get("name", ""),
                "validation_source":    p.get("validation_source", "GeoJSON"),
                "page":                 p.get("source_page", np.nan),
                "bbox_west":            coords[0] if coords[0] else np.nan,
                "bbox_east":            coords[0] if coords[0] else np.nan,
                "bbox_south":           coords[1] if len(coords)>1 else np.nan,
                "bbox_north":           coords[1] if len(coords)>1 else np.nan,
                "context_first":        contexts[0] if contexts else "",
                "environmental_context": contexts,
                "note":                 p.get("ground_truth_class",""),
            })
    else:
        # Original config_spatial_anchors.json format (dict of anchors)
        for anchor_id, rec in data.items():
            if not isinstance(rec, dict):
                continue
            bbox     = rec.get("spatial_bounding_box", {}) or {}
            contexts = rec.get("environmental_context", []) or []
            rows.append({
                "anchor_id":            anchor_id,
                "validation_source":    rec.get("validation_source", ""),
                "page":                 rec.get("discovered_on_page", np.nan),
                "bbox_west":            bbox.get("west", np.nan),
                "bbox_east":            bbox.get("east", np.nan),
                "bbox_south":           bbox.get("south", np.nan),
                "bbox_north":           bbox.get("north", np.nan),
                "context_first":        contexts[0] if contexts else "",
                "environmental_context": contexts,
                "note": "Registry evidence only; bbox may be AOI-wide.",
            })

    return pd.DataFrame(rows)


def registry_bbox_layer(registry: pd.DataFrame) -> gpd.GeoDataFrame:
    """Optional polygons from registry bboxes; useful for seeing that most are AOI-wide."""
    if registry.empty:
        return gpd.GeoDataFrame(geometry=[], crs=CRS_WGS84)

    rows, geoms = [], []
    for _, r in registry.iterrows():
        vals = [r.get("bbox_west"), r.get("bbox_east"), r.get("bbox_south"), r.get("bbox_north")]
        if any(pd.isna(v) for v in vals):
            continue
        rows.append(r.to_dict())
        geoms.append(box(float(r["bbox_west"]), float(r["bbox_south"]), float(r["bbox_east"]), float(r["bbox_north"])))

    return gpd.GeoDataFrame(rows, geometry=geoms, crs=CRS_WGS84)


def classify_raster(path: Path) -> str:
    s = path.name.lower()
    if "risk_classified" in s:
        return "risk_classified"
    if "risk_score" in s:
        return "risk_score"
    if "ead" in s or "exposure" in s:
        return "tidal_exposure"
    if "mndwi" in s or "instability" in s:
        return "mndwi_instability"
    if "ndvi" in s:
        return "ndvi_loss"
    if "slope" in s:
        return "terrain_slope"
    if "plan_curvature" in s:
        return "terrain_plan_curvature"
    if "curvature" in s:
        return "terrain_curvature"
    return "other"


def find_rasters(root: Path, raster_root: Path) -> pd.DataFrame:
    candidates: list[Path] = []

    # Search first in requested raster root.
    if raster_root.exists():
        candidates.extend(sorted(raster_root.rglob("*.tif")))

    # Also find known filenames anywhere under root, but avoid huge recursive surprises if possible.
    for name in KNOWN_RASTERS:
        candidates.extend(root.rglob(name))

    # De-duplicate while preserving order.
    seen, rasters = set(), []
    for p in candidates:
        p = p.resolve()
        if p not in seen and p.exists():
            seen.add(p)
            rasters.append(p)

    rows = []
    for i, tif in enumerate(rasters, start=1):
        rec: dict[str, Any] = {
            "raster_id": f"r_{i:03d}",
            "filename": tif.name,
            "category": classify_raster(tif),
            "absolute_path": str(tif),
            "relative_path": str(tif.relative_to(root)) if str(tif).startswith(str(root)) else str(tif),
            "qgis_action": "Load as external raster layer; do not expect it embedded in GeoPackage.",
        }
        if rasterio is not None:
            try:
                with rasterio.open(tif) as src:
                    rec.update(
                        {
                            "crs": str(src.crs),
                            "width": src.width,
                            "height": src.height,
                            "count": src.count,
                            "res_x": src.res[0],
                            "res_y": src.res[1],
                            "bounds_left": src.bounds.left,
                            "bounds_bottom": src.bounds.bottom,
                            "bounds_right": src.bounds.right,
                            "bounds_top": src.bounds.top,
                            "nodata": src.nodata,
                        }
                    )
            except Exception as e:
                rec["read_error"] = str(e)
        rows.append(rec)

    return pd.DataFrame(rows)


def sample_rasters_at_sites(sites: gpd.GeoDataFrame, catalog: pd.DataFrame, buffer_m: float) -> pd.DataFrame:
    if rasterio is None or rio_mask is None:
        print("  ! rasterio not available; raster sampling skipped")
        return pd.DataFrame()
    if sites.empty or catalog.empty:
        return pd.DataFrame()

    rows = []
    for _, rr in catalog.iterrows():
        tif = Path(rr["absolute_path"])
        if not tif.exists():
            continue
        try:
            with rasterio.open(tif) as src:
                sites_r = sites.to_crs(src.crs)
                for _, site in sites_r.iterrows():
                    x, y = site.geometry.x, site.geometry.y
                    point_value = np.nan
                    try:
                        val = float(next(src.sample([(x, y)]))[0])
                        if src.nodata is None or val != src.nodata:
                            point_value = val
                    except Exception:
                        pass

                    buffer_mean = np.nan
                    buffer_p90 = np.nan
                    if buffer_m > 0:
                        try:
                            geom = site.geometry.buffer(buffer_m)
                            arr, _ = rio_mask(src, [geom], crop=True, filled=True)
                            data = arr[0].astype("float64")
                            if src.nodata is not None:
                                data[data == src.nodata] = np.nan
                            vals = data[np.isfinite(data)]
                            if vals.size:
                                buffer_mean = float(np.nanmean(vals))
                                buffer_p90 = float(np.nanpercentile(vals, 90))
                        except Exception:
                            pass

                    rows.append(
                        {
                            "site_id": site.get("site_id"),
                            "site_name": site.get("name"),
                            "site_status": site.get("status"),
                            "raster_id": rr["raster_id"],
                            "raster_filename": rr["filename"],
                            "raster_category": rr["category"],
                            "point_value": point_value,
                            f"buffer_mean_{int(buffer_m)}m": buffer_mean,
                            f"buffer_p90_{int(buffer_m)}m": buffer_p90,
                        }
                    )
        except Exception as e:
            print(f"  ! sampling failed for {tif.name}: {e}")

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Scheldt Observatory GeoPackage from config JSON + GeoTIFFs.")
    ap.add_argument("--root", default=str(PROJECT_ROOT), help="Project root. Default: folder containing this script.")
    ap.add_argument("--ground-truth", default="results/jsons/config_spatial_anchors.json", help="Ground-truth GeoJSON")
    ap.add_argument("--summary", default="data/scheldt/monitoring/analysis_summary.json", help="analysis_summary.json")
    ap.add_argument("--raster-root", default="data/scheldt/risk", help="Folder to scan for GeoTIFFs")
    ap.add_argument("--out", default="results/maps/scheldt_observatory.gpkg", help="Output GeoPackage")
    ap.add_argument("--sample-rasters", action="store_true", help="Sample GeoTIFFs at ground-truth anchors")
    ap.add_argument("--buffer-m", type=float, default=50.0, help="Buffer radius for raster samples, in raster CRS metres")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    config_path = resolve(root, args.ground_truth)
    summary_path = resolve(root, args.summary)
    raster_root = resolve(root, args.raster_root)
    out = resolve(root, args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    print("=== Scheldt Observatory GeoPackage builder ===")
    print(f"Root       : {root}")
    print(f"Config     : {config_path}  exists={config_path.exists()}")
    print(f"Summary    : {summary_path}  exists={summary_path.exists()}")
    print(f"Raster root: {raster_root}  exists={raster_root.exists()}")
    print(f"Output     : {out}\n")

    aoi = make_aoi()
    stations = load_tide_stations(summary_path)
    sites = make_risk_sites()
    registry = load_config_registry(config_path)
    registry_bboxes = registry_bbox_layer(registry)
    raster_catalog = find_rasters(root, raster_root)

    print("Writing spatial layers:")
    write_layer(aoi, out, "aoi")
    write_layer(stations, out, "tide_stations")
    write_layer(sites, out, "risk_sites_manual")
    write_layer(registry_bboxes, out, "registry_bbox_polygons")

    print("\nWriting tables:")
    write_table(registry, out, "ground_truth_registry")
    write_table(raster_catalog, out, "raster_catalog")

    if args.sample_rasters:
        samples = sample_rasters_at_sites(sites, raster_catalog, args.buffer_m)
        write_table(samples, out, "raster_samples_at_risk_sites")
    else:
        print("  - raster sampling skipped; add --sample-rasters if needed")

    print("\nDone.")
    print(f"Open in QGIS: {out}")
    print("Load external GeoTIFFs listed in raster_catalog as raster layers.")


if __name__ == "__main__":
    main()
