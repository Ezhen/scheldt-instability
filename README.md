# Scheldt Estuary — Coastal Instability Detection
### Remote Sensing Portfolio Project | Eugène Ivanov | 2025–2026

---

## Overview

A fully automated coastal instability risk assessment for the **Saeftinghe / Western Scheldt** estuary (Belgium / Netherlands), combining multi-temporal satellite imagery, airborne lidar, and in-situ tide gauge data into a composite instability index.

**Study area:** 3.80–4.25°E, 51.28–51.45°N (~30 × 20 km)  
**Period:** 2017–2025  
**Output:** Pixel-level instability risk map (5m resolution, 4-class)

---

## Key Results

| Metric | Value |
|---|---|
| Sentinel-2 snapshots | 27 (seasonal composites) |
| AHN4 lidar resolution | 5m (Dutch bank) |
| Tidal range (Vlissingen) | 3.68 m |
| Mean High Water | +1.979 m NAP |
| Valid pixels in risk index | 3,871,191 |
| Mean risk score | 0.356 |
| Critical risk zones | 10.0% of AOI |

**Scientific finding:** Over 2017–2025, the Saeftinghe marsh shows no significant NDVI browning trend — vegetation is stable. The dominant instability signal is driven by tidal exposure (EAD) and bank geometry (slope/curvature), consistent with a macrotidal system under active management.

---

## Pipeline Architecture

```
DATA ACQUISITION
├── acquire_scheldt.py      Sentinel-2 L2A + Sentinel-1 GRD via CDSE openEO
├── acquire_dem.py          Copernicus GLO-30 DEM via CDSE OData
├── acquire_dem_ahn4.py     AHN4 lidar DTM via PDOK WCS (Netherlands only)
└── acquire_tides.py        Vlissingen tide gauge via Rijkswaterstaat DDL API

ANALYSIS — OPTICAL
├── explore_snapshot.py     Single-snapshot visualiser (7 indices)
├── tier1_dynamics.py       NDVI trend, waterline migration, instability map
└── tier1_save_tiffs.py     Export analysis layers as GeoTIFFs

TERRAIN
├── recompute_derivatives.py  Evans-Young slope/curvature/TWI (--source ahn4)
└── compute_ead.py            Elevation Above Datum (EAD = elev − MHW)

RISK SCORING
└── risk_score.py           Weighted composite of 5 evidence layers → classified map

UTILITIES
├── debug_dem.py            DEM diagnostic
└── debug_curv.py           Curvature diagnostic
```

---

## Data Sources

| Dataset | Source | Resolution | Access |
|---|---|---|---|
| Sentinel-2 L2A | CDSE / ESA | 10m | Free, openEO |
| Sentinel-1 GRD | CDSE / ESA | 20m | Free, openEO |
| Copernicus DEM GLO-30 | CDSE | 30m | Free, OData |
| AHN4 DTM | PDOK / RWS | 5m | Free, WCS |
| Vlissingen tide gauge | Rijkswaterstaat DDL | Hourly | Free, REST API |

All data sources are **open access**, **no registration required** (except standard CDSE account).

---

## Risk Index

```
Risk = w₁ × MNDWI_instability  (0.25)
     + w₂ × NDVI_loss           (0.20)
     + w₃ × Slope               (0.20)
     + w₄ × Plan_curvature      (0.15)
     + w₅ × EAD_exposure        (0.20)
```

All layers normalised to [0,1] before weighting.  
Classification thresholds: Low <0.25 / Medium 0.25–0.35 / High 0.35–0.45 / Critical >0.45

---

## Environment Setup

```bash
# Create environment
conda create -n scheldt python=3.10
conda activate scheldt

# Install dependencies
pip install openeo rioxarray rasterio numpy scipy matplotlib
pip install scikit-image tqdm pandas ddlpy owslib pyproj richdem

# Yoda HPC note — suppress GDAL/PROJ warnings
export CPL_LOG=/dev/null
```

---

## Running the Pipeline

```bash
# 1. Acquire Sentinel-2 data (openEO — ~60 min)
python pipeline/acquisition/acquire_scheldt.py

# 2. Explore a snapshot
python pipeline/analysis/explore_snapshot.py data/scheldt/sentinel2/S2_2020_summer.tif

# 3. Tier 1 dynamics
python pipeline/analysis/tier1_dynamics.py ./data/scheldt/sentinel2/
python pipeline/analysis/tier1_save_tiffs.py

# 4. Download AHN4 lidar DEM
python pipeline/terrain/acquire_dem_ahn4.py
python pipeline/terrain/recompute_derivatives.py --source ahn4

# 5. Tide gauge + EAD
python pipeline/tides/acquire_tides.py
python pipeline/tides/compute_ead.py

# 6. Risk score
python pipeline/risk/risk_score.py
# Custom weights:
python pipeline/risk/risk_score.py --w1 0.30 --w2 0.15 --w3 0.20 --w4 0.15 --w5 0.20
```

---

## Methodological Notes

**Evans-Young curvature** — second-order polynomial fitting to 3×3 windows, preferred over `np.gradient` on integer/low-resolution DEMs to avoid denominator instability on flat terrain.

**MNDWI over NDWI** — Modified NDWI (Green−SWIR)/(Green+SWIR) preferred for the turbid macrotidal Scheldt over standard McFeeters NDWI; sharper waterline extraction in high-suspended-sediment conditions.

**Tidal correction** — all S2 waterline analyses should be cross-referenced against Vlissingen gauge timestamps. Winter composites show systematically higher water area due to tidal stage bias.

**AHN4 coverage** — lidar data covers the Dutch bank only. The Belgian Sea Scheldt upstream of the border uses GLO-30 (30m, integer metres). Combined analysis should note the resolution discontinuity.

**TWI caveat** — Topographic Wetness Index computed via richdem D8 flow accumulation. Not physically rigorous for tidal marshes (TWI assumes gravity-driven hydrology); used here as a flatness/wetness proxy only.

---

## Caveats & Limitations

- AHN4 covers NL bank only — Belgian bank uses 30m GLO-30
- NDVI browning signal is near-zero for this AOI (stable marsh) — not a bug
- Tidal stage at S2 acquisition not corrected — affects waterline position estimates
- Risk index weights are expert-judged, not empirically calibrated
- No ground truth validation — index is a relative ranking, not absolute probability

---

## Citation

```
Ivanov, E. (2026). Coastal instability detection using multi-source remote 
sensing: Scheldt estuary case study. Portfolio project, University of Liège.
Data: ESA Copernicus, Rijkswaterstaat, AHN/PDOK.
```
