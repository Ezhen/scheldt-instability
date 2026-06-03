# Scheldt Bank Instability Observatory

**Hybrid coastal monitoring system** combining satellite remote sensing with
hydrodynamic modelling to assess saltmarsh bank instability and sediment
dynamics in the Western Scheldt estuary (Westerschelde), Netherlands.

Two independent lines of evidence for the same phenomenon:
- **Satellite pipeline** — multi-decadal Sentinel-2/AHN4 instability index
- **Hydrodynamic model** — Delft3D FM tidal forcing and erosion stress diagnostics

---

## Scientific context

The Western Scheldt is a macrotidal estuary with a ~4m tidal range and
significant anthropogenic pressure from the third navigational deepening (2010).
Saltmarsh banks at Saeftinghe, Plaat van Walsoorden, and Rug van Baarland show
signs of accelerating erosion documented in the IMDC Monitoringprogramma
Flexibel Storten reports (2016–2019).

This repository reproduces and extends that monitoring with:
- Automated satellite time series (2017–2025)
- A calibrated 2D barotropic DFM tidal model for January 2020
- ZES.1 ecotope classification aligned with Dutch monitoring standards

---

## Repository structure

```
pipeline/
├── acquisition/        Data download scripts (Sentinel, AHN4, tides, ERA5)
├── analysis/           Satellite analysis (NDVI trend, SAR, instability)
├── terrain/            DEM derivatives (slope, curvature, TWI, EAD)
├── tides/              Tidal datums and extremes
├── risk/               Composite risk score pipeline
├── model/              Delft3D FM hydrodynamic model pipeline
└── utils/              Visualisation and diagnostics

dfm_v5_final/           Final DFM model configuration (200m mesh, Jan 2020)
data/scheldt/
├── figures/
│   ├── erosion/        ZES.1 diagnostics, shear stress, current roses
│   └── model/          Validation figures, harmonic analysis, bed level
├── monitoring/         RWS tidal station analysis
├── risk/               Risk score layers and validation
└── wind/               ERA5 wind forcing diagnostics

docs/                   Methods, data sources, results guide
results/figures/        Publication-ready summary figures
```

---

## Pipeline 1 — Satellite Observatory

### Data acquisition
```
pipeline/acquisition/
├── acquire_scheldt.py      Sentinel-2 L2A + Sentinel-1 GRD via CDSE openEO
├── acquire_dem.py          Copernicus GLO-30 DEM via CDSE OData
├── acquire_dem_ahn4.py     AHN4 lidar DTM via PDOK WCS (Netherlands only)
└── acquire_tides.py        Vlissingen tide gauge via Rijkswaterstaat DDL API
```

### Analysis — optical
```
pipeline/analysis/
├── explore_snapshot.py     Single-snapshot visualiser (7 indices)
├── tier1_dynamics.py       NDVI trend, waterline migration, instability map
└── tier1_save_tiffs.py     Export analysis layers as GeoTIFFs
```

### Terrain
```
pipeline/terrain/
├── recompute_derivatives.py  Evans-Young slope/curvature/TWI (--source ahn4)
└── compute_ead.py            Elevation Above Datum (EAD = elev − MHW)
```

### Risk scoring
```
pipeline/risk/
├── risk_score.py           Weighted composite of 5 evidence layers
├── pioneer_correction.py   Pioneer zone threshold correction
├── transect_analysis.py    Shoreline retreat rate analysis
└── validate_risk.py        Ground-truth validation against field data
```

---

## Pipeline 2 — Hydrodynamic Model

Delft3D FM (D-Flow FM 1.2.184) barotropic 2D model of the Western Scheldt
for January 2020. Model domain: x=[−5000, 80000], y=[368000, 401000] RD New.

### Model setup scripts
```
pipeline/model/
├── acquire_bathymetry.py   EMODnet WCS, 29m resolution, EPSG:4326
├── generate_mesh.py        meshkernel 8.3.0 variable-resolution mesh
│                           Casulli refinement: channel 200m, flat 400m, outer 500m
├── generate_roughness.py   Spatially varying Manning n from AHN4 + Sentinel-2 NDVI
│                           ZES.1-aligned classification (6 classes, n=0.019–0.045)
├── generate_spatial_bc.py  Obs-driven tidal BC from RWS monitoring.db
├── acquire_wind_forcing.py ERA5 10m wind → DFM meteo_on_equidistant_grid
├── prepare_dfm_model.py    Full DFM input generation (MDU, fieldFile.ini, dimr)
├── compare_obs.py          Skill metrics vs RWS observations (RMSE, r², skill score)
├── visualise_dfm.py        Time series, map snapshots, bed level figures
├── tidal_harmonics.py      Least-squares M2/S2/N2/M4 harmonic analysis
└── erosion_diagnostics.py  ZES.1 erosion stress diagnostic suite
```

### Final model configuration (`dfm_v5_final/`)
| Parameter | Value |
|---|---|
| Mesh | 103,224 faces, 200m channel / 400m flat / 500m outer |
| Bathymetry | EMODnet 29m, IDW onto mesh nodes (3.96M source points) |
| Manning | Spatial: AHN4 + S2 NDVI, n = 0.019 (channel) to 0.045 (marsh) |
| Tidal BC | RWS Vlissingen obs, direct interpolation, t₀ = −1.030 m NAP |
| Discharge | 300 m³/s constant, eastern boundary (Kallosluis) |
| Wind | Uniform SW 5.66 m/s |
| Period | 2020-01-01 to 2020-02-01 UTC |
| Output | Hourly map + 10-min his; τ, ucx/ucy, water depth, Chezy |

### Validation
| Station | RMSE (m) | r² | Range ratio | Skill |
|---|---|---|---|---|
| Vlissingen | 0.241 | 0.947 | 0.97 | 0.945 |
| Terneuzen | 0.288 | 0.946 | 0.88 | 0.936 |
| Hansweert | 0.368 | 0.936 | 0.81 | 0.908 |
| Bath | 0.640 | 0.777 | 0.74 | 0.758 |
| Kallosluis | 0.689 | 0.773 | 0.70 | 0.748 |

Mesh refinement (500m → 200m): Bath RMSE 1.198 → 0.640 m (−47%), r² 0.505 → 0.777.

### Erosion diagnostics (ZES.1 aligned)
Figures in `data/scheldt/figures/erosion/`:

| Figure | Description |
|---|---|
| `zes_ecotope_map.png` | ZES.1 ecotope classification (Bouma et al. 2005) |
| `erosion_peak_speed.png` | Peak current speed + 0.2/0.8/1.5 m/s threshold contours |
| `erosion_tau_mean.png` | Mean bed shear stress + τ_cr contours (sand/mud/marsh) |
| `erosion_exceedance_mud.png` | Fraction of time τ > 0.40 Pa (cohesive erosion) |
| `erosion_exceedance_marsh.png` | Fraction of time τ > 1.00 Pa (root zone failure) |
| `erosion_flood_ebb_ratio.png` | Flood/ebb velocity ratio (transport direction) |
| `erosion_inundation_freq.png` | Inundation frequency (hydroperiod) |
| `erosion_residual_speed.png` | Residual current (Eulerian mean) |
| `erosion_current_rose.png` | Current roses by ZES.1 dynamic class |
| `erosion_station_timeseries.png` | Speed + shear stress at RWS stations |

---

## The hybrid connection

The satellite risk score and model erosion diagnostics are independent evidence
for the same processes:

```
Satellite (2017–2025)              Model (January 2020)
─────────────────────              ────────────────────
NDVI loss trend           ←→       High τ > τ_marsh exceedance
Waterline retreat         ←→       Flood/ebb dominance (landward transport)
Morphological instability ←→       Peak U > 0.8 m/s (ZES.1 hoogdynamisch)
Pioneer zone extent       ←→       Inundation frequency (hydroperiod)
```

Comparison figure: `data/scheldt/risk/speed_correlation.png` —
correlation between satellite-derived instability index and
model peak current speed across the intertidal zone.

---

## Data sources

| Dataset | Source | Resolution | Coverage |
|---|---|---|---|
| Sentinel-2 L2A | CDSE openEO | 10m | 2017–2025, 3 seasons/yr |
| Sentinel-1 GRD | CDSE openEO | 10m | 2017–2025, 3 seasons/yr |
| AHN4 DTM | PDOK WCS | 5m | Eastern Scheldt only |
| EMODnet bathymetry | EMODnet WCS | 29m | Full domain |
| RWS tidal observations | Rijkswaterstaat DDL | 10-min | 5 stations, 2020 |
| ERA5 wind | ECMWF CDS | ~31km | January 2020 |

Monitoring database (`data/scheldt/monitoring.db`) built by
`pipeline/acquisition/acquire_monitoring_db.py` — schema in
`data/scheldt/monitoring_schema.sql`.

---

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies: `rasterio`, `pyproj`, `netCDF4`, `meshkernel==8.3.0`,
`scipy`, `numpy`, `matplotlib`, `pandas`, `openeo`, `cdsapi`

**PROJ fix** required on systems with conda-managed environments —
see top of any `pipeline/model/` script for the environment variable pattern.

---

## SLURM / HPC

Model runs on CECI/Yoda cluster (nic5). Submit with:
```bash
sbatch run_dfm_final.slurm   # 31-day final run, ~3.5h wall time
```

---

## Reference

IMDC (2019). Monitoringprogramma Flexibel Storten — Voortgangsrapportage
2016–2017. Lanckriet T., Pandelaers C., Pieterse A., Van Holland G.
Report I/RA/11498/18.126/API, Vlaamse Overheid / Afdeling Maritieme Toegang.

Bouma H., de Jong D.J., Twisk F. & Wolfstein F. (2005). Zoute wateren
EcotopenStelsel (ZES.1). RIKZ/2005.024, Rijkswaterstaat.
