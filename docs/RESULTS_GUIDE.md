# Results Guide — What Each Output Means

## Output Files

### Sentinel-2 Analysis (`data/scheldt/sentinel2/analysis/`)

| File | Description | Portfolio use |
|---|---|---|
| `mndwi_std.tif` | MNDWI temporal std dev | Instability heatmap — primary change signal |
| `ndvi_slope.tif` | NDVI linear trend (NDVI/timestep) | Full greening/browning map |
| `ndvi_slope_loss.tif` | Browning signal only (neg. slopes) | Risk layer input |
| `water_frequency.tif` | Fraction of snapshots as water [0–1] | Dynamic fringe identification |
| `ndvi_mean.tif` | Mean NDVI across all snapshots | Baseline vegetation map |
| `mndwi_mean.tif` | Mean MNDWI | Baseline water/mud map |
| `qa/snapshot_stats.csv` | Per-snapshot quality metrics | Data quality documentation |
| `qa/extreme_pixels_*.tif` | Corruption maps | QA — shows artefact locations |

### Terrain (`data/scheldt/dem_ahn4/derivatives/`)

| File | Description | Portfolio use |
|---|---|---|
| `slope_degrees.tif` | Terrain slope in degrees | Bank steepness — dike faces show as lines |
| `plan_curvature.tif` | Plan curvature (Evans-Young) | Negative = convergent = erosion-prone |
| `profile_curvature.tif` | Profile curvature | Concave banks = accelerating flow |
| `TWI.tif` | Topographic Wetness Index (D8) | Wetness/flatness proxy |
| `EAD.tif` | Elevation above MHW | Primary tidal risk layer |
| `intertidal_zone.tif` | Binary: pixels between MLW and MHW | Intertidal extent map |
| `vulnerable_fringe.tif` | Binary: pixels MHW to MHW+1m | At-risk but currently protected |
| `resampled_10m/` | All derivatives at S2 pixel grid | For risk score combination |

### Tides (`data/scheldt/tides/`)

| File | Description |
|---|---|
| `vlissingen_hourly.csv` | 2017–2025 hourly water levels (m NAP) |
| `vlissingen_extremes.csv` | HW/LW peak detections |
| `tidal_datums.json` | MHW, MLW, MTL, TR, MHWS, MLWS |
| `tides_overview.png` | Three-panel figure |
| `EAD_overview.png` | EAD map + zone classification + histogram |

### Risk (`data/scheldt/risk/`)

| File | Description |
|---|---|
| `risk_score.tif` | Composite instability index [0–1] |
| `risk_classified.tif` | 4-class map (1=Low 2=Med 3=High 4=Critical) |
| `risk_weights.json` | Layer weights and classification thresholds used |
| `layer_*.tif` | Individual normalised layers |
| `risk_overview.png` | **Portfolio main figure** |

---

## Reading the Risk Map

```
Class 1 — Low      (#2196F3 blue)    : Stable elevated polder interior
Class 2 — Medium   (#4CAF50 green)   : Intertidal marsh, regularly exposed but stable
Class 3 — High     (#FF9800 orange)  : Dynamic margins, bank faces, channel edges
Class 4 — Critical (#F44336 red)     : Highest instability — active erosion zones,
                                       tidal creek banks, dike toes
```

**What drives Critical classification:**
Pixels scoring Critical typically have 3+ of: low EAD (below MHW), steep slope
(dike face), high MNDWI variability (dynamic waterline), convergent plan curvature
(bank concavity), and detectable NDVI browning.

---

## Portfolio Narrative Suggestion

> "The composite risk index reveals that 10% of the Dutch Scheldt bank is
> classified as Critical — concentrated along tidal creek margins within
> Saeftinghe and at the toe of managed dike faces. The dominant drivers are
> tidal exposure (EAD) and bank geometry (slope), rather than vegetation loss
> — consistent with a well-established macrotidal marsh under active management.
> The MNDWI temporal variability layer highlights the most hydrodynamically
> active zones, particularly the bifurcating tidal creek network in the
> Saeftinghe interior."
