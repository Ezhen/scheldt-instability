# Methods Documentation

## 1. Study Area

The Verdronken Land van Saeftinghe and adjacent Sea Scheldt banks constitute one of the
most dynamic intertidal environments in NW Europe. The macrotidal regime (TR ≈ 3.68m at
Vlissingen) drives strong ebb/flood asymmetry and sustained sediment exchange between
the channel, mudflat, and salt marsh zones. The area has been subject to repeated dredging
of the navigation channel to the Port of Antwerp, altering tidal prism and bank exposure.

## 2. Sentinel-2 Processing

L2A surface reflectance products accessed via CDSE openEO.
Seasonal composites: spring (May–Jun), summer (Aug–Sep), winter (Dec–Jan).
Cloud masking: SCL band, classes 8/9/10 (cloud) and 3 (shadow) removed.
Temporal reducer: median composite per season.
DN to reflectance: divide by 10000 (ESA standard scaling).
Artefact removal: index values outside [−1, 1] masked as invalid.

### Indices computed
- NDVI  = (B08 − B04) / (B08 + B04)
- NDWI  = (B03 − B08) / (B03 + B08)  [McFeeters 1996]
- MNDWI = (B03 − B11) / (B03 + B11)  [Xu 2006]
- SWIR  = B11 raw reflectance

## 3. Terrain Analysis

### DEM sources
- GLO-30: TanDEM-X radar DSM, 30m, integer metres, EGM2008 datum
- AHN4: airborne lidar DTM, 5m, float32 cm precision, NAP datum

### Derivative computation
Evans-Young (1979) polynomial method: fits 6-parameter quadratic to
3×3 pixel neighbourhoods. Returns slope, plan curvature, profile curvature.
Preferred over finite-difference (np.gradient) for integer/low-resolution DEMs.

Gaussian pre-smoothing: sigma=0.5px (AHN4 5m), sigma=1.0px (GLO-30 30m).
Water pixels filled with neighbourhood interpolation before smoothing
to prevent channel artefacts contaminating bank derivatives.

## 4. Tidal Datums

Source: Rijkswaterstaat DDL API, station VLISSGN (Vlissingen).
Period: 2017-01-01 to 2025-12-31.
Sampling: 10-minute observations, resampled to 1-hour mean.
Peak detection: scipy.find_peaks, min distance 10 hours (semidiurnal cycle ~12.4h).

| Datum | Value (m NAP) |
|-------|---------------|
| MHW   | +1.979 |
| MLW   | −1.697 |
| MTL   | +0.141 |
| TR    | 3.676  |
| MHWS  | +2.407 |
| MLWS  | −2.052 |
| MSL   | +0.010 |

## 5. Elevation Above Datum

EAD = AHN4 elevation (m NAP) − MHW (+1.979 m NAP)

Interpretation:
- EAD > +1.0m  : above spring tide level, low tidal exposure
- 0 < EAD ≤ +1.0m : floods on spring tides, vulnerable fringe
- −TR < EAD ≤ 0   : regular intertidal, floods every tidal cycle
- EAD ≤ −TR       : permanent water / deep tidal flat

## 6. Risk Index

### Layer normalisation
Percentile stretch to [0,1]: p5–p95 for normally distributed layers.
Zero-heavy layers (NDVI loss, MNDWI std when mostly stable):
uses p90 of non-zero values as upper bound.

### Inversion
EAD: inverted (low elevation = high risk)
Plan curvature: inverted (convergent = negative = erosion-prone)

### Spatial smoothing
Gaussian sigma=1 pixel (10m) applied after weighted sum
to remove salt-and-pepper noise while preserving spatial patterns.

### Classification thresholds
Absolute thresholds calibrated to Scheldt macrotidal context:
Low <0.25 / Medium 0.25–0.35 / High 0.35–0.45 / Critical >0.45

## 7. Known Limitations

See README.md Caveats section.
