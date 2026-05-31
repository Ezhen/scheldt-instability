# Data Sources & Access Instructions

## 1. Sentinel-2 L2A — CDSE openEO

**URL:** https://openeo.dataspace.copernicus.eu  
**Registration:** https://dataspace.copernicus.eu (free)  
**Authentication:** OAuth2 OIDC (browser-based, cached after first use)  
**Script:** `pipeline/acquisition/acquire_scheldt.py`

```python
import openeo
conn = openeo.connect("https://openeo.dataspace.copernicus.eu")
conn.authenticate_oidc()
```

Collection ID: `SENTINEL2_L2A`  
Max cloud cover filter: 20% (relax to 40% for winter scenes in Belgium)

---

## 2. Copernicus DEM GLO-30 — CDSE OData

**URL:** https://catalogue.dataspace.copernicus.eu/odata/v1  
**Authentication:** Same CDSE account, username/password token  
**Script:** `pipeline/terrain/acquire_dem.py`

Collection: `COP-DEM_GLO-30-DGED/2024_1`  
Note: EEA-10 (10m) requires institutional access — not publicly available.

---

## 3. AHN4 DTM — PDOK WCS

**URL:** https://service.pdok.nl/rws/ahn/wcs/v1_0  
**Authentication:** None required  
**Script:** `pipeline/terrain/acquire_dem_ahn4.py`

Coverage ID: `dtm_05m` (bare ground, 0.5m native, request at 5m)  
CRS: EPSG:28992 (RD New) — reproject to WGS84 for S2 alignment  
Coverage: Netherlands only

Manual test URL:
```
https://service.pdok.nl/rws/ahn/wcs/v1_0?SERVICE=WCS&VERSION=1.0.0
&REQUEST=GetCoverage&COVERAGE=dtm_05m
&BBOX=44264,366251,75952,385747&CRS=EPSG:28992&RESPONSE_CRS=EPSG:28992
&FORMAT=GEOTIFF_FLOAT32&WIDTH=500&HEIGHT=308
```

---

## 4. Vlissingen Tide Gauge — Rijkswaterstaat DDL

**URL:** https://waterwebservices.rijkswaterstaat.nl  
**Package:** `pip install ddlpy`  
**Authentication:** None required  
**Script:** `pipeline/tides/acquire_tides.py`

Station: VLISSGN (Vlissingen)  
Parameter: WATHTE (water level)  
Unit: cm NAP → divide by 100 for metres  
Column name (ddlpy v2+): `Meetwaarde.Waarde_Numeriek`  
Sampling: 10-minute observations  

Note: ddlpy station codes changed in v2. Station is found via
`locations["Naam"].str.contains("Vlissingen")` filter if VLISSGN
direct lookup fails.

---

## 5. Yoda HPC Notes

```bash
# Activate environment
conda activate Yoda

# Load X11 for matplotlib display
module load X11

# Suppress GDAL/PROJ warnings
export CPL_LOG=/dev/null

# PROJ conflict workaround
# Never use rasterio.CRS.from_epsg() directly
# Use string "EPSG:XXXX" or rioxarray.rio.reproject() instead
```
