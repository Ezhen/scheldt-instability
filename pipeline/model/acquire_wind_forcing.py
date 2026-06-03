"""
Western Scheldt — ERA5 Wind Forcing for Delft3D FM
====================================================
Downloads ERA5 10m wind (u, v components) for January 2020 over the
Western Scheldt domain and converts to DFM meteo forcing format.

ERA5 wind → DFM wind forcing pipeline:
  1. Download ERA5 netCDF via CDS API
  2. Reproject ERA5 grid (WGS84) to RD New (EPSG:28992)
  3. Write DFM .amu (u-wind) and .amv (v-wind) grid files
  4. Write .amgrid file (meteo grid definition)
  5. Update bnd.ext with wind forcing entry

DFM meteo format: ARCINFO ASCII grid, time-varying
  Header: grid definition (nrows, ncols, cellsize, corner coords)
  Data:   one 2D field per timestep

Expected impact on bank erosion simulation:
  - Storm surge reproduction (missing without wind)
  - Wave setup on intertidal flats
  - Wind-driven currents during SW storms (dominant January direction)

Requirements:
  pip install cdsapi netCDF4 numpy

CDS registration (free):
  https://cds.climate.copernicus.eu/user/register
  Then create ~/.cdsapirc with url and key

Usage:
    python acquire_wind_forcing.py
    python acquire_wind_forcing.py --year 2020 --month 01
    python acquire_wind_forcing.py --skip-download  # if ERA5 file exists
"""

import os
_P = "/home/ulg/mast/eivanov/.conda/envs/Yoda/lib/python3.10/site-packages/pyproj/proj_dir/share/proj"
_R = "/home/ulg/mast/eivanov/.conda/envs/Yoda/lib/python3.10/site-packages/rasterio/proj_data"
os.environ["CPL_LOG"]   = "/dev/null"
os.environ["PROJ_DATA"] = _R
os.environ["PROJ_LIB"]  = _R
import pyproj; pyproj.datadir.set_data_dir(_P)

import sys
import argparse
import warnings
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

try:
    import netCDF4 as nc
except ImportError:
    sys.exit("pip install netCDF4")

# ── CONFIG ────────────────────────────────────────────────────────────────────

ap = argparse.ArgumentParser()
ap.add_argument("--year",  default="2020")
ap.add_argument("--month", default="01")
ap.add_argument("--out",   default="dfm_clean/dflowfm",
                help="DFM input directory")
ap.add_argument("--era5",  default="data/scheldt/wind/era5_wind_jan2020.nc",
                help="ERA5 netCDF file path")
ap.add_argument("--skip-download", action="store_true",
                help="Skip CDS download if ERA5 file already exists")
ap.add_argument("--dt-hours", type=float, default=1.0,
                help="Output timestep in hours (default 1h)")
args = ap.parse_args()

OUT_DIR  = Path(args.out)
ERA5_DIR = Path(args.era5).parent
ERA5_DIR.mkdir(parents=True, exist_ok=True)
ERA5_NC  = Path(args.era5)

# ERA5 download domain — wider than model for interpolation buffer
ERA5_BBOX = [51.7, 3.0, 51.0, 4.6]   # N, W, S, E (WGS84)

# DFM meteo grid definition (RD New, EPSG:28992)
# Coarser than model mesh — ~5km grid sufficient for ERA5 resolution (~31km)
GRID_X_MIN =  -10000
GRID_X_MAX =   85000
GRID_Y_MIN =  365000
GRID_Y_MAX =  405000
GRID_DX    =    5000   # 5km
GRID_DY    =    5000

# Simulation reference
SIM_START  = datetime(2020, 1, 1, 0, 0, 0)
SIM_STOP   = datetime(2020, 2, 1, 0, 0, 0)

print("=== ERA5 Wind Forcing — Western Scheldt DFM ===\n")
print(f"Period: {SIM_START} → {SIM_STOP}")
print(f"ERA5 domain: {ERA5_BBOX}")
print(f"Output grid: RD New {GRID_DX}m × {GRID_DY}m")

# ── STEP 1: DOWNLOAD ERA5 ────────────────────────────────────────────────────

if ERA5_NC.exists() and args.skip_download:
    print(f"\n[1] Using existing ERA5 file: {ERA5_NC}")
elif ERA5_NC.exists():
    print(f"\n[1] ERA5 file found: {ERA5_NC} — skipping download")
    print(f"    (use --skip-download to suppress this message)")
else:
    print(f"\n[1] Downloading ERA5 wind data ...")
    try:
        import cdsapi
    except ImportError:
        print("  ✗ cdsapi not installed")
        print("    pip install cdsapi --break-system-packages")
        print("    Then create ~/.cdsapirc:")
        print("    url: https://cds.climate.copernicus.eu/api/v2")
        print("    key: YOUR-UID:YOUR-API-KEY")
        print("    (Register at https://cds.climate.copernicus.eu/user/register)")
        sys.exit(1)

    year  = args.year
    month = args.month
    import calendar
    n_days = calendar.monthrange(int(year), int(month))[1]

    c = cdsapi.Client()
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": [
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
            ],
            "year":  year,
            "month": month,
            "day":   [f"{d:02d}" for d in range(1, n_days+1)],
            "time":  [f"{h:02d}:00" for h in range(24)],
            "area":  ERA5_BBOX,    # N, W, S, E
            "format": "netcdf",
        },
        str(ERA5_NC)
    )
    print(f"  ✓ Downloaded: {ERA5_NC}")

# ── STEP 2: LOAD ERA5 DATA ────────────────────────────────────────────────────

print(f"\n[2] Loading ERA5 data ...")
with nc.Dataset(str(ERA5_NC)) as ds:
    # ERA5 variable names may vary by version
    # Try standard names
    u_var = None
    v_var = None
    for vname in ds.variables:
        if "u10" in vname.lower() or "u_component" in vname.lower():
            u_var = vname
        if "v10" in vname.lower() or "v_component" in vname.lower():
            v_var = vname

    if u_var is None or v_var is None:
        print(f"  Variables found: {list(ds.variables.keys())}")
        sys.exit("Could not find u10/v10 wind variables")

    print(f"  Wind variables: {u_var}, {v_var}")

    # Time
    t_nc   = nc.num2date(ds["time"][:], ds["time"].units,
                          only_use_cftime_datetimes=False)
    t_dt   = np.array([datetime(d.year, d.month, d.day, d.hour) for d in t_nc])

    # Coordinates
    lons = np.array(ds["longitude"][:])
    lats = np.array(ds["latitude"][:])

    # Wind components — shape (time, lat, lon) or (time, lat, lon)
    u10  = np.array(ds[u_var][:])
    v10  = np.array(ds[v_var][:])

    # Handle scale/offset if present
    if hasattr(ds[u_var], "scale_factor"):
        u10 = u10 * ds[u_var].scale_factor + ds[u_var].add_offset
        v10 = v10 * ds[v_var].scale_factor + ds[v_var].add_offset

    # Mask fill values
    fill = getattr(ds[u_var], "_FillValue", 9.969e+36)
    u10  = np.where(np.abs(u10) > 200, np.nan, u10.astype(np.float32))
    v10  = np.where(np.abs(v10) > 200, np.nan, v10.astype(np.float32))

print(f"  ERA5 shape: {u10.shape}  (time × lat × lon)")
print(f"  ERA5 time : {t_dt[0]} → {t_dt[-1]}")
print(f"  ERA5 lons : [{lons.min():.2f}, {lons.max():.2f}]°E")
print(f"  ERA5 lats : [{lats.min():.2f}, {lats.max():.2f}]°N")
wspd = np.sqrt(u10**2 + v10**2)
print(f"  Wind speed: [{np.nanmin(wspd):.1f}, {np.nanmax(wspd):.1f}] m/s")

# Filter to simulation period
t_mask = (t_dt >= SIM_START - timedelta(hours=1)) & \
         (t_dt <= SIM_STOP  + timedelta(hours=1))
t_dt   = t_dt[t_mask]
u10    = u10[t_mask, :, :]
v10    = v10[t_mask, :, :]
print(f"  Filtered: {len(t_dt)} timesteps for simulation period")

# ── STEP 3: BUILD DFM METEO GRID ─────────────────────────────────────────────

print(f"\n[3] Building DFM meteo grid (RD New) ...")
from pyproj import Transformer

# DFM grid centres in RD New
grid_x = np.arange(GRID_X_MIN + GRID_DX/2,
                   GRID_X_MAX, GRID_DX, dtype=np.float64)
grid_y = np.arange(GRID_Y_MIN + GRID_DY/2,
                   GRID_Y_MAX, GRID_DY, dtype=np.float64)
nx = len(grid_x)
ny = len(grid_y)

# Grid centres as 2D arrays
GX, GY = np.meshgrid(grid_x, grid_y)   # (ny, nx)

# Convert grid centres to WGS84 for ERA5 interpolation
tr_rd_to_wgs = Transformer.from_crs(28992, 4326, always_xy=True)
glon, glat = tr_rd_to_wgs.transform(GX.ravel(), GY.ravel())
glon = glon.reshape(ny, nx)
glat = glat.reshape(ny, nx)

print(f"  DFM grid: {ny} rows × {nx} cols")
print(f"  Grid lon : [{glon.min():.2f}, {glon.max():.2f}]°E")
print(f"  Grid lat : [{glat.min():.2f}, {glat.max():.2f}]°N")

# ── STEP 4: INTERPOLATE ERA5 → DFM GRID ──────────────────────────────────────

print(f"\n[4] Interpolating ERA5 → DFM grid ...")
from scipy.interpolate import RegularGridInterpolator

# ERA5 may have decreasing latitudes — ensure ascending order
lat_asc = lats[::-1] if lats[0] > lats[-1] else lats
flip_lat = lats[0] > lats[-1]

n_steps   = len(t_dt)
u10_grid  = np.full((n_steps, ny, nx), np.nan, dtype=np.float32)
v10_grid  = np.full((n_steps, ny, nx), np.nan, dtype=np.float32)

for i in range(n_steps):
    u_field = u10[i, ::-1, :] if flip_lat else u10[i, :, :]
    v_field = v10[i, ::-1, :] if flip_lat else v10[i, :, :]

    # Replace NaN with nearest valid
    u_field = np.where(np.isfinite(u_field), u_field, 0.0)
    v_field = np.where(np.isfinite(v_field), v_field, 0.0)

    interp_u = RegularGridInterpolator(
        (lat_asc, lons), u_field,
        method="linear", bounds_error=False, fill_value=None
    )
    interp_v = RegularGridInterpolator(
        (lat_asc, lons), v_field,
        method="linear", bounds_error=False, fill_value=None
    )

    pts = np.column_stack([glat.ravel(), glon.ravel()])
    u10_grid[i] = interp_u(pts).reshape(ny, nx)
    v10_grid[i] = interp_v(pts).reshape(ny, nx)

wspd_grid = np.sqrt(u10_grid**2 + v10_grid**2)
print(f"  Interpolated wind speed: [{wspd_grid.min():.1f}, {wspd_grid.max():.1f}] m/s")

# ── STEP 5: WRITE DFM METEO FILES ────────────────────────────────────────────

print(f"\n[5] Writing DFM meteo files ...")

def write_arcinfo_wind(path, u_or_v_stack, t_arr, x_llcorner, y_llcorner,
                       cellsize, quantity="x_wind", nodata=-999.0):
    """
    Write DFM meteo_on_equidistant_grid ASCII file (.amu or .amv).
    Format: global header once, then TIME blocks with data.
    Confirmed working format for DFM 1.2.184.
    """
    n_rows, n_cols = u_or_v_stack.shape[1], u_or_v_stack.shape[2]
    x_llc = x_llcorner + cellsize / 2.0
    y_llc = y_llcorner + cellsize / 2.0

    with open(str(path), "w") as f:
        # Global header — written once
        f.write("FileVersion  = 1.03\n")
        f.write("Filetype     = meteo_on_equidistant_grid\n")
        f.write(f"NODATA_value = {nodata:.1f}\n")
        f.write(f"n_cols       = {n_cols}\n")
        f.write(f"n_rows       = {n_rows}\n")
        f.write("grid_unit    = m\n")
        f.write(f"x_llcenter   = {x_llc:.1f}\n")
        f.write(f"y_llcenter   = {y_llc:.1f}\n")
        f.write(f"dx           = {cellsize:.1f}\n")
        f.write(f"dy           = {cellsize:.1f}\n")
        f.write("n_quantity   = 1\n")
        f.write(f"quantity1    = {quantity}\n")
        f.write("unit1        = m/s\n")
        f.write("\n")

        # Time blocks
        for i, t in enumerate(t_arr):
            t_hr = (t - SIM_START).total_seconds() / 3600.0
            f.write(f"TIME = {t_hr:.4f} hours since "
                    f"{SIM_START.strftime('%Y-%m-%d %H:%M:%S')} +00:00\n")
            # Rows north to south (flip y axis)
            grid_slice = u_or_v_stack[i, ::-1, :]
            for row in grid_slice:
                f.write(" ".join(f"{v:.4f}" for v in row) + "\n")

# Write u-wind (.amu)
amu_path = OUT_DIR / "wind_jan2020.amu"
write_arcinfo_wind(amu_path, u10_grid, t_dt,
                   GRID_X_MIN, GRID_Y_MIN, GRID_DX, quantity="x_wind")
print(f"  ✓ {amu_path.name}  ({len(t_dt)} timesteps)")

# Write v-wind (.amv)
amv_path = OUT_DIR / "wind_jan2020.amv"
write_arcinfo_wind(amv_path, v10_grid, t_dt,
                   GRID_X_MIN, GRID_Y_MIN, GRID_DY, quantity="y_wind")
print(f"  ✓ {amv_path.name}")

# ── STEP 6: WRITE WIND GRID FILE (.amgrid) ────────────────────────────────────

print(f"\n[6] Writing meteo grid definition ...")
amgrid_path = OUT_DIR / "wind_jan2020.amgrid"
with open(str(amgrid_path), "w") as f:
    f.write("### Wind meteo grid — Western Scheldt DFM\n")
    f.write("### ERA5 reanalysis, 10m wind, ECMWF\n")
    f.write("### Interpolated to RD New (EPSG:28992) 5km grid\n")
    f.write(f"x_llcenter = {GRID_X_MIN + GRID_DX/2:.1f}\n")
    f.write(f"y_llcenter = {GRID_Y_MIN + GRID_DY/2:.1f}\n")
    f.write(f"n_cols = {nx}\n")
    f.write(f"n_rows = {ny}\n")
    f.write(f"dx = {GRID_DX:.1f}\n")
    f.write(f"dy = {GRID_DY:.1f}\n")
    f.write("value_pos = CENTRE\n")
    f.write("x_unit = m\n")
    f.write("y_unit = m\n")
print(f"  ✓ {amgrid_path.name}")

# ── STEP 7: UPDATE bnd.ext ────────────────────────────────────────────────────

print(f"\n[7] Updating bnd.ext with wind forcing ...")
ext_path = OUT_DIR / "bnd.ext"

wind_block = """
[Meteo]
quantity              = airpressure_windx_windy
forcingFile           = wind_jan2020.amu
"""

# Check if wind already in bnd.ext
if ext_path.exists():
    with open(ext_path) as f:
        content = f.read()
    if "airpressure_windx_windy" not in content and "amu" not in content:
        # Append wind block
        with open(ext_path, "a") as f:
            f.write("\n[Meteo]\n")
            f.write("quantity              = windx\n")
            f.write(f"forcingFile           = {amu_path.name}\n")
            f.write("forcingFileType       = arcinfo\n")
            f.write("interpolationMethod   = linearSpaceTime\n")
            f.write("interpolationMethod   = linearSpaceTime\n\n")
            f.write("[Meteo]\n")
            f.write("quantity              = windy\n")
            f.write(f"forcingFile           = {amv_path.name}\n")
            f.write("forcingFileType       = arcinfo\n")
            f.write("interpolationMethod   = linearSpaceTime\n")
        print(f"  ✓ Wind blocks added to {ext_path.name}")
    else:
        print(f"  Wind already in {ext_path.name} — skipping")
else:
    print(f"  {ext_path.name} not found — skipping")

# ── STEP 8: UPDATE MDU ────────────────────────────────────────────────────────

print(f"\n[8] Checking MDU wind settings ...")
mdu_path = OUT_DIR / "WesternScheldt.mdu"
if mdu_path.exists():
    with open(mdu_path) as f:
        mdu = f.read()

    changes = []

    # Enable wind drag
    if "ICdtyp" not in mdu:
        mdu = mdu.replace(
            "[wind]",
            "[wind]\nICdtyp                = 2\n"
            "Cdbreakpoints         = 6.3d-4  7.23d-3\n"
            "Windspeedbreakpoints  = 0.      100.\n"
            "Rhoair                = 1.205\n"
            "Relativewind          = 0.\n"
            "Windpartialdry        = 1"
        )
        changes.append("Added wind drag coefficients")

    if changes:
        with open(mdu_path, "w") as f:
            f.write(mdu)
        print(f"  ✓ MDU updated: {', '.join(changes)}")
    else:
        print(f"  MDU already has wind settings")

# ── STEP 9: DIAGNOSTIC FIGURE ────────────────────────────────────────────────

print(f"\n[9] Generating wind diagnostic ...")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

DARK_BG = "#0e0e0e"

fig, axes = plt.subplots(3, 1, figsize=(14, 10), facecolor=DARK_BG,
                          sharex=True)
fig.subplots_adjust(hspace=0.15)

# Panel 1: U wind at domain centre
centre_j = ny // 2
centre_i = nx // 2
ax = axes[0]
ax.set_facecolor("#1a1a1a")
ax.plot(t_dt, u10_grid[:, centre_j, centre_i],
        color="#42A5F5", lw=0.8, label="U-wind (W→E)")
ax.axhline(0, color="#555", lw=0.6, ls="--")
ax.set_ylabel("U10 (m/s)", color="white", fontsize=9)
ax.tick_params(colors="white", labelsize=8)
ax.spines[:].set_color("#333")
ax.legend(fontsize=8, facecolor="#1a1a1a", labelcolor="white")

# Panel 2: V wind
ax = axes[1]
ax.set_facecolor("#1a1a1a")
ax.plot(t_dt, v10_grid[:, centre_j, centre_i],
        color="#FF7043", lw=0.8, label="V-wind (S→N)")
ax.axhline(0, color="#555", lw=0.6, ls="--")
ax.set_ylabel("V10 (m/s)", color="white", fontsize=9)
ax.tick_params(colors="white", labelsize=8)
ax.spines[:].set_color("#333")
ax.legend(fontsize=8, facecolor="#1a1a1a", labelcolor="white")

# Panel 3: Wind speed
ax = axes[2]
ax.set_facecolor("#1a1a1a")
wspd_centre = np.sqrt(u10_grid[:, centre_j, centre_i]**2 +
                      v10_grid[:, centre_j, centre_i]**2)
ax.fill_between(t_dt, 0, wspd_centre,
                color="#66BB6A", alpha=0.7, label="|U| (m/s)")
ax.axhline(10, color="#EF5350", lw=0.8, ls="--", alpha=0.7,
           label="Beaufort 6 (10 m/s)")
ax.set_ylabel("|Wind| (m/s)", color="white", fontsize=9)
ax.set_xlabel("Date (2020)", color="white", fontsize=9)
ax.tick_params(colors="white", labelsize=8)
ax.spines[:].set_color("#333")
ax.legend(fontsize=8, facecolor="#1a1a1a", labelcolor="white")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))

fig.suptitle(
    "ERA5 10m Wind — Western Scheldt Domain Centre\n"
    "January 2020 | ECMWF Reanalysis",
    color="white", fontsize=11, fontweight="bold"
)

out_png = ERA5_DIR / "era5_wind_diagnostic.png"
fig.savefig(str(out_png), dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close(fig)
print(f"  ✓ {out_png}")

print(f"""
{'='*55}
✓ Wind forcing complete

Files written:
  {amu_path}      (u-wind)
  {amv_path}      (v-wind)
  {amgrid_path}
  {out_png}
  {ext_path} updated

Next steps:
  1. Clear cache and rerun DFM:
       rm -f {OUT_DIR}/WesternScheldt.cache
       rm -f {OUT_DIR}/DFM_OUTPUT_WesternScheldt/*
       sbatch run_dfm.slurm

  2. Expected improvements:
     - Storm surge events reproduced (Jan 2020 had 3 surges)
     - Wind setup on intertidal flats during SW storms
     - More realistic bed shear stress during storms
     - Better RMSE at all stations during storm periods

  3. ERA5 resolution: ~31km — adequate for domain-scale wind
     For higher resolution consider KNMI HARMONIE (~2km)
     available on request from KNMI data portal
""")
