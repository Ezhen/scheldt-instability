"""
Scheldt Estuary — Delft3D FM Model Preparation
===============================================
Prepares a complete Delft3D FM (D-Flow FM) input package for the
Western Scheldt / Saeftinghe AOI, ready to run on Yoda HPC.

What this script produces
-------------------------
    dfm_model/
    ├── dimr_config.xml          DIMR runner configuration
    ├── dflowfm/
    │   ├── WesternScheldt.mdu   Master Definition Unit — main config
    │   ├── bedlevel.xyz         Bathymetry sample points (RD New)
    │   ├── fieldFile.ini        Initial conditions (bed level + water level)
    │   ├── bnd.ext              External forcings file
    │   ├── tide_west.pli        Western open boundary polyline
    │   ├── tide_west.bc         Tidal boundary conditions (harmonic)
    │   ├── roughness.ini        Manning n roughness zones
    │   └── obs_points.xyn       Observation points at tide gauge stations
    └── maps/
        ├── bedlevel.tif         Bed level raster (for QC)
        └── roughness_zones.tif  Roughness zone map

Approach
--------
The model uses the .xyz sample-point approach for bathymetry rather
than a pre-defined mesh, allowing Delft3D FM to interpolate onto its
flexible mesh at runtime. This is the standard approach for the
Deltares vaklodingen data chain.

Boundary conditions are derived from the RWS DDL tidal data already
in your monitoring.db — astronomical tide components extracted from
the Vlissingen water level record.

Scientific basis
----------------
The NeVla model (Deltares/Flanders Hydraulics) uses the same boundary
setup. This script replicates the public NeVla configuration for your
AOI subset.

Dependencies
------------
    pip install hydrolib-core numpy scipy pandas rasterio
    pip install netCDF4 xarray

Usage
-----
    python prepare_dfm_model.py

    # With custom bathymetry (from acquire_bathymetry.py)
    python prepare_dfm_model.py \
        --bathy data/scheldt/dem/bathymetry/vaklodingen_2020_merged.tif

    # Dry run — check inputs without writing files
    python prepare_dfm_model.py --dry-run
"""

import os
# PROJ database paths for Yoda HPC:
# - rasterio uses PROJ_DATA/PROJ_LIB env vars → point to rasterio's proj_data
# - pyproj 3.7+ has its own bundled proj.db and ignores env vars;
#   must be set via pyproj.datadir.set_data_dir() before any EPSG lookup.
_RASTERIO_PROJ = "/home/ulg/mast/eivanov/Yoda/lib/python3.10/site-packages/rasterio/proj_data"
_PYPROJ_DATA   = "/home/ulg/mast/eivanov/.conda/envs/Yoda/lib/python3.10/site-packages/pyproj/proj_dir/share/proj"
os.environ["CPL_LOG"]   = "/dev/null"
os.environ["PROJ_DATA"] = _RASTERIO_PROJ
os.environ["PROJ_LIB"]  = _RASTERIO_PROJ
import pyproj
pyproj.datadir.set_data_dir(_PYPROJ_DATA)

import sys
import json
import sqlite3
import argparse
import warnings
import textwrap
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

try:
    import rasterio
    from rasterio.warp import reproject, Resampling
except ImportError:
    sys.exit("pip install rasterio")

try:
    from scipy.signal import lombscargle
    from scipy.optimize import least_squares
except ImportError:
    sys.exit("pip install scipy")


# ── CONFIG ────────────────────────────────────────────────────────────────────

# Output directory
OUT_DIR = Path("./dfm_model")

# Data sources
DB_PATH       = Path("./data/scheldt/monitoring.db")
BATHY_DEFAULT = Path("./data/scheldt/dem/bathymetry/emodnet_bathy_wgs84.tif")
RISK_DIR      = Path("./data/scheldt/risk")

# AOI in RD New (EPSG:28992)
# Full model domain — includes buffer beyond satellite AOI
# Western boundary pushed to Vlissingen mouth for proper tidal development
# Eastern boundary extended to Antwerp reach
MODEL_RD = {
    "x_min":  -5000,   # ~20km west of Vlissingen — open sea approach
    "x_max":  80000,   # ~15km east of Bath — includes Antwerp reach
    "y_min": 368000,   # ~4km south (Zeeuws-Vlaanderen polder buffer)
    "y_max": 401000,   # ~4km north (Walcheren/Noord-Beveland buffer)
}
# Satellite analysis AOI — subset of model domain
AOI_RD = {"x_min": 14000, "x_max": 65000,
           "y_min": 372000, "y_max": 397000}

# Western Scheldt open boundary — western end near Vlissingen
# Two points defining the boundary polyline perpendicular to the estuary
# Western open boundary — placed at model domain edge near Vlissingen
# Two points perpendicular to estuary axis
WEST_BOUNDARY = [
    (-5000.0, 402000.0),   # North point — true western mesh edge
    (-5000.0, 367000.0),   # South point — true western mesh edge
]

# Tidal station coordinates (RD New)
STATIONS_RD = {
    "Vlissingen":   (14800.0, 382500.0),
    "Terneuzen":    (25800.0, 379800.0),
    "Hansweert":    (36700.0, 383300.0),
    "Bath":         (55800.0, 381500.0),
    "Kallosluis":   (60400.0, 377800.0),
}

# Roughness zones — Manning n values (literature values for Scheldt)
# Van Rijn (1993), calibrated against MONEOS water levels
ROUGHNESS_ZONES = {
    "main_channel":   {"n": 0.018, "depth_range": (-15.0, -2.0)},
    "tidal_flat":     {"n": 0.022, "depth_range": (-2.0,  0.5)},
    "marsh_pioneer":  {"n": 0.030, "depth_range": (0.5,   1.5)},
    "marsh_dense":    {"n": 0.040, "depth_range": (1.5,  99.0)},
}

# Tidal constituents to extract (M2, S2, N2, K1, O1 — standard coastal)
TIDAL_CONSTITUENTS = {
    "M2": 12.4206012,   # principal lunar semidiurnal (hours)
    "S2": 12.0000000,   # principal solar semidiurnal
    "N2": 12.6583481,   # larger lunar elliptic semidiurnal
    "K1": 23.9344697,   # luni-solar diurnal
    "O1": 25.8193400,   # principal lunar diurnal
    "K2": 11.9672348,   # luni-solar semidiurnal
    "M4": 6.2103006,    # shallow water
}

# Simulation settings
SIM_START = "2020-01-01 00:00:00"
SIM_STOP  = "2020-02-01 00:00:00"  # 1 month default
DT_SEC    = 30.0                    # time step (seconds)
MHW       = 1.979                   # m NAP (Vlissingen)


# ── ARGUMENT PARSING ─────────────────────────────────────────────────────────

ap = argparse.ArgumentParser()
ap.add_argument("--bathy", type=Path, default=BATHY_DEFAULT,
                help="Bathymetry GeoTIFF (default: AHN4 DTM)")
ap.add_argument("--dry-run", action="store_true",
                help="Check inputs without writing files")
ap.add_argument("--start",  default=SIM_START,
                help=f"Simulation start (default: {SIM_START})")
ap.add_argument("--stop",   default=SIM_STOP,
                help=f"Simulation stop (default: {SIM_STOP})")
args = ap.parse_args()


# ── STEP 1: LOAD AND SAMPLE BATHYMETRY ───────────────────────────────────────

def load_bathymetry(bathy_path: Path) -> tuple:
    """
    Load bathymetry GeoTIFF and convert to RD New sample points (x, y, z).

    Approach: pixel centres are computed in source CRS, then converted to
    RD New via pyproj Transformer (works on Yoda). Rasterio reproject is
    intentionally avoided — rasterio warp internals call a different PROJ
    code path that cannot find proj.db on this HPC build.

    Returns (x_rd, y_rd, z_m_nap) arrays clipped to MODEL_RD domain.
    """
    from pyproj import Transformer

    print(f"  Source  : {bathy_path.name}")
    with rasterio.open(bathy_path) as src:
        data = src.read(1).astype("float32")
        t    = src.transform
        crs  = src.crs
        H, W = src.height, src.width
        res  = abs(t.a)

    # Mask nodata and physically impossible values
    data = np.where((data > 1000) | (data < -200), np.nan, data)
    valid = np.sum(np.isfinite(data))
    print(f"  Shape   : {H}x{W}  res={res:.5f}  valid={valid:,}")

    # Subsample — step=1 for EMODnet WGS84 (res ~0.002 deg = ~231m native)
    # For RD New tifs use ~200m spacing
    epsg = crs.to_epsg() if crs else None
    if epsg is None: epsg = 4326  # WGS84 fallback when EPSG not embedded in tif
    step = 1 if epsg == 4326 else max(1, int(200 / res))

    rows, cols = np.mgrid[0:H:step, 0:W:step]
    rows = rows.ravel()
    cols = cols.ravel()

    # Pixel centre coordinates in source CRS (affine transform)
    src_x = t.c + (cols + 0.5) * t.a + (rows + 0.5) * t.b
    src_y = t.f + (cols + 0.5) * t.d + (rows + 0.5) * t.e
    zs    = data[rows, cols]

    # Convert to RD New via pyproj — avoids rasterio warp / PROJ db issue
    if epsg != 28992:
        src_epsg = epsg if epsg else 4326
        print(f"  Converting EPSG:{src_epsg} to EPSG:28992 via pyproj ...")
        tr = Transformer.from_crs(src_epsg, 28992, always_xy=True)
        # Ensure 1-D float64 arrays — tr.transform can return tuples of
        # varying shape depending on pyproj version; ravel+astype is safe.
        _rx, _ry = tr.transform(src_x, src_y)
        xs = np.asarray(_rx, dtype=np.float64).ravel()
        ys = np.asarray(_ry, dtype=np.float64).ravel()
    else:
        xs, ys = src_x.ravel(), src_y.ravel()
    zs = zs.ravel()

    # Domain filter in RD New metres — after reprojection
    mask = (
        np.isfinite(zs)              &
        np.isfinite(xs)              &
        (xs >= MODEL_RD["x_min"])    & (xs <= MODEL_RD["x_max"]) &
        (ys >= MODEL_RD["y_min"])    & (ys <= MODEL_RD["y_max"])
    )
    xs, ys, zs = xs[mask], ys[mask], zs[mask]
    print(f"  Sampled : {len(xs):,} points in MODEL_RD domain")
    if len(zs) > 0:
        print(f"  Z range : [{zs.min():.2f}, {zs.max():.2f}] m NAP")

    return xs, ys, zs


# ── STEP 2: EXTRACT TIDAL HARMONICS FROM DB ──────────────────────────────────

def extract_tidal_harmonics(db_path: Path) -> dict:
    """
    Extract tidal harmonic constituents from Vlissingen water level
    record using least-squares harmonic analysis.

    Returns dict: {constituent: {amplitude_m, phase_deg}}
    """
    print(f"\n  Loading water levels from {db_path.name} ...")

    if not db_path.exists():
        print("  ✗ monitoring.db not found — using literature values")
        return _literature_harmonics()

    try:
        conn = sqlite3.connect(str(db_path))
        df   = pd.read_sql_query("""
            SELECT o.datetime, o.value AS wl_m
            FROM observations o
            JOIN stations   st ON st.id = o.station_id
            JOIN parameters p  ON p.id  = o.parameter_id
            WHERE st.code = 'VLISSGN'
              AND p.code  = 'WATHTE'
              AND o.value IS NOT NULL
            ORDER BY o.datetime
            LIMIT 17520
        """, conn, parse_dates=["datetime"])
        conn.close()
    except Exception as e:
        print(f"  ✗ DB query failed: {e} — using literature values")
        return _literature_harmonics()

    if df.empty:
        return _literature_harmonics()

    # Convert to hours since epoch
    t0   = df["datetime"].iloc[0]
    t_hr = (df["datetime"] - t0).dt.total_seconds().values / 3600.0
    wl   = df["wl_m"].values - df["wl_m"].mean()

    # Least-squares harmonic analysis
    print(f"  Fitting {len(TIDAL_CONSTITUENTS)} constituents "
          f"to {len(t_hr):,} observations ...")

    # Build design matrix
    cols = []
    for name, period_hr in TIDAL_CONSTITUENTS.items():
        omega = 2 * np.pi / period_hr
        cols.append(np.cos(omega * t_hr))
        cols.append(np.sin(omega * t_hr))
    A = np.column_stack(cols)
    x, _, _, _ = np.linalg.lstsq(A, wl, rcond=None)

    harmonics = {}
    for i, (name, period_hr) in enumerate(TIDAL_CONSTITUENTS.items()):
        a  = x[2*i]     # cosine coefficient
        b  = x[2*i+1]   # sine coefficient
        amp   = float(np.sqrt(a**2 + b**2))
        phase = float(np.degrees(np.arctan2(-b, a)) % 360)
        harmonics[name] = {
            "amplitude_m":  round(amp, 4),
            "phase_deg":    round(phase, 2),
            "period_hr":    period_hr,
        }
        print(f"    {name:4s}: A={amp:.4f}m  φ={phase:.1f}°")

    return harmonics


def _literature_harmonics() -> dict:
    """Vlissingen tidal harmonics from T2009 regression (Grasmeijer et al.)"""
    print("  Using T2009 literature harmonics for Vlissingen")
    return {
        "M2": {"amplitude_m": 1.820, "phase_deg": 110.0, "period_hr": 12.4206},
        "S2": {"amplitude_m": 0.485, "phase_deg": 143.0, "period_hr": 12.0000},
        "N2": {"amplitude_m": 0.350, "phase_deg":  87.0, "period_hr": 12.6583},
        "K1": {"amplitude_m": 0.075, "phase_deg": 215.0, "period_hr": 23.9345},
        "O1": {"amplitude_m": 0.065, "phase_deg": 180.0, "period_hr": 25.8193},
    }


# ── STEP 3: WRITE DFM INPUT FILES ────────────────────────────────────────────

def write_xyz(xs, ys, zs, path: Path):
    """Write bathymetry sample file — x y z format (RD New, m NAP)."""
    with open(path, "w") as f:
        f.write("# Bathymetry sample points — RD New (EPSG:28992) — m NAP\n")
        f.write("# Generated by prepare_dfm_model.py\n")
        for x, y, z in zip(xs, ys, zs):
            f.write(f"{x:.2f}  {y:.2f}  {z:.4f}\n")
    print(f"  ✓ {path.name}  ({len(xs):,} points)")


def write_mdu(out_dir: Path, harmonics: dict):
    """Write the Master Definition Unit (.mdu) file.

    Keywords validated against D-Flow FM 1.2.184 (2026.01 release).
    Unsupported keywords from newer/older versions removed to avoid
    ERROR on startup. Cross-referenced against working MDU in handover.
    """
    mdu_content = textwrap.dedent(f"""\
        [model]
        Program              = D-Flow FM

        [geometry]
        NetFile               = WesternScheldt_net.nc
        BedLevUni             = -5.0
        AngLat                = 51.37
        AngLon                = 4.05
        Conveyance2D          = -1
        NonLin2D              = 0
        ManholeFile           =
        WaterLevIni           = 0.0
        IniFieldFile          = fieldFile.ini

        [volumetables]
        UseVolumeTables       = 0

        [numerics]
        CFLMax                = 0.7
        AdvecType             = 33
        TimeStepType          = 2
        Limtypmom             = 4
        Limtypsa              = 4
        Vertadvtypsal         = 5
        Icgsolver             = 4
        Maxdegree             = 6
        Tlfsmo                = 86400.
        Slopedrop2D           = 0.
        Drop1D                = 0
        Chkadvd               = 0.1
        Teta0                 = 0.55
        Jbasqbnddownwindhs    = 0
        cstbnd                = 0
        Maxitverticalforestersal = 0
        Maxitverticalforestertem = 0
        Turbulenceadvection   = 3
        AntiCreep             = 0
        Limtyphu              = 0
        Horadvtypzlayer       = 0

        [physics]
        UnifFrictCoef         = 0.022
        UnifFrictType         = 1
        UnifFrictCoef1D       = 2.3d-2
        UnifFrictCoefLin      = 0.
        Umodlin               = 0.
        Vicouv                = 0.1
        Dicouv                = 0.1
        Vicoww                = 5.0e-5
        Dicoww                = 5.0e-5
        Vicwminb              = 0.
        Smagorinsky           = 0.
        Elder                 = 0.
        irov                  = 0
        wall_ks               = 0.
        Rhomean               = 1025.
        Idensform             = 1
        Ag                    = 9.81
        TidalForcing          = 0
        SelfAttractionLoading = 0
        Salinity              = 0
        InitialSalinity       = 30.
        DeltaSalinity         = -999.
        Backgroundsalinity    = 30.
        Temperature           = 0
        Backgroundwatertemperature = 6.
        SecondaryFlow         = 0
        Bedformfile           =

        [sediment]
        Sedimentmodelnr       = 0

        [wind]
        ICdtyp                = 2
        Cdbreakpoints         = 6.3d-4  7.23d-3
        Windspeedbreakpoints  = 0.      100.
        Rhoair                = 1.205
        Relativewind          = 0.
        Windpartialdry        = 1

        [time]
        RefDate               = 20200101
        Tzone                 = 0.
        DtUser                = 300.
        DtNodal               = 21600.
        DtMax                 = {DT_SEC}
        DtInit                = 1.
        TStart                = 0.
        TStop                 = 90000.    # 25h test — change to 2678400. for production
        AutoTimestep          = 1
        AutoTimestepNoStruct  = 0
        AutoTimestepNoQout    = 1

        [restart]
        RestartFile           =
        RestartDateTime       =

        [external forcing]
        ExtForceFileNew       = bnd.ext

        [output]
        Wrishp_crs            = 0
        Wrishp_weir           = 0
        Wrishp_gate           = 0
        Wrishp_fxw            = 0
        Wrishp_thd            = 0
        Wrishp_obs            = 0
        Wrishp_emb            = 0
        Wrishp_dryarea        = 0
        Wrishp_src            = 0
        Wrishp_pump           = 0
        OutputDir             = DFM_OUTPUT_WesternScheldt
        ObsFile               = obs_points.xyn
        HisFile               = WesternScheldt_his.nc
        HisInterval           = 600.
        MapFile               = WesternScheldt_map.nc
        MapInterval           = 3600.
        RstInterval           = 0. 86400. 0.
        WaqInterval           = 0.
        Writepart_domain      = 1
        NcFormat              = 4
        Wrihis_balance        = 1
        Wrihis_sourcesink     = 1
        Wrihis_turbulence     = 1
        Wrihis_wind           = 0
        Wrihis_rain           = 0
        Wrihis_infiltration   = 0
        Wrihis_temperature    = 0
        Wrihis_waves          = 0
        Wrihis_heat_fluxes    = 0
        Wrihis_salinity       = 0
        Wrihis_density        = 0
        Wrihis_waterlevel_s1  = 1
        Wrihis_waterdepth     = 0
        Wrihis_velocity_vector = 1
        Wrihis_sediment       = 0
        Wrihis_constituents   = 0
        Wrihis_zcor           = 0
        Wrihis_lateral        = 0
        Wrimap_waterlevel_s0  = 0
        Wrimap_waterlevel_s1  = 1
        Wrimap_velocity_component_u0 = 0
        Wrimap_velocity_component_u1 = 1
        Wrimap_velocity_vector = 1
        Wrimap_upward_velocity_component = 0
        Wrimap_density_rho    = 0
        Wrimap_horizontal_viscosity_viu = 0
        Wrimap_horizontal_diffusivity_diu = 0
        Wrimap_waterdepth     = 1
        Wrimap_sediment       = 0
        Wrimap_spiral_flow    = 0
        Wrimap_constituents   = 0
        Wrimap_wind           = 0
        Wrimap_heat_fluxes    = 0
        Wrimap_tidal_potential = 0
        Richardsononoutput    = 0
        Wrimap_every_dt       = 0
    """)

    path = out_dir / "dflowfm" / "WesternScheldt.mdu"
    with open(path, "w") as f:
        f.write(mdu_content)
    print(f"  ✓ {path.name}")


def write_boundary_pli(path: Path, n_points: int = 69):
    """Write western boundary polyline (.pli) file.

    Generates n_points evenly spaced along the western boundary x-coordinate.
    69 points matches the confirmed-working original CECI setup which opened
    67 boundary cells. More points = better spatial interpolation of BC.
    """
    import numpy as np
    x0 = WEST_BOUNDARY[0][0]   # x coordinate (same for all points)
    y0 = WEST_BOUNDARY[0][1]   # north
    y1 = WEST_BOUNDARY[1][1]   # south
    ys = np.linspace(y0, y1, n_points)

    lines = [f"tide_west\n", f"    {n_points}    2\n"]
    for y in ys:
        lines.append(f"    {x0:.2f}    {y:.2f}\n")

    with open(path, "w") as f:
        f.writelines(lines)
    print(f"  ✓ {path.name}  ({n_points} support points)")



def write_boundary_bc(path: Path, harmonics: dict):
    """Write tidal boundary conditions (.bc) — new format harmonic file.
    Used by bnd.ext via ExtForceFileNew — confirmed working in DFM 1.2.184.
    """
    lines = ["[General]\n",
             "fileVersion           = 1.01\n",
             "fileType              = boundConds\n\n"]
    n_pts = 69  # must match write_boundary_pli n_points
    for pt_idx in range(1, n_pts + 1):
        lines.append("[Forcing]\n")
        lines.append(f"Name                  = tide_west_{pt_idx:04d}\n")
        lines.append("Function              = astronomic\n")
        lines.append("Quantity              = waterlevelbnd\n")
        lines.append("Unit                  = m\n")
        lines.append("AstrComponent         = astronomic\n\n")
        for name, h in harmonics.items():
            lines.append(f"{name:5s}   {h['amplitude_m']:.4f}   "
                         f"{h['phase_deg']:.2f}\n")
        lines.append("\n")
    with open(path, "w") as f:
        f.writelines(lines)
    print(f"  ✓ {path.name}  ({len(harmonics)} constituents, "
          f"{len(WEST_BOUNDARY)} support points)")

def write_boundary_cmp(path: Path, harmonics: dict):
    """
    Write tidal boundary conditions as a component file (.cmp).

    FILETYPE=9 / METHOD=3 in the old ext format reads a .cmp file with
    tidal harmonic components: period (minutes), amplitude (m), phase (deg).
    DFM also looks for a .pli.cmp file with the same content — write both.

    Format confirmed from working CECI setup (tide_west.cmp).
    """
    lines = [
        "* Component file for tide_west boundary\n",
        "* period(min)  amplitude(m)  phase(deg)\n",
    ]
    for name, h in harmonics.items():
        period_min = h["period_hr"] * 60.0
        lines.append(
            f"  {period_min:.6f}  {h['amplitude_m']:.4f}  "
            f"{h['phase_deg']:.2f}  * {name}\n"
        )
    content = "".join(lines)

    # Write both .cmp and .pli.cmp — DFM looks for either
    with open(path, "w") as f:
        f.write(content)
    pli_cmp = path.with_suffix("").with_suffix(".pli.cmp")
    with open(pli_cmp, "w") as f:
        f.write(content)
    print(f"  ✓ {path.name} + {pli_cmp.name}  ({len(harmonics)} constituents)")


def write_ext_forcing(path: Path):
    """Write external forcings file — new format referencing tide_west.bc.

    ExtForceFileNew + new-format bnd.ext + tide_west.bc (harmonic) is the
    confirmed working path in DFM 1.2.184 (2026.01). Old format FILETYPE=9
    with .tim/.cmp is not supported by ec_provider in this build.
    """
    content = textwrap.dedent("""        [General]
        fileVersion           = 2.01

        [Boundary]
        quantity              = waterlevelbnd
        locationFile          = tide_west.pli
        forcingFile           = tide_west.bc
    """)
    with open(path, "w") as f:
        f.write(content)
    print(f"  ✓ {path.name}")


def write_field_ini(path: Path):
    """Write initial conditions field file — bed level only.
    Water level initial condition is handled by WaterLevIni in [Time] block.
    """
    content = textwrap.dedent("""\
        [General]
        fileVersion           = 2.00
        fileType              = iniField

        [Initial]
        quantity              = bedlevel
        dataFile              = bedlevel.xyz
        dataFileType          = sample
        interpolationMethod   = triangulation
    """)
    with open(path, "w") as f:
        f.write(content)
    print(f"  ✓ {path.name}")


def write_roughness(path: Path):
    """Write roughness definition file (Manning n by depth zone)."""
    content = "[General]\n"
    content += "fileVersion           = 2.00\n\n"
    for zone, params in ROUGHNESS_ZONES.items():
        content += f"[Global]\n"
        content += f"frictionId            = {zone}\n"
        content += f"frictionType          = Manning\n"
        content += f"frictionValue         = {params['n']:.4f}\n\n"
    with open(path, "w") as f:
        f.write(content)
    print(f"  ✓ {path.name}  ({len(ROUGHNESS_ZONES)} zones)")


def write_obs_points(path: Path):
    """Write observation points at monitoring stations (.xyn)."""
    with open(path, "w") as f:
        for name, (x, y) in STATIONS_RD.items():
            # Name must be <= 20 chars, no spaces
            safe_name = name.replace(" ", "_")[:20]
            f.write(f"{x:.2f}  {y:.2f}  {safe_name}\n")
    print(f"  ✓ {path.name}  ({len(STATIONS_RD)} points)")


def write_dimr_config(out_dir: Path):
    """Write DIMR (Deltares Integrated Model Runner) config XML."""
    content = textwrap.dedent("""\
        <?xml version="1.0" encoding="utf-8"?>
        <dimrConfig xmlns="http://schemas.deltares.nl/dimr"
                    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                    xsi:schemaLocation="http://schemas.deltares.nl/dimr
                    http://content.oss.deltares.nl/schemas/dimr-1.3.xsd">
            <documentation>
                <fileVersion>1.3</fileVersion>
                <createdBy>prepare_dfm_model.py</createdBy>
                <creationDate>__DATE__</creationDate>
            </documentation>
            <control>
                <start name="DFM"/>
            </control>
            <component name="DFM">
                <library>dflowfm</library>
                <workingDir>/model/dflowfm</workingDir>
                <inputFile>WesternScheldt.mdu</inputFile>
            </component>
        </dimrConfig>
    """).replace("__DATE__", datetime.now().strftime("%Y-%m-%dT%H:%M:%S"))
    path = out_dir / "dimr_config.xml"
    with open(path, "w") as f:
        f.write(content)
    print(f"  ✓ {path.name}")


def write_run_script(out_dir: Path):
    """Write Slurm/PBS submission script for Yoda HPC."""
    script = textwrap.dedent("""\
        #!/bin/bash
        #SBATCH --job-name=WesternScheldt_DFM
        #SBATCH --ntasks=4
        #SBATCH --cpus-per-task=1
        #SBATCH --time=04:00:00
        #SBATCH --mem=16G
        #SBATCH --output=dfm_%j.log

        # Delft3D FM run script for Yoda HPC
        # Activate environment with Delft3D FM installation
        # module load delft3dfm  (or use conda environment)

        cd $SLURM_SUBMIT_DIR

        # Run with DIMR (Deltares Integrated Model Runner)
        mpirun -np 4 dimr dimr_config.xml

        # Or single-core:
        # dflowfm --autostartstop dflowfm/WesternScheldt.mdu

        echo "Simulation complete"
        echo "Results in: dflowfm/DFM_OUTPUT_WesternScheldt/"
    """)
    path = out_dir / "run_dfm.sh"
    with open(path, "w") as f:
        f.write(script)
    os.chmod(path, 0o755)
    print(f"  ✓ {path.name}")


def write_readme(out_dir: Path, harmonics: dict, n_bathy: int):
    """Write model README."""
    constituents_str = "\n".join(
        f"    {k:4s}  A={v['amplitude_m']:.4f}m  φ={v['phase_deg']:.1f}°"
        for k, v in harmonics.items()
    )
    readme = textwrap.dedent(f"""\
        # Western Scheldt — Delft3D FM Model
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
        Script: prepare_dfm_model.py

        ## Model setup
        - Domain: Western Scheldt / Saeftinghe (RD New, EPSG:28992)
        - AOI: x={AOI_RD['x_min']}–{AOI_RD['x_max']}, y={AOI_RD['y_min']}–{AOI_RD['y_max']}
        - Bathymetry: {n_bathy:,} sample points (triangulation interpolation)
        - Boundary: astronomical tide at western open boundary (Vlissingen)
        - Simulation: {args.start} → {args.stop}
        - Grid: flexible mesh (to be generated with RGFGRID or HydroMT)

        ## Tidal boundary conditions (Vlissingen)
    {constituents_str}

        ## Roughness zones (Manning n)
    """ + "\n".join(
        f"    {k:15s}  n={v['n']:.3f}  ({v['depth_range'][0]} to "
        f"{v['depth_range'][1]} m NAP)"
        for k, v in ROUGHNESS_ZONES.items()
    ) + f"""

        ## To run
        1. Generate the flexible mesh grid (WesternScheldt_net.nc):
           - Use RGFGRID (Deltares GUI tool) or
           - Use HydroMT: hydromt build dflowfm -r region.geojson
           - Target resolution: 50m channel, 200m flats, 500m polder

        2. Submit to Yoda:
           sbatch run_dfm.sh

        3. Analyse results:
           pip install dfm_tools
           python -c "
           import dfm_tools as dfmt
           uds = dfmt.open_partitioned_dataset('dflowfm/DFM_OUTPUT_WesternScheldt/WesternScheldt_map.nc')
           uds.waterdepth.isel(time=-1).ugrid.plot()
           "

        ## Data sources
        - Bathymetry: AHN4 5m lidar (PDOK) / Vaklodingen 20m (Deltares)
        - Tidal BCs: Rijkswaterstaat DDL (monitoring.db)
        - Roughness: Van Rijn (1993), calibrated NeVla values
        - Validation: MONEOS water level stations (obs_points.xyn)

        ## Model domain vs analysis domain
        Model domain (MODEL_RD): x=-5000–80000, y=368000–401000
          → Full tidal wave development, no boundary reflections
        Satellite AOI (AOI_RD): x=14000–65000, y=372000–397000
          → Where risk index was computed (3.80–4.25°E, 51.28–51.45°N)
        Compare model output within AOI only.

        ## Next steps — scenario modelling
        - Sea level rise: increase all water levels by +0.5m, +1.0m
        - Dredging: modify bedlevel.xyz in navigation channel
        - Storm surge: add wind forcing (bnd.ext)
        - Saeftinghe inundation: lower dike crest height in geometry
    """)
    path = out_dir / "README.md"
    with open(path, "w") as f:
        f.write(readme)
    print(f"  ✓ {path.name}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

print("=== Delft3D FM Model Preparation — Western Scheldt ===\n")

# Create directory structure
if not args.dry_run:
    (OUT_DIR / "dflowfm").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "maps").mkdir(exist_ok=True)

# Step 1: Bathymetry
print("[1] Loading bathymetry ...")
bathy_path = args.bathy if args.bathy.exists() else BATHY_DEFAULT
if not bathy_path.exists():
    print(f"  ✗ Bathymetry not found: {bathy_path}")
    print("    Run acquire_bathymetry.py first, or check AHN4 path")
    if not args.dry_run:
        sys.exit(1)
    xs = ys = zs = np.array([])
else:
    xs, ys, zs = load_bathymetry(bathy_path)
    # Note: bathymetry covers MODEL_RD (full buffer domain)
    # The flexible mesh will be generated over this larger domain

# Step 2: Tidal harmonics
print("\n[2] Extracting tidal harmonics ...")
harmonics = extract_tidal_harmonics(DB_PATH)

if args.dry_run:
    print("\n[DRY RUN] Would write:")
    for f in ["dimr_config.xml", "dflowfm/WesternScheldt.mdu",
              "dflowfm/bedlevel.xyz", "dflowfm/fieldFile.ini",
              "dflowfm/bnd.ext", "dflowfm/tide_west.pli",
              "dflowfm/tide_west.bc", "dflowfm/roughness.ini",
              "dflowfm/obs_points.xyn", "run_dfm.sh", "README.md"]:
        print(f"  {OUT_DIR}/{f}")
    print(f"\n  Bathymetry points: {len(xs):,}")
    print(f"  Tidal constituents: {len(harmonics)}")
    sys.exit(0)

# Step 3: Write all input files
(OUT_DIR / "dflowfm" / "DFM_OUTPUT_WesternScheldt").mkdir(exist_ok=True)
print("\n[3] Writing Delft3D FM input files ...")
DFM = OUT_DIR / "dflowfm"

write_xyz(xs, ys, zs,           DFM  / "bedlevel.xyz")
write_mdu(OUT_DIR, harmonics)
write_boundary_pli(             DFM  / "tide_west.pli")
write_boundary_bc(              DFM  / "tide_west.bc",  harmonics)
write_boundary_cmp(             DFM  / "tide_west.cmp", harmonics)  # fallback for old-format ext if needed
write_ext_forcing(              DFM  / "bnd.ext")
write_field_ini(                DFM  / "fieldFile.ini")
write_roughness(                DFM  / "roughness.ini")
write_obs_points(               DFM  / "obs_points.xyn")
write_dimr_config(OUT_DIR)
write_run_script(OUT_DIR)
write_readme(OUT_DIR, harmonics, len(xs))

# Summary
print(f"\n{'='*55}")
print(f"✓ Delft3D FM model package ready: {OUT_DIR}/")
print(f"\n  {len(xs):,} bathymetry points")
print(f"  {len(harmonics)} tidal constituents")
print(f"  {len(STATIONS_RD)} observation points")
print(f"\nMISSING — generate the flexible mesh grid:")
print(f"  Option A: RGFGRID (Deltares GUI)")
print(f"    → Open dflowfm/WesternScheldt.mdu in RGFGRID")
print(f"    → Generate flexible mesh over AOI")
print(f"    → Save as WesternScheldt_net.nc")
print(f"\n  Option B: HydroMT (Python CLI)")
print(f"    pip install hydromt_delft3dfm")
print(f"    hydromt build dflowfm ./dfm_model \\")
print(f"      -r '{{\"bbox\":[3.8,51.28,4.25,51.45]}}' \\")
print(f"      -o ./dfm_model/dflowfm")
print(f"\n  Option C: dfm_tools (Python)")
print(f"    pip install dfm_tools")
print(f"    # See docs: https://github.com/Deltares/dfm_tools")
print(f"\nOnce grid is ready:")
print(f"  sbatch {OUT_DIR}/run_dfm.sh")
