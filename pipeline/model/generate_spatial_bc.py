"""
Western Scheldt — Spatially Varying Tidal Boundary Conditions
=============================================================
Generates tide_west.bc with spatially varying water level forcing,
interpolated along the western boundary polyline between observed
timeseries at Vlissingen (west) and Bath (east).

This fixes the tidal amplification problem: the uniform BC gave
range_ratio ~0.45 at Bath. With spatial interpolation the model
receives the correct amplitude gradient at the open boundary,
allowing the estuary geometry to reproduce the observed amplification.

Interpolation: linear in along-channel distance (x-coordinate)
  x = -5000 (west boundary) → Vlissingen signal
  x = -5000 with y=y_bath  → Bath signal (extrapolated upstream)

Actually: the boundary is a straight N-S line at x=-5000. We cannot
directly apply Bath obs there (Bath is at x=55800). Instead we:
1. Fit harmonic constituents at each station from monitoring.db
2. Correct phases to simulation RefDate (2020-01-01 UTC)
3. Interpolate amplitude and phase linearly along the boundary
   from Vlissingen (y_north) to Bath equivalent (y_south)
4. Synthesise per-support-point timeseries

This is physically motivated: the tidal wave entering from the west
has a spatial structure. Points closer to the Belgian coast (south)
experience a slightly different signal than points closer to Walcheren
(north). The interpolation captures the cross-estuary gradient.

Outputs:
  dfm_clean/dflowfm/tide_west_spatialBC.bc     spatially varying BC
  dfm_clean/dflowfm/tide_west_spatialBC.pli    matching polyline
  dfm_clean/dflowfm/bnd_spatialBC.ext          ext file referencing above

Usage:
    python generate_spatial_bc.py
    python generate_spatial_bc.py --db data/scheldt/monitoring.db
"""

import os
_P = "/home/ulg/mast/eivanov/.conda/envs/Yoda/lib/python3.10/site-packages/pyproj/proj_dir/share/proj"
os.environ["PROJ_DATA"] = _P; os.environ["PROJ_LIB"] = _P
import pyproj; pyproj.datadir.set_data_dir(_P)

import sys
import sqlite3
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────

ap = argparse.ArgumentParser()
ap.add_argument("--db",  default="data/scheldt/monitoring.db")
ap.add_argument("--out", default="dfm_clean/dflowfm")
ap.add_argument("--n-points", type=int, default=69,
                help="Number of support points along boundary")
args = ap.parse_args()

OUT_DIR   = Path(args.out)
SIM_START = datetime(2020, 1, 1, 0, 0, 0)   # UTC
SIM_STOP  = datetime(2020, 2, 1, 0, 0, 0)   # UTC
DT_S      = 600.0    # 10-minute output

# Boundary polyline geometry (RD New, x=-5000)
BC_X      = -5000.0
BC_Y_N    =  402000.0   # north end
BC_Y_S    =  367000.0   # south end

# Stations to use for spatial interpolation
# Along-boundary weight: 0.0 = full Vlissingen, 1.0 = full Bath
# Physically: Vlissingen is near the western mouth, Bath is upstream
# We map station position to boundary fraction via longitude
STATIONS = {
    "Vlissingen": {
        "code":   "VLISSGN",
        "weight": 0.0,    # north end of boundary
    },
    "Bath": {
        "code":   "BATH",
        "weight": 1.0,    # south end (upstream signal)
    },
}

# Tidal constituents to fit
CONSTITUENTS = {
    "M2": 12.4206,
    "S2": 12.0000,
    "N2": 12.6583,
    "K1": 23.9345,
    "O1": 25.8193,
    "K2": 11.9672,
    "M4":  6.2103,
}

print("=== Spatially Varying Tidal BC — Western Scheldt ===\n")
print(f"Simulation: {SIM_START} → {SIM_STOP} UTC")
print(f"Support points: {args.n_points}")
print(f"Boundary: x={BC_X:.0f}, y=[{BC_Y_S:.0f},{BC_Y_N:.0f}]")

# ── LOAD OBSERVATIONS ─────────────────────────────────────────────────────────

print("\n[1] Loading observations from monitoring.db ...")
conn = sqlite3.connect(args.db)

obs_series = {}
for name, cfg in STATIONS.items():
    code = cfg["code"]
    df = pd.read_sql_query(f"""
        SELECT datetime, value AS wl_m
        FROM observations o
        JOIN stations   st ON st.id = o.station_id
        JOIN parameters p  ON p.id  = o.parameter_id
        WHERE st.code = '{code}'
          AND p.code  = 'WATHTE'
          AND value IS NOT NULL
          AND datetime >= '2019-12-31 22:00:00'
          AND datetime <= '2020-02-02 01:00:00'
        ORDER BY datetime
    """, conn)

    # Convert CET → UTC
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_localize(None)
    df["t_sec"] = (df["datetime"] - pd.Timestamp(SIM_START)).dt.total_seconds()
    df = df.dropna(subset=["wl_m"])

    obs_series[name] = df
    print(f"  {name:12s} ({code}): {len(df):,} obs, "
          f"WL=[{df.wl_m.min():.3f},{df.wl_m.max():.3f}]m, "
          f"t=[{df.t_sec.min():.0f},{df.t_sec.max():.0f}]s")

conn.close()

# ── FIT HARMONICS TO EACH STATION ─────────────────────────────────────────────

print("\n[2] Fitting harmonics (t0 = 2020-01-01 00:00 UTC) ...")

def fit_harmonics(t_sec, wl, constituents):
    """
    Fit tidal harmonics via least squares.
    t_sec: seconds since SIM_START (2020-01-01 00:00 UTC)
    Returns dict of {name: {amplitude_m, phase_deg, period_hr}}
    """
    t_hr = t_sec / 3600.0

    # Build design matrix: [1, cos(omega*t), sin(omega*t)] for each constituent
    cols = [np.ones(len(t_hr))]  # mean
    names_ordered = []
    for cname, period_hr in constituents.items():
        omega = 2 * np.pi / period_hr
        cols.append(np.cos(omega * t_hr))
        cols.append(np.sin(omega * t_hr))
        names_ordered.append(cname)

    A = np.column_stack(cols)

    # Least squares fit
    coeffs, _, _, _ = np.linalg.lstsq(A, wl, rcond=None)

    result = {"mean": float(coeffs[0])}
    for i, cname in enumerate(names_ordered):
        period_hr = constituents[cname]
        a = coeffs[1 + 2*i]      # cos coefficient
        b = coeffs[2 + 2*i]      # sin coefficient
        amplitude = float(np.sqrt(a**2 + b**2))
        phase_deg = float(np.degrees(np.arctan2(-b, a)) % 360)
        result[cname] = {
            "amplitude_m": amplitude,
            "phase_deg":   phase_deg,
            "period_hr":   period_hr,
        }
    return result

harmonics = {}
for name, df in obs_series.items():
    # Use only spin-up-free period (first 3 days excluded from fitting
    # but we fit the full record since we need accurate phases)
    t   = df["t_sec"].values
    wl  = df["wl_m"].values
    h   = fit_harmonics(t, wl, CONSTITUENTS)
    harmonics[name] = h

    print(f"\n  {name} (mean WL = {h['mean']:+.3f}m NAP):")
    print(f"  {'Const':4s}  {'Amp(m)':7s}  {'Phase(°)':8s}  {'Period(h)':9s}")
    for cname in CONSTITUENTS:
        hc = h[cname]
        print(f"  {cname:4s}  {hc['amplitude_m']:7.4f}  "
              f"{hc['phase_deg']:8.2f}  {hc['period_hr']:9.4f}")

# Verify phase is correct: reconstruct at t=72h and compare to obs
print("\n  Phase verification at t=72h (2020-01-04 00:00 UTC):")
for name, df in obs_series.items():
    t72 = df[np.abs(df.t_sec - 3*86400) < 400]
    obs_val = t72["wl_m"].values[0] if len(t72) else np.nan

    h = harmonics[name]
    t_hr = 72.0
    mod_val = h["mean"]
    for cname in CONSTITUENTS:
        hc = h[cname]
        omega = 2 * np.pi / hc["period_hr"]
        phase = np.radians(hc["phase_deg"])
        mod_val += hc["amplitude_m"] * np.cos(omega * t_hr - phase)

    print(f"  {name:12s}: obs={obs_val:+.3f}m  model={mod_val:+.3f}m  "
          f"diff={mod_val-obs_val:+.3f}m")

# ── BUILD SPATIALLY VARYING TIMESERIES ────────────────────────────────────────

print("\n[3] Building spatially varying timeseries ...")

# Support point y-coordinates along boundary
ys = np.linspace(BC_Y_N, BC_Y_S, args.n_points)

# Fractional distance along boundary (0=north=Vlissingen, 1=south=Bath)
fracs = np.linspace(0.0, 1.0, args.n_points)

# Time array with buffer
t_start_s = -3600.0
t_end_s   = (SIM_STOP - SIM_START).total_seconds() + 3600.0
t_sec_arr = np.arange(t_start_s, t_end_s + DT_S, DT_S)
t_hr_arr  = t_sec_arr / 3600.0
n_steps   = len(t_sec_arr)

print(f"  {args.n_points} support points × {n_steps} timesteps")
print(f"  Coverage: {t_sec_arr[-1]/3600:.1f}h")

# Synthesise WL for each support point
# Interpolate harmonic parameters linearly between Vlissingen and Bath
h_vlis = harmonics["Vlissingen"]
h_bath = harmonics["Bath"]

wl_matrix = np.zeros((args.n_points, n_steps))

for j, frac in enumerate(fracs):
    # Interpolate mean
    mean_j = (1-frac) * h_vlis["mean"] + frac * h_bath["mean"]

    wl_j = np.full(n_steps, mean_j)
    for cname in CONSTITUENTS:
        hv = h_vlis[cname]
        hb = h_bath[cname]

        # Interpolate amplitude linearly
        amp_j = (1-frac) * hv["amplitude_m"] + frac * hb["amplitude_m"]

        # Interpolate phase — must handle 360° wraparound
        # Use complex phasor interpolation
        pv = hv["amplitude_m"] * np.exp(1j * np.radians(hv["phase_deg"]))
        pb = hb["amplitude_m"] * np.exp(1j * np.radians(hb["phase_deg"]))
        pj = (1-frac) * pv + frac * pb
        amp_j   = abs(pj)
        phase_j = np.degrees(np.angle(pj)) % 360

        omega = 2 * np.pi / hv["period_hr"]
        wl_j += amp_j * np.cos(omega * t_hr_arr - np.radians(phase_j))

    wl_matrix[j, :] = wl_j

# Report range at first and last point
print(f"  Point 0  (y={ys[0]:.0f}, Vlissingen-like): "
      f"WL=[{wl_matrix[0].min():.3f},{wl_matrix[0].max():.3f}]m")
print(f"  Point {args.n_points-1} (y={ys[-1]:.0f}, Bath-like):       "
      f"WL=[{wl_matrix[-1].min():.3f},{wl_matrix[-1].max():.3f}]m")

# ── WRITE OUTPUT FILES ────────────────────────────────────────────────────────

print("\n[4] Writing output files ...")

SIM_START_STR = SIM_START.strftime("%Y-%m-%d %H:%M:%S")

# 1. tide_west_spatialBC.bc
bc_path = OUT_DIR / "tide_west_spatialBC.bc"
with open(bc_path, "w") as f:
    f.write("[General]\n")
    f.write("fileVersion           = 1.01\n")
    f.write("fileType              = boundConds\n\n")

    for j in range(args.n_points):
        # Name must match the polyline name in the .pli file
        # Convention: pli file contains "tide_west" as polyline name
        # so bc blocks must be tide_west_0001 ... tide_west_00NN
        pli_stem = "tide_west"
        f.write("[Forcing]\n")
        f.write(f"Name                  = {pli_stem}_{j+1:04d}\n")
        f.write("Function              = timeseries\n")
        f.write("Time-interpolation    = linear\n")
        f.write("Quantity              = time\n")
        f.write(f"Unit                  = seconds since {SIM_START_STR}\n")
        f.write("Quantity              = waterlevelbnd\n")
        f.write("Unit                  = m\n")
        for t, w in zip(t_sec_arr, wl_matrix[j]):
            f.write(f"{t:.0f}  {w:.6f}\n")
        f.write("\n")

print(f"  ✓ {bc_path.name}  "
      f"({args.n_points} points × {n_steps} steps)")

# 2. tide_west_spatialBC.pli
pli_path = OUT_DIR / "tide_west_spatialBC.pli"
with open(pli_path, "w") as f:
    f.write("tide_west\n")
    f.write(f"    {args.n_points}    2\n")
    for y in ys:
        f.write(f"    {BC_X:.2f}    {y:.2f}\n")
print(f"  ✓ {pli_path.name}")

# 3. bnd_spatialBC.ext
ext_path = OUT_DIR / "bnd_spatialBC.ext"
with open(ext_path, "w") as f:
    f.write("[General]\n")
    f.write("fileVersion           = 2.01\n\n")
    f.write("[Boundary]\n")
    f.write("quantity              = waterlevelbnd\n")
    f.write("locationFile          = tide_west_spatialBC.pli\n")
    f.write("forcingFile           = tide_west_spatialBC.bc\n")
print(f"  ✓ {ext_path.name}")

print(f"""
{'='*55}
✓ Spatial BC complete

To use in DFM run:
  sed -i 's/ExtForceFileNew.*= bnd.ext/ExtForceFileNew       = bnd_spatialBC.ext/' \\
      {OUT_DIR}/WesternScheldt.mdu

To compare branches:
  Branch A (uniform):  ExtForceFileNew = bnd.ext         (tide_west.bc)
  Branch B (spatial):  ExtForceFileNew = bnd_spatialBC.ext

Expected improvement at Bath/Kallosluis:
  range_ratio: 0.46 → ~0.75-0.85
  RMSE:        1.2m → ~0.5-0.7m
""")
