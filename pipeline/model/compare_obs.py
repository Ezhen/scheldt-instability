"""
Western Scheldt — DFM vs Observations Comparison
==================================================
Compares DFM model output (his.nc) against RWS water level observations
from monitoring.db for January 2020 (simulation period).

Produces:
  dfm_obs_comparison.png   — 5-panel time series model vs obs per station
  dfm_obs_scatter.png      — scatter plots + RMSE/bias/r² per station
  dfm_obs_metrics.json     — skill metrics for all stations

Usage:
    python compare_obs.py
    python compare_obs.py --base dfm_clean/dflowfm --db data/scheldt/monitoring.db
"""

import os
_P = "/home/ulg/mast/eivanov/.conda/envs/Yoda/lib/python3.10/site-packages/pyproj/proj_dir/share/proj"
os.environ["PROJ_DATA"] = _P; os.environ["PROJ_LIB"] = _P
import pyproj; pyproj.datadir.set_data_dir(_P)

import sys
import json
import sqlite3
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

try:
    import netCDF4 as nc
except ImportError:
    sys.exit("pip install netCDF4")

# ── CONFIG ────────────────────────────────────────────────────────────────────

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="dfm_clean/dflowfm")
ap.add_argument("--db",   default="data/scheldt/monitoring.db")
args = ap.parse_args()

BASE    = Path(args.base)
DB_PATH = Path(args.db)
HIS_NC  = BASE / "DFM_OUTPUT_WesternScheldt/WesternScheldt_his.nc"
FIG_DIR = Path(".")

SIM_START = datetime(2020, 1, 1)
SIM_STOP  = datetime(2020, 2, 1)

# Station codes in monitoring.db → his.nc station index
# Order must match obs_points.xyn
STATIONS = {
    "Vlissingen":  {"db_code": "VLISSGN",  "his_idx": 0},
    "Terneuzen":   {"db_code": "TERNZN",   "his_idx": 1},
    "Hansweert":   {"db_code": "HANSWT",   "his_idx": 2},
    "Bath":        {"db_code": "BATH",     "his_idx": 3},
    "Kallosluis":  {"db_code": "KALLSLZS", "his_idx": 4},
}

DARK_BG = "#0e0e0e"
DARK_AX = "#1a1a1a"
COLORS  = {"model": "#42A5F5", "obs": "#FF7043"}

# ── LOAD MODEL OUTPUT ─────────────────────────────────────────────────────────

print("Loading model output (map.nc at corrected channel positions) ...")
MAP_NC = BASE / "DFM_OUTPUT_WesternScheldt/WesternScheldt_map.nc"

# Corrected station face indices — nearest deep channel cell (bed < -5m)
# Determined by find_channel_cells diagnostic (avoid intertidal sampling)
CORRECTED_XY = {
    "Vlissingen": (14750, 382250),
    "Terneuzen":  (26750, 381250),
    "Hansweert":  (36750, 383250),
    "Bath":       (55750, 381750),
    "Kallosluis": (60750, 377750),
}

from scipy.spatial import cKDTree

with nc.Dataset(str(MAP_NC)) as ds:
    t_sec  = np.array(ds["time"][:])
    fx     = np.array(ds["mesh2d_face_x"][:])
    fy     = np.array(ds["mesh2d_face_y"][:])
    bl_f   = np.array(ds["mesh2d_flowelem_bl"][:])
    wl_map = np.array(ds["mesh2d_s1"][:])      # (time, face)

tree = cKDTree(np.column_stack([fx, fy]))

# Build wl_mod array (time, station) sampled at corrected positions
face_indices = []
for name in STATIONS:
    xy = CORRECTED_XY[name]
    _, idx = tree.query(xy)
    face_indices.append(idx)
    print(f"  {name:12s}: face {idx}, bed={bl_f[idx]:.2f}m, "
          f"x={fx[idx]:.0f}, y={fy[idx]:.0f}")

wl_mod = wl_map[:, face_indices]   # (time, station)

t_mod = pd.DatetimeIndex([SIM_START + timedelta(seconds=float(s)) for s in t_sec])
print(f"  {len(t_mod)} map timesteps, {wl_mod.shape[1]} stations")
print(f"  Period: {t_mod[0]} → {t_mod[-1]}")

# ── LOAD OBSERVATIONS ─────────────────────────────────────────────────────────

print("\nLoading observations from monitoring.db ...")
if not DB_PATH.exists():
    sys.exit(f"DB not found: {DB_PATH}")

conn = sqlite3.connect(str(DB_PATH))
obs_data = {}

for name, cfg in STATIONS.items():
    code = cfg["db_code"]
    try:
        df = pd.read_sql_query(f"""
            SELECT o.datetime, o.value AS wl_m
            FROM observations o
            JOIN stations   st ON st.id = o.station_id
            JOIN parameters p  ON p.id  = o.parameter_id
            WHERE st.code = '{code}'
              AND p.code  = 'WATHTE'
              AND o.value IS NOT NULL
              AND o.datetime >= '2020-01-01'
              AND o.datetime <  '2020-02-01'
            ORDER BY o.datetime
        """, conn, parse_dates=["datetime"])
        # Strip timezone (obs in CET=UTC+1, model in UTC — align to UTC)
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_localize(None)
        df = df.set_index("datetime").rename(columns={"wl_m": "wl"})
        # Skip first 3 days — model spin-up from WaterLevIni=0
        spinup_end = SIM_START + timedelta(days=3)
        df = df[df.index >= pd.Timestamp(spinup_end)]
        obs_data[name] = df
        print(f"  {name:12s} ({code:8s}): {len(df):5d} obs after spin-up, "
              f"WL=[{df.wl.min():.2f},{df.wl.max():.2f}]m")
    except Exception as e:
        print(f"  {name}: DB query failed — {e}")
        obs_data[name] = pd.DataFrame()

conn.close()

# ── COMPUTE SKILL METRICS ─────────────────────────────────────────────────────

def skill_metrics(mod_series, obs_series):
    """Interpolate model to obs times, compute RMSE/bias/r²/skill."""
    obs_clean = obs_series.dropna()
    if len(obs_clean) < 10:
        return None

    # mod_series may be filtered (spin-up removed) — use its own index
    mod_t  = mod_series.index.astype(np.int64)
    mod_wl = mod_series.values

    # Only compare obs within model time range
    obs_clean = obs_clean[
        (obs_clean.index >= mod_series.index[0]) &
        (obs_clean.index <= mod_series.index[-1])
    ]
    if len(obs_clean) < 10:
        return None

    mod_vals = np.interp(
        obs_clean.index.astype(np.int64),
        mod_t,
        mod_wl
    )
    obs_vals = obs_clean.values

    diff  = mod_vals - obs_vals
    rmse  = float(np.sqrt(np.mean(diff**2)))
    bias  = float(np.mean(diff))
    mae   = float(np.mean(np.abs(diff)))
    ss    = 1 - np.sum(diff**2) / np.sum((obs_vals - obs_vals.mean())**2)

    # Correlation
    if np.std(mod_vals) > 0 and np.std(obs_vals) > 0:
        r = float(np.corrcoef(mod_vals, obs_vals)[0, 1])
    else:
        r = 0.0

    # Tidal range comparison
    mod_range = float(np.percentile(mod_vals, 95) - np.percentile(mod_vals, 5))
    obs_range = float(np.percentile(obs_vals, 95) - np.percentile(obs_vals, 5))

    return {
        "rmse": round(rmse, 4),
        "bias": round(bias, 4),
        "mae":  round(mae, 4),
        "r":    round(r, 4),
        "r2":   round(r**2, 4),
        "skill_score": round(float(ss), 4),
        "mod_range_m": round(mod_range, 3),
        "obs_range_m": round(obs_range, 3),
        "range_ratio": round(mod_range / obs_range if obs_range > 0 else 0, 3),
        "n_obs": len(obs_clean),
    }

print("\nComputing skill metrics ...")
metrics = {}
for name, cfg in STATIONS.items():
    idx = cfg["his_idx"]
    if idx >= wl_mod.shape[1]:
        print(f"  {name}: station index {idx} out of range")
        continue
    spinup_end = pd.Timestamp(SIM_START + timedelta(days=3))
    mod_s = pd.Series(wl_mod[:, idx], index=t_mod)
    mod_s = mod_s[mod_s.index >= spinup_end]
    if obs_data[name].empty:
        print(f"  {name}: no observations")
        continue
    m = skill_metrics(mod_s, obs_data[name]["wl"])
    if m:
        metrics[name] = m
        print(f"  {name:12s}: RMSE={m['rmse']:.3f}m  bias={m['bias']:+.3f}m  "
              f"r²={m['r2']:.3f}  range ratio={m['range_ratio']:.2f}  "
              f"skill={m['skill_score']:.3f}")

# Save metrics
out_json = FIG_DIR / "dfm_obs_metrics.json"
with open(out_json, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"  ✓ {out_json}")

# ── FIGURE 1: TIME SERIES COMPARISON ─────────────────────────────────────────

print("\n[1] Time series comparison ...")
n_sta = len(STATIONS)
fig, axes = plt.subplots(n_sta, 1, figsize=(16, 3*n_sta),
                          facecolor=DARK_BG, sharex=True)
fig.subplots_adjust(hspace=0.08)

for i, (name, cfg) in enumerate(STATIONS.items()):
    ax = axes[i]
    ax.set_facecolor(DARK_AX)
    idx = cfg["his_idx"]

    # Model
    if idx < wl_mod.shape[1]:
        ax.plot(t_mod, wl_mod[:, idx],
                color=COLORS["model"], lw=0.8, label="Model", alpha=0.9)

    # Observations
    if not obs_data[name].empty:
        ax.plot(obs_data[name].index, obs_data[name]["wl"],
                color=COLORS["obs"], lw=0.7, label="Obs (RWS)",
                alpha=0.85)

    # Metrics annotation
    if name in metrics:
        m = metrics[name]
        ax.text(0.01, 0.93,
                f"RMSE={m['rmse']:.3f}m  bias={m['bias']:+.3f}m  "
                f"r²={m['r2']:.3f}  range ratio={m['range_ratio']:.2f}",
                transform=ax.transAxes, color="white", fontsize=7.5,
                va="top", ha="left",
                bbox=dict(facecolor="#111", alpha=0.7, pad=2))

    ax.set_ylabel(f"{name}\n(m NAP)", color="white", fontsize=8)
    ax.tick_params(colors="white", labelsize=7)
    ax.spines[:].set_color("#333")
    ax.axhline(0, color="#555", lw=0.6, ls="--")
    if i == 0:
        ax.legend(fontsize=8, facecolor=DARK_AX, labelcolor="white",
                  loc="upper right", framealpha=0.8)

axes[-1].set_xlabel("Date (2020)", color="white", fontsize=9)
fig.suptitle("DFM Model vs RWS Observations — Western Scheldt January 2020",
             color="white", fontsize=11, fontweight="bold", y=0.995)

out = FIG_DIR / "dfm_obs_comparison.png"
fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close(fig)
print(f"  ✓ {out}")

# ── FIGURE 2: SCATTER PLOTS ───────────────────────────────────────────────────

print("\n[2] Scatter plots ...")
n_cols = 3
n_rows = (n_sta + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols,
                          figsize=(5*n_cols, 4.5*n_rows),
                          facecolor=DARK_BG)
axes = axes.ravel()

for i, (name, cfg) in enumerate(STATIONS.items()):
    ax = axes[i]
    ax.set_facecolor(DARK_AX)
    idx = cfg["his_idx"]

    if obs_data[name].empty or idx >= wl_mod.shape[1]:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                color="white", ha="center", va="center")
        continue

    obs_clean = obs_data[name]["wl"].dropna()
    mod_interp = np.interp(
        obs_clean.index.astype(np.int64),
        t_mod.astype(np.int64),
        wl_mod[:, idx]
    )

    # Density scatter
    ax.scatter(obs_clean.values, mod_interp,
               c=COLORS["model"], s=1.5, alpha=0.3, rasterized=True)

    # 1:1 line
    lim = max(abs(obs_clean.values).max(), abs(mod_interp).max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], color="white", lw=1.0,
            ls="--", alpha=0.6, label="1:1")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)

    if name in metrics:
        m = metrics[name]
        ax.text(0.04, 0.96,
                f"RMSE={m['rmse']:.3f}m\nbias={m['bias']:+.3f}m\n"
                f"r²={m['r2']:.3f}\nrange={m['range_ratio']:.2f}×",
                transform=ax.transAxes, color="white", fontsize=8,
                va="top", ha="left",
                bbox=dict(facecolor="#111", alpha=0.8, pad=3))

    ax.set_title(name, color="white", fontsize=9, fontweight="bold")
    ax.set_xlabel("Observed (m NAP)", color="white", fontsize=8)
    ax.set_ylabel("Modelled (m NAP)", color="white", fontsize=8)
    ax.tick_params(colors="white", labelsize=7)
    ax.spines[:].set_color("#333")

# Hide unused axes
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle("DFM vs Observations — Scatter  |  Western Scheldt Jan 2020",
             color="white", fontsize=11, fontweight="bold")
fig.tight_layout()

out = FIG_DIR / "dfm_obs_scatter.png"
fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close(fig)
print(f"  ✓ {out}")

print("\n✓ Comparison complete.")
print("\nSkill summary:")
print(f"{'Station':12s}  {'RMSE':>6}  {'Bias':>7}  {'r²':>6}  "
      f"{'Range ratio':>11}  {'Skill':>6}")
print("-" * 60)
for name, m in metrics.items():
    print(f"{name:12s}  {m['rmse']:6.3f}  {m['bias']:+7.3f}  "
          f"{m['r2']:6.3f}  {m['range_ratio']:11.2f}  "
          f"{m['skill_score']:6.3f}")
