"""
Western Scheldt — DFM Run Visualisation
========================================
Produces three figures from the 31-day DFM output:

1. dfm_map_snapshots.png  — 6-panel map of water level at selected timesteps
                             showing tidal channel morphology
2. dfm_timeseries.png     — Water level time series at 5 obs stations
                             with tidal range amplification upstream
3. dfm_bedlevel.png       — Bed level map showing channel/flat morphology

Usage:
    python visualise_dfm.py
    python visualise_dfm.py --base dfm_clean/dflowfm
"""

import os
os.environ["CPL_LOG"]   = "/dev/null"
_RASTERIO_PROJ = "/home/ulg/mast/eivanov/Yoda/lib/python3.10/site-packages/rasterio/proj_data"
_PYPROJ_DATA   = "/home/ulg/mast/eivanov/.conda/envs/Yoda/lib/python3.10/site-packages/pyproj/proj_dir/share/proj"
os.environ["PROJ_DATA"] = _RASTERIO_PROJ
os.environ["PROJ_LIB"]  = _RASTERIO_PROJ
import pyproj; pyproj.datadir.set_data_dir(_PYPROJ_DATA)

import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from pathlib import Path

try:
    import netCDF4 as nc
except ImportError:
    sys.exit("pip install netCDF4")

# ── CONFIG ────────────────────────────────────────────────────────────────────

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="dfm_clean/dflowfm",
                help="Path to dflowfm directory")
args = ap.parse_args()

BASE    = Path(args.base)
OUT_DIR = BASE / "DFM_OUTPUT_WesternScheldt"
NET     = BASE / "WesternScheldt_net_v3.nc"
MAP_NC  = OUT_DIR / "WesternScheldt_map.nc"
HIS_NC  = OUT_DIR / "WesternScheldt_his.nc"
FIG_DIR = BASE.parent.parent / "data/scheldt" if (BASE.parent.parent / "data").exists() \
          else Path(".")

STATION_NAMES = ["Vlissingen", "Terneuzen", "Hansweert", "Bath", "Kallosluis"]
DARK_BG  = "#0e0e0e"
DARK_AX  = "#1a1a1a"
ACCENT   = "#42A5F5"

# Zoom to satellite analysis AOI — chops open sea and polder buffer
AOI_X = (14000, 65000)
AOI_Y = (372000, 397000)
# Depth display cap: values above this are saturated white (deep channel)
# Set lower than true max to reveal intertidal detail
DEPTH_CAP = 10.0   # m — navigation channel ~17m but flats ~1-3m
BED_ZMIN  = -20.0  # m NAP — clip deep channel for bed level display
BED_ZMAX  =   5.0  # m NAP — clip high polders

# ── LOAD MESH ────────────────────────────────────────────────────────────────

print("Loading mesh ...")
with nc.Dataset(str(NET)) as ds:
    node_x  = np.array(ds["mesh2d_node_x"][:])
    node_y  = np.array(ds["mesh2d_node_y"][:])
    face_nodes = np.array(ds["mesh2d_face_nodes"][:]) - 1  # 0-based

# Build triangulation from face_nodes (may be quads — triangulate)
tris = []
for face in face_nodes:
    valid = face[face >= 0]
    if len(valid) == 3:
        tris.append(valid)
    elif len(valid) == 4:
        tris.append([valid[0], valid[1], valid[2]])
        tris.append([valid[0], valid[2], valid[3]])
tris = np.array(tris)
triang = mtri.Triangulation(node_x, node_y, tris)
print(f"  Mesh: {len(node_x)} nodes, {len(tris)} triangles")

# ── LOAD MAP OUTPUT ──────────────────────────────────────────────────────────

print("Loading map output ...")
with nc.Dataset(str(MAP_NC)) as ds:
    t_map   = np.array(ds["time"][:])              # seconds since RefDate
    wl_map  = np.array(ds["mesh2d_s1"][:])         # (time, face)
    bl_face = np.array(ds["mesh2d_flowelem_bl"][:]) # (face,) bed level
    fx      = np.array(ds["mesh2d_face_x"][:])
    fy      = np.array(ds["mesh2d_face_y"][:])

t_hr = t_map / 3600.0
print(f"  Map: {len(t_map)} timesteps, "
      f"WL range [{wl_map.min():.2f},{wl_map.max():.2f}]m")
print(f"  BL range: [{bl_face.min():.2f},{bl_face.max():.2f}]m")

# ── LOAD HIS OUTPUT ──────────────────────────────────────────────────────────

print("Loading his output ...")
with nc.Dataset(str(HIS_NC)) as ds:
    t_his   = np.array(ds["time"][:])
    wl_his  = np.array(ds["waterlevel"][:])        # (time, station)

t_his_hr = t_his / 3600.0
print(f"  His: {len(t_his)} timesteps, {wl_his.shape[1]} stations")

# ── FIGURE 1: BED LEVEL MAP ───────────────────────────────────────────────────

print("\n[1] Bed level map ...")
fig, ax = plt.subplots(figsize=(14, 7), facecolor=DARK_BG)
ax.set_facecolor(DARK_BG)

# Interpolate face bed level to nodes for smooth tricontourf
# Use nearest face value per node
from scipy.spatial import cKDTree
tree = cKDTree(np.column_stack([fx, fy]))
_, idx = tree.query(np.column_stack([node_x, node_y]))
bl_node = bl_face[idx]
bl_node = np.clip(bl_node, BED_ZMIN, BED_ZMAX)

# Clip for display
vmin, vmax = np.percentile(bl_face, 2), np.percentile(bl_face, 98)
cmap = plt.cm.RdBu_r

tcf = ax.tricontourf(triang, bl_node, levels=50,
                     cmap=cmap, vmin=BED_ZMIN, vmax=BED_ZMAX)
cb = plt.colorbar(tcf, ax=ax, fraction=0.03, pad=0.02)
cb.set_label("Bed level (m NAP)", color="white", fontsize=9)
cb.ax.tick_params(colors="white", labelsize=8)

# Mark 0m NAP contour (approx MSL)
ax.tricontour(triang, bl_node, levels=[0.0],
              colors=["white"], linewidths=0.8, linestyles="--", alpha=0.6)

ax.set_xlim(*AOI_X); ax.set_ylim(*AOI_Y)
ax.set_xlabel("RD New X (m)", color="white", fontsize=9)
ax.set_ylabel("RD New Y (m)", color="white", fontsize=9)
ax.tick_params(colors="white", labelsize=8)
ax.spines[:].set_color("#333")
ax.set_title("Western Scheldt — Bed Level (EMODnet DTM 2024)\n"
             "White dashed = 0m NAP (approx MSL)  |  clipped [-20,+5]m",
             color="white", fontsize=10, fontweight="bold")

fig.tight_layout()
out = FIG_DIR / "dfm_bedlevel.png"
fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close(fig)
print(f"  ✓ {out}")

# ── FIGURE 2: WATER LEVEL TIME SERIES ────────────────────────────────────────

print("\n[2] Time series ...")
fig, ax = plt.subplots(figsize=(14, 5), facecolor=DARK_BG)
ax.set_facecolor(DARK_AX)

colors = ["#42A5F5", "#FF7043", "#66BB6A", "#EF5350", "#AB47BC"]
for i, (name, col) in enumerate(zip(STATION_NAMES, colors)):
    if i < wl_his.shape[1]:
        ax.plot(t_his_hr, wl_his[:, i], color=col, lw=0.9,
                label=name, alpha=0.9)

ax.axhline(0, color="#555", lw=0.8, ls="--")
ax.set_xlabel("Time (hours since 2020-01-01)", color="white", fontsize=9)
ax.set_ylabel("Water level (m NAP)", color="white", fontsize=9)
ax.tick_params(colors="white", labelsize=8)
ax.spines[:].set_color("#333")
ax.legend(fontsize=8, facecolor=DARK_AX, labelcolor="white",
          loc="upper right", framealpha=0.8)
ax.set_title("Water level time series — 31-day DFM run\n"
             "Western Scheldt observation stations",
             color="white", fontsize=10, fontweight="bold")

fig.tight_layout()
out = FIG_DIR / "dfm_timeseries_31d.png"
fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close(fig)
print(f"  ✓ {out}")

# ── FIGURE 3: MAP SNAPSHOTS ───────────────────────────────────────────────────

print("\n[3] Map snapshots ...")

# Pick 6 timesteps: HW, LW, flood, ebb at ~day 15 (mid spring tide)
# Find approximate high/low water near station 0 (Vlissingen) at day 15
t15_idx = np.argmin(np.abs(t_his_hr - 15*24))
search_start = t15_idx
search_end   = min(t15_idx + 50, len(t_his_hr))
wl_v = wl_his[search_start:search_end, 0]

# HW and LW indices in his
hw_rel = np.argmax(wl_v)
lw_rel = np.argmin(wl_v)
hw_t   = t_his_hr[search_start + hw_rel]
lw_t   = t_his_hr[search_start + lw_rel]

# Quarter-tidal-cycle offsets (~3h = M2/4)
snap_times_hr = [
    lw_t,
    lw_t + 3.1,
    hw_t,
    hw_t + 3.1,
    lw_t + 12.4,
    lw_t + 15.5,
]
snap_labels = ["Low water", "Flood (+3h)", "High water",
               "Ebb (+3h)", "Low water +12h", "Flood +15h"]

# Find nearest map timesteps
snap_map_idx = [np.argmin(np.abs(t_hr - t)) for t in snap_times_hr]

fig = plt.figure(figsize=(18, 10), facecolor=DARK_BG)
gs  = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.08)

wl_valid  = wl_map[wl_map > -100]
wl_vmin   = np.percentile(wl_valid, 2)
wl_vmax   = np.percentile(wl_valid, 98)
# Water depth colorscale: 0 to 95th percentile of positive depths
depth_all = wl_map - bl_face[np.newaxis, :]
depth_wet = depth_all[(wl_map > -100) & (depth_all > 0)]
depth_vmax = np.percentile(depth_wet, 95) if len(depth_wet) else 10.0

# Interpolate face wl to nodes for each snapshot
for i, (tidx, label) in enumerate(zip(snap_map_idx, snap_labels)):
    ax = fig.add_subplot(gs[i//3, i%3])
    ax.set_facecolor(DARK_BG)

    wl_face_t = wl_map[tidx, :].copy()
    dry_mask  = wl_face_t < -100
    wl_face_t[dry_mask] = np.nan
    wl_node_t = wl_face_t[idx]  # face→node

    # Compute water depth = wl - bed; mask dry cells (depth <= 0)
    depth_node = wl_node_t - bl_node
    dry_node   = ~np.isfinite(wl_node_t) | (depth_node <= 0)
    tri_mask   = np.any(dry_node[tris], axis=1)
    triang_t   = mtri.Triangulation(node_x, node_y, tris, mask=tri_mask)
    depth_plot = np.where(~dry_node, depth_node, 0.0)

    tcf = ax.tricontourf(triang_t, depth_plot, levels=40,
                         cmap="Blues_r", vmin=0, vmax=DEPTH_CAP)
    ax.set_xlim(*AOI_X); ax.set_ylim(*AOI_Y)

    # Inundation line: where wl > bed level (wet cells)
    # Shoreline contour (edge of wet region) already defined by tri_mask

    t_day = t_hr[tidx] / 24.0
    wl_valid_t = wl_face_t[np.isfinite(wl_face_t)]
    wl_str = f"[{wl_valid_t.min():.2f},{wl_valid_t.max():.2f}]m" if len(wl_valid_t) else "all dry"
    ax.set_title(f"{label}\nt = {t_day:.1f} d  |  WL={wl_str}",
                 color="white", fontsize=7.5, pad=3)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines[:].set_color("#333")

# Shared colorbar
sm = plt.cm.ScalarMappable(cmap="Blues_r",
     norm=mcolors.Normalize(vmin=0, vmax=DEPTH_CAP))
sm.set_array([])
cb = fig.colorbar(sm, ax=fig.axes, fraction=0.02, pad=0.02)
cb.set_label("Water depth (m)", color="white", fontsize=9)
cb.ax.tick_params(colors="white", labelsize=8)

fig.suptitle("Western Scheldt — Delft3D FM 2026.01  |  31-day run  |  "
             "EMODnet bathymetry + DB tidal forcing",
             color="white", fontsize=11, fontweight="bold", y=0.98)

out = FIG_DIR / "dfm_map_snapshots.png"
fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close(fig)
print(f"  ✓ {out}")

print("\n✓ All figures written.")
