"""
Western Scheldt — Tidal Forcing Variability & Bank Erosion Analysis
====================================================================
Analyses how spring-neap tidal variability drives spatial patterns of
bed shear stress and erosion potential on the Scheldt banks.

Four analyses:
  1. Spring vs neap shear stress maps + ratio
  2. Exceedance duration curves per ZES.1 ecotope zone
  3. Tidal stage-dependent stress (flood/HW/ebb/LW bins)
  4. Extreme event analysis (return levels, storm vs calm)

Scientific framing (IMDC 2016-2017 context):
  The spring-neap modulation controls whether intertidal banks are in
  the laagdynamisch or hoogdynamisch regime. Banks near the 0.8 m/s
  ZES.1 threshold oscillate between classes over the fortnightly cycle —
  these are the morphologically most active and ecologically most
  vulnerable zones.

  Spring-neap τ ratio > 3 identifies cells where neap conditions allow
  sediment deposition but spring conditions drive net erosion — the
  morphodynamic engine behind the plaatrand evolution documented in
  IMDC monitoring reports.

Outputs (all in --out directory):
  tidal_spring_tau.png          Mean τ during spring tide period
  tidal_neap_tau.png            Mean τ during neap tide period
  tidal_spring_neap_ratio.png   Spring/neap τ ratio — morphodynamically active zones
  tidal_exceedance_curves.png   Duration exceedance curves per ecotope zone
  tidal_stage_stress.png        τ binned by tidal stage (4 panels)
  tidal_storm_vs_calm.png       Jan 13-14 storm vs calm tidal conditions
  tidal_critical_duration.png   Hours/spring-tide above τ_cr thresholds
  tidal_forcing_summary.png     6-panel overview figure
  tidal_forcing_report.json     Numerical summary for each analysis

Usage:
    python tidal_forcing_analysis.py
    python tidal_forcing_analysis.py --base dfm_v5_final/dflowfm --spinup 3
"""

import os
_P = "/home/ulg/mast/eivanov/.conda/envs/Yoda/lib/python3.10/site-packages/pyproj/proj_dir/share/proj"
os.environ["PROJ_DATA"] = _P; os.environ["PROJ_LIB"] = _P
import pyproj; pyproj.datadir.set_data_dir(_P)

import sys, argparse, warnings, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")
try:
    import netCDF4 as nc
except ImportError:
    sys.exit("pip install netCDF4")

# ── CONFIG ────────────────────────────────────────────────────────────────────

ap = argparse.ArgumentParser()
ap.add_argument("--base",   default="dfm_v5_final/dflowfm")
ap.add_argument("--spinup", type=float, default=3.0)
ap.add_argument("--out",    default="data/scheldt/figures/tidal_forcing")
args = ap.parse_args()

BASE    = Path(args.base)
MAP_NC  = BASE / "DFM_OUTPUT_WesternScheldt/WesternScheldt_map.nc"
NET_NC  = next(BASE.glob("WesternScheldt_net_v*.nc"))
OUT_DIR = Path(args.out)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# January 2020 tidal periods (days since 2020-01-01)
# Spring tides: ~Jan 10-11 and ~Jan 25-26 (new/full moon)
# Neap tides:   ~Jan 3-4 and ~Jan 17-18 (quarter moon)
SPRING1 = (9.0,  13.0)   # first spring (Jan 10-13)
STORM   = (12.5, 14.5)   # Jan 13-14 storm event
NEAP1   = (2.5,  6.0)    # first neap (Jan 3-6)
NEAP2   = (16.5, 20.0)   # second neap (Jan 17-20)
SPRING2 = (24.0, 28.0)   # second spring (Jan 25-28)

# ZES.1 thresholds
TAU_SAND    = 0.15
TAU_MUD     = 0.40
TAU_MARSH   = 1.00
U_LOW       = 0.80   # laag/hoogdynamisch boundary

# Monitoring stations
STATIONS = {
    "Vlissingen":  (14743, 382503),
    "Terneuzen":   (26373, 381501),
    "Hansweert":   (36750, 383248),
    "Bath":        (55759, 381499),
    "Kallosluis":  (60765, 378104),
}

DARK_BG  = "#0e0e0e"
DARK_PAN = "#111111"
AOI_X    = (14000, 65000)
AOI_Y    = (372000, 397000)

print("=== Tidal Forcing Variability & Bank Erosion Analysis ===\n")
print(f"  Base   : {BASE}")
print(f"  Spinup : {args.spinup} days")
print(f"  Output : {OUT_DIR}")

# ── LOAD DATA ─────────────────────────────────────────────────────────────────

print("\n[1] Loading map.nc ...")
with nc.Dataset(str(MAP_NC)) as ds:
    t_sec  = np.array(ds["time"][:])
    fx     = np.array(ds["mesh2d_face_x"][:])
    fy     = np.array(ds["mesh2d_face_y"][:])
    bl     = np.array(ds["mesh2d_flowelem_bl"][:])
    wl_all = np.array(ds["mesh2d_s1"][:])
    ux_all = np.array(ds["mesh2d_ucx"][:])
    uy_all = np.array(ds["mesh2d_ucy"][:])
    has_tau = "mesh2d_taus" in ds.variables
    if has_tau:
        tau_all = np.array(ds["mesh2d_taus"][:])
        print("  Bed shear stress ✓")

dt_hr = np.median(np.diff(t_sec)) / 3600.0
t_hr  = t_sec / 3600.0
t_day = t_hr / 24.0
n_t, n_f = wl_all.shape
print(f"  {n_t} timesteps × {n_f:,} faces")
print(f"  Period: day {t_day[0]:.1f} → {t_day[-1]:.1f}")

wl_all = np.where(wl_all < -100, np.nan, wl_all)
ux_all = np.where(np.abs(ux_all) > 100, np.nan, ux_all)
uy_all = np.where(np.abs(uy_all) > 100, np.nan, uy_all)

# Spinup exclusion
spinup_mask = t_day >= args.spinup
t_day_fit   = t_day[spinup_mask]
wl  = wl_all[spinup_mask, :]
ux  = ux_all[spinup_mask, :]
uy  = uy_all[spinup_mask, :]
n_fit = spinup_mask.sum()

if has_tau:
    tau = tau_all[spinup_mask, :]
    tau = np.where(tau < 0, 0.0, tau)
else:
    rho, g, C = 1025.0, 9.81, 50.0
    tau = rho * g / C**2 * (ux**2 + uy**2)

depth = np.where(np.isfinite(wl), wl - bl[np.newaxis, :], np.nan)
wet   = depth > 0.05
inund = wet.mean(axis=0)
perm_dry = inund < 0.01
spd = np.sqrt(ux**2 + uy**2)
peak_spd = np.nanmax(spd, axis=0)

print(f"  Analysis: {n_fit} steps ({t_day_fit[-1]:.1f} days)")

# Tidal period masks (relative to analysis start)
def period_mask(t, t_start, t_end):
    return (t >= t_start) & (t <= t_end)

spring1_mask = period_mask(t_day_fit, *SPRING1)
spring2_mask = period_mask(t_day_fit, *SPRING2)
neap1_mask   = period_mask(t_day_fit, *NEAP1)
neap2_mask   = period_mask(t_day_fit, *NEAP2)
storm_mask   = period_mask(t_day_fit, *STORM)
spring_mask  = spring1_mask | spring2_mask
neap_mask    = neap1_mask | neap2_mask
calm_mask    = ~storm_mask  # non-storm timesteps

print(f"  Spring period: {spring_mask.sum()} steps")
print(f"  Neap period  : {neap_mask.sum()} steps")
print(f"  Storm window : {storm_mask.sum()} steps")

# ── COMPUTE INDICATORS ───────────────────────────────────────────────────────

print("\n[2] Computing spring/neap/storm indicators ...")

def masked_mean(arr, mask):
    m = np.broadcast_to(mask[:, np.newaxis], arr.shape)
    return np.nanmean(np.where(m, arr, np.nan), axis=0)

tau_spring = masked_mean(tau, spring_mask)
tau_neap   = masked_mean(tau, neap_mask)
tau_storm  = masked_mean(tau, storm_mask)
tau_calm   = masked_mean(tau, ~storm_mask)
tau_all_m  = np.nanmean(tau, axis=0)

with np.errstate(divide="ignore", invalid="ignore"):
    spring_neap_ratio = np.where(
        (tau_neap > 0.05) & np.isfinite(tau_spring),
        tau_spring / tau_neap,
        np.nan
    )
    storm_calm_ratio = np.where(
        (tau_calm > 0.05) & np.isfinite(tau_storm),
        tau_storm / tau_calm,
        np.nan
    )

# Tidal stage classification using local tidal range
# Slack = within 5% of local tidal range per cell — avoids over-classifying
wl_diff  = np.diff(wl, axis=0, prepend=wl[:1, :])
wl_range = (np.nanpercentile(wl, 95, axis=0) -
            np.nanpercentile(wl,  5, axis=0))
wl_range = np.where(wl_range > 0.1, wl_range, 0.5)   # minimum 0.5m
slack_thr = 0.05 * wl_range[np.newaxis, :]             # 5% of local range

stage_flood = wl_diff >  slack_thr    # actively rising
stage_ebb   = wl_diff < -slack_thr    # actively falling
stage_hw    = (np.abs(wl_diff) <= slack_thr) & (wl > 0)   # HW slack
stage_lw    = (np.abs(wl_diff) <= slack_thr) & (wl <= 0)  # LW slack

# For stage means: average tau only at timesteps/cells in that stage
# Use spatial mean of the 2D boolean mask to get a time mask
tau_flood = np.nanmean(np.where(stage_flood, tau, np.nan), axis=0)
tau_ebb   = np.nanmean(np.where(stage_ebb,   tau, np.nan), axis=0)
tau_hw    = np.nanmean(np.where(stage_hw,    tau, np.nan), axis=0)
tau_lw    = np.nanmean(np.where(stage_lw,    tau, np.nan), axis=0)

# Critical duration: hours per spring tide above thresholds
# Number of spring tides in analysis period
n_spring_tides = 2  # two spring tides in January
dt_h = dt_hr

def critical_hours(tau_arr, threshold, wet_arr, period_m):
    """Hours per spring tide above threshold during wet conditions."""
    wet_period = wet_arr & np.broadcast_to(
        period_m[:, np.newaxis], wet_arr.shape)
    exceed = (tau_arr > threshold) & wet_period
    total_hours = exceed.sum(axis=0) * dt_h
    return total_hours / n_spring_tides

crit_sand_spring  = critical_hours(tau, TAU_SAND,  wet, spring_mask)
crit_mud_spring   = critical_hours(tau, TAU_MUD,   wet, spring_mask)
crit_marsh_spring = critical_hours(tau, TAU_MARSH, wet, spring_mask)

print(f"  Spring τ range: [{np.nanmin(tau_spring):.3f}, {np.nanmax(tau_spring):.3f}] Pa")
print(f"  Neap   τ range: [{np.nanmin(tau_neap):.3f}, {np.nanmax(tau_neap):.3f}] Pa")
print(f"  Spring/neap ratio max: {np.nanmax(spring_neap_ratio):.1f}×")
print(f"  Storm/calm ratio max:  {np.nanmax(storm_calm_ratio):.1f}×")

# ── MESH ─────────────────────────────────────────────────────────────────────

print("\n[3] Building triangulation ...")
with nc.Dataset(str(NET_NC)) as ds:
    node_x     = np.array(ds["mesh2d_node_x"][:])
    node_y     = np.array(ds["mesh2d_node_y"][:])
    face_nodes = np.array(ds["mesh2d_face_nodes"][:]) - 1

tris = []
for fn in face_nodes:
    v = fn[fn >= 0]
    if len(v) == 3:   tris.append(v)
    elif len(v) == 4: tris.append([v[0],v[1],v[2]]); tris.append([v[0],v[2],v[3]])
tris   = np.array(tris)
tree_n = cKDTree(np.column_stack([fx, fy]))
_, f2n = tree_n.query(np.column_stack([node_x, node_y]))

sta_tree  = cKDTree(np.column_stack([fx, fy]))
sta_faces = {name: sta_tree.query([sx, sy])[1]
             for name, (sx, sy) in STATIONS.items()}

# ── PLOTTING HELPERS ──────────────────────────────────────────────────────────

def make_triang(field_face):
    fn  = field_face[f2n].astype(float)
    bad = ~np.isfinite(fn) | (inund[f2n] < 0.01)
    tri_mask = np.any(bad[tris], axis=1)
    fn  = np.where(np.isfinite(fn), fn, 0.0)
    return mtri.Triangulation(node_x, node_y, tris, mask=tri_mask), fn

def add_base(ax):
    """Shoreline contour + dry mask."""
    bl_n = np.where(np.isfinite(bl[f2n]), bl[f2n], 0.0)
    tr_b = mtri.Triangulation(node_x, node_y, tris)
    try:
        ax.tricontour(tr_b, bl_n, levels=[0.0], colors=["white"],
                      linewidths=[1.2], alpha=0.85, zorder=4)
    except Exception: pass
    dry_n = inund[f2n] < 0.01
    tri_dry = np.all(inund[tris] < 0.01, axis=1)
    if tri_dry.any():
        tr_d = mtri.Triangulation(node_x, node_y, tris, mask=~tri_dry)
        ax.tripcolor(tr_d, np.ones(len(node_x)),
                     cmap=mcolors.ListedColormap(["#1a1a1a"]),
                     vmin=0, vmax=1, zorder=5, alpha=0.95)

def add_stations(ax, values=None, fmt="{:.2f}"):
    for name, (sx, sy) in STATIONS.items():
        ax.plot(sx, sy, "w^", ms=5, zorder=8)
        if values is not None:
            idx = sta_faces[name]
            v   = values[idx]
            if np.isfinite(v):
                ax.annotate(f"{name}\n{fmt.format(v)}", (sx, sy),
                            color="white", fontsize=6,
                            xytext=(3,3), textcoords="offset points",
                            zorder=9, bbox=dict(boxstyle="round,pad=0.2",
                                                fc="#222", ec="none", alpha=0.7))

def style_ax(ax):
    ax.set_facecolor(DARK_PAN)
    ax.set_xlim(*AOI_X); ax.set_ylim(*AOI_Y)
    ax.tick_params(colors="white", labelsize=7)
    ax.spines[:].set_color("#333")

def save(fig, name):
    fig.savefig(str(OUT_DIR / name), dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ {name}")

# ── FIG 1: SPRING vs NEAP τ ───────────────────────────────────────────────────

print("\n[4] Generating figures ...")
print("  Spring vs neap shear stress ...")

fig, axes = plt.subplots(1, 3, figsize=(20, 7), facecolor=DARK_BG)
fig.subplots_adjust(wspace=0.05)

titles    = ["Springtij: Gemiddelde τ (Pa)",
             "Doodtij: Gemiddelde τ (Pa)",
             "Springtij/Doodtij verhouding"]
fields    = [tau_spring, tau_neap,
                 np.clip(spring_neap_ratio, 0.5, 5.0)]
cmaps     = ["YlOrRd", "YlOrRd", "RdYlBu_r"]
tau_p98   = np.nanpercentile(
    tau_spring[np.isfinite(tau_spring) & ~perm_dry], 97)
vmins     = [0, 0, 0.5]
vmaxs     = [tau_p98, tau_p98, 5.0]
fmts      = ["{:.2f} Pa", "{:.2f} Pa", "{:.1f}×"]

for ax, title, field, cmap, vmin, vmax, fmt in \
        zip(axes, titles, fields, cmaps, vmins, vmaxs, fmts):
    style_ax(ax)
    tr, fn = make_triang(field)
    tcf = ax.tricontourf(tr, fn, levels=50, cmap=cmap, vmin=vmin, vmax=vmax)
    cb  = plt.colorbar(tcf, ax=ax, fraction=0.04, pad=0.01)
    cb.ax.tick_params(colors="white", labelsize=7)
    add_base(ax)
    add_stations(ax, field, fmt)
    ax.set_title(title, color="white", fontsize=9, fontweight="bold")
    ax.set_xlabel("RD New X (m)", color="white", fontsize=8)

axes[0].set_ylabel("RD New Y (m)", color="white", fontsize=8)

# Add threshold contours on spring map
tr_s, fn_s = make_triang(tau_spring)
axes[0].tricontour(tr_s, fn_s, levels=[TAU_SAND, TAU_MUD, TAU_MARSH],
                   colors=["#FFF176","#FF8F00","#EF5350"],
                   linewidths=[0.8, 1.0, 1.2], linestyles="dashed", zorder=3)

# Highlight ratio > 3 on ratio map
tr_r, fn_r = make_triang(np.clip(spring_neap_ratio, 0.5, 5.0))
axes[2].tricontour(tr_r, fn_r, levels=[3.0],
                   colors=["white"], linewidths=[1.5], zorder=3)
axes[2].text(0.02, 0.02, "Witte contour = ratio 3×\n(morfodynamisch actief)",
             transform=axes[2].transAxes, color="white", fontsize=7,
             va="bottom", bbox=dict(fc="#222", ec="none", alpha=0.7))

fig.suptitle(
    "Springtij vs Doodtij Bodemschuifspanning — Western Scheldt DFM\n"
    "Januari 2020 | Links: springtij | Midden: doodtij | Rechts: verhouding",
    color="white", fontsize=10, fontweight="bold")
save(fig, "tidal_spring_neap_ratio.png")

# ── FIG 2: TIDAL STAGE STRESS ─────────────────────────────────────────────────

print("  Tidal stage stress ...")
fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor=DARK_BG)
fig.subplots_adjust(hspace=0.12, wspace=0.05)
axes = axes.ravel()

stage_fields = [tau_flood, tau_hw, tau_ebb, tau_lw]
stage_titles = ["Vloedstroom (rijzend water)",
                "Hoogwater slack",
                "Ebstroom (dalend water)",
                "Laagwater slack"]
tau_max = np.nanpercentile(
    tau_flood[np.isfinite(tau_flood) & ~perm_dry], 97)

for ax, field, title in zip(axes, stage_fields, stage_titles):
    style_ax(ax)
    tr, fn = make_triang(field)
    tcf = ax.tricontourf(tr, fn, levels=50,
                          cmap="plasma", vmin=0, vmax=tau_max)
    cb  = plt.colorbar(tcf, ax=ax, fraction=0.035, pad=0.01)
    cb.ax.tick_params(colors="white", labelsize=7)
    add_base(ax)
    add_stations(ax, field, "{:.2f}")
    ax.set_title(title, color="white", fontsize=9, fontweight="bold")

for ax in axes[2:]:
    ax.set_xlabel("RD New X (m)", color="white", fontsize=8)
for ax in [axes[0], axes[2]]:
    ax.set_ylabel("RD New Y (m)", color="white", fontsize=8)

fig.suptitle(
    "Bodemschuifspanning per Getijfase — Western Scheldt DFM\n"
    "Vloed / HW / Eb / LW | Pa",
    color="white", fontsize=10, fontweight="bold")
save(fig, "tidal_stage_stress.png")

# ── FIG 3: CRITICAL DURATION MAP ─────────────────────────────────────────────

print("  Critical duration ...")
fig, axes = plt.subplots(1, 3, figsize=(20, 7), facecolor=DARK_BG)
fig.subplots_adjust(wspace=0.05)

crit_fields = [crit_sand_spring, crit_mud_spring, crit_marsh_spring]
crit_titles = [f"Uren/springtij τ > {TAU_SAND} Pa\n(zand entrainment)",
               f"Uren/springtij τ > {TAU_MUD} Pa\n(slik erosie)",
               f"Uren/springtij τ > {TAU_MARSH} Pa\n(schor falendrempel)"]
crit_cmaps  = ["YlOrRd", "YlOrRd", "Reds"]
crit_vmaxs  = [
    np.nanpercentile(crit_sand_spring[~perm_dry & np.isfinite(crit_sand_spring)], 97),
    np.nanpercentile(crit_mud_spring[~perm_dry & np.isfinite(crit_mud_spring)], 97),
    np.nanpercentile(crit_marsh_spring[~perm_dry & np.isfinite(crit_marsh_spring)], 99),
]

for ax, field, title, cmap, vmax in \
        zip(axes, crit_fields, crit_titles, crit_cmaps, crit_vmaxs):
    style_ax(ax)
    tr, fn = make_triang(field)
    tcf = ax.tricontourf(tr, fn, levels=50, cmap=cmap, vmin=0, vmax=max(vmax, 0.1))
    cb  = plt.colorbar(tcf, ax=ax, fraction=0.04, pad=0.01)
    cb.set_label("uren/springtij", color="white", fontsize=8)
    cb.ax.tick_params(colors="white", labelsize=7)
    add_base(ax)
    add_stations(ax, field, "{:.1f} h")
    ax.set_title(title, color="white", fontsize=9, fontweight="bold")
    ax.set_xlabel("RD New X (m)", color="white", fontsize=8)

axes[0].set_ylabel("RD New Y (m)", color="white", fontsize=8)
fig.suptitle(
    "Kritische Belastingsduur per Springtij — Western Scheldt DFM\n"
    "Aantal uren per springtij dat drempelwaarde overschreden wordt",
    color="white", fontsize=10, fontweight="bold")
save(fig, "tidal_critical_duration.png")

# ── FIG 4: STORM vs CALM ──────────────────────────────────────────────────────

print("  Storm vs calm ...")
fig, axes = plt.subplots(1, 3, figsize=(20, 7), facecolor=DARK_BG)
fig.subplots_adjust(wspace=0.05)

storm_fields = [tau_storm, tau_calm,
                np.clip(storm_calm_ratio, 0.5, 5.0)]
storm_titles = ["Storm τ (Jan 13-14, Pa)",
                "Kalm getij τ (Pa)",
                "Storm/kalm verhouding"]
storm_cmaps  = ["YlOrRd", "YlOrRd", "RdYlBu_r"]
tau_sv = np.nanpercentile(tau_storm[np.isfinite(tau_storm) & ~perm_dry], 97)
storm_vmins  = [0, 0, 0.5]
storm_vmaxs  = [tau_sv, tau_sv, 4.0]
storm_fmts   = ["{:.2f} Pa", "{:.2f} Pa", "{:.1f}×"]

for ax, field, title, cmap, vmin, vmax, fmt in \
        zip(axes, storm_fields, storm_titles,
            storm_cmaps, storm_vmins, storm_vmaxs, storm_fmts):
    style_ax(ax)
    tr, fn = make_triang(field)
    tcf = ax.tricontourf(tr, fn, levels=50, cmap=cmap, vmin=vmin, vmax=vmax)
    cb  = plt.colorbar(tcf, ax=ax, fraction=0.04, pad=0.01)
    cb.ax.tick_params(colors="white", labelsize=7)
    add_base(ax)
    add_stations(ax, field, fmt)
    ax.set_title(title, color="white", fontsize=9, fontweight="bold")
    ax.set_xlabel("RD New X (m)", color="white", fontsize=8)

axes[0].set_ylabel("RD New Y (m)", color="white", fontsize=8)
axes[2].tricontour(*make_triang(storm_calm_ratio),
                   levels=[2.0], colors=["white"],
                   linewidths=[1.5], zorder=3)
axes[2].text(0.02, 0.02, "Witte contour = storm 2× kalm",
             transform=axes[2].transAxes, color="white", fontsize=7,
             va="bottom", bbox=dict(fc="#222", ec="none", alpha=0.7))

fig.suptitle(
    "Storm (13-14 jan) vs Kalm Getij — Western Scheldt DFM\n"
    "ERA5 storm: V_wind = +10.5 m/s (ZW) | Beaufort 6",
    color="white", fontsize=10, fontweight="bold")
save(fig, "tidal_storm_vs_calm.png")

# ── FIG 5: EXCEEDANCE DURATION CURVES ────────────────────────────────────────

print("  Exceedance duration curves ...")

# ZES.1 zone masks
intertidal_low  = (inund > 0.01) & (inund < 0.50) & (bl >= -2) & (bl < 0)
intertidal_mid  = (inund > 0.01) & (inund < 0.85) & (bl >= 0)  & (bl < 1)
channel_shallow = (inund > 0.95) & (bl >= -5) & (bl < -2)
channel_deep    = (inund > 0.99) & (bl < -5)

zone_masks  = [channel_deep, channel_shallow, intertidal_low, intertidal_mid]
zone_labels = ["Geul (z < -5m)", "Ondiep sublitoraal (-5 to -2m)",
               "Laaglitoraal (-2 to 0m)", "Middenlitoraal (0 to +1m)"]
zone_colors = ["#1565C0", "#42A5F5", "#4CAF50", "#8D6E63"]

fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor=DARK_BG)
fig.subplots_adjust(wspace=0.25)

tau_levels = np.logspace(-2, 1.5, 100)

for zone_mask, label, color in \
        zip(zone_masks, zone_labels, zone_colors):
    if zone_mask.sum() < 10:
        continue
    # Sample up to 500 faces from this zone
    zone_idx = np.where(zone_mask)[0]
    sample   = zone_idx[::max(1, len(zone_idx)//500)]
    tau_zone = tau[:, sample]

    # Full period exceedance
    exc_full = np.array([
        (tau_zone > t).mean() for t in tau_levels
    ])
    # Spring only
    tau_spring_zone = tau[spring_mask, :][:, sample]
    exc_spring = np.array([
        (tau_spring_zone > t).mean() for t in tau_levels
    ])
    # Neap only
    tau_neap_zone = tau[neap_mask, :][:, sample]
    exc_neap = np.array([
        (tau_neap_zone > t).mean() for t in tau_levels
    ])

    axes[0].semilogx(tau_levels, exc_full * 100,
                     color=color, lw=2.0, label=label)
    axes[1].semilogx(tau_levels, exc_spring * 100,
                     color=color, lw=2.0, ls="-", label=f"{label} (spring)")
    axes[1].semilogx(tau_levels, exc_neap * 100,
                     color=color, lw=1.0, ls="--")

for ax, title in zip(axes, [
        "Overschrijdingsduurkromme — Volledige periode",
        "Spring (—) vs Doodtij (--) vergelijking"]):
    ax.set_facecolor(DARK_PAN)
    ax.set_xlabel("Bodemschuifspanning τ (Pa)", color="white", fontsize=9)
    ax.set_ylabel("Overschrijdingskans (%)", color="white", fontsize=9)
    ax.tick_params(colors="white", labelsize=8)
    ax.spines[:].set_color("#444")
    ax.set_xlim(0.05, 30)
    ax.set_ylim(0, 100)
    ax.set_title(title, color="white", fontsize=9, fontweight="bold")
    ax.grid(color="#333", lw=0.5, alpha=0.7)

    # Critical threshold lines
    for tau_c, label_c, col_c in [
        (TAU_SAND,  "zand", "#FFF176"),
        (TAU_MUD,   "slik", "#FF8F00"),
        (TAU_MARSH, "schor", "#EF5350")
    ]:
        ax.axvline(tau_c, color=col_c, lw=1.0, ls=":", alpha=0.8)
        ax.text(tau_c*1.05, 95, label_c, color=col_c, fontsize=7, va="top")

axes[0].legend(fontsize=7.5, facecolor="#1a1a1a", labelcolor="white",
               loc="upper right")
axes[1].legend(fontsize=7.0, facecolor="#1a1a1a", labelcolor="white",
               loc="upper right")
fig.suptitle(
    "Overschrijdingsduurkromme per ZES.1 Zone — Western Scheldt DFM\n"
    "Fractie tijd dat bodemschuifspanning drempelwaarde overschrijdt",
    color="white", fontsize=10, fontweight="bold")
save(fig, "tidal_exceedance_curves.png")

# ── FIG 6: STATION TIME SERIES WITH SPRING/NEAP SHADING ──────────────────────

print("  Station time series with tidal period shading ...")

fig, axes = plt.subplots(5, 1, figsize=(16, 14), facecolor=DARK_BG,
                          sharex=True)
fig.subplots_adjust(hspace=0.15)

for i, (name, (sx, sy)) in enumerate(STATIONS.items()):
    idx = sta_faces[name]
    ax  = axes[i]
    ax.set_facecolor("#151515")

    # Shade spring and neap periods
    for s, e in [SPRING1, SPRING2]:
        ax.axvspan(s, e, alpha=0.15, color="#FFC107", zorder=1)
    for s, e in [NEAP1, NEAP2]:
        ax.axvspan(s, e, alpha=0.10, color="#42A5F5", zorder=1)
    ax.axvspan(*STORM, alpha=0.20, color="#EF5350", zorder=1)

    tau_sta = tau[:, idx]
    ax.fill_between(t_day_fit, 0, tau_sta,
                    color="#FF7043", alpha=0.85, zorder=2)
    ax.axhline(TAU_SAND,  color="#FFF176", lw=0.8, ls="--", zorder=3)
    ax.axhline(TAU_MUD,   color="#FF8F00", lw=1.0, ls="--", zorder=3)
    ax.axhline(TAU_MARSH, color="#EF5350", lw=1.2, ls="-",  zorder=3)

    # Annotate spring/neap means
    t_spring = tau_sta[spring_mask].mean()
    t_neap   = tau_sta[neap_mask].mean()
    ax.text(0.01, 0.92,
            f"{name} | spring: {t_spring:.2f} Pa | dood: {t_neap:.2f} Pa | "
            f"ratio: {t_spring/max(t_neap,0.01):.1f}×",
            transform=ax.transAxes, color="white", fontsize=8,
            va="top",
            bbox=dict(fc="#222", ec="none", alpha=0.7))

    ax.set_ylabel("τ (Pa)", color="white", fontsize=7)
    ax.tick_params(colors="white", labelsize=7)
    ax.spines[:].set_color("#333")

axes[-1].set_xlabel("Dag (januari 2020)", color="white", fontsize=9)

# Legend
handles = [
    mpatches.Patch(color="#FFC107", alpha=0.3, label="Springtij periode"),
    mpatches.Patch(color="#42A5F5", alpha=0.2, label="Doodtij periode"),
    mpatches.Patch(color="#EF5350", alpha=0.3, label="Storm 13-14 jan"),
    mpatches.Patch(color="#FFF176", label=f"τ_zand ({TAU_SAND} Pa)"),
    mpatches.Patch(color="#FF8F00", label=f"τ_slik ({TAU_MUD} Pa)"),
    mpatches.Patch(color="#EF5350", label=f"τ_schor ({TAU_MARSH} Pa)"),
]
axes[0].legend(handles=handles, fontsize=7, facecolor="#1a1a1a",
               labelcolor="white", loc="upper right", ncol=3)

fig.suptitle(
    "Bodemschuifspanning Tijdreeks — Meetstations Western Scheldt\n"
    "Geel = springtij | Blauw = doodtij | Rood = storm 13-14 jan",
    color="white", fontsize=10, fontweight="bold")
save(fig, "tidal_station_timeseries.png")

# ── SUMMARY REPORT ────────────────────────────────────────────────────────────

print("\n[5] Writing summary report ...")

report = {
    "period": {
        "start_day": float(t_day_fit[0]),
        "end_day":   float(t_day_fit[-1]),
        "spinup_days": args.spinup,
    },
    "spring_neap": {
        "spring_period_days": f"{SPRING1[0]}-{SPRING1[1]} and {SPRING2[0]}-{SPRING2[1]}",
        "neap_period_days":   f"{NEAP1[0]}-{NEAP1[1]} and {NEAP2[0]}-{NEAP2[1]}",
        "mean_ratio_channel": float(np.nanmean(
            spring_neap_ratio[~perm_dry & (bl < -2)])),
        "mean_ratio_intertidal": float(np.nanmean(
            spring_neap_ratio[~perm_dry & (bl >= -2) & (bl < 1)])),
        "max_ratio": float(np.nanmax(spring_neap_ratio)),
    },
    "storm_jan13_14": {
        "storm_period_days": f"{STORM[0]}-{STORM[1]}",
        "mean_storm_calm_ratio": float(np.nanmean(
            storm_calm_ratio[~perm_dry])),
        "max_storm_calm_ratio": float(np.nanmax(storm_calm_ratio)),
    },
    "critical_duration_per_spring_tide": {
        name: {
            "tau_sand_h":  float(crit_sand_spring[idx]),
            "tau_mud_h":   float(crit_mud_spring[idx]),
            "tau_marsh_h": float(crit_marsh_spring[idx]),
        }
        for name, (_, _) in STATIONS.items()
        for idx in [sta_faces[name]]
    },
    "station_spring_neap": {
        name: {
            "tau_spring_Pa": float(tau_spring[sta_faces[name]]),
            "tau_neap_Pa":   float(tau_neap[sta_faces[name]]),
            "ratio":         float(tau_spring[sta_faces[name]] /
                                   max(tau_neap[sta_faces[name]], 0.01)),
        }
        for name in STATIONS
    }
}

report_path = OUT_DIR / "tidal_forcing_report.json"
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)
print(f"  ✓ tidal_forcing_report.json")

# ── CONSOLE SUMMARY ───────────────────────────────────────────────────────────

print(f"\n{'='*65}")
print("Station spring-neap stress summary:")
print(f"{'Station':12s}  {'τ_spring':9s}  {'τ_neap':8s}  "
      f"{'ratio':6s}  {'h/spring>mud':12s}  {'h/spring>marsh':14s}")
print("-"*65)
for name in STATIONS:
    idx = sta_faces[name]
    print(f"{name:12s}  "
          f"{tau_spring[idx]:9.3f}  "
          f"{tau_neap[idx]:8.3f}  "
          f"{tau_spring[idx]/max(tau_neap[idx],0.01):6.1f}×  "
          f"{crit_mud_spring[idx]:12.1f}  "
          f"{crit_marsh_spring[idx]:14.1f}")

print(f"\n✓ All figures written to {OUT_DIR}")
