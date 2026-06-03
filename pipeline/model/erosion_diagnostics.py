"""
Western Scheldt — Erosion Stress & Current Diagnostics
=======================================================
Computes and plots hydrodynamic indicators relevant to bank instability,
saltmarsh erosion, and sediment transport. Figures are aligned with the
ZES.1 ecotope classification (Bouma et al. 2005) used in IMDC monitoring
reports for the Westerschelde.

ZES.1 velocity thresholds:
  < 0.2 m/s peak  → zeer laagdynamisch (potential schor development)
  < 0.8 m/s peak  → laagdynamisch (ecologically valuable intertidal)
  > 0.8 m/s peak  → hoogdynamisch (physically stressed)

Depth thresholds (NAP):
  > +1.0 m  → supralitoraal / schor
  0 to +1.0 m  → hooglitoraal
  -2 to 0 m   → laag/middenlitoraal
  -5 to -2 m  → ondiep sublitoraal
  < -5 m     → sublitoraal / geul

Critical shear stress (Van Rijn 2007):
  0.15 Pa  → fine sand entrainment
  0.40 Pa  → cohesive mudflat erosion
  1.00 Pa  → saltmarsh root zone failure

Usage:
    python erosion_diagnostics.py
    python erosion_diagnostics.py --base dfm_v5_manning/dflowfm --spinup 2
"""

import os
_P = "/home/ulg/mast/eivanov/.conda/envs/Yoda/lib/python3.10/site-packages/pyproj/proj_dir/share/proj"
os.environ["PROJ_DATA"] = _P; os.environ["PROJ_LIB"] = _P
import pyproj; pyproj.datadir.set_data_dir(_P)

import sys, argparse, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")
try:
    import netCDF4 as nc
except ImportError:
    sys.exit("pip install netCDF4")

# ── CONFIG ────────────────────────────────────────────────────────────────────

ap = argparse.ArgumentParser()
ap.add_argument("--base",   default="dfm_clean/dflowfm")
ap.add_argument("--spinup", type=float, default=2.0)
ap.add_argument("--out",    default=".")
args = ap.parse_args()

BASE    = Path(args.base)
MAP_NC  = BASE / "DFM_OUTPUT_WesternScheldt/WesternScheldt_map.nc"
NET_NC  = next(BASE.glob("WesternScheldt_net_v*.nc"))
OUT_DIR = Path(args.out)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ZES.1 velocity thresholds (m/s, peak tidal current)
U_VERY_LOW  = 0.20   # zeer laagdynamisch / schor potential
U_LOW       = 0.80   # laagdynamisch / hoogdynamisch boundary (key IMDC metric)
U_HIGH      = 1.50   # strongly energetic channel

# Critical shear stress thresholds (Pa)
TAU_SAND    = 0.15
TAU_MUD     = 0.40
TAU_MARSH   = 1.00

# ZES.1 depth thresholds (m NAP)
Z_GEUL      = -5.0   # geul / ondiep sublitoraal boundary
Z_SUBLITT   = -2.0   # ondiep sublitoraal / litoraal boundary
Z_MID       =  0.0   # laaglitoraal / middenlitoraal boundary
Z_HIGH      =  1.0   # hooglitoraal / supralitoraal boundary
Z_POLDER    =  2.5   # supralitoraal / polder

# Inundation threshold for "can be flooded"
INUND_MIN   = 0.01   # mask cells wet < 1% of time

# Monitoring stations (RD New)
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

# ZES.1 ecotope colour scheme (matches IMDC reports)
ZES_COLORS = {
    "Geul (z < -5m)":           "#1a237e",   # deep blue
    "Ondiep sublitoraal":        "#1565C0",   # medium blue
    "Laagdyn. litoraal":         "#4CAF50",   # green
    "Hoogdyn. litoraal":         "#FFC107",   # amber
    "Schor / supralitoraal":     "#8D6E63",   # brown
    "Permanent droog":           "#212121",   # near black
}

print("=== Erosion Stress & Current Diagnostics — Western Scheldt ===\n")
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

dt_hr = np.median(np.diff(t_sec)) / 3600.0
t_hr  = t_sec / 3600.0
n_t, n_f = wl_all.shape
print(f"  {n_t} timesteps × {n_f:,} faces, dt={dt_hr:.1f}h")

wl_all = np.where(wl_all < -100, np.nan, wl_all)
ux_all = np.where(np.abs(ux_all) > 100, np.nan, ux_all)
uy_all = np.where(np.abs(uy_all) > 100, np.nan, uy_all)

# Exclude spinup
spinup_h = args.spinup * 24.0
t_mask   = t_hr >= spinup_h
wl = wl_all[t_mask, :]
ux = ux_all[t_mask, :]
uy = uy_all[t_mask, :]
n_fit = t_mask.sum()
print(f"  Analysis: {n_fit} steps ({t_hr[t_mask][-1]/24:.1f} days)")

if has_tau:
    tau = tau_all[t_mask, :]
    tau = np.where(tau < 0, 0.0, tau)
else:
    rho, g, C = 1025.0, 9.81, 50.0
    spd_tmp   = np.sqrt(ux**2 + uy**2)
    tau       = rho * g / C**2 * spd_tmp**2

# ── COMPUTE INDICATORS ───────────────────────────────────────────────────────

print("\n[2] Computing indicators ...")

depth    = np.where(np.isfinite(wl), wl - bl[np.newaxis,:], np.nan)
wet      = depth > 0.05

# Inundation frequency
inund    = wet.mean(axis=0)

# Permanent mask: cells never flooded OR always flooded (polder/open sea boundary)
can_flood = (inund >= INUND_MIN) & (inund <= 0.999)
perm_dry  = inund < INUND_MIN

# Mean bed shear stress (wet periods only)
tau_wet  = np.where(wet, tau, np.nan)
tau_mean = np.nanmean(tau_wet, axis=0)

# Shear stress exceedance fractions
n_wet = np.where(wet.sum(axis=0) > 0, wet.sum(axis=0).astype(float), np.nan)
exc_sand  = np.nansum((tau > TAU_SAND)  & wet, axis=0) / n_wet
exc_mud   = np.nansum((tau > TAU_MUD)   & wet, axis=0) / n_wet
exc_marsh = np.nansum((tau > TAU_MARSH) & wet, axis=0) / n_wet

# Peak and residual currents
spd_all  = np.sqrt(ux**2 + uy**2)
peak_spd = np.nanmax(spd_all, axis=0)

ux_resid = np.nanmean(ux, axis=0)
uy_resid = np.nanmean(uy, axis=0)
resid_spd = np.sqrt(ux_resid**2 + uy_resid**2)

# Flood/ebb asymmetry (clipped ratio)
u_flood = np.nanmean(np.where(ux > 0, ux, np.nan), axis=0)
u_ebb   = np.nanmean(np.where(ux < 0, np.abs(ux), np.nan), axis=0)
with np.errstate(divide="ignore", invalid="ignore"):
    fe_ratio = np.where((u_ebb > 0.01) & np.isfinite(u_flood),
                        np.clip(u_flood / u_ebb, 0.3, 3.0), np.nan)

# ZES.1 ecotope classification per face
def zes_classify(bl_face, peak_u, inund_frac):
    """Classify each face into ZES.1 ecotope based on depth + dynamics."""
    cls = np.full(len(bl_face), -1, dtype=np.int8)
    cls[bl_face < Z_GEUL]                                   = 0  # geul
    cls[(bl_face >= Z_GEUL) & (bl_face < Z_SUBLITT)]       = 1  # ondiep sublitoraal
    intertidal = (inund_frac > 0.01) & (inund_frac < 0.999) & (bl_face >= Z_SUBLITT)
    ld_intertidal = intertidal & (peak_u < U_LOW)
    hd_intertidal = intertidal & (peak_u >= U_LOW)
    cls[ld_intertidal] = 2  # laagdynamisch litoraal
    cls[hd_intertidal] = 3  # hoogdynamisch litoraal
    cls[bl_face >= Z_HIGH] = 4  # schor / supralitoraal
    cls[inund_frac < INUND_MIN] = 5  # permanent dry
    return cls

zes = zes_classify(bl, peak_spd, inund)

# Laagdynamisch areaal (key IMDC metric)
ld_area_faces = (zes == 2).sum()
cell_area_approx = 200**2  # ~200m cells
ld_area_ha = ld_area_faces * cell_area_approx / 10000
print(f"  Laagdynamisch litoraal: {ld_area_faces:,} faces (~{ld_area_ha:.0f} ha)")
print(f"  Peak speed: [{peak_spd.min():.3f}, {peak_spd.max():.3f}] m/s")
print(f"  Mean τ:     [{np.nanmin(tau_mean):.3f}, {np.nanmax(tau_mean):.3f}] Pa")

# Station face indices
sta_tree = cKDTree(np.column_stack([fx, fy]))
sta_faces = {name: sta_tree.query([sx, sy])[1]
             for name, (sx, sy) in STATIONS.items()}

# ── LOAD MESH ─────────────────────────────────────────────────────────────────

print("\n[3] Building triangulation ...")
with nc.Dataset(str(NET_NC)) as ds:
    node_x     = np.array(ds["mesh2d_node_x"][:])
    node_y     = np.array(ds["mesh2d_node_y"][:])
    face_nodes = np.array(ds["mesh2d_face_nodes"][:]) - 1

tris = []
for fn in face_nodes:
    v = fn[fn >= 0]
    if len(v) == 3:
        tris.append(v)
    elif len(v) == 4:
        tris.append([v[0], v[1], v[2]])
        tris.append([v[0], v[2], v[3]])
tris   = np.array(tris)
tree_n = cKDTree(np.column_stack([fx, fy]))
_, f2n = tree_n.query(np.column_stack([node_x, node_y]))
print(f"  {len(node_x):,} nodes, {len(tris):,} triangles")

# Shoreline: mean WL contour ≈ 0m NAP
wl_mean = np.nanmean(wl, axis=0)

# ── PLOTTING HELPERS ──────────────────────────────────────────────────────────

def apply_mask(field_face, mask_dry=True):
    """Interpolate field to nodes, build masked triangulation."""
    f = field_face.copy().astype(float)
    if mask_dry:
        f[perm_dry] = np.nan
    fn  = f[f2n]
    bad = ~np.isfinite(fn)
    tri_mask = np.any(bad[tris], axis=1)
    fn  = np.where(np.isfinite(fn), fn, 0.0)
    tr  = mtri.Triangulation(node_x, node_y, tris, mask=tri_mask)
    return tr, fn

def add_estuary_base(ax, show_shoreline=True, show_depth_contours=True):
    """Add shoreline and depth reference contours to any map axis."""
    bl_n = bl[f2n]
    bl_n = np.where(np.isfinite(bl_n), bl_n, 0.0)
    tr_bl = mtri.Triangulation(node_x, node_y, tris)

    if show_depth_contours:
        # Depth contours: geul boundary (-5m), sublittoral (-2m), intertidal (0m)
        try:
            cs = ax.tricontour(tr_bl, bl_n,
                               levels=[Z_GEUL, Z_SUBLITT, Z_MID, Z_HIGH],
                               colors=["#1565C0", "#42A5F5", "#A5D6A7", "#8D6E63"],
                               linewidths=[1.2, 0.8, 1.0, 0.8],
                               linestyles=["solid","dashed","solid","dashed"],
                               alpha=0.7, zorder=3)
            # Label the 0m contour
            for coll in cs.collections:
                coll.set_zorder(3)
        except Exception:
            pass

    if show_shoreline:
        # Bold 0m NAP contour = mean water line / estuary edge
        try:
            ax.tricontour(tr_bl, bl_n, levels=[0.0],
                          colors=["white"], linewidths=[1.5],
                          linestyles=["solid"], alpha=0.9, zorder=4)
        except Exception:
            pass

    # Mask permanently dry cells with dark fill
    dry_n = (inund[f2n] < INUND_MIN).astype(float)
    tri_mask_dry = np.all(
        inund[tris] < INUND_MIN, axis=1
    )
    if tri_mask_dry.any():
        tr_dry = mtri.Triangulation(node_x, node_y, tris, mask=~tri_mask_dry)
        ax.tripcolor(tr_dry, np.ones(len(node_x)),
                     cmap=mcolors.ListedColormap(["#1a1a1a"]),
                     vmin=0, vmax=1, zorder=5, alpha=0.95)

def add_stations(ax, field_face=None, fmt="{:.2f}"):
    """Add station markers with optional field value annotation."""
    for name, (sx, sy) in STATIONS.items():
        ax.plot(sx, sy, "w^", ms=6, zorder=8, mec="white", mew=0.8)
        label = name
        if field_face is not None:
            idx = sta_faces[name]
            val = field_face[idx]
            if np.isfinite(val):
                label = f"{name}\n{fmt.format(val)}"
        ax.annotate(label, (sx, sy), color="white", fontsize=6.5,
                    xytext=(4, 4), textcoords="offset points", zorder=9,
                    bbox=dict(boxstyle="round,pad=0.2", fc="#222", ec="none", alpha=0.7))

def make_ax(facecolor=DARK_BG):
    fig, ax = plt.subplots(figsize=(14, 7), facecolor=facecolor)
    ax.set_facecolor(DARK_PAN)
    ax.set_xlim(*AOI_X); ax.set_ylim(*AOI_Y)
    ax.set_xlabel("RD New X (m)", color="white", fontsize=9)
    ax.set_ylabel("RD New Y (m)", color="white", fontsize=9)
    ax.tick_params(colors="white", labelsize=8)
    ax.spines[:].set_color("#333")
    return fig, ax

def save(fig, filename):
    out = OUT_DIR / filename
    fig.savefig(str(out), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ {filename}")

def depth_contour_legend(ax):
    """Add legend for depth reference contours."""
    handles = [
        mpatches.Patch(color="white",    label="0 m NAP (gemiddeld zeeniveau)"),
        mpatches.Patch(color="#A5D6A7",  label="0 m NAP (intertidaal grens)"),
        mpatches.Patch(color="#42A5F5",  label="-2 m NAP (sublitoraal)"),
        mpatches.Patch(color="#1565C0",  label="-5 m NAP (geul grens)"),
        mpatches.Patch(color="#1a1a1a",  label="Permanent droog"),
    ]
    ax.legend(handles=handles, fontsize=6.5, facecolor="#1a1a1a",
              labelcolor="white", loc="lower right", framealpha=0.9,
              handlelength=1.2, borderpad=0.6)

# ── FIG 1: ZES.1 ECOTOPE MAP ─────────────────────────────────────────────────

print("\n[4] Generating figures ...")
print("  ZES.1 ecotope map ...")

zes_cmap   = mcolors.ListedColormap(
    ["#1a237e","#1565C0","#4CAF50","#FFC107","#8D6E63","#1a1a1a"])
zes_bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
zes_norm   = mcolors.BoundaryNorm(zes_bounds, zes_cmap.N)

fig, ax = make_ax()
zes_n = zes[f2n].astype(float)
tr_z  = mtri.Triangulation(node_x, node_y, tris)
ax.tricontourf(tr_z, zes_n, cmap=zes_cmap, norm=zes_norm, zorder=1)
add_estuary_base(ax, show_shoreline=True, show_depth_contours=False)
add_stations(ax)

zes_labels = list(ZES_COLORS.keys())
zes_cols   = list(ZES_COLORS.values())
handles = [mpatches.Patch(color=c, label=l)
           for c, l in zip(zes_cols, zes_labels)]
ax.legend(handles=handles, fontsize=7, facecolor="#1a1a1a",
          labelcolor="white", loc="lower right", framealpha=0.9)
ax.set_title(
    "ZES.1 Ecotooptypen — Western Scheldt DFM\n"
    f"Laagdynamisch litoraal: ~{ld_area_ha:.0f} ha  |  "
    f"Grens laag/hoogdynamisch: U_peak = {U_LOW} m/s  (Bouma et al. 2005)",
    color="white", fontsize=9, fontweight="bold")
save(fig, "zes_ecotope_map.png")

# ── FIG 2: PEAK CURRENT SPEED + ZES THRESHOLDS ───────────────────────────────

print("  Peak current speed ...")
fig, ax = make_ax()
tr_p, fn_p = apply_mask(peak_spd)
tcf = ax.tricontourf(tr_p, fn_p, levels=50,
                      cmap="plasma", vmin=0, vmax=2.5, zorder=1)
cb  = plt.colorbar(tcf, ax=ax, fraction=0.03, pad=0.02)
cb.set_label("|U|_peak (m/s)", color="white", fontsize=9)
cb.ax.tick_params(colors="white", labelsize=8)

# ZES velocity threshold contours
ax.tricontour(tr_p, fn_p, levels=[U_VERY_LOW, U_LOW, U_HIGH],
              colors=["#A5D6A7","#FFC107","#EF5350"],
              linewidths=[1.2, 2.0, 1.2], linestyles=["dashed","solid","dashed"],
              zorder=3)

add_estuary_base(ax, show_depth_contours=False)
add_stations(ax, field_face=peak_spd, fmt="{:.2f} m/s")

handles = [
    mpatches.Patch(color="#A5D6A7", label=f"0.2 m/s — zeer laagdynamisch grens"),
    mpatches.Patch(color="#FFC107", label=f"0.8 m/s — laag/hoogdynamisch grens (ZES.1)"),
    mpatches.Patch(color="#EF5350", label=f"1.5 m/s — sterk energetisch"),
    mpatches.Patch(color="white",   label="0 m NAP (kustlijn)"),
]
ax.legend(handles=handles, fontsize=7, facecolor="#1a1a1a",
          labelcolor="white", loc="lower right", framealpha=0.9)
ax.set_title(
    "Maximale Stroomsnelheid (m/s) — Western Scheldt DFM\n"
    "Gekleurde contouren = ZES.1 dynamiekklassen (Bouma et al. 2005)",
    color="white", fontsize=9, fontweight="bold")
save(fig, "erosion_peak_speed.png")

# ── FIG 3: MEAN BED SHEAR STRESS ─────────────────────────────────────────────

print("  Mean bed shear stress ...")
fig, ax = make_ax()
tr_t, fn_t = apply_mask(tau_mean)
vmax_tau = np.nanpercentile(tau_mean[np.isfinite(tau_mean) & ~perm_dry], 97)
tcf = ax.tricontourf(tr_t, fn_t, levels=50,
                      cmap="YlOrRd", vmin=0, vmax=vmax_tau, zorder=1)
cb  = plt.colorbar(tcf, ax=ax, fraction=0.03, pad=0.02)
cb.set_label("τ_mean (Pa)", color="white", fontsize=9)
cb.ax.tick_params(colors="white", labelsize=8)
ax.tricontour(tr_t, fn_t, levels=[TAU_SAND, TAU_MUD, TAU_MARSH],
              colors=["#FFF176","#FF8F00","#EF5350"],
              linewidths=[0.8, 1.2, 1.5], linestyles=["dashed","dashed","solid"],
              zorder=3)
add_estuary_base(ax)
add_stations(ax, field_face=tau_mean, fmt="{:.2f} Pa")
handles = [
    mpatches.Patch(color="#FFF176", label=f"{TAU_SAND} Pa — zand entrainment"),
    mpatches.Patch(color="#FF8F00", label=f"{TAU_MUD} Pa — cohesief slik erosie"),
    mpatches.Patch(color="#EF5350", label=f"{TAU_MARSH} Pa — schor wortelzone falen"),
    mpatches.Patch(color="white",   label="0 m NAP (kustlijn)"),
]
ax.legend(handles=handles, fontsize=7, facecolor="#1a1a1a",
          labelcolor="white", loc="lower right", framealpha=0.9)
ax.set_title(
    "Gemiddelde Bodemschuifspanning (Pa) — Western Scheldt DFM\n"
    "Tijdsgemiddeld over analyseperiode | Contouren = kritische drempelwaarden",
    color="white", fontsize=9, fontweight="bold")
save(fig, "erosion_tau_mean.png")

# ── FIG 4: MUD EXCEEDANCE ─────────────────────────────────────────────────────

print("  Shear stress exceedance (mud) ...")
fig, ax = make_ax()
tr_e, fn_e = apply_mask(exc_mud)
tcf = ax.tricontourf(tr_e, fn_e, levels=50,
                      cmap="plasma", vmin=0, vmax=1, zorder=1)
cb  = plt.colorbar(tcf, ax=ax, fraction=0.03, pad=0.02)
cb.set_label(f"Overschrijdingsfractie τ > {TAU_MUD} Pa", color="white", fontsize=9)
cb.ax.tick_params(colors="white", labelsize=8)
ax.tricontour(tr_e, fn_e, levels=[0.10, 0.25, 0.50, 0.75],
              colors=["white"], linewidths=[0.7], alpha=0.6, zorder=3)
add_estuary_base(ax)
add_stations(ax, field_face=exc_mud, fmt="{:.0%}")
ax.set_title(
    f"Bodemschuifspanning Overschrijdingsfractie (τ > {TAU_MUD} Pa)\n"
    "Cohesief slik erosiedrempel | Witte contouren = 10%, 25%, 50%, 75%",
    color="white", fontsize=9, fontweight="bold")
save(fig, "erosion_exceedance_mud.png")

# ── FIG 5: MARSH EXCEEDANCE ───────────────────────────────────────────────────

print("  Shear stress exceedance (marsh) ...")
fig, ax = make_ax()
tr_m, fn_m = apply_mask(exc_marsh)
vmax_m = max(np.nanpercentile(exc_marsh[np.isfinite(exc_marsh) & ~perm_dry], 99), 0.1)
tcf = ax.tricontourf(tr_m, fn_m, levels=50,
                      cmap="Reds", vmin=0, vmax=vmax_m, zorder=1)
cb  = plt.colorbar(tcf, ax=ax, fraction=0.03, pad=0.02)
cb.set_label(f"Overschrijdingsfractie τ > {TAU_MARSH} Pa", color="white", fontsize=9)
cb.ax.tick_params(colors="white", labelsize=8)
ax.tricontour(tr_m, fn_m, levels=[0.01, 0.05, 0.10],
              colors=["white"], linewidths=[0.7], alpha=0.6, zorder=3)
add_estuary_base(ax)
add_stations(ax, field_face=exc_marsh, fmt="{:.1%}")
ax.set_title(
    f"Bodemschuifspanning Overschrijdingsfractie (τ > {TAU_MARSH} Pa)\n"
    "Schor wortelzone falendrempel | Contouren = 1%, 5%, 10%",
    color="white", fontsize=9, fontweight="bold")
save(fig, "erosion_exceedance_marsh.png")

# ── FIG 6: INUNDATION FREQUENCY ──────────────────────────────────────────────

print("  Inundation frequency ...")
fig, ax = make_ax()
inund_plot = np.where(~perm_dry, inund, np.nan)
tr_i, fn_i = apply_mask(inund_plot, mask_dry=False)
tcf = ax.tricontourf(tr_i, fn_i, levels=50,
                      cmap="Blues", vmin=0, vmax=1, zorder=1)
cb  = plt.colorbar(tcf, ax=ax, fraction=0.03, pad=0.02)
cb.set_label("Overstromingsfrequentie (fractie tijd nat)", color="white", fontsize=9)
cb.ax.tick_params(colors="white", labelsize=8)
ax.tricontour(tr_i, fn_i, levels=[0.25, 0.50, 0.75],
              colors=["white"], linewidths=[0.7], alpha=0.6, zorder=3)
add_estuary_base(ax, show_depth_contours=False)
add_stations(ax, field_face=inund_plot, fmt="{:.0%}")
ax.set_title(
    "Overstromingsfrequentie — Western Scheldt DFM\n"
    "Fractie tijd diepte > 5 cm | Contouren = 25%, 50%, 75%",
    color="white", fontsize=9, fontweight="bold")
save(fig, "erosion_inundation_freq.png")

# ── FIG 7: RESIDUAL CURRENT ───────────────────────────────────────────────────

print("  Residual current ...")
fig, ax = make_ax()
tr_r, fn_r = apply_mask(resid_spd)
vmax_r = np.nanpercentile(resid_spd[np.isfinite(resid_spd) & ~perm_dry], 97)
tcf = ax.tricontourf(tr_r, fn_r, levels=50,
                      cmap="RdPu", vmin=0, vmax=vmax_r, zorder=1)
cb  = plt.colorbar(tcf, ax=ax, fraction=0.03, pad=0.02)
cb.set_label("|U|_residueel (m/s)", color="white", fontsize=9)
cb.ax.tick_params(colors="white", labelsize=8)
# Residual current vectors (subsampled)
step = max(1, len(fx) // 400)
ax.quiver(fx[::step], fy[::step],
          ux_resid[::step], uy_resid[::step],
          color="white", alpha=0.4, scale=3.0, width=0.002, zorder=4)
add_estuary_base(ax)
add_stations(ax, field_face=resid_spd, fmt="{:.3f} m/s")
ax.set_title(
    "Residuele Stroomsnelheid (m/s) — Western Scheldt DFM\n"
    "Tijdgemiddelde snelheid (netto drift) | Pijlen = richting",
    color="white", fontsize=9, fontweight="bold")
save(fig, "erosion_residual_speed.png")

# ── FIG 8: FLOOD/EBB RATIO ────────────────────────────────────────────────────

print("  Flood/ebb ratio ...")
fig, ax = make_ax()
tr_f, fn_f = apply_mask(fe_ratio)
tcf = ax.tricontourf(tr_f, fn_f, levels=50,
                      cmap="RdBu_r", vmin=0.5, vmax=2.0, zorder=1)
cb  = plt.colorbar(tcf, ax=ax, fraction=0.03, pad=0.02)
cb.set_label("Vloed/eb snelheidsverhouding", color="white", fontsize=9)
cb.ax.tick_params(colors="white", labelsize=8)
ax.tricontour(tr_f, fn_f, levels=[1.0],
              colors=["white"], linewidths=[1.5], zorder=3)
add_estuary_base(ax)
add_stations(ax, field_face=fe_ratio, fmt="{:.2f}")
handles = [
    mpatches.Patch(color="#EF5350", label="> 1.0 — vloed dominant (landwaarts sedimenttransport)"),
    mpatches.Patch(color="#1565C0", label="< 1.0 — eb dominant (zeewaarts transport)"),
    mpatches.Patch(color="white",   label="1.0 — symmetrisch"),
]
ax.legend(handles=handles, fontsize=7, facecolor="#1a1a1a",
          labelcolor="white", loc="lower right", framealpha=0.9)
ax.set_title(
    "Vloed/Eb Snelheidsverhouding — Western Scheldt DFM\n"
    "Rood = vloed dominant (landwaarts) | Blauw = eb dominant | Witte contour = symmetriegrens",
    color="white", fontsize=9, fontweight="bold")
save(fig, "erosion_flood_ebb_ratio.png")

# ── FIG 9: STATION TIME SERIES ────────────────────────────────────────────────

print("  Station time series ...")
t_days = t_hr[t_mask] / 24.0

fig = plt.figure(figsize=(16, 14), facecolor=DARK_BG)
fig.subplots_adjust(hspace=0.35, wspace=0.25)

for i, (name, (sx, sy)) in enumerate(STATIONS.items()):
    idx = sta_faces[name]
    spd_sta = np.sqrt(ux[:,idx]**2 + uy[:,idx]**2)
    tau_sta = tau[:, idx]

    # Current speed
    ax1 = fig.add_subplot(5, 2, 2*i+1)
    ax1.set_facecolor("#151515")
    ax1.fill_between(t_days, 0, spd_sta, color="#42A5F5", alpha=0.8)
    ax1.axhline(U_VERY_LOW, color="#A5D6A7", lw=1.0, ls="--",
                label=f"0.2 m/s (zeer laagdyn.)")
    ax1.axhline(U_LOW, color="#FFC107", lw=1.5, ls="-",
                label=f"0.8 m/s (ZES.1 grens)")
    ax1.set_ylabel("|U| (m/s)", color="white", fontsize=7)
    ax1.set_title(f"{name} — Stroomsnelheid", color="white", fontsize=8)
    ax1.tick_params(colors="white", labelsize=7)
    ax1.spines[:].set_color("#333")
    if i == 0:
        ax1.legend(fontsize=6, facecolor="#1a1a1a", labelcolor="white",
                   loc="upper right")

    # Bed shear stress
    ax2 = fig.add_subplot(5, 2, 2*i+2)
    ax2.set_facecolor("#151515")
    ax2.fill_between(t_days, 0, tau_sta, color="#FF7043", alpha=0.8)
    ax2.axhline(TAU_SAND,  color="#FFF176", lw=0.8, ls="--",
                label=f"{TAU_SAND} Pa (zand)")
    ax2.axhline(TAU_MUD,   color="#FF8F00", lw=1.0, ls="--",
                label=f"{TAU_MUD} Pa (slik)")
    ax2.axhline(TAU_MARSH, color="#EF5350", lw=1.2, ls="-",
                label=f"{TAU_MARSH} Pa (schor)")
    ax2.set_ylabel("τ (Pa)", color="white", fontsize=7)
    ax2.set_title(f"{name} — Bodemschuifspanning", color="white", fontsize=8)
    ax2.tick_params(colors="white", labelsize=7)
    ax2.spines[:].set_color("#333")
    if i == 0:
        ax2.legend(fontsize=6, facecolor="#1a1a1a", labelcolor="white",
                   loc="upper right")

fig.text(0.5, 0.01, "Tijd (dagen)", ha="center", color="white", fontsize=9)
fig.suptitle(
    "Stroomsnelheid & Bodemschuifspanning — Meetstations Western Scheldt\n"
    "DFM model | Horizontale lijnen = ZES.1 en erosiedrempelwaarden",
    color="white", fontsize=10, fontweight="bold", y=0.99)
save(fig, "erosion_station_timeseries.png")

# ── FIG 10: CURRENT ROSE ─────────────────────────────────────────────────────

print("  Current roses ...")
fig = plt.figure(figsize=(18, 4), facecolor=DARK_BG)
fig.patch.set_facecolor(DARK_BG)

speed_bins  = [0.0, 0.2, 0.8, 1.5, 99]
speed_cols  = ["#1565C0", "#4CAF50", "#FFC107", "#EF5350"]
speed_labs  = ["0–0.2 m/s\n(zeer laagdyn.)","0.2–0.8 m/s\n(laagdyn.)",
               "0.8–1.5 m/s\n(hoogdyn.)",">1.5 m/s\n(sterk)"]
n_bins = 36

for i, (name, (sx, sy)) in enumerate(STATIONS.items()):
    idx = sta_faces[name]
    ax  = fig.add_subplot(1, 5, i+1, projection="polar")
    ax.set_facecolor("#1a1a1a")
    u   = ux[:, idx]
    v   = uy[:, idx]
    spd = np.sqrt(u**2 + v**2)
    ang = np.arctan2(u, v)
    bins_a = np.linspace(-np.pi, np.pi, n_bins+1)

    for k in range(len(speed_bins)-1):
        mask_k = (spd >= speed_bins[k]) & (spd < speed_bins[k+1])
        counts, _ = np.histogram(ang[mask_k], bins=bins_a)
        counts = counts / max(len(ang), 1) * 100
        ax.bar(bins_a[:-1], counts, width=2*np.pi/n_bins,
               color=speed_cols[k], alpha=0.85, linewidth=0, label=speed_labs[k])

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.tick_params(colors="white", labelsize=5.5)
    ax.set_title(name, color="white", fontsize=8, pad=8)
    ax.spines["polar"].set_color("#444")
    ax.set_yticklabels([])

    if i == 0:
        ax.legend(fontsize=5.5, facecolor="#1a1a1a", labelcolor="white",
                  loc="lower left", bbox_to_anchor=(-0.5, -0.3),
                  title="Snelheidsklasse", title_fontsize=6)

fig.suptitle(
    "Stroomroos — Western Scheldt Meetstations\n"
    "Richting stroom naartoe | % frequentie | Kleur = ZES.1 dynamiekklasse",
    color="white", fontsize=10, fontweight="bold")
save(fig, "erosion_current_rose.png")

# ── SUMMARY TABLE ─────────────────────────────────────────────────────────────

print(f"\n{'='*75}")
print("Erosie stressoverzicht per station (vergelijk met IMDC 2016-2017):")
print(f"{'Station':12s}  {'τ_mean':7s}  {'U_peak':7s}  "
      f"{'exc_slik':8s}  {'exc_schor':9s}  {'inund':6s}  "
      f"{'ZES klasse':20s}")
print("-"*75)
zes_names = ["Geul","Ondiep sub.","Laagdyn. lit.","Hoogdyn. lit.",
             "Schor/supra","Droog"]
for name, (sx, sy) in STATIONS.items():
    idx = sta_faces[name]
    zes_cls = zes_names[zes[idx]] if 0 <= zes[idx] <= 5 else "?"
    print(f"{name:12s}  "
          f"{tau_mean[idx]:7.3f}  "
          f"{peak_spd[idx]:7.3f}  "
          f"{exc_mud[idx]:8.3f}  "
          f"{exc_marsh[idx]:9.4f}  "
          f"{inund[idx]:6.3f}  "
          f"{zes_cls:20s}")

print(f"\n✓ All figures written to {OUT_DIR}")
