"""
Western Scheldt — Tidal Harmonic Analysis of DFM Output
========================================================
Performs least-squares harmonic analysis on the DFM map.nc output,
producing spatial maps of tidal constituent amplitudes and phases.

Constituents analysed: M2, S2, N2, K1, O1, M4, MS4
Outputs:
  dfm_tidal_harmonics.nc     — amplitudes and phases at every face
  dfm_M2_amplitude.png       — M2 amplitude map (dominant constituent)
  dfm_M2_phase.png           — M2 phase map (tidal wave propagation)
  dfm_tidal_asymmetry.png    — M4/M2 ratio (flood/ebb dominance)
  dfm_tidal_range.png        — tidal range = HW-LW envelope
  dfm_velocity_M2.png        — M2 current ellipses (if velocity in map.nc)

Scientific context:
  M2 amplitude amplification from west (Vlissingen) to east (Bath)
  is the primary calibration target. M4/M2 ratio > 0.1 indicates
  tidal distortion and net sediment transport direction.
  Phase propagation speed reveals estuarine resonance effects.

Usage:
    python tidal_harmonics.py
    python tidal_harmonics.py --base dfm_clean/dflowfm --spinup 3
"""

import os
_P = "/home/ulg/mast/eivanov/.conda/envs/Yoda/lib/python3.10/site-packages/pyproj/proj_dir/share/proj"
os.environ["PROJ_DATA"] = _P; os.environ["PROJ_LIB"] = _P
import pyproj; pyproj.datadir.set_data_dir(_P)

import sys
import argparse
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.colors as mcolors
from pathlib import Path

warnings.filterwarnings("ignore")

try:
    import netCDF4 as nc
except ImportError:
    sys.exit("pip install netCDF4")

# ── CONFIG ────────────────────────────────────────────────────────────────────

ap = argparse.ArgumentParser()
ap.add_argument("--base",   default="dfm_clean/dflowfm")
ap.add_argument("--spinup", type=float, default=3.0,
                help="Spin-up period to exclude (days, default 3)")
ap.add_argument("--chunk",  type=int, default=5000,
                help="Faces per chunk for memory management")
args = ap.parse_args()

BASE    = Path(args.base)
MAP_NC  = BASE / "DFM_OUTPUT_WesternScheldt/WesternScheldt_map.nc"
NET_NC  = BASE / "WesternScheldt_net_v5.nc"
FIG_DIR = Path(".")
OUT_NC  = FIG_DIR / "dfm_tidal_harmonics.nc"

# Tidal constituents: name → period in hours
CONSTITUENTS = {
    "M2":  12.4206,
    "S2":  12.0000,
    "N2":  12.6583,
    "K1":  23.9345,
    "O1":  25.8193,
    "M4":   6.2103,
    "MS4":  6.1033,
}

DARK_BG = "#0e0e0e"
AOI_X   = (14000, 65000)
AOI_Y   = (372000, 397000)

print("=== Tidal Harmonic Analysis — Western Scheldt DFM ===\n")

# ── LOAD MAP OUTPUT ───────────────────────────────────────────────────────────

print("[1] Loading map.nc ...")
with nc.Dataset(str(MAP_NC)) as ds:
    t_sec  = np.array(ds["time"][:])
    fx     = np.array(ds["mesh2d_face_x"][:])
    fy     = np.array(ds["mesh2d_face_y"][:])
    wl_all = np.array(ds["mesh2d_s1"][:])          # (time, face)
    bl     = np.array(ds["mesh2d_flowelem_bl"][:])  # (face,)

    # Check for velocity
    has_vel = "mesh2d_ucx" in ds.variables
    if has_vel:
        ux_all = np.array(ds["mesh2d_ucx"][:])
        uy_all = np.array(ds["mesh2d_ucy"][:])
        print("  Velocity data found ✓")

    # Check for bed shear stress
    has_tau = "mesh2d_taucurrent" in ds.variables
    if has_tau:
        tau_all = np.array(ds["mesh2d_taucurrent"][:])
        print("  Bed shear stress found ✓")

dt_hr  = np.median(np.diff(t_sec)) / 3600.0
t_hr   = t_sec / 3600.0
n_t, n_f = wl_all.shape

print(f"  {n_t} timesteps × {n_f:,} faces")
print(f"  dt = {dt_hr:.2f}h  →  {t_hr[-1]/24:.1f} days")
print(f"  Spin-up exclusion: {args.spinup} days")

# Mask nodata
wl_all = np.where(wl_all < -100, np.nan, wl_all)

# Exclude spin-up
spinup_h  = args.spinup * 24.0
t_mask    = t_hr >= spinup_h
t_hr_fit  = t_hr[t_mask]
wl_fit    = wl_all[t_mask, :]
n_fit     = t_mask.sum()
print(f"  Fitting on {n_fit} timesteps ({t_hr_fit[-1]/24:.1f} days)")

# ── BUILD DESIGN MATRIX ───────────────────────────────────────────────────────

print("\n[2] Building harmonic design matrix ...")

# A = [1, cos(ω₁t), sin(ω₁t), cos(ω₂t), sin(ω₂t), ...]
cols = [np.ones(n_fit)]
con_names = list(CONSTITUENTS.keys())
for period_hr in CONSTITUENTS.values():
    omega = 2 * np.pi / period_hr
    cols.append(np.cos(omega * t_hr_fit))
    cols.append(np.sin(omega * t_hr_fit))

A = np.column_stack(cols)   # (n_fit, 1 + 2*n_con)
n_con = len(CONSTITUENTS)
print(f"  Design matrix: {A.shape}  ({n_con} constituents)")

# Pre-compute pseudo-inverse — same for all faces
AtA_inv = np.linalg.pinv(A.T @ A)
AtA_inv_At = AtA_inv @ A.T   # (n_params, n_fit)

# ── HARMONIC ANALYSIS — CHUNKED ───────────────────────────────────────────────

print(f"\n[3] Fitting harmonics (chunk={args.chunk} faces) ...")

# Output arrays
amp  = np.full((n_con, n_f), np.nan)   # amplitude per constituent per face
pha  = np.full((n_con, n_f), np.nan)   # phase (degrees) per constituent per face
mean_wl = np.full(n_f, np.nan)          # mean water level

if has_vel:
    amp_ux = np.full((n_con, n_f), np.nan)
    pha_ux = np.full((n_con, n_f), np.nan)
    amp_uy = np.full((n_con, n_f), np.nan)
    pha_uy = np.full((n_con, n_f), np.nan)

n_chunks = (n_f + args.chunk - 1) // args.chunk
for c in range(n_chunks):
    i0 = c * args.chunk
    i1 = min(i0 + args.chunk, n_f)

    wl_chunk = wl_fit[:, i0:i1]   # (n_fit, chunk)

    # Mask faces with too many NaN (dry for most of simulation)
    valid = np.sum(np.isfinite(wl_chunk), axis=0) > n_fit * 0.5

    if valid.sum() == 0:
        continue

    # Solve least squares for valid faces only
    wl_v = wl_chunk[:, valid]
    # Replace remaining NaN with column mean for fitting
    col_means = np.nanmean(wl_v, axis=0)
    nan_mask  = ~np.isfinite(wl_v)
    wl_v      = np.where(nan_mask, col_means[np.newaxis, :], wl_v)

    coeffs = AtA_inv_At @ wl_v   # (n_params, n_valid)

    mean_wl[i0:i1][valid] = coeffs[0]

    for k, name in enumerate(con_names):
        a_k = coeffs[1 + 2*k]    # cos coefficient
        b_k = coeffs[2 + 2*k]    # sin coefficient
        idx = np.where(valid)[0]
        amp[k, i0 + idx] = np.sqrt(a_k**2 + b_k**2)
        pha[k, i0 + idx] = np.degrees(np.arctan2(-b_k, a_k)) % 360

    if has_vel:
        valid_abs = i0 + np.where(valid)[0]  # absolute face indices
        for var_all, amp_arr, pha_arr in [
            (ux_all, amp_ux, pha_ux),
            (uy_all, amp_uy, pha_uy),
        ]:
            var_chunk = var_all[t_mask, i0:i1]
            var_v = var_chunk[:, valid].copy()
            var_v = np.where(~np.isfinite(var_v),
                             np.nanmean(var_v, axis=0), var_v)
            c_v   = AtA_inv_At @ var_v
            for k in range(n_con):
                a_k = c_v[1 + 2*k]
                b_k = c_v[2 + 2*k]
                amp_arr[k, valid_abs] = np.sqrt(a_k**2 + b_k**2)
                pha_arr[k, valid_abs] = (
                    np.degrees(np.arctan2(-b_k, a_k)) % 360)

    if (c + 1) % 10 == 0 or c == n_chunks - 1:
        print(f"  Chunk {c+1}/{n_chunks} done")

# Print domain-mean statistics
print("\n  Constituent summary (median over wet faces):")
print(f"  {'Name':4s}  {'Amp (m)':8s}  {'Phase (°)':10s}")
wet = np.isfinite(amp[0])
for k, name in enumerate(con_names):
    a_med = np.nanmedian(amp[k, wet])
    p_med = np.nanmedian(pha[k, wet])
    print(f"  {name:4s}  {a_med:8.4f}  {p_med:10.2f}")

# ── COMPUTE TIDAL RANGE ───────────────────────────────────────────────────────

print("\n[4] Computing tidal range envelope ...")
# Tidal range from percentiles (robust vs harmonic reconstruction)
wl_p95 = np.nanpercentile(wl_fit, 95, axis=0)
wl_p05 = np.nanpercentile(wl_fit,  5, axis=0)
tidal_range = wl_p95 - wl_p05
tidal_range = np.where(np.isfinite(tidal_range), tidal_range, np.nan)

# M4/M2 ratio — tidal asymmetry index
m2_idx = con_names.index("M2")
m4_idx = con_names.index("M4")
with np.errstate(divide="ignore", invalid="ignore"):
    asymmetry = np.where(
        amp[m2_idx] > 0.01,
        amp[m4_idx] / amp[m2_idx],
        np.nan
    )

print(f"  Tidal range: [{np.nanmin(tidal_range):.2f}, {np.nanmax(tidal_range):.2f}] m")
print(f"  M4/M2 ratio: [{np.nanmin(asymmetry):.3f}, {np.nanmax(asymmetry):.3f}]")

# ── SAVE NETCDF ───────────────────────────────────────────────────────────────

print(f"\n[5] Saving {OUT_NC.name} ...")
with nc.Dataset(str(OUT_NC), "w") as ds:
    ds.source = "tidal_harmonics.py — DFM harmonic analysis"
    ds.constituents = " ".join(con_names)

    ds.createDimension("face",        n_f)
    ds.createDimension("constituent", n_con)

    v = ds.createVariable("face_x", "f4", ("face",))
    v.units = "m"; v[:] = fx
    v = ds.createVariable("face_y", "f4", ("face",))
    v.units = "m"; v[:] = fy
    v = ds.createVariable("bed_level", "f4", ("face",))
    v.units = "m NAP"; v[:] = bl
    v = ds.createVariable("mean_waterlevel", "f4", ("face",))
    v.units = "m NAP"; v[:] = mean_wl
    v = ds.createVariable("tidal_range", "f4", ("face",))
    v.units = "m"; v.long_name = "P95-P05 water level range"; v[:] = tidal_range
    v = ds.createVariable("M4_M2_ratio", "f4", ("face",))
    v.long_name = "Tidal asymmetry index M4/M2"; v[:] = asymmetry

    v = ds.createVariable("amplitude", "f4", ("constituent", "face"))
    v.units = "m"; v.long_name = "Tidal amplitude per constituent"
    v[:] = amp
    v = ds.createVariable("phase", "f4", ("constituent", "face"))
    v.units = "degrees"; v.long_name = "Tidal phase per constituent (t0=RefDate)"
    v[:] = pha

    if has_vel:
        for name, arr_x, arr_y in [
            ("amplitude_ux", amp_ux, amp_uy),
            ("phase_ux",     pha_ux, pha_uy),
        ]:
            v = ds.createVariable(f"{name}", "f4", ("constituent", "face"))
            v[:] = arr_x
            v = ds.createVariable(f"{name.replace('ux','uy')}", "f4",
                                  ("constituent", "face"))
            v[:] = arr_y

print(f"  ✓ {OUT_NC}")

# ── LOAD MESH FOR PLOTTING ────────────────────────────────────────────────────

print("\n[6] Building triangulation for plots ...")
try:
    net_path = str(NET_NC) if NET_NC.exists() else str(
        next(BASE.glob("WesternScheldt_net_v*.nc")))
    with nc.Dataset(net_path) as ds:
        node_x     = np.array(ds["mesh2d_node_x"][:])
        node_y     = np.array(ds["mesh2d_node_y"][:])
        face_nodes = np.array(ds["mesh2d_face_nodes"][:]) - 1

    tris = []
    for fn in face_nodes:
        valid = fn[fn >= 0]
        if len(valid) == 3:
            tris.append(valid)
        elif len(valid) == 4:
            tris.append([valid[0], valid[1], valid[2]])
            tris.append([valid[0], valid[2], valid[3]])
    tris    = np.array(tris)
    triang  = mtri.Triangulation(node_x, node_y, tris)

    # Face → node interpolation via nearest face
    from scipy.spatial import cKDTree
    tree = cKDTree(np.column_stack([fx, fy]))
    _, f2n = tree.query(np.column_stack([node_x, node_y]))
    print(f"  ✓ {len(node_x):,} nodes, {len(tris):,} triangles")
    has_mesh = True
except Exception as e:
    print(f"  Mesh load failed: {e} — skipping plots")
    has_mesh = False

# ── PLOTTING HELPER ───────────────────────────────────────────────────────────

def plot_field(field_face, title, cbar_label, cmap, vmin, vmax,
               filename, contour_levels=None):
    """Plot a face-centred field using triangulation."""
    if not has_mesh:
        return

    field_node = field_face[f2n]
    nan_nodes  = ~np.isfinite(field_node)
    tri_mask   = np.any(nan_nodes[tris], axis=1)
    triang_m   = mtri.Triangulation(node_x, node_y, tris, mask=tri_mask)
    field_node = np.where(np.isfinite(field_node), field_node, 0.0)

    fig, ax = plt.subplots(figsize=(14, 7), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)

    tcf = ax.tricontourf(triang_m, field_node, levels=50,
                          cmap=cmap, vmin=vmin, vmax=vmax)
    cb  = plt.colorbar(tcf, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label(cbar_label, color="white", fontsize=9)
    cb.ax.tick_params(colors="white", labelsize=8)

    if contour_levels is not None:
        try:
            ax.tricontour(triang_m, field_node, levels=contour_levels,
                          colors=["white"], linewidths=0.7, alpha=0.6)
        except Exception:
            pass

    ax.set_xlim(*AOI_X); ax.set_ylim(*AOI_Y)
    ax.set_xlabel("RD New X (m)", color="white", fontsize=9)
    ax.set_ylabel("RD New Y (m)", color="white", fontsize=9)
    ax.tick_params(colors="white", labelsize=8)
    ax.spines[:].set_color("#333")
    ax.set_title(title, color="white", fontsize=10, fontweight="bold")

    out = FIG_DIR / filename
    fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  ✓ {filename}")

# ── FIGURES ───────────────────────────────────────────────────────────────────

print("\n[7] Generating figures ...")

# M2 amplitude
m2_amp = amp[m2_idx]
plot_field(
    m2_amp,
    "M2 Tidal Amplitude (m)\nWestern Scheldt — DFM harmonic analysis",
    "M2 Amplitude (m)",
    "plasma",
    vmin=0, vmax=np.nanpercentile(m2_amp[np.isfinite(m2_amp)], 98),
    filename="dfm_M2_amplitude.png",
    contour_levels=np.arange(0.5, 3.0, 0.25),
)

# M2 phase
m2_pha = pha[m2_idx]
plot_field(
    m2_pha,
    "M2 Tidal Phase (°, t₀ = 2020-01-01 UTC)\nPhase increases eastward → tidal wave propagation",
    "M2 Phase (°)",
    "hsv",
    vmin=0, vmax=360,
    filename="dfm_M2_phase.png",
    contour_levels=np.arange(0, 360, 30),
)

# Tidal range
plot_field(
    tidal_range,
    "Tidal Range (m) — P95 minus P05 water level\nAmplification from Vlissingen to Bath",
    "Tidal range (m)",
    "YlOrRd",
    vmin=np.nanpercentile(tidal_range[np.isfinite(tidal_range)], 2),
    vmax=np.nanpercentile(tidal_range[np.isfinite(tidal_range)], 98),
    filename="dfm_tidal_range.png",
    contour_levels=np.arange(2.0, 6.0, 0.5),
)

# Tidal asymmetry
asym_plot = np.clip(asymmetry, 0, 0.5)
plot_field(
    asym_plot,
    "Tidal Asymmetry Index M4/M2\n>0.1 = significant distortion | Red = flood-dominant",
    "M4/M2 ratio",
    "RdYlBu_r",
    vmin=0, vmax=0.3,
    filename="dfm_tidal_asymmetry.png",
    contour_levels=[0.05, 0.10, 0.15, 0.20],
)

# S2 amplitude (surge contribution)
s2_idx = con_names.index("S2")
plot_field(
    amp[s2_idx],
    "S2 Tidal Amplitude (m)\nSpring-neap modulation",
    "S2 Amplitude (m)",
    "viridis",
    vmin=0,
    vmax=np.nanpercentile(amp[s2_idx][np.isfinite(amp[s2_idx])], 98),
    filename="dfm_S2_amplitude.png",
)

# 6-panel constituent summary
if has_mesh:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor=DARK_BG)
    axes = axes.ravel()
    plot_cons = ["M2", "S2", "N2", "M4", "K1", "O1"]

    for i, cname in enumerate(plot_cons):
        ax  = axes[i]
        ax.set_facecolor(DARK_BG)
        k   = con_names.index(cname)
        fld = amp[k]
        fn  = fld[f2n]
        nan_n = ~np.isfinite(fn)
        tri_m = np.any(nan_n[tris], axis=1)
        tr_m  = mtri.Triangulation(node_x, node_y, tris, mask=tri_m)
        fn    = np.where(np.isfinite(fn), fn, 0.0)
        vmax  = np.nanpercentile(fld[np.isfinite(fld)], 98) if np.isfinite(fld).any() else 1.0

        tcf = ax.tricontourf(tr_m, fn, levels=40,
                              cmap="plasma", vmin=0, vmax=vmax)
        plt.colorbar(tcf, ax=ax, fraction=0.04).ax.tick_params(
            colors="white", labelsize=7)
        ax.set_xlim(*AOI_X); ax.set_ylim(*AOI_Y)
        ax.set_title(f"{cname}  (max={np.nanmax(fld):.3f}m)",
                     color="white", fontsize=9, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        ax.spines[:].set_color("#333")

    fig.suptitle(
        "Tidal Constituent Amplitudes — Western Scheldt DFM\n"
        "Least-squares harmonic analysis",
        color="white", fontsize=11, fontweight="bold"
    )
    out = FIG_DIR / "dfm_constituents_all.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  ✓ dfm_constituents_all.png")

print(f"\n✓ Harmonic analysis complete")
print(f"  NetCDF: {OUT_NC}")
print(f"  Figures: dfm_M2_amplitude.png, dfm_M2_phase.png,")
print(f"           dfm_tidal_range.png, dfm_tidal_asymmetry.png,")
print(f"           dfm_S2_amplitude.png, dfm_constituents_all.png")
