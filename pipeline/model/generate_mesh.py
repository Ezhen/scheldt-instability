"""
Western Scheldt — Variable-Resolution Mesh Generation
======================================================
Generates a flexible mesh for Delft3D FM with refinement zones:
  - Navigation channel (z < -5m NAP):  ~150m cells
  - Intertidal zone  (-5m to +1m NAP): ~300m cells
  - Open sea / polder buffer:           ~500m cells

The refinement polygons are derived automatically from bedlevel.xyz
— no manual digitising needed.

Output:
  dfm_clean/dflowfm/WesternScheldt_net_v5.nc   variable-resolution mesh

Usage:
    python generate_mesh.py
    python generate_mesh.py --channel-res 150 --flat-res 300 --outer-res 500
"""

import os
_P = "/home/ulg/mast/eivanov/.conda/envs/Yoda/lib/python3.10/site-packages/pyproj/proj_dir/share/proj"
os.environ["PROJ_DATA"] = _P; os.environ["PROJ_LIB"] = _P
import pyproj; pyproj.datadir.set_data_dir(_P)

import sys
import argparse
import numpy as np
import netCDF4 as nc
from pathlib import Path
from scipy.spatial import cKDTree

try:
    import meshkernel as mk
    from meshkernel import (
        MeshKernel, MeshKernelError,
        MakeGridParameters, GeometryList,
        MeshRefinementParameters, RefinementType,
        ProjectionType,
    )
except ImportError as e:
    sys.exit(f"meshkernel import failed: {e}")

# ── CONFIG ────────────────────────────────────────────────────────────────────

ap = argparse.ArgumentParser()
ap.add_argument("--channel-res", type=float, default=150.0,
                help="Cell size in navigation channel (m, default 150)")
ap.add_argument("--flat-res",    type=float, default=300.0,
                help="Cell size on intertidal flats (m, default 300)")
ap.add_argument("--outer-res",   type=float, default=500.0,
                help="Cell size in outer domain (m, default 500)")
ap.add_argument("--xyz",  default="dfm_clean/dflowfm/bedlevel.xyz",
                help="Path to bedlevel.xyz")
ap.add_argument("--out",  default="dfm_clean/dflowfm/WesternScheldt_net_v5.nc",
                help="Output net-file path")
args = ap.parse_args()

# Model domain (RD New, EPSG:28992)
MODEL_RD = {
    "x_min": -5000, "x_max": 80000,
    "y_min": 368000, "y_max": 401000,
}

# Depth thresholds for refinement zones
Z_CHANNEL  = -5.0   # deeper than this → channel refinement
Z_FLAT_MAX =  1.0   # shallower than this (and > Z_CHANNEL) → flat refinement

print("=== Western Scheldt — Variable-Resolution Mesh Generation ===\n")
print(f"  Channel resolution : {args.channel_res:.0f}m  (z < {Z_CHANNEL}m)")
print(f"  Flat resolution    : {args.flat_res:.0f}m  ({Z_CHANNEL} ≤ z ≤ {Z_FLAT_MAX}m)")
print(f"  Outer resolution   : {args.outer_res:.0f}m")

# ── STEP 1: LOAD BATHYMETRY AND EXTRACT ZONE POLYGONS ────────────────────────

print("\n[1] Loading bathymetry ...")
xyz = np.loadtxt(args.xyz, comments="#")
bx, by, bz = xyz[:,0], xyz[:,1], xyz[:,2]
print(f"  {len(bx):,} points, z=[{bz.min():.2f},{bz.max():.2f}]m")

def points_to_convex_hull_polygon(px, py, buffer_m=500.0):
    """
    Compute convex hull of point cloud and return as (x, y) polygon arrays.
    Adds a buffer by expanding each vertex outward from centroid.
    """
    from scipy.spatial import ConvexHull
    pts = np.column_stack([px, py])
    if len(pts) < 4:
        return None, None
    hull = ConvexHull(pts)
    hx = pts[hull.vertices, 0]
    hy = pts[hull.vertices, 1]
    # Buffer: expand from centroid
    cx, cy = hx.mean(), hy.mean()
    dirs = np.column_stack([hx - cx, hy - cy])
    norms = np.sqrt(dirs[:,0]**2 + dirs[:,1]**2)
    hx = hx + dirs[:,0] / norms * buffer_m
    hy = hy + dirs[:,1] / norms * buffer_m
    # Close polygon
    hx = np.append(hx, hx[0])
    hy = np.append(hy, hy[0])
    return hx, hy

def extract_zone_polygon(bx, by, bz, z_min, z_max, buffer_m=300.0):
    """Extract convex hull polygon for bathymetry zone z_min < z <= z_max."""
    mask = (bz >= z_min) & (bz < z_max)
    if mask.sum() < 10:
        print(f"  Warning: only {mask.sum()} points in zone [{z_min},{z_max}]m")
        return None, None
    px, py = bx[mask], by[mask]
    print(f"  Zone [{z_min:+.0f},{z_max:+.0f}]m: {mask.sum():,} points")

    # Use alpha-shape-like approach: grid the points and find boundary
    # For simplicity use convex hull with buffer
    hx, hy = points_to_convex_hull_polygon(px, py, buffer_m)
    return hx, hy

print("\n  Extracting channel polygon (z < -5m) ...")
ch_x, ch_y = extract_zone_polygon(bx, by, bz, -200.0, Z_CHANNEL, buffer_m=400.0)

print("  Extracting intertidal polygon (-5m to +1m) ...")
fl_x, fl_y = extract_zone_polygon(bx, by, bz, Z_CHANNEL, Z_FLAT_MAX, buffer_m=300.0)

# ── STEP 2: GENERATE COARSE BASE MESH ────────────────────────────────────────

print(f"\n[2] Generating base mesh at {args.outer_res:.0f}m ...")

mk_instance = MeshKernel(projection=ProjectionType.CARTESIAN)

# Domain polygon (full model extent)
dom_x = np.array([
    MODEL_RD["x_min"], MODEL_RD["x_max"],
    MODEL_RD["x_max"], MODEL_RD["x_min"],
    MODEL_RD["x_min"]
], dtype=np.float64)
dom_y = np.array([
    MODEL_RD["y_min"], MODEL_RD["y_min"],
    MODEL_RD["y_max"], MODEL_RD["y_max"],
    MODEL_RD["y_min"]
], dtype=np.float64)

grid_params = MakeGridParameters()
grid_params.origin_x         = float(MODEL_RD["x_min"])
grid_params.origin_y         = float(MODEL_RD["y_min"])
grid_params.upper_right_x    = float(MODEL_RD["x_max"])
grid_params.upper_right_y    = float(MODEL_RD["y_max"])
grid_params.block_size_x     = float(args.outer_res)
grid_params.block_size_y     = float(args.outer_res)

geometry_list = GeometryList(
    x_coordinates=dom_x,
    y_coordinates=dom_y
)

mk_instance.mesh2d_make_rectangular_mesh_from_polygon(
    grid_params, geometry_list
)
mesh = mk_instance.mesh2d_get()
print(f"  Base mesh: {mesh.node_x.size} nodes, {len(mesh.face_nodes)//4} faces")

# ── STEP 3: CASULLI REFINEMENT — INTERTIDAL ZONE ────────────────────────────
# Casulli refinement maintains orthogonality — preferred over polygon refine

if fl_x is not None:
    print(f"\n[3] Casulli refinement — intertidal zone to ~{args.flat_res:.0f}m ...")
    n_flat = max(1, round(np.log2(args.outer_res / args.flat_res)))
    flat_polygon = GeometryList(
        x_coordinates=fl_x.astype(np.float64),
        y_coordinates=fl_y.astype(np.float64)
    )
    for i in range(n_flat):
        try:
            mk_instance.mesh2d_casulli_refinement_on_polygon(flat_polygon)
            mesh = mk_instance.mesh2d_get()
            print(f"  Pass {i+1}: {mesh.node_x.size} nodes")
        except Exception as e:
            print(f"  Pass {i+1} failed: {e}")
            break
else:
    print("\n[3] Skipping flat refinement")

# ── STEP 4: CASULLI REFINEMENT — NAVIGATION CHANNEL ─────────────────────────

if ch_x is not None:
    print(f"\n[4] Casulli refinement — channel zone to ~{args.channel_res:.0f}m ...")
    n_ch = max(1, round(np.log2(args.outer_res / args.channel_res)))
    channel_polygon = GeometryList(
        x_coordinates=ch_x.astype(np.float64),
        y_coordinates=ch_y.astype(np.float64)
    )
    for i in range(n_ch):
        try:
            mk_instance.mesh2d_casulli_refinement_on_polygon(channel_polygon)
            mesh = mk_instance.mesh2d_get()
            print(f"  Pass {i+1}: {mesh.node_x.size} nodes")
        except Exception as e:
            print(f"  Pass {i+1} failed: {e}")
            break
else:
    print("\n[4] Skipping channel refinement")

# ── STEP 4b: ORTHOGONALIZATION ───────────────────────────────────────────────

print("\n[4b] Orthogonalizing mesh ...")
try:
    from meshkernel import OrthogonalizationParameters
    orth_params = OrthogonalizationParameters()
    orth_params.outer_iterations = 25
    orth_params.boundary_iterations = 25
    orth_params.inner_iterations = 25
    orth_params.orthogonalization_to_smoothing_factor = 0.975
    domain_gl = GeometryList(
        x_coordinates=dom_x.astype(np.float64),
        y_coordinates=dom_y.astype(np.float64)
    )
    empty_gl = GeometryList(
        x_coordinates=np.array([], dtype=np.float64),
        y_coordinates=np.array([], dtype=np.float64)
    )
    mk_instance.mesh2d_compute_orthogonalization(
        ProjectionType.CARTESIAN, orth_params, empty_gl, domain_gl
    )
    mesh = mk_instance.mesh2d_get()
    print(f"  ✓ Done — {mesh.node_x.size} nodes")
except Exception as e:
    print(f"  Orthogonalization failed: {e}")

# ── STEP 5: BAKE BATHYMETRY ONTO NODES ───────────────────────────────────────

print("\n[5] Baking bathymetry onto mesh nodes ...")
mesh = mk_instance.mesh2d_get()
nx = np.array(mesh.node_x)
ny = np.array(mesh.node_y)
print(f"  Final mesh: {len(nx)} nodes")

tree = cKDTree(np.column_stack([bx, by]))
dist, idx = tree.query(np.column_stack([nx, ny]), k=4)
weights = 1.0 / np.maximum(dist, 1e-6)
weights /= weights.sum(axis=1, keepdims=True)
nz = (weights * bz[idx]).sum(axis=1)
print(f"  Node z: [{nz.min():.2f},{nz.max():.2f}]m")

# ── STEP 6: WRITE NET-FILE ────────────────────────────────────────────────────

print(f"\n[6] Writing net-file: {args.out} ...")

out_path = Path(args.out)
out_path.parent.mkdir(parents=True, exist_ok=True)

edge_nodes = np.array(mesh.edge_nodes).reshape(-1, 2) + 1  # 1-based
face_nodes_flat = np.array(mesh.face_nodes)
nodes_per_face  = np.array(mesh.nodes_per_face)

max_nodes = int(nodes_per_face.max()) if len(nodes_per_face) > 0 else 4
n_faces = len(nodes_per_face)
face_nodes_2d = np.full((n_faces, max_nodes), -1, dtype=np.int32)
ptr = 0
for i, npf in enumerate(nodes_per_face):
    face_nodes_2d[i, :npf] = face_nodes_flat[ptr:ptr+npf] + 1  # 1-based
    ptr += npf

with nc.Dataset(str(out_path), "w", format="NETCDF4") as ds:
    ds.Conventions     = "CF-1.8 UGRID-1.0"
    ds.institution     = "University of Liège / MAST"
    ds.source          = "generate_mesh.py — meshkernel variable-resolution"
    ds.mesh_resolution = (f"channel={args.channel_res:.0f}m "
                          f"flat={args.flat_res:.0f}m "
                          f"outer={args.outer_res:.0f}m")

    # Dimensions
    n_nodes = len(nx)
    n_edges = len(edge_nodes)
    ds.createDimension("mesh2d_nNodes", n_nodes)
    ds.createDimension("mesh2d_nEdges", n_edges)
    ds.createDimension("mesh2d_nFaces", n_faces)
    ds.createDimension("mesh2d_nMax_face_nodes", max_nodes)
    ds.createDimension("Two", 2)

    # Mesh topology
    mesh_var = ds.createVariable("mesh2d", "i4", ())
    mesh_var.cf_role            = "mesh_topology"
    mesh_var.topology_dimension = 2
    mesh_var.node_coordinates   = "mesh2d_node_x mesh2d_node_y"
    mesh_var.edge_node_connectivity = "mesh2d_edge_nodes"
    mesh_var.face_node_connectivity = "mesh2d_face_nodes"

    # Node coordinates
    vx = ds.createVariable("mesh2d_node_x", "f8", ("mesh2d_nNodes",))
    vx.units     = "m"; vx.standard_name = "projection_x_coordinate"
    vx[:]        = nx

    vy = ds.createVariable("mesh2d_node_y", "f8", ("mesh2d_nNodes",))
    vy.units     = "m"; vy.standard_name = "projection_y_coordinate"
    vy[:]        = ny

    vz = ds.createVariable("mesh2d_node_z", "f8", ("mesh2d_nNodes",))
    vz.units     = "m"; vz.standard_name = "altitude"
    vz.positive  = "up"
    vz[:]        = nz

    # Edge nodes
    ve = ds.createVariable("mesh2d_edge_nodes", "i4",
                           ("mesh2d_nEdges", "Two"))
    ve.cf_role     = "edge_node_connectivity"
    ve.start_index = 1
    ve[:]          = edge_nodes

    # Face nodes
    vf = ds.createVariable("mesh2d_face_nodes", "i4",
                           ("mesh2d_nFaces", "mesh2d_nMax_face_nodes"),
                           fill_value=-1)
    vf.cf_role       = "face_node_connectivity"
    vf.start_index   = 1
    vf[:]            = face_nodes_2d

    # Face centres
    fc_x = ds.createVariable("mesh2d_face_x", "f8", ("mesh2d_nFaces",))
    fc_y = ds.createVariable("mesh2d_face_y", "f8", ("mesh2d_nFaces",))
    # Compute face centres as mean of valid nodes
    for i in range(n_faces):
        valid = face_nodes_2d[i, :nodes_per_face[i]] - 1
        fc_x[i] = nx[valid].mean()
        fc_y[i] = ny[valid].mean()
    fc_x.units = "m"; fc_y.units = "m"

    # CRS
    crs_var = ds.createVariable("projected_coordinate_system", "i4", ())
    crs_var.epsg                    = 28992
    crs_var.grid_mapping_name       = "Unknown projected"
    crs_var.longitude_of_prime_meridian = 0.0
    crs_var.semi_major_axis         = 6378137.0
    crs_var.semi_minor_axis         = 6356752.314245

print(f"  ✓ {out_path.name}")
print(f"\n{'='*55}")
print(f"✓ Mesh complete")
print(f"  Nodes : {n_nodes:,}")
print(f"  Edges : {n_edges:,}")
print(f"  Faces : {n_faces:,}")
print(f"\nUpdate MDU:")
print(f"  NetFile = {out_path.name}")
print(f"\nVerify in DFM:")
print(f"  grep 'netnodes\\|netcells' dfm_run.log")
