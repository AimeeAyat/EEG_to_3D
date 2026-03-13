"""
pixel2mesh_reconstruction.py

Pixel2Mesh-inspired 3D mesh reconstruction from Wonder3D multi-view RGB + normal maps.

Approach (inspired by Wang et al., ECCV 2018 "Pixel2Mesh"):
  - Pixel2Mesh deforms an ellipsoid using graph CNNs conditioned on multi-scale
    perceptual image features. Here we have something BETTER: we have cross-domain
    diffusion outputs (6 RGB views + 6 surface normal maps) from Wonder3D.
  - We use the 6 silhouettes for space carving (visual hull), then use the
    normal maps as geometric supervision to assign surface normals to every vertex
    (analogous to Pixel2Mesh's perceptual feature projection at each vertex).
  - Two reconstruction methods are provided:
      A) Space-Carving + Marching Cubes + Normal-guided refinement
      B) Ray-surface intersection + Poisson reconstruction (Open3D)

Camera convention (Wonder3D / BlenderProc, orthographic, elevation=0):
  Azimuth theta -> camera at [sin(theta), 0, cos(theta)] x radius, looking at origin
  Camera axes:
    x_cam (right)    = [ cos(theta), 0, -sin(theta)]
    y_cam (up)       = [0, 1, 0]
    z_cam (backward) = [sin(theta), 0,  cos(theta)]
  -> R_c2w = rotation around Y by theta

Normal map convention (Wonder3D):
  Stored as uint8 RGB in [0,255]: n = pixel/127.5 - 1 -> [-1,1]
  R -> nx (right in camera space)
  G -> ny (up in camera space)
  B -> nz (toward camera, positive = front-facing)

Usage:
    # List available objects
    python pixel2mesh_reconstruction.py --list

    # Reconstruct one object (both methods)
    python pixel2mesh_reconstruction.py --object airliner_airliner_004

    # Fast carving-only, lower resolution
    python pixel2mesh_reconstruction.py --object airliner_airliner_004 --method carving --resolution 96

    # High-quality Poisson only
    python pixel2mesh_reconstruction.py --object airliner_airliner_004 --method poisson --poisson_depth 10

    # Reconstruct all objects
    python pixel2mesh_reconstruction.py

    # Open visualizer after reconstruction
    python pixel2mesh_reconstruction.py --object airliner_airliner_004 --visualize

Requirements:
    open3d >= 0.19, numpy, Pillow, scipy, scikit-image
    All available in a standard conda/pip environment.
    GPU (RTX 5090) is used automatically by Open3D for Poisson if available.
"""

import os
import sys
import time
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import open3d as o3d
from scipy import ndimage
from skimage.measure import marching_cubes


# =============================================================================
# CAMERA CONFIGURATION
# =============================================================================

# 6 orthographic views used by Wonder3D (azimuths in degrees, elevation=0)
VIEWS = [
    ("front",       0.0),
    ("front_left",  45.0),
    ("left",        90.0),
    ("back",        180.0),
    ("right",       270.0),
    ("front_right", 315.0),
]

# Default Wonder3D outputs directory
_HERE = Path(__file__).parent
DEFAULT_WONDER3D_DIR = _HERE / "Wonder3D/outputs/cropsize-192-cfg3.0"
DEFAULT_OUTPUT_DIR   = _HERE / "outputs/meshes"


# =============================================================================
# CAMERA MATH
# =============================================================================

def rotation_y(azimuth_deg: float) -> np.ndarray:
    """
    Camera-to-world rotation matrix R_c2w for camera at azimuth_deg around Y axis.

    Column i of R_c2w = i-th camera axis expressed in world coordinates:
      col 0 = x_cam in world (right)
      col 1 = y_cam in world (up)
      col 2 = z_cam in world (backward, away from scene)

    world_vector = R_c2w @ camera_vector
    camera_vector = R_c2w.T @ world_vector
    """
    t = np.radians(azimuth_deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([
        [ c, 0.0,  s],
        [0.0, 1.0, 0.0],
        [-s, 0.0,  c],
    ], dtype=np.float64)


def project_orthographic(
    world_pts: np.ndarray,   # [N, 3]
    R_w2c: np.ndarray,       # [3, 3]  R_c2w.T
    img_size: int,
    orth_scale: float,       # object occupies [-orth_scale, orth_scale] in camera XY
) -> tuple:
    """
    Project world points into image pixel coordinates via orthographic projection.

    Returns:
        u_px  [N] float  ? horizontal pixel (0 = left, img_size-1 = right)
        v_px  [N] float  ? vertical pixel   (0 = top,  img_size-1 = bottom)
        depth [N] float  ? camera-space z (positive = toward camera)
    """
    cam = world_pts @ R_w2c.T          # [N, 3]

    # camera x  ->  image u  (right = increasing u)
    # camera y  ->  image v  (up = decreasing v, image 0 is top)
    u_norm = cam[:, 0] / orth_scale    # [-1, 1]
    v_norm = cam[:, 1] / orth_scale    # [-1, 1]

    u_px = (u_norm + 1.0) * 0.5 * img_size        # [0, img_size]
    v_px = (1.0 - (v_norm + 1.0) * 0.5) * img_size  # [0, img_size]  (y flipped)
    depth = cam[:, 2]                              # z in camera space

    return u_px.astype(np.float32), v_px.astype(np.float32), depth.astype(np.float32)


def sample_image(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Nearest-neighbour sampling of img [H, W, C] at fractional pixel coords u, v [N].
    Returns [N, C] or [N] (if img is 2D).
    """
    H, W = img.shape[:2]
    ui = np.clip(np.round(u).astype(int), 0, W - 1)
    vi = np.clip(np.round(v).astype(int), 0, H - 1)
    return img[vi, ui]


# =============================================================================
# DATA LOADING
# =============================================================================

def load_views(obj_dir: Path) -> list:
    """
    Load all 6 Wonder3D views.  Returns list of dicts with keys:
        name, azimuth, rgb [H,W,3] float32 [0,1],
        alpha [H,W] float32 [0,1], normal [H,W,3] float32 [-1,1],
        R_c2w [3,3] float64
    """
    views = []
    for view_name, azimuth in VIEWS:
        rgb_path  = obj_dir / "masked_colors" / f"rgb_000_{view_name}.png"
        norm_path = obj_dir / "normals"        / f"normals_000_{view_name}.png"

        if not rgb_path.exists():
            # Fall back to root-level files
            rgb_path  = obj_dir / f"rgb_000_{view_name}.png"
            norm_path = obj_dir / f"normals_000_{view_name}.png"

        if not rgb_path.exists() or not norm_path.exists():
            print(f"  [WARN] Missing files for view '{view_name}' ? skipping")
            continue

        # --- RGB + alpha ---
        rgba      = np.array(Image.open(rgb_path).convert("RGBA"), dtype=np.float32)
        rgb       = rgba[:, :, :3] / 255.0
        alpha     = rgba[:, :, 3]  / 255.0

        # --- Normal map ---
        norm_rgb  = np.array(Image.open(norm_path).convert("RGB"), dtype=np.float32)
        normal    = norm_rgb / 127.5 - 1.0   # [0,255] -> [-1,1]

        views.append({
            "name":    view_name,
            "azimuth": azimuth,
            "rgb":     rgb,
            "alpha":   alpha,
            "normal":  normal,
            "R_c2w":   rotation_y(azimuth),
        })
        print(f"  Loaded '{view_name}' (azimuth={azimuth:.0f}deg)  "
              f"img={rgb.shape[1]}x{rgb.shape[0]}")

    return views


# =============================================================================
# STAGE 1: SPACE CARVING  ->  VISUAL HULL OCCUPANCY GRID
# =============================================================================

def space_carve(
    views: list,
    grid_res:        int   = 128,
    orth_scale:      float = 1.05,
    alpha_threshold: float = 0.3,
    closing_iters:   int   = 3,
) -> np.ndarray:
    """
    Multi-view space carving using silhouette (alpha) masks.

    Voxel grid spans [-orth_scale, orth_scale]^3.
    A voxel is OCCUPIED iff it projects INSIDE every view's silhouette.

    Post-processing:
      1. Morphological closing (binary_closing) to fill gaps in thin parts
      2. Keep only the largest connected component (removes floating fragments)

    Returns bool occupancy grid [G, G, G].
    """
    print(f"  Grid: {grid_res}^3  scale={orth_scale:.2f}  alpha_thr={alpha_threshold}")

    coords    = np.linspace(-orth_scale, orth_scale, grid_res, dtype=np.float32)
    X, Y, Z   = np.meshgrid(coords, coords, coords, indexing="ij")
    world_pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)  # [G^3, 3]

    occupancy = np.ones(grid_res ** 3, dtype=bool)
    img_size  = views[0]["alpha"].shape[0]

    for vd in views:
        R_w2c = vd["R_c2w"].T
        alpha = vd["alpha"]

        u, v, _ = project_orthographic(world_pts, R_w2c, img_size, orth_scale)

        in_bounds   = (u >= 0) & (u < img_size) & (v >= 0) & (v < img_size)
        alpha_vals  = sample_image(alpha, u, v)        # [G^3]
        in_sil      = in_bounds & (alpha_vals >= alpha_threshold)

        occupancy  &= in_sil
        print(f"    After '{vd['name']}': {occupancy.sum():,} voxels")

    occupancy = occupancy.reshape(grid_res, grid_res, grid_res)

    # --- Morphological closing: fills gaps in thin parts (wings, handles, etc.) ---
    if closing_iters > 0:
        struct = ndimage.generate_binary_structure(3, 2)   # 26-connectivity
        occupancy = ndimage.binary_closing(occupancy, structure=struct,
                                           iterations=closing_iters)
        print(f"  After closing (iters={closing_iters}): {occupancy.sum():,} voxels")

    # --- Keep only largest connected component (removes floating fragments) ---
    labeled, n_labels = ndimage.label(occupancy)
    if n_labels > 1:
        counts  = ndimage.sum(occupancy, labeled, range(1, n_labels + 1))
        largest = int(np.argmax(counts)) + 1
        occupancy = (labeled == largest)
        print(f"  Kept largest of {n_labels} components: {occupancy.sum():,} voxels")

    return occupancy


# =============================================================================
# STAGE 2A: SDF + MARCHING CUBES  ->  INITIAL MESH
# =============================================================================

def occupancy_to_mesh(
    occupancy:    np.ndarray,   # [G, G, G] bool
    orth_scale:   float = 1.05,
    gaussian_sigma: float = 1.0,
) -> tuple:
    """
    Binary occupancy -> watertight mesh via Eikonal-approximate SDF + Marching Cubes.

    1. SDF  = dist_inside ? dist_outside   (Euclidean distance transform)
    2. Gaussian smooth to remove staircase artefacts
    3. Marching cubes at level 0

    Returns (vertices [N,3] world-space float32, faces [M,3] int32).
    """
    print("  Distance transform (SDF)...")
    dist_in  = ndimage.distance_transform_edt( occupancy).astype(np.float32)
    dist_out = ndimage.distance_transform_edt(~occupancy).astype(np.float32)
    sdf      = dist_in - dist_out          # + inside, ? outside

    if gaussian_sigma > 0:
        print(f"  Gaussian smooth sigma={gaussian_sigma}")
        sdf = ndimage.gaussian_filter(sdf, sigma=gaussian_sigma).astype(np.float32)

    print("  Marching cubes...")
    G = occupancy.shape[0]
    verts_vox, faces, _, _ = marching_cubes(sdf, level=0.0, spacing=(1, 1, 1))

    # Voxel index -> world coordinate: [0, G-1] -> [-orth_scale, orth_scale]
    verts_world = (verts_vox / (G - 1)) * 2.0 * orth_scale - orth_scale

    print(f"  Mesh: {len(verts_world):,} vertices, {len(faces):,} triangles")
    return verts_world.astype(np.float32), faces.astype(np.int32)


# =============================================================================
# STAGE 3: PIXEL2MESH-STYLE NORMAL PROJECTION  ->  VERTEX NORMALS
# =============================================================================

def assign_normals(
    vertices:   np.ndarray,   # [N, 3]
    views:      list,
    orth_scale: float = 1.05,
) -> np.ndarray:
    """
    Pixel2Mesh-inspired step: project each vertex into every camera view,
    sample the normal map, convert to world space, and blend weighted by:
      - visibility (alpha mask)
      - facing angle (how directly the camera sees that surface point)

    This replaces Pixel2Mesh's "perceptual feature unprojection" with the
    directly supervised normal maps from cross-domain diffusion.

    Returns vertex_normals [N, 3] in world space (unit vectors).
    """
    img_size = views[0]["normal"].shape[0]
    N        = len(vertices)
    n_views  = len(views)

    accum_normals = np.zeros((N, 3), dtype=np.float64)
    accum_weights = np.zeros(N,      dtype=np.float64)

    for vd in views:
        R_c2w = vd["R_c2w"]
        R_w2c = R_c2w.T
        norm_map = vd["normal"]   # [H, W, 3]  camera space [-1,1]
        alpha    = vd["alpha"]

        u, v, depth = project_orthographic(vertices, R_w2c, img_size, orth_scale)

        in_bounds  = (u >= 0) & (u < img_size) & (v >= 0) & (v < img_size)
        alpha_vals = sample_image(alpha, u, v)
        visible    = in_bounds & (alpha_vals > 0.3)

        # Sample normal in camera space
        cam_nrm = sample_image(norm_map, u, v).astype(np.float64)  # [N, 3]

        # Normalise sampled normals
        nrm_len = np.linalg.norm(cam_nrm, axis=1, keepdims=True).clip(min=1e-6)
        cam_nrm = cam_nrm / nrm_len

        # Convert camera -> world: world_n = R_c2w @ cam_n
        world_nrm = cam_nrm @ R_c2w.T          # [N, 3]

        # Weight = visibility x |cos(facing angle)|
        # Camera looks along -z_cam in world = -(R_c2w col 2) = -R_c2w[:,2]
        cam_fwd_world = -R_c2w[:, 2]           # world vector camera looks toward
        facing = np.clip(world_nrm @ (-cam_fwd_world), 0.0, 1.0)

        weight = visible.astype(np.float64) * (facing + 0.05)

        accum_normals += world_nrm * weight[:, None]
        accum_weights += weight

    # Weighted average, normalise
    safe_w = accum_weights[:, None].clip(min=1e-6)
    vertex_normals = accum_normals / safe_w
    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True).clip(min=1e-6)
    vertex_normals = vertex_normals / norms

    return vertex_normals.astype(np.float32)


# =============================================================================
# STAGE 4: VERTEX COLOR ASSIGNMENT
# =============================================================================

def assign_colors(
    vertices:       np.ndarray,   # [N, 3]
    vertex_normals: np.ndarray,   # [N, 3]
    views:          list,
    orth_scale:     float = 1.05,
) -> np.ndarray:
    """
    Project each vertex to the best-facing camera view and sample RGB.
    Returns vertex_colors [N, 3] in [0, 1].
    """
    img_size = views[0]["rgb"].shape[0]
    N        = len(vertices)

    accum_colors  = np.zeros((N, 3), dtype=np.float64)
    accum_weights = np.zeros(N,      dtype=np.float64)

    for vd in views:
        R_c2w = vd["R_c2w"]
        R_w2c = R_c2w.T
        rgb   = vd["rgb"]
        alpha = vd["alpha"]

        u, v, _ = project_orthographic(vertices, R_w2c, img_size, orth_scale)

        in_bounds  = (u >= 0) & (u < img_size) & (v >= 0) & (v < img_size)
        alpha_vals = sample_image(alpha, u, v)
        visible    = in_bounds & (alpha_vals > 0.3)

        colors = sample_image(rgb, u, v).astype(np.float64)    # [N, 3]

        cam_fwd_world = -R_c2w[:, 2]
        facing = np.clip(vertex_normals @ (-cam_fwd_world), 0.0, 1.0)
        weight = visible.astype(np.float64) * (facing + 0.05)

        accum_colors  += colors * weight[:, None]
        accum_weights += weight

    safe_w = accum_weights[:, None].clip(min=1e-6)
    return np.clip(accum_colors / safe_w, 0.0, 1.0).astype(np.float32)


# =============================================================================
# STAGE 2B: RAY-CASTING + POISSON RECONSTRUCTION
# =============================================================================

def build_surface_point_cloud(
    views:      list,
    occupancy:  np.ndarray,    # [G, G, G] bool  (visual hull from space carving)
    grid_res:   int,
    orth_scale: float = 1.05,
) -> o3d.geometry.PointCloud:
    """
    Build an oriented point cloud for Poisson reconstruction.

    For each foreground pixel in each view:
      1. Cast an orthographic ray through the visual hull.
      2. Take the first voxel entry point as the 3D surface sample.
      3. Use the normal map as the surface normal at that point.
      4. Use the RGB map as the color.

    This avoids the "all points on a flat plane" problem of naive orthographic
    back-projection without depth information.
    """
    all_pts  = []
    all_nrms = []
    all_clrs = []
    img_size = views[0]["rgb"].shape[0]

    # Precompute 3D coordinates of each voxel centre
    coords = np.linspace(-orth_scale, orth_scale, grid_res, dtype=np.float32)

    for vd in views:
        R_c2w  = vd["R_c2w"]
        R_w2c  = R_c2w.T
        rgb    = vd["rgb"]
        alpha  = vd["alpha"]
        normal = vd["normal"]

        # Pixel grid
        pu = np.arange(img_size, dtype=np.float32)
        pv = np.arange(img_size, dtype=np.float32)
        UU, VV = np.meshgrid(pu, pv)         # [H, W]

        # Foreground pixels only
        mask = (alpha > 0.4).ravel()
        UU_f = UU.ravel()[mask]
        VV_f = VV.ravel()[mask]

        # Camera-space XY for each foreground pixel (orthographic image plane)
        x_cam = ((UU_f / img_size) * 2.0 - 1.0) * orth_scale
        y_cam = (1.0 - (VV_f / img_size) * 2.0) * orth_scale

        # Ray direction in world space = camera viewing direction = -R_c2w[:,2]
        # (camera z-axis is BACKWARD; camera looks along -z_cam = -R_c2w[:,2] in world)
        ray_dir_world = -R_c2w[:, 2]          # [3]  pointing INTO scene

        # Ray starts on the CAMERA SIDE of the bounding box: z_cam = +orth_scale
        # (camera z+ is backward, so starting at z_cam = +orth_scale puts us
        #  on the camera side; the ray then travels into the scene toward z_cam = -orth_scale)
        ray_orig_cam  = np.stack([x_cam, y_cam, np.ones_like(x_cam) * orth_scale], axis=1)
        ray_orig_world = ray_orig_cam @ R_c2w.T    # [N_px, 3]

        # Step along the ray through the voxel grid
        # t spans [0, 2*orth_scale], covering the full bounding box depth
        n_px = ray_orig_world.shape[0]
        t_vals = np.linspace(0, 2 * orth_scale, grid_res, dtype=np.float32)

        # Broadcast: [N_px, 1, 3] + [1, G, 3]
        ray_pts = ray_orig_world[:, None, :] + t_vals[None, :, None] * ray_dir_world[None, None, :]
        # ray_pts: [N_px, G, 3]

        # Convert world coords -> voxel indices
        def world_to_vox(wc):  # wc: [..., 3]
            return np.clip(
                ((wc + orth_scale) / (2 * orth_scale) * (grid_res - 1)).astype(int),
                0, grid_res - 1,
            )

        vox_idx = world_to_vox(ray_pts)   # [N_px, G, 3]
        # occupancy lookup [N_px, G]
        occ_along = occupancy[vox_idx[:, :, 0], vox_idx[:, :, 1], vox_idx[:, :, 2]]

        # Find first entry into occupied region
        first_hit = occ_along.argmax(axis=1)   # [N_px]  (index of first True)
        has_hit   = occ_along.any(axis=1)       # [N_px]

        valid     = has_hit
        hit_pts   = ray_pts[np.arange(n_px), first_hit, :]   # [N_px, 3]

        surface_pts  = hit_pts[valid]            # [M, 3]
        cam_nrm_flat = normal.reshape(-1, 3)[mask][valid]     # [M, 3]  camera-space
        rgb_flat     = rgb.reshape(-1, 3)[mask][valid]        # [M, 3]

        # Normalise and convert normals to world space
        cn_len      = np.linalg.norm(cam_nrm_flat, axis=1, keepdims=True).clip(min=1e-6)
        cam_nrm_flat = cam_nrm_flat / cn_len
        world_nrm   = cam_nrm_flat @ R_c2w.T      # [M, 3]

        all_pts.append(surface_pts)
        all_nrms.append(world_nrm)
        all_clrs.append(rgb_flat)

        print(f"    '{vd['name']}': {valid.sum():,} surface points")

    pts  = np.concatenate(all_pts,  axis=0).astype(np.float64)
    nrms = np.concatenate(all_nrms, axis=0).astype(np.float64)
    clrs = np.concatenate(all_clrs, axis=0).astype(np.float64)

    # Normalise world normals
    nn = np.linalg.norm(nrms, axis=1, keepdims=True).clip(min=1e-6)
    nrms = nrms / nn

    pcd = o3d.geometry.PointCloud()
    pcd.points  = o3d.utility.Vector3dVector(pts)
    pcd.normals = o3d.utility.Vector3dVector(nrms)
    pcd.colors  = o3d.utility.Vector3dVector(np.clip(clrs, 0, 1))

    print(f"  Total: {len(pts):,} surface points")
    return pcd


def poisson_from_pcd(
    pcd:           o3d.geometry.PointCloud,
    poisson_depth: int   = 9,
    density_trim:  float = 0.05,
    voxel_downsample: float = 0.02,
) -> o3d.geometry.TriangleMesh:
    """Run Open3D Poisson reconstruction with density trimming."""

    if voxel_downsample > 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_downsample)
        print(f"  After voxel downsample ({voxel_downsample}): {len(pcd.points):,} pts")

    print(f"  Poisson reconstruction (depth={poisson_depth})...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=poisson_depth, n_threads=-1
    )

    # Remove low-density artefact vertices
    d = np.asarray(densities)
    remove = d < np.quantile(d, density_trim)
    mesh.remove_vertices_by_mask(remove)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()

    print(f"  Poisson mesh: {len(mesh.vertices):,} verts, "
          f"{len(mesh.triangles):,} tris")
    return mesh


# =============================================================================
# SAVE + VISUALISE
# =============================================================================

def make_o3d_mesh(
    vertices:       np.ndarray,
    faces:          np.ndarray,
    vertex_colors:  np.ndarray,
    vertex_normals: np.ndarray,
) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices       = o3d.utility.Vector3dVector(vertices.astype(np.float64))
    mesh.triangles      = o3d.utility.Vector3iVector(faces.astype(np.int32))
    mesh.vertex_colors  = o3d.utility.Vector3dVector(vertex_colors.astype(np.float64))
    mesh.vertex_normals = o3d.utility.Vector3dVector(vertex_normals.astype(np.float64))
    return mesh


def save_mesh(mesh: o3d.geometry.TriangleMesh, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ply = out_dir / f"{stem}.ply"
    obj = out_dir / f"{stem}.obj"
    o3d.io.write_triangle_mesh(str(ply), mesh, write_ascii=False, compressed=True)
    o3d.io.write_triangle_mesh(str(obj), mesh)
    print(f"  Saved -> {ply.name}  |  {obj.name}")
    # Print mesh stats
    print(f"  Stats: {len(mesh.vertices):,} vertices  |  {len(mesh.triangles):,} triangles")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def reconstruct(
    wonder3d_dir:  str,
    object_name:   str,
    output_dir:    str,
    method:        str   = "both",
    grid_res:      int   = 128,
    poisson_depth: int   = 9,
    visualize:     bool  = False,
) -> None:
    t0      = time.perf_counter()
    obj_dir = Path(wonder3d_dir) / object_name
    out_dir = Path(output_dir)   / object_name

    print(f"\n{'='*62}")
    print(f"  Object  : {object_name}")
    print(f"  Method  : {method}")
    print(f"  Grid    : {grid_res}^3")
    print(f"  Poisson : depth={poisson_depth}")
    print(f"  Output  : {out_dir}")
    print(f"{'='*62}")

    # ?? 1. Load all views ?????????????????????????????????????????????????
    print("\n[1/4] Loading Wonder3D views...")
    views = load_views(obj_dir)
    if not views:
        print("  ERROR: No views loaded. Check object directory.")
        return

    orth_scale = 1.05

    # ?? 2. Space carving (needed for both methods) ????????????????????????
    print("\n[2/4] Space carving (visual hull)...")
    occupancy = space_carve(views, grid_res=grid_res, orth_scale=orth_scale)
    n_occupied = occupancy.sum()
    print(f"  Visual hull: {n_occupied:,} voxels  ({n_occupied/grid_res**3*100:.1f}% fill)")

    if n_occupied == 0:
        print("  ERROR: Empty visual hull. Check alpha masks.")
        return

    meshes_to_show = []

    # ?? 3A. Space-Carving + Marching Cubes ??????????????????????????????
    if method in ("carving", "both"):
        print("\n[3/4] Marching cubes...")
        verts, faces = occupancy_to_mesh(occupancy, orth_scale=orth_scale, gaussian_sigma=1.0)

        print("\n[3b/4] Pixel2Mesh-style normal projection...")
        v_nrm = assign_normals(verts, views, orth_scale)

        print("\n[3c/4] Vertex color projection...")
        v_clr = assign_colors(verts, v_nrm, views, orth_scale)

        mesh_carv = make_o3d_mesh(verts, faces, v_clr, v_nrm)

        # Light smoothing to reduce staircase
        mesh_carv = mesh_carv.filter_smooth_laplacian(number_of_iterations=3)
        mesh_carv.compute_vertex_normals()

        save_mesh(mesh_carv, out_dir, "mesh_carving")
        meshes_to_show.append(("Space-Carving + Marching Cubes", mesh_carv))

    # ?? 3B. Ray-casting + Poisson ?????????????????????????????????????????
    if method in ("poisson", "both"):
        print("\n[3/4] Building surface point cloud via ray-casting...")
        pcd = build_surface_point_cloud(views, occupancy, grid_res, orth_scale)

        print("\n[4/4] Poisson surface reconstruction...")
        mesh_pois = poisson_from_pcd(pcd, poisson_depth=poisson_depth)

        save_mesh(mesh_pois, out_dir, "mesh_poisson")
        meshes_to_show.append(("Poisson", mesh_pois))

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.1f}s  ->  {out_dir}")

    # ?? Visualise ??????????????????????????????????????????????????????????
    if visualize and meshes_to_show:
        for title, mesh in meshes_to_show:
            print(f"  [Viewer] {title}")
            o3d.visualization.draw_geometries(
                [mesh],
                window_name=f"{title} ? {object_name}",
                width=1200, height=900,
                mesh_show_back_face=True,
            )


# =============================================================================
# CLI
# =============================================================================

def list_objects(wonder3d_dir: str) -> list:
    p = Path(wonder3d_dir)
    if not p.exists():
        print(f"Directory not found: {wonder3d_dir}")
        return []
    objs = sorted(d for d in os.listdir(p) if (p / d).is_dir())
    print(f"\nAvailable objects in:\n  {wonder3d_dir}\n")
    for i, o in enumerate(objs):
        print(f"  [{i:3d}]  {o}")
    return objs


def main():
    parser = argparse.ArgumentParser(
        description="Pixel2Mesh-inspired 3D reconstruction from Wonder3D outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--wonder3d_dir", default=DEFAULT_WONDER3D_DIR,
        help="Path to Wonder3D outputs/cropsize-*/  directory",
    )
    parser.add_argument(
        "--object", default=None,
        help="Object folder name to reconstruct. If omitted, reconstruct all.",
    )
    parser.add_argument(
        "--output_dir", default=DEFAULT_OUTPUT_DIR,
        help="Root output directory for meshes",
    )
    parser.add_argument(
        "--method", default="both", choices=["carving", "poisson", "both"],
        help="Reconstruction method (default: both)",
    )
    parser.add_argument(
        "--resolution", type=int, default=128,
        help="Voxel grid resolution (carving method, default: 128)",
    )
    parser.add_argument(
        "--poisson_depth", type=int, default=9,
        help="Poisson octree depth (default: 9, higher = more detail)",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Open Open3D viewer after reconstruction",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available objects and exit",
    )
    parser.add_argument(
        "--outputs_subdir", default=None,
        help="Override the outputs subdirectory (e.g. 'outputs000/cropsize-192-cfg3.0')",
    )

    args = parser.parse_args()

    if args.outputs_subdir:
        wonder3d_dir = str(Path(args.wonder3d_dir).parent.parent / args.outputs_subdir)
    else:
        wonder3d_dir = args.wonder3d_dir

    if args.list or args.object is None:
        objs = list_objects(wonder3d_dir)
        if args.list or not objs:
            return
        # Reconstruct all
        print(f"\nReconstructing all {len(objs)} objects...")
        for obj in objs:
            try:
                reconstruct(
                    wonder3d_dir, obj, args.output_dir,
                    method=args.method,
                    grid_res=args.resolution,
                    poisson_depth=args.poisson_depth,
                    visualize=args.visualize,
                )
            except Exception as e:
                print(f"  [ERROR] {obj}: {e}")
    else:
        reconstruct(
            wonder3d_dir, args.object, args.output_dir,
            method=args.method,
            grid_res=args.resolution,
            poisson_depth=args.poisson_depth,
            visualize=args.visualize,
        )


if __name__ == "__main__":
    main()
