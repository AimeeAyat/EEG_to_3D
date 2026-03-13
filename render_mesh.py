"""
render_mesh.py

Render reconstructed 3D meshes as paper-quality figures.
Uses Open3D Visualizer with hidden window (standard Windows OpenGL, not EGL).

Produces per object:
  1. carving_6views.png   - 6-angle grid of the space-carving mesh
  2. poisson_6views.png   - 6-angle grid of the Poisson mesh
  3. comparison.png       - Paper-style: RGB inputs | normals | carving | Poisson

Usage:
    python render_mesh.py --object airliner_airliner_004
    python render_mesh.py --object eeg_sample_0000
    python render_mesh.py                    # render all objects
    python render_mesh.py --size 768 --dpi 200
"""

import os
import sys
import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image
import open3d as o3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# =============================================================================
# DEFAULTS
# =============================================================================
_HERE = Path(__file__).parent
DEFAULT_MESHES_DIR   = _HERE / "outputs/meshes"
DEFAULT_WONDER3D_DIR = _HERE / "Wonder3D/outputs/cropsize-192-cfg3.0"
VIEWS_ORDER = ["front", "front_left", "left", "back", "right", "front_right"]

# Render viewpoints: (label, azimuth_deg, elevation_deg)
# Azimuth convention matches Wonder3D: 0=front, 90=left, 180=back, 270=right
RENDER_CAMERAS = [
    ("Front",        0,   20),
    ("Front-Left",  45,   20),
    ("Left",        90,   20),
    ("Back",       180,   20),
    ("Right",      270,   20),
    ("Top",         45,   70),
]


# =============================================================================
# OPEN3D VISUALIZER RENDERER  (hidden window, standard Windows OpenGL)
# =============================================================================

def prepare_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """Centre, normalise scale, compute normals."""
    m = o3d.geometry.TriangleMesh(mesh)   # copy
    bbox   = m.get_axis_aligned_bounding_box()
    centre = bbox.get_center()
    extent = bbox.get_extent()
    scale  = 1.6 / max(extent)
    m.translate(-centre)
    m.scale(scale, [0, 0, 0])
    m.compute_vertex_normals()
    return m


def camera_lookat(azimuth_deg: float, elevation_deg: float, radius: float = 2.5):
    """Compute camera eye position for given azimuth/elevation, looking at origin."""
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)
    eye = radius * np.array([
        np.cos(el) * np.sin(az),   # x
        np.sin(el),                # y
        np.cos(el) * np.cos(az),   # z
    ])
    return eye


def render_view(
    mesh:          o3d.geometry.TriangleMesh,
    azimuth_deg:   float,
    elevation_deg: float,
    img_size:      int   = 512,
    bg_color:      list  = None,
    neutral_color: bool  = True,     # True = silver-grey like paper figures
) -> np.ndarray:
    """
    Render one view of a mesh using Open3D Visualizer (hidden window, GPU rendering).
    Returns uint8 RGB array [H, W, 3].

    neutral_color=True paints the mesh silver-grey (matches paper figure style).
    neutral_color=False uses vertex colors from the PLY file.
    """
    if bg_color is None:
        bg_color = [1.0, 1.0, 1.0]

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=img_size, height=img_size)

    # Optionally override vertex colors with a neutral silver-grey
    render_mesh = o3d.geometry.TriangleMesh(mesh)
    if neutral_color:
        n = len(render_mesh.vertices)
        render_mesh.vertex_colors = o3d.utility.Vector3dVector(
            np.full((n, 3), 0.75)   # silver-grey
        )

    vis.add_geometry(render_mesh)

    opt = vis.get_render_option()
    opt.background_color = np.array(bg_color)
    opt.mesh_show_back_face  = True
    opt.mesh_show_wireframe  = False
    opt.light_on             = True

    # Set camera
    ctr = vis.get_view_control()
    eye = camera_lookat(azimuth_deg, elevation_deg, radius=2.5)
    up  = [0.0, 1.0, 0.0]
    if abs(elevation_deg) > 80:
        up = [0.0, 0.0, -1.0]   # avoid gimbal lock at top view

    ctr.set_lookat([0.0, 0.0, 0.0])
    ctr.set_front((-eye / np.linalg.norm(eye)).tolist())
    ctr.set_up(up)
    ctr.set_zoom(0.55)

    vis.poll_events()
    vis.update_renderer()

    frame = vis.capture_screen_float_buffer(do_render=True)
    vis.destroy_window()

    img = (np.asarray(frame) * 255).clip(0, 255).astype(np.uint8)
    return img


def render_all_views(
    mesh:          o3d.geometry.TriangleMesh,
    cameras:       list,
    img_size:      int  = 512,
    bg_color:      list = None,
    neutral_color: bool = True,
) -> list:
    """Render mesh from all camera positions. Returns [(label, PIL.Image), ...]."""
    m   = prepare_mesh(mesh)
    out = []
    for label, az, el in cameras:
        arr = render_view(m, az, el, img_size=img_size, bg_color=bg_color,
                          neutral_color=neutral_color)
        out.append((label, Image.fromarray(arr)))
    return out


# =============================================================================
# FIGURE BUILDERS
# =============================================================================

def make_mesh_grid(
    renders:    list,          # [(label, PIL.Image), ...]
    title:      str  = "",
    cols:       int  = 3,
    dpi:        int  = 150,
    panel_in:   float = 3.2,   # inches per panel
    bg:         str  = "white",
) -> plt.Figure:
    """6-view mesh grid, paper-style."""
    n    = len(renders)
    rows = (n + cols - 1) // cols
    bgc  = (1, 1, 1) if bg == "white" else (0.10, 0.10, 0.13)

    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * panel_in, rows * panel_in),
                             facecolor=bgc)
    axes = np.array(axes).ravel()

    for i, (label, img) in enumerate(renders):
        axes[i].imshow(img)
        axes[i].set_title(label, fontsize=11, fontweight="bold", pad=4,
                          color="white" if bg == "dark" else "#111")
        axes[i].axis("off")

    for j in range(len(renders), len(axes)):
        axes[j].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01,
                     color="white" if bg == "dark" else "#111")

    fig.tight_layout(pad=0.3)
    return fig


def make_comparison_figure(
    wonder3d_dir:  Path,
    object_name:   str,
    renders_carv:  list,   # [(label, PIL.Image), ...]
    renders_pois:  list,
    n_views:       int = 3,
    dpi:           int = 150,
) -> plt.Figure:
    """
    Paper-style 4-row comparison figure:
      Row 1: Wonder3D RGB inputs
      Row 2: Wonder3D surface normal maps
      Row 3: Space-carving mesh renders
      Row 4: Poisson mesh renders
    """
    obj_dir    = wonder3d_dir / object_name
    row_labels = ["RGB Input", "Normal Map", "Space-Carving\nMesh", "Poisson\nMesh"]
    view_names = VIEWS_ORDER[:n_views]
    cam_subset = RENDER_CAMERAS[:n_views]

    n_rows  = 4
    n_cols  = n_views
    panel_w = 2.7
    panel_h = 2.7
    fig_w   = 1.2 + n_cols * panel_w
    fig_h   = n_rows * panel_h + 0.55
    bg      = (1.0, 1.0, 1.0)

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=bg, dpi=dpi)
    gs  = gridspec.GridSpec(
        n_rows, n_cols + 1,
        figure=fig,
        width_ratios=[0.28] + [1.0] * n_cols,
        hspace=0.06, wspace=0.04,
        top=0.94, bottom=0.01, left=0.01, right=0.99,
    )

    carv_dict = {lbl: img for lbl, img in renders_carv}
    pois_dict = {lbl: img for lbl, img in renders_pois}

    for r, row_lbl in enumerate(row_labels):
        # Row label on left
        ax_lbl = fig.add_subplot(gs[r, 0])
        ax_lbl.set_axis_off()
        ax_lbl.text(0.5, 0.5, row_lbl, fontsize=9, fontweight="bold",
                    ha="center", va="center", transform=ax_lbl.transAxes,
                    rotation=90, color="#1a1a1a",
                    linespacing=1.4)

        for c in range(n_cols):
            ax = fig.add_subplot(gs[r, c + 1])
            ax.axis("off")

            if r == 0:   # RGB
                view_name = view_names[c]
                p = obj_dir / "masked_colors" / f"rgb_000_{view_name}.png"
                if not p.exists():
                    p = obj_dir / f"rgb_000_{view_name}.png"
                if p.exists():
                    img = Image.open(p).convert("RGB")
                    # Crop to object bounding box (remove empty border)
                    arr = np.array(img)
                    bg_px = np.all(arr < 15, axis=2)
                    rows_ok = ~np.all(bg_px, axis=1)
                    cols_ok = ~np.all(bg_px, axis=0)
                    if rows_ok.any() and cols_ok.any():
                        r0, r1 = np.where(rows_ok)[0][[0, -1]]
                        c0, c1 = np.where(cols_ok)[0][[0, -1]]
                        pad = 8
                        r0 = max(0, r0 - pad); r1 = min(arr.shape[0] - 1, r1 + pad)
                        c0 = max(0, c0 - pad); c1 = min(arr.shape[1] - 1, c1 + pad)
                        img = Image.fromarray(arr[r0:r1, c0:c1])
                    ax.imshow(img)
                if r == 0:
                    ax.set_title(view_name.replace("_", " ").title(),
                                 fontsize=9, pad=3, color="#111")

            elif r == 1:   # Normals
                view_name = view_names[c]
                p = obj_dir / "normals" / f"normals_000_{view_name}.png"
                if not p.exists():
                    p = obj_dir / f"normals_000_{view_name}.png"
                if p.exists():
                    ax.imshow(Image.open(p).convert("RGB"))

            elif r == 2:   # Carving
                label = cam_subset[c][0]
                if label in carv_dict:
                    ax.imshow(carv_dict[label])

            else:          # Poisson
                label = cam_subset[c][0]
                if label in pois_dict:
                    ax.imshow(pois_dict[label])

    fig.suptitle(
        f"3D Reconstruction — {object_name.replace('_', ' ')}",
        fontsize=12, fontweight="bold",
    )
    return fig


# =============================================================================
# MAIN
# =============================================================================

def render_object(
    object_name:  str,
    meshes_dir:   str,
    wonder3d_dir: str,
    output_dir:   str,
    mesh_type:    str   = "both",
    img_size:     int   = 512,
    dpi:          int   = 150,
    bg_color:     list  = None,
) -> None:
    t0 = time.perf_counter()
    obj_mesh_dir = Path(meshes_dir) / object_name
    obj_out_dir  = Path(output_dir) / object_name
    obj_out_dir.mkdir(parents=True, exist_ok=True)
    wonder3d_path = Path(wonder3d_dir)

    if bg_color is None:
        bg_color = [1.0, 1.0, 1.0]

    print(f"\n  {object_name}")

    # Load meshes
    data = {}
    for mtype, stem in [("carving", "mesh_carving"), ("poisson", "mesh_poisson")]:
        if mesh_type not in (mtype, "both"):
            continue
        ply = obj_mesh_dir / f"{stem}.ply"
        if not ply.exists():
            print(f"    [skip] {stem}.ply not found")
            continue
        mesh = o3d.io.read_triangle_mesh(str(ply))
        data[mtype] = mesh
        print(f"    {stem}: {len(mesh.vertices):,} verts, {len(mesh.triangles):,} tris")

    if not data:
        print("    No meshes. Run pixel2mesh_reconstruction.py first.")
        return

    # Render each mesh from 6 angles
    all_renders = {}
    for mtype, mesh in data.items():
        print(f"    Rendering {mtype} ({len(RENDER_CAMERAS)} views) ...", end=" ", flush=True)
        renders = render_all_views(mesh, RENDER_CAMERAS, img_size=img_size, bg_color=bg_color)
        all_renders[mtype] = renders

        fig = make_mesh_grid(
            renders,
            title=f"{object_name.replace('_', ' ')} — {mtype.title()} Mesh",
            cols=3,
        )
        out = obj_out_dir / f"{mtype}_6views.png"
        fig.savefig(str(out), dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"saved {out.name}")

    # Comparison figure (requires both meshes)
    if len(all_renders) == 2 and (wonder3d_path / object_name).exists():
        print(f"    Building comparison figure ...", end=" ", flush=True)
        fig2 = make_comparison_figure(
            wonder3d_path, object_name,
            all_renders.get("carving", []),
            all_renders.get("poisson", []),
            n_views=3,
            dpi=dpi,
        )
        out2 = obj_out_dir / "comparison.png"
        fig2.savefig(str(out2), dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig2)
        print(f"saved {out2.name}")

    print(f"    Done in {time.perf_counter() - t0:.1f}s")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Render reconstructed meshes as paper figures (Open3D GPU renderer)",
    )
    parser.add_argument("--object",       default=None)
    parser.add_argument("--meshes_dir",   default=DEFAULT_MESHES_DIR)
    parser.add_argument("--wonder3d_dir", default=DEFAULT_WONDER3D_DIR)
    parser.add_argument("--output_dir",   default=DEFAULT_MESHES_DIR)
    parser.add_argument("--mesh_type",    default="both",
                        choices=["carving", "poisson", "both"])
    parser.add_argument("--size",         type=int, default=512,
                        help="Render resolution per panel (default 512)")
    parser.add_argument("--dpi",          type=int, default=150)
    parser.add_argument("--dark",         action="store_true",
                        help="Dark background (default: white)")
    parser.add_argument("--list",         action="store_true")
    args = parser.parse_args()

    bg = [0.10, 0.10, 0.13] if args.dark else [1.0, 1.0, 1.0]

    meshes_dir = Path(args.meshes_dir)
    if args.list or args.object is None:
        objs = sorted(d for d in os.listdir(meshes_dir) if (meshes_dir / d).is_dir())
        print(f"Reconstructed objects in {meshes_dir}:")
        for i, o in enumerate(objs):
            print(f"  [{i:3d}]  {o}")
        if args.list:
            return
        print(f"\nRendering all {len(objs)} objects...")
        for obj in objs:
            try:
                render_object(obj, args.meshes_dir, args.wonder3d_dir, args.output_dir,
                              args.mesh_type, args.size, args.dpi, bg)
            except Exception as e:
                import traceback
                print(f"  [ERROR] {obj}: {e}")
                traceback.print_exc()
    else:
        render_object(args.object, args.meshes_dir, args.wonder3d_dir, args.output_dir,
                      args.mesh_type, args.size, args.dpi, bg)


if __name__ == "__main__":
    main()
