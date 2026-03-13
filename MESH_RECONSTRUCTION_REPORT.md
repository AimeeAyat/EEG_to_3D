# 3D Mesh Reconstruction from Multi-View Normal Maps
## Method Documentation

---

## 1. Overview

The reconstruction pipeline (`pixel2mesh_reconstruction.py`) generates a textured
3D mesh from the multi-view outputs of Wonder3D — six RGB images and six surface
normal maps rendered at evenly spaced azimuths around the object.

---

## 2. Input Data

Wonder3D produces **6 orthographic views** at azimuths
{ 0°, 45°, 90°, 180°, 270°, 315° } around the vertical Y axis at zero elevation.
Each view yields:

| File | Content |
|---|---|
| `masked_colors/rgb_000_<view>.png` | RGBA, 256×256; alpha channel is the soft silhouette mask |
| `normals/normals_000_<view>.png`   | RGB, 256×256; encodes camera-space surface normals |

**Normal map decoding:**

```
n = pixel / 127.5 − 1     ∈  [−1, +1]³
```

Channel mapping in camera space: R → nx (right), G → ny (up), B → nz (toward camera).

---

## 3. Camera Model

All six cameras share the same **orthographic** projection and zero elevation.
The camera-to-world rotation for azimuth θ is a pure Y-axis rotation:

```
        ⎡ cos θ    0    sin θ ⎤
R_c2w = ⎢   0      1      0   ⎥
        ⎣−sin θ    0    cos θ ⎦
```

- Camera x-axis (right) maps to world `[cos θ, 0, −sin θ]`
- Camera y-axis (up) maps to world `[0, 1, 0]`
- Camera z-axis (backward) maps to world `[sin θ, 0, cos θ]`

Orthographic projection of a world point **p** into pixel coordinates:

```
p_cam = R_c2w^T · p

u = (p_cam.x / s  +  1) / 2 · W          (s = orth_scale = 1.05)
v = (1  −  p_cam.y / s  +  1) / 2) · H   (image y flipped)
```

---

## 4. Method — Space Carving + Marching Cubes

### 4.1 Stage 1: Space Carving (Visual Hull)

A cubic voxel grid of resolution N³ (default N = 128) spans world space
[−s, +s]³. Each voxel centre is projected into every camera. A voxel is marked
**occupied** if and only if it projects inside the foreground silhouette (alpha
> 0.3) in **all six views**. This computes the *visual hull* — the tightest
shape consistent with all silhouette outlines.

```
occupied(v) = ∧  [ alpha_i( proj_i(v) ) ≥ 0.3 ]
              i=1..6
```

**Post-processing:**

1. **Morphological closing** (26-connectivity, 3 iterations) — fills gaps caused
   by thin parts such as wings or handles whose projections barely overlap.
2. **Largest connected component** — removes floating voxel fragments that
   occasionally survive the intersection.

### 4.2 Stage 2: Signed Distance Field + Marching Cubes

The boolean occupancy grid is converted to a signed distance field (SDF):

```
SDF(v) = dist_inside(v) − dist_outside(v)
```

where `dist_inside` and `dist_outside` are Euclidean distance transforms of
the occupied and unoccupied regions respectively. The SDF is positive inside
the object and negative outside. A Gaussian smooth (σ = 1 voxel) suppresses
staircase artefacts before the isosurface is extracted.

Marching Cubes (Lorensen & Cline, 1987) is applied at level SDF = 0 to extract
the triangular surface mesh. Vertex coordinates in voxel-index space are mapped
back to world coordinates:

```
p_world = (p_voxel / (N − 1)) · 2s − s
```

### 4.3 Stage 3: Pixel2Mesh-Style Normal Projection

Each mesh vertex is projected into all six cameras. The camera-space normal
sampled from the normal map at that projection is converted to world space and
blended across views weighted by:

```
w_i = alpha_i · ( n_world · (−z_cam_i) + ε )
```

where `n_world` is the sampled world normal and `−z_cam_i` is the viewing
direction of camera i. This weighting gives more influence to views that see
the surface nearly head-on (high cosine similarity), directly analogous to
the perceptual feature unprojection step in the original Pixel2Mesh paper
(Wang et al., ECCV 2018) — but using directly supervised normal maps from
Wonder3D instead of learned CNN features.

### 4.4 Stage 4: Vertex Colour Projection

Vertex colours are assigned by the same projection mechanism: each view
contributes its RGB sample weighted by visibility and facing angle; the
weighted average gives the final vertex colour.

A final Laplacian smooth (3 iterations) is applied to reduce mesh staircase
artefacts.

---

## 5. Output Files

Per object, in `outputs/meshes/<object_name>/`:

| File | Content |
|---|---|
| `mesh_carving.ply` | Space carving + Marching Cubes mesh, vertex-coloured (binary PLY) |
| `mesh_carving.obj` | Same mesh in OBJ format (Blender / MeshLab compatible) |

---

## 6. Key Parameters

| Parameter | Default | Effect |
|---|---|---|
| `--resolution` | 128 | Voxel grid N³. Higher → finer silhouette detail, more memory. |

---

## 7. Limitations

The visual hull produced by space carving is the *tightest convex-like shape
consistent with all silhouettes*; it cannot recover concavities that are not
visible from any of the six views. Geometry details finer than one voxel
(2s/N ≈ 0.016 world units at N = 128) are also lost.

---

## 8. References

1. Wang, N., Zhang, Y., Li, Z., Fu, Y., Liu, W., & Jiang, Y. (2018).
   *Pixel2Mesh: Generating 3D Mesh Models from Single RGB Images.*
   ECCV 2018.

2. Long, X., et al. (2023).
   *Wonder3D: Single Image to 3D using Cross-Domain Diffusion.*
   arXiv:2310.15008.

3. Lorensen, W. E., & Cline, H. E. (1987).
   *Marching cubes: A high resolution 3D surface construction algorithm.*
   SIGGRAPH 1987.

4. Curless, B., & Levoy, M. (1996).
   *A volumetric method for building complex models from range images.*
   SIGGRAPH 1996.
