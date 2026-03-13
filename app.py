import sys
import os
import random
import shutil
import subprocess
import threading
from pathlib import Path

# Set working directory before all project imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

import streamlit as st
import torch
import yaml
import numpy as np
from PIL import Image

st.set_page_config(page_title="MV-EEG", layout="wide", page_icon="🧠")



def is_caption_relevant(predicted_class: str, caption: str) -> bool:
    class_words = predicted_class.lower().replace("-", " ").replace("_", " ").split()
    caption_lower = caption.lower()
    return any(len(w) > 3 and w in caption_lower for w in class_words)


def create_enhanced_prompt(predicted_class: str, caption: str) -> tuple:
    use_caption = is_caption_relevant(predicted_class, caption)
    if use_caption:
        c = caption
        for src, dst in [("A group of","A single"),("group of","single"),
                         ("Several","One"),("several","one"),("Many","One"),("many","one"),
                         ("Two","One"),("two","one"),("Three","One"),("three","one"),
                         ("people",""),("persons",""),("men",""),("women","")]:
            c = c.replace(src, dst)
        prompt = (f"ONE single {predicted_class}, exactly one object, {c}, "
                  f"full body visible, complete anatomy, whole body in frame, "
                  f"solo subject only, no other objects, isolated on pure white background, "
                  f"centered, studio product photography, professional lighting, sharp focus, 8k")
    else:
        prompt = (f"ONE single {predicted_class}, exactly one object, "
                  f"full body visible, complete anatomy, whole body in frame, "
                  f"solo subject only, no other objects, isolated on pure white background, "
                  f"centered, studio product photography, professional lighting, sharp focus, 8k, photorealistic")
    return prompt, use_caption


def create_negative_prompt() -> str:
    return ("multiple objects, two objects, three objects, many objects, "
            "group, collection, crowd, pair, couple, several, duplicate, "
            "second object, additional items, extra objects, "
            "people, person, human, hands, face, "
            "busy background, cluttered, complex scene, "
            "text, watermark, logo, signature, "
            "blurry, low quality, distorted, deformed, "
            "cropped, cut off, partial object")


# ── Cached resource loaders ──

@st.cache_resource(show_spinner="Loading EEG models...")
def load_eeg_models():
    with open('config/config.yaml') as f:
        config = yaml.safe_load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    classifier = EEGClassifier(config).to(device)
    classifier.model.load_state_dict(
        torch.load('checkpoints/eeg_classifier_best.pth', map_location=device))
    classifier.eval()

    contrastive = ContrastiveEncoder(config).to(device)
    contrastive.model.load_state_dict(
        torch.load('checkpoints/contrastive_model_best.pth', map_location=device))
    contrastive.eval()

    return config, device, classifier, contrastive


@st.cache_resource(show_spinner="Loading dataset...")
def load_dataset(_config):
    from src.data.data_loader import CATVisDataLoader
    from src.data.preprocessor import DataPreprocessor
    data_loader = CATVisDataLoader(_config)
    df = data_loader.get_dataset_dataframe()
    _, _, test_df = data_loader.get_train_val_test_splits(df)
    preprocessor = DataPreprocessor(_config)
    test_dataset = preprocessor.create_pipeline_dataset(test_df)
    return data_loader, test_df, test_dataset


@st.cache_resource(show_spinner="Setting up caption retrieval...")
def load_retrieval(config, _contrastive, _device, _test_df):
    from src.pipeline.retrieval import TextRetrieval
    retrieval = TextRetrieval(config, _contrastive, _device)
    retrieval.setup_retrieval_corpus(_test_df)
    return retrieval


@st.cache_resource(show_spinner="Loading Flux.1-schnell ")
def load_flux():
    from diffusers import FluxPipeline
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    return pipe


# ── Pipeline steps ──

def run_eeg_inference(sample, config, device, classifier, data_loader):
    eeg_data = sample['eeg'].unsqueeze(0).to(device)
    gt_idx = sample['label'].item() if torch.is_tensor(sample['label']) else sample['label']
    img_idx = sample['image'].item() if torch.is_tensor(sample['image']) else sample['image']
    with torch.no_grad():
        outputs, _ = classifier(eeg_data)
        pred_idx = outputs.argmax().item()
    labels_order = data_loader.raw_data['labels']
    predicted_class = config['class_prompts'][labels_order[pred_idx]]
    ground_truth_class = config['class_prompts'][labels_order[gt_idx]]
    return eeg_data, predicted_class, ground_truth_class, img_idx


def get_gt_image(img_idx, data_loader, config) -> Image.Image | None:
    rel_path = data_loader.image_to_path.get(img_idx)
    if rel_path is None:
        return None
    full_path = (Path(config['data']['root_dir']) /
                 config['data']['imagenet_images'] / rel_path)
    return Image.open(full_path).convert('RGB') if full_path.exists() else None


def generate_image(prompt, seed, pipe, num_steps: int = 4) -> Image.Image:
    result = pipe(
        prompt,
        guidance_scale=0.0,
        num_inference_steps=num_steps,
        max_sequence_length=256,
        generator=torch.Generator("cpu").manual_seed(seed),
        height=1024, width=1024,
    )
    return result.images[0]


def remove_background(img: Image.Image) -> Image.Image:
    from rembg import remove as rembg_remove
    return rembg_remove(img)


def run_wonder3d(img_no_bg: Image.Image, scene_name: str) -> Path | None:
    wonder3d_dir = project_root / 'Wonder3D'
    input_dir = wonder3d_dir / 'example_images'
    input_dir.mkdir(parents=True, exist_ok=True)

    img_filename = f"{scene_name}.png"
    img_no_bg.save(input_dir / img_filename)

    try:
        result = subprocess.run([
            'python', 'test_mvdiffusion_seq.py',
            '--config', 'configs/mvdiffusion-joint-ortho-6views.yaml',
            f'validation_dataset.filepaths=[{img_filename}]'
        ], cwd=str(wonder3d_dir), check=True,
           capture_output=True, text=True, timeout=300)
    except subprocess.CalledProcessError as e:
        st.error("Wonder3D failed")
        st.code(e.stderr[-3000:] if e.stderr else "No stderr output")
        return None
    except subprocess.TimeoutExpired:
        st.error("Wonder3D timed out after 5 minutes")
        return None

    scene_dir = wonder3d_dir / 'outputs' / 'cropsize-192-cfg3.0' / scene_name
    return scene_dir if scene_dir.exists() else None


def build_360_gif(scene_dir: Path, out_path: Path,
                  use_rife: bool = False, n_interp: int = 3) -> bool:
    from generate_mesh_360 import make_gif_from_views
    return make_gif_from_views(
        scene_dir, out_path, size=384, fps=8,
        n_interp=n_interp, use_rife=use_rife)


def load_wonder3d_views(scene_dir: Path) -> dict:
    view_order = ['front', 'front_left', 'left', 'back', 'right', 'front_right']
    views = {}
    for view in view_order:
        p = scene_dir / 'masked_colors' / f'rgb_000_{view}.png'
        if not p.exists():
            p = scene_dir / f'rgb_000_{view}.png'
        if p.exists():
            img = Image.open(p).convert('RGBA')
            bg = Image.new('RGB', img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            views[view] = bg
    return views


# ── UI ──

st.title("EEG-based multi-view 3D Image Generation")
st.caption("Reconstruct what you saw from brain signals")

# ── Sidebar ──
with st.sidebar:
    st.header("⚙️ Settings")
    seed = st.number_input("Generation seed", value=42, min_value=0)
    num_steps = st.slider(
        "Flux inference steps", min_value=1, max_value=50, value=4,
        help="Flux.1-schnell is optimised for 4 steps. More steps = higher quality but slower.")
    use_rife = st.toggle(
        "RIFE frame interpolation",
        value=False,
        help="Use neural RIFE interpolation for smoother GIF. Requires: pip install ccvfi")
    n_interp = st.slider(
        "Interpolated frames per gap", min_value=1, max_value=8, value=3,
        disabled=not use_rife,
        help="Frames inserted between each pair of Wonder3D views using RIFE. "
             "3 = 40 frames total, 8 = 90 frames total.")
    st.divider()
    st.subheader("Pipeline Status")
    status_placeholder = st.empty()

# ── Load core models immediately (cached after first run) ──
try:
    from src.models.eeg_classifier import EEGClassifier
    from src.models.contrastive_encoder import ContrastiveEncoder
    config, device, classifier, contrastive = load_eeg_models()
    data_loader, test_df, test_dataset = load_dataset(config)
    retrieval = load_retrieval(config, contrastive, device, test_df)
    with st.sidebar:
        st.success(f"EEG models loaded")
        st.info(f"Test samples: {len(test_dataset)}")
except Exception as e:
    st.error(f"Failed to load EEG models: {e}")
    st.stop()

# ── Sample selector ──
col_ctrl, col_spacer = st.columns([2, 3])
with col_ctrl:
    st.subheader("Select EEG Sample")
    c1, c2 = st.columns([3, 1])
    with c1:
        sample_idx = st.number_input(
            "Sample index", min_value=0,
            max_value=len(test_dataset) - 1, value=0, step=1)
    with c2:
        st.write("")
        st.write("")
        if st.button("🎲 Random"):
            sample_idx = random.randint(0, len(test_dataset) - 1)
            st.rerun()

    generate_btn = st.button("▶ Generate 360° View", type="primary", use_container_width=True)

# Persist scene_dir across reruns so the 3D button stays active
if "scene_dir" not in st.session_state:
    st.session_state.scene_dir = None

# ── Preview GT on sample change ──
sample = test_dataset[int(sample_idx)]
_, predicted_class, ground_truth_class, img_idx = run_eeg_inference(
    sample, config, device, classifier, data_loader)
gt_image = get_gt_image(img_idx, data_loader, config)

with col_ctrl:
    st.divider()
    if gt_image:
        st.image(gt_image, caption=f"Ground truth: {ground_truth_class}", width='stretch')
    else:
        st.warning("Ground truth image not found")
    st.metric("Predicted class", predicted_class)
    correct = predicted_class.lower() == ground_truth_class.lower()
    st.metric("Classification", "✓ Correct" if correct else "✗ Wrong",
              delta=None)

# ── Main generation pipeline ──
if generate_btn:
    st.divider()
    st.subheader("Pipeline Output")

    scene_name = f"eeg_sample_{int(sample_idx):04d}"
    out_dir = project_root / 'outputs' / 'streamlit_runs'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1 — Retrieve caption
    with st.status("Running pipeline...", expanded=True) as status:
        st.write("🔍 Retrieving caption...")
        eeg_data = sample['eeg'].unsqueeze(0).to(device)
        retrieved = retrieval.retrieve_top_k_from_eeg(eeg_data.squeeze(0), k=1)
        caption = retrieved[0][0]
        prompt, caption_used = create_enhanced_prompt(predicted_class, caption)

        # Step 2 — Generate image
        st.write(f"🎨 Generating image with Flux.1 ({num_steps} steps)...")
        flux_pipe = load_flux()
        import time as _time
        _t0 = _time.perf_counter()
        gen_image = generate_image(prompt, seed + int(sample_idx), flux_pipe, num_steps)
        gen_time = _time.perf_counter() - _t0
        st.write(f"   ⏱️ Image generated in **{gen_time:.1f}s**")

        # Step 3 — Remove background
        st.write("✂️ Removing background...")
        gen_no_bg = remove_background(gen_image)
        gen_path = out_dir / f"{scene_name}.png"
        gen_no_bg.save(gen_path)

        # Step 4 — Wonder3D
        st.write("🌐 Running Wonder3D (multi-view generation)...")
        scene_dir = run_wonder3d(gen_no_bg, scene_name)
        if scene_dir:
            st.session_state.scene_dir = scene_dir
            st.session_state.scene_name = scene_name

        # Step 5 — 360° GIF
        gif_path = out_dir / f"{scene_name}_360.gif"
        if scene_dir:
            gif_label = (f"🔄 Building 360° GIF with RIFE ({10 * (n_interp + 1)} frames)..."
                         if use_rife else "🔄 Building 360° GIF (6 keyframes)...")
            st.write(gif_label)
            build_360_gif(scene_dir, gif_path, use_rife=use_rife, n_interp=n_interp)
            status.update(label="✅ Done!", state="complete")
        else:
            status.update(label="⚠️ Wonder3D failed", state="error")

    # ── Display results ──
    st.subheader("Results")

    # Caption info
    with st.expander("Prompt details", expanded=False):
        st.write(f"**Retrieved caption:** {caption}")
        st.write(f"**Caption used in prompt:** {caption_used}")
        st.code(prompt)

    # Images side by side
    img_col1, img_col2 = st.columns(2)
    with img_col1:
        if gt_image:
            st.image(gt_image, caption=f"Ground Truth — {ground_truth_class}",
                     width='stretch')
    with img_col2:
        st.image(gen_no_bg, caption=f"Generated — {predicted_class}",
                 width='stretch')

    # Wonder3D views
    if scene_dir:
        st.subheader("Wonder3D Multi-View")
        views = load_wonder3d_views(scene_dir)
        if views:
            view_cols = st.columns(len(views))
            for col, (view_name, view_img) in zip(view_cols, views.items()):
                with col:
                    st.image(view_img,
                             caption=view_name.replace('_', ' ').title(),
                             width='stretch')

        # 360° GIF
        if gif_path.exists():
            st.subheader("360° Rotation")
            st.image(str(gif_path), width=400)
            with open(gif_path, 'rb') as f:
                st.download_button(
                    "⬇️ Download 360° GIF",
                    data=f.read(),
                    file_name=f"{predicted_class.replace(' ', '_')}_360.gif",
                    mime="image/gif"
                )

# ── 3D Mesh Reconstruction (Method A: Space Carving + Marching Cubes) ──
if st.session_state.get("scene_dir"):
    st.divider()
    st.subheader("3D Mesh Reconstruction")
    st.caption("Space Carving + Marching Cubes from 6 Wonder3D views")

    mesh_out_dir = project_root / "outputs" / "meshes" / st.session_state.scene_name
    mesh_ply = mesh_out_dir / "mesh_carving.ply"

    col_mesh1, col_mesh2 = st.columns([2, 3])
    with col_mesh1:
        if st.button("🧊 Reconstruct & View 3D Mesh", use_container_width=True):
            from pixel2mesh_reconstruction import (
                load_views, space_carve, occupancy_to_mesh,
                assign_normals, assign_colors, make_o3d_mesh, save_mesh,
            )
            import open3d as o3d

            with st.spinner("Running Method A reconstruction..."):
                _views = load_views(Path(st.session_state.scene_dir))
                _occ   = space_carve(_views, grid_res=128, orth_scale=1.05)
                _v, _f = occupancy_to_mesh(_occ, orth_scale=1.05, gaussian_sigma=1.0)
                _nrm   = assign_normals(_v, _views, orth_scale=1.05)
                _clr   = assign_colors(_v, _nrm, _views, orth_scale=1.05)
                _mesh  = make_o3d_mesh(_v, _f, _clr, _nrm)
                _mesh  = _mesh.filter_smooth_laplacian(number_of_iterations=3)
                _mesh.compute_vertex_normals()
                save_mesh(_mesh, mesh_out_dir, "mesh_carving")

            st.success(f"Mesh saved: {mesh_ply.name}  "
                       f"({len(_mesh.vertices):,} verts)")

            # Open Open3D viewer in a background thread so Streamlit stays responsive
            def _open_viewer(mesh, title):
                o3d.visualization.draw_geometries(
                    [mesh],
                    window_name=title,
                    width=1200, height=900,
                    mesh_show_back_face=True,
                )
            threading.Thread(
                target=_open_viewer,
                args=(_mesh, f"3D Mesh — {st.session_state.scene_name}"),
                daemon=True,
            ).start()
            st.info("Open3D viewer launched in a separate window.")

        # Re-open viewer from saved mesh (if already reconstructed)
        if mesh_ply.exists():
            if st.button("🔁 Re-open Viewer (cached mesh)", use_container_width=True):
                import open3d as o3d
                _mesh = o3d.io.read_triangle_mesh(str(mesh_ply))
                _mesh.compute_vertex_normals()
                def _open_viewer(mesh, title):
                    o3d.visualization.draw_geometries(
                        [mesh],
                        window_name=title,
                        width=1200, height=900,
                        mesh_show_back_face=True,
                    )
                threading.Thread(
                    target=_open_viewer,
                    args=(_mesh, f"3D Mesh — {st.session_state.scene_name}"),
                    daemon=True,
                ).start()
                st.info("Open3D viewer launched.")

    with col_mesh2:
        if mesh_ply.exists():
            st.success(f"Cached mesh: `{mesh_ply}`")
        else:
            st.info("No mesh yet — click Reconstruct to generate.")
