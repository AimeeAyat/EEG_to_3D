#!/usr/bin/env python3
"""
EEG Spatio-Temporal Analysis for CATVis Dataset
================================================
Visualises how EEG brain responses unfold over time alongside the stimulus images.

Five figure types are produced:

  1. deep_dive          — one image, full butterfly + channel-time heatmap + grand ERP
  2. multi_subject      — same image, one channel-time heatmap per subject
  3. intersubject       — cosine-similarity matrix between subjects + ERP overlays
  4. class_erp_gallery  — grand-average ERP for each ImageNet class
  5. multi_image_grid   — image thumbnail + heatmap + ERP for 8 diverse images

Run:
    python eeg_timeseries_analysis.py

All figures are saved to:  outputs/eeg_timeseries/
"""

import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless-safe; swap to "TkAgg" if you want interactive
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from PIL import Image

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# Paths  (all relative to the script location)
# ──────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = BASE_DIR / "data"
IMAGES_DIR = DATA_DIR / "imageNet_images"
EEG_FILE   = DATA_DIR / "eeg_55_95_std.pth"
OUT_DIR    = BASE_DIR / "outputs" / "eeg_timeseries"

# EEG window matching config.yaml  (time_low=20, time_high=460 → 440 samples)
TIME_LOW, TIME_HIGH = 20, 460
N_TIMES = TIME_HIGH - TIME_LOW           # 440 time samples

# Approximate time axis in ms.
# The 55–95 Hz band-pass + 500-sample epoch suggests a stimulus-locked
# window of roughly -200 ms (pre) to +800 ms (post stimulus).
TIME_MS = np.linspace(-200, 800, N_TIMES)   # [440,]

DARK_BG   = "#0d0d0d"
PANEL_BG  = "#111111"
SPINE_COL = "#444444"
GREY_TEXT = "#aaaaaa"


# ──────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────

def load_eeg_data() -> dict:
    """
    Load eeg_55_95_std.pth and organise into convenient structures.

    Returns
    -------
    dict with keys:
        by_image   : {image_idx → list of {'eeg':[128,440], 'label':int, 'subject':int}}
        labels     : list[str]  — 40 ImageNet synset IDs
        img_names  : list[str]  — 1 996 image filenames (synset_img#)
        all_eeg    : ndarray [N, 128, 440]
        all_labels : ndarray [N,]
        all_images : ndarray [N,]
        all_subjects: ndarray [N,]
    """
    print("Loading EEG data ...")
    raw = torch.load(str(EEG_FILE), weights_only=False)

    dataset   = raw["dataset"]     # list[dict]
    labels    = raw["labels"]      # list[str], len=40
    img_names = raw["images"]      # list[str], len=1996

    by_image     = defaultdict(list)
    all_eeg      = []
    all_labels   = []
    all_images   = []
    all_subjects = []

    for sample in dataset:
        eeg     = sample["eeg"].numpy().astype(np.float32)[:, TIME_LOW:TIME_HIGH]  # [128, 440]
        img_idx = int(sample["image"])
        lbl     = int(sample["label"])
        subj    = int(sample["subject"])

        by_image[img_idx].append({"eeg": eeg, "label": lbl, "subject": subj})
        all_eeg.append(eeg)
        all_labels.append(lbl)
        all_images.append(img_idx)
        all_subjects.append(subj)

    print(
        f"  {len(dataset):,} samples | "
        f"{len(by_image):,} unique images | "
        f"{len(labels)} classes | "
        f"{len(set(all_subjects))} subjects"
    )

    return {
        "by_image"    : by_image,
        "labels"      : labels,
        "img_names"   : img_names,
        "all_eeg"     : np.stack(all_eeg),           # [N, 128, 440]
        "all_labels"  : np.array(all_labels),
        "all_images"  : np.array(all_images),
        "all_subjects": np.array(all_subjects),
    }


# ──────────────────────────────────────────────────────────────
# Image helpers
# ──────────────────────────────────────────────────────────────

def get_image_path(img_name: str) -> Path:
    """e.g. 'n02106662_10232'  →  imageNet_images/n02106662/n02106662_10232.JPEG"""
    synset = img_name.split("_")[0]
    return IMAGES_DIR / synset / f"{img_name}.JPEG"


def load_pil(img_name: str):
    p = get_image_path(img_name)
    return Image.open(p).convert("RGB") if p.exists() else None


# ──────────────────────────────────────────────────────────────
# Style helpers
# ──────────────────────────────────────────────────────────────

def _dark(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL_BG)
    for sp in ax.spines.values():
        sp.set_edgecolor(SPINE_COL)
    ax.tick_params(colors=GREY_TEXT, labelsize=7)
    if title:
        ax.set_title(title, color="white", fontsize=9, pad=4)
    if xlabel:
        ax.set_xlabel(xlabel, color=GREY_TEXT, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=GREY_TEXT, fontsize=8)


def _colorbar(mappable, ax, label=""):
    cb = plt.colorbar(mappable, ax=ax, shrink=0.85, pad=0.02)
    cb.ax.tick_params(colors=GREY_TEXT, labelsize=6)
    cb.set_label(label, color=GREY_TEXT, fontsize=7)
    cb.outline.set_edgecolor(SPINE_COL)


def _show_image(ax, img_name, title=""):
    pil = load_pil(img_name)
    if pil:
        ax.imshow(pil)
    else:
        ax.text(0.5, 0.5, "not found", ha="center", va="center",
                color="grey", transform=ax.transAxes)
    _dark(ax, title=title)
    ax.axis("off")


# ──────────────────────────────────────────────────────────────
# Figure 1 — Single-image deep dive
# ──────────────────────────────────────────────────────────────

def fig_deep_dive(data: dict, image_idx: int, save_dir: Path):
    """
    Four-panel figure for one image:
      A  Stimulus image
      B  Butterfly plot — all 128 channels, every trial, colour-coded by subject
      C  Channel × Time heatmap (grand average over subjects)
      D  Grand-average ERP ± 1 SD
    """
    by_image  = data["by_image"]
    img_names = data["img_names"]
    labels    = data["labels"]

    trials    = by_image[image_idx]
    img_name  = img_names[image_idx]
    synset    = labels[trials[0]["label"]]
    n_trials  = len(trials)

    eegs     = np.stack([t["eeg"] for t in trials])   # [T, 128, 440]
    subjects = [t["subject"] for t in trials]
    uniq_s   = sorted(set(subjects))
    cmap_s   = plt.cm.get_cmap("tab20", max(len(uniq_s), 1))

    mean_eeg = eegs.mean(axis=0)                       # [128, 440]
    grand    = eegs.mean(axis=(0, 1))                  # [440,]
    grand_sd = eegs.std(axis=(0, 1))

    fig = plt.figure(figsize=(22, 13), facecolor=DARK_BG)
    fig.suptitle(
        f"EEG Spatio-Temporal Deep Dive\n"
        f"Image: {img_name}  ·  Class: {synset}  ·  "
        f"{n_trials} trials  ·  {len(uniq_s)} subjects",
        color="white", fontsize=13, y=0.99,
    )

    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        hspace=0.40, wspace=0.36,
        left=0.06, right=0.97, top=0.91, bottom=0.08,
    )
    ax_img    = fig.add_subplot(gs[0, 0])
    ax_butter = fig.add_subplot(gs[0, 1:])
    ax_heat   = fig.add_subplot(gs[1, :2])
    ax_erp    = fig.add_subplot(gs[1, 2])

    # ── A: image ──────────────────────────────────────────
    _show_image(ax_img, img_name, title=f"{img_name}\n({synset})")

    # ── B: butterfly ──────────────────────────────────────
    for trial in trials:
        s   = trial["subject"]
        col = cmap_s(uniq_s.index(s))
        for ch in range(trial["eeg"].shape[0]):
            ax_butter.plot(TIME_MS, trial["eeg"][ch],
                           color=col, alpha=0.05, linewidth=0.35, rasterized=True)

    for s in uniq_s:
        s_eegs = np.stack([t["eeg"] for t in trials if t["subject"] == s])
        s_mean = s_eegs.mean(axis=(0, 1))           # [440,]
        col    = cmap_s(uniq_s.index(s))
        ax_butter.plot(TIME_MS, s_mean, color=col, linewidth=1.6,
                       label=f"Subj {s}", zorder=5)

    ax_butter.axvline(0, color="gold", linestyle="--", linewidth=0.9, alpha=0.8)
    ax_butter.axhline(0, color="white", linestyle=":", linewidth=0.5, alpha=0.3)
    _dark(ax_butter,
          title=f"Butterfly Plot — 128 channels × {n_trials} trials (bold = per-subject mean)",
          xlabel="Time (ms)", ylabel="Amplitude (σ)")
    ax_butter.legend(
        fontsize=7, loc="upper right",
        facecolor="#1e1e1e", labelcolor="white",
        framealpha=0.8, ncol=max(1, len(uniq_s) // 5),
        edgecolor=SPINE_COL,
    )

    # ── C: channel-time heatmap ───────────────────────────
    vmax = np.percentile(np.abs(mean_eeg), 97)
    im = ax_heat.imshow(
        mean_eeg, aspect="auto", origin="lower",
        extent=[TIME_MS[0], TIME_MS[-1], 0.5, 128.5],
        cmap="RdBu_r", vmin=-vmax, vmax=vmax,
        interpolation="nearest",
    )
    ax_heat.axvline(0, color="gold", linestyle="--", linewidth=0.9)
    _dark(ax_heat,
          title="Channel × Time Heatmap  (grand average across subjects)",
          xlabel="Time (ms)", ylabel="EEG Channel #")
    _colorbar(im, ax_heat, label="Amplitude (σ)")

    # ── D: grand ERP ──────────────────────────────────────
    ax_erp.plot(TIME_MS, grand, color="#00d4ff", linewidth=1.9, label="Mean")
    ax_erp.fill_between(TIME_MS,
                         grand - grand_sd, grand + grand_sd,
                         color="#00d4ff", alpha=0.18, label="±1 SD")
    ax_erp.axvline(0, color="gold", linestyle="--", linewidth=0.9)
    ax_erp.axhline(0, color="white", linestyle=":", linewidth=0.5, alpha=0.35)
    _dark(ax_erp,
          title="Grand-Average ERP\n(all channels · all subjects)",
          xlabel="Time (ms)", ylabel="Amplitude (σ)")
    ax_erp.legend(fontsize=8, facecolor="#1e1e1e",
                  labelcolor="white", edgecolor=SPINE_COL)

    _save(fig, save_dir / f"1_deep_dive_img{image_idx:04d}.png")


# ──────────────────────────────────────────────────────────────
# Figure 2 — Same image, per-subject heatmaps
# ──────────────────────────────────────────────────────────────

def fig_multi_subject(data: dict, image_idx: int, save_dir: Path, max_subjects: int = 6):
    """
    One column per subject: stimulus image + channel-time heatmap.
    Directly shows how different brains respond to the same stimulus.
    """
    by_image  = data["by_image"]
    img_names = data["img_names"]
    labels    = data["labels"]

    trials   = by_image[image_idx]
    img_name = img_names[image_idx]
    synset   = labels[trials[0]["label"]]

    # Keep one representative trial per subject
    subj_trial = {}
    for t in trials:
        s = t["subject"]
        if s not in subj_trial:
            subj_trial[s] = t

    subjects = sorted(subj_trial)[:max_subjects]
    n_s      = len(subjects)

    # Global colour scale
    all_eegs = np.stack([subj_trial[s]["eeg"] for s in subjects])
    vmax     = np.percentile(np.abs(all_eegs), 97)

    ncols = n_s + 1
    fig, axes = plt.subplots(
        1, ncols, figsize=(3.8 * ncols, 5.5), facecolor=DARK_BG,
        gridspec_kw={"wspace": 0.08},
    )
    fig.suptitle(
        f"Same Image → {n_s} Subjects' EEG\n{img_name}  ({synset})",
        color="white", fontsize=12, y=1.02,
    )

    _show_image(axes[0], img_name, title="Stimulus")

    for i, s in enumerate(subjects):
        ax  = axes[i + 1]
        eeg = subj_trial[s]["eeg"]   # [128, 440]
        im  = ax.imshow(
            eeg, aspect="auto", origin="lower",
            extent=[TIME_MS[0], TIME_MS[-1], 0.5, 128.5],
            cmap="RdBu_r", vmin=-vmax, vmax=vmax,
            interpolation="nearest",
        )
        ax.axvline(0, color="gold", linestyle="--", linewidth=0.8)
        _dark(ax, title=f"Subject {s}", xlabel="Time (ms)")
        if i == 0:
            ax.set_ylabel("EEG Channel #", color=GREY_TEXT, fontsize=7)
        else:
            ax.set_yticks([])

    _colorbar(im, axes[-1], label="Amplitude (σ)")
    _save(fig, save_dir / f"2_multi_subject_img{image_idx:04d}.png")


# ──────────────────────────────────────────────────────────────
# Figure 3 — Inter-subject correlation
# ──────────────────────────────────────────────────────────────

def fig_intersubject(data: dict, image_idx: int, save_dir: Path):
    """
    Left:  cosine-similarity matrix between all subjects (for one image).
    Right: per-subject mean ERP overlaid — shows where subjects agree/differ.
    """
    by_image  = data["by_image"]
    img_names = data["img_names"]
    labels    = data["labels"]

    trials   = by_image[image_idx]
    img_name = img_names[image_idx]
    synset   = labels[trials[0]["label"]]

    # Average EEG per subject
    subj_eegs: dict = defaultdict(list)
    for t in trials:
        subj_eegs[t["subject"]].append(t["eeg"])

    subjects = sorted(subj_eegs)
    means    = np.stack(
        [np.mean(subj_eegs[s], axis=0) for s in subjects]
    )  # [S, 128, 440]

    # Cosine similarity between subjects (flatten channel×time)
    flat = means.reshape(len(subjects), -1).astype(np.float64)
    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    flat_n = flat / (norms + 1e-12)
    corr = flat_n @ flat_n.T   # [S, S]

    n_s   = len(subjects)
    cmap_s = plt.cm.get_cmap("tab20", max(n_s, 1))

    fig, (ax_c, ax_e) = plt.subplots(
        1, 2, figsize=(15, 5.5), facecolor=DARK_BG,
        gridspec_kw={"wspace": 0.30},
    )
    fig.suptitle(
        f"Inter-Subject Analysis  ·  {img_name}  ({synset})  ·  {n_s} subjects",
        color="white", fontsize=12,
    )

    # Similarity matrix
    im = ax_c.imshow(corr, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax_c.set_xticks(range(n_s))
    ax_c.set_xticklabels([f"S{s}" for s in subjects], color="white", fontsize=7, rotation=45)
    ax_c.set_yticks(range(n_s))
    ax_c.set_yticklabels([f"S{s}" for s in subjects], color="white", fontsize=7)
    _dark(ax_c, title="Inter-Subject Cosine Similarity\n(EEG flattened: 128 ch × 440 t)")
    _colorbar(im, ax_c, label="Cosine similarity")

    # Numeric annotations
    for i in range(n_s):
        for j in range(n_s):
            ax_c.text(j, i, f"{corr[i, j]:.2f}",
                      ha="center", va="center",
                      color="white" if corr[i, j] < 0.7 else "black",
                      fontsize=max(4, 7 - n_s // 3))

    # ERP overlay
    for i, s in enumerate(subjects):
        mean_erp = means[i].mean(axis=0)   # [440,]
        ax_e.plot(TIME_MS, mean_erp,
                  color=cmap_s(i), linewidth=1.4,
                  alpha=0.85, label=f"S{s}")

    ax_e.axvline(0, color="gold", linestyle="--", linewidth=0.9)
    ax_e.axhline(0, color="white", linestyle=":", linewidth=0.5, alpha=0.35)
    _dark(ax_e,
          title="Per-Subject Mean ERP\n(averaged over 128 channels)",
          xlabel="Time (ms)", ylabel="Amplitude (σ)")
    ax_e.legend(
        fontsize=7, facecolor="#1e1e1e", labelcolor="white",
        edgecolor=SPINE_COL, ncol=max(1, n_s // 6),
        loc="upper right",
    )

    _save(fig, save_dir / f"3_intersubject_img{image_idx:04d}.png")


# ──────────────────────────────────────────────────────────────
# Figure 4 — Class-level ERP gallery
# ──────────────────────────────────────────────────────────────

def fig_class_erp_gallery(data: dict, save_dir: Path, n_classes: int = 16):
    """
    One sub-panel per class: grand-average ERP ± 1 SD (collapsed over channels).
    Lets you compare temporal brain-response patterns across object categories.
    """
    all_eeg    = data["all_eeg"]     # [N, 128, 440]
    all_labels = data["all_labels"]
    labels     = data["labels"]

    unique_labels = sorted(np.unique(all_labels))[:n_classes]
    ncols = 4
    nrows = (n_classes + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5.5 * ncols, 3.8 * nrows),
        facecolor=DARK_BG, sharex=True,
        gridspec_kw={"hspace": 0.50, "wspace": 0.30},
    )
    fig.suptitle(
        "Class-Level Grand-Average ERP  (mean ± 1 SD, all channels, all subjects)",
        color="white", fontsize=14, y=1.01,
    )

    axes_flat = axes.flatten()

    for i, lbl in enumerate(unique_labels):
        ax     = axes_flat[i]
        mask   = all_labels == lbl
        eegs   = all_eeg[mask]              # [M, 128, 440]
        mean_t = eegs.mean(axis=(0, 1))     # [440,]
        std_t  = eegs.std(axis=(0, 1))

        ax.plot(TIME_MS, mean_t, color="#00d4ff", linewidth=1.6)
        ax.fill_between(TIME_MS,
                         mean_t - std_t, mean_t + std_t,
                         color="#00d4ff", alpha=0.18)
        ax.axvline(0, color="gold", linestyle="--", linewidth=0.7)
        ax.axhline(0, color="white", linestyle=":", linewidth=0.5, alpha=0.25)

        synset = labels[lbl]
        _dark(ax, title=f"Class {lbl}: {synset}  (n={mask.sum()})")
        ax.tick_params(colors=GREY_TEXT, labelsize=6)

        if i % ncols == 0:
            ax.set_ylabel("Amplitude (σ)", color=GREY_TEXT, fontsize=7)
        if i >= (nrows - 1) * ncols:
            ax.set_xlabel("Time (ms)", color=GREY_TEXT, fontsize=7)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    _save(fig, save_dir / "4_class_erp_gallery.png")


# ──────────────────────────────────────────────────────────────
# Figure 5 — Multi-image EEG grid
# ──────────────────────────────────────────────────────────────

def fig_multi_image_grid(data: dict, image_indices: list, save_dir: Path):
    """
    Three-row layout for each selected image:
      Row 0: stimulus thumbnail
      Row 1: grand-average channel-time heatmap
      Row 2: grand-average ERP line
    Enables quick visual comparison across stimulus categories.
    """
    by_image  = data["by_image"]
    img_names = data["img_names"]
    labels    = data["labels"]

    n = len(image_indices)
    fig, axes = plt.subplots(
        3, n, figsize=(3.5 * n, 10), facecolor=DARK_BG,
        gridspec_kw={"hspace": 0.28, "wspace": 0.10},
    )
    fig.suptitle(
        "EEG Spatio-Temporal Signals across Multiple Stimulus Images",
        color="white", fontsize=13, y=1.01,
    )

    for col, img_idx in enumerate(image_indices):
        trials   = by_image[img_idx]
        eegs     = np.stack([t["eeg"] for t in trials])  # [T, 128, 440]
        mean_eeg = eegs.mean(axis=0)                      # [128, 440]
        grand    = mean_eeg.mean(axis=0)                  # [440,]
        grand_sd = eegs.std(axis=(0, 1))

        img_name  = img_names[img_idx]
        synset    = labels[trials[0]["label"]]
        n_trials  = len(trials)

        # Row 0: image
        ax_i = axes[0, col]
        _show_image(ax_i, img_name,
                    title=f"{synset}\n({n_trials} trials)")

        # Row 1: heatmap
        ax_h = axes[1, col]
        vmax = np.percentile(np.abs(mean_eeg), 97)
        ax_h.imshow(
            mean_eeg, aspect="auto", origin="lower",
            extent=[TIME_MS[0], TIME_MS[-1], 0.5, 128.5],
            cmap="RdBu_r", vmin=-vmax, vmax=vmax,
            interpolation="nearest",
        )
        ax_h.axvline(0, color="gold", linestyle="--", linewidth=0.7)
        _dark(ax_h)
        ax_h.tick_params(colors=GREY_TEXT, labelsize=5)
        if col == 0:
            ax_h.set_ylabel("Channel #", color=GREY_TEXT, fontsize=7)
        else:
            ax_h.set_yticks([])

        # Row 2: ERP
        ax_e = axes[2, col]
        ax_e.plot(TIME_MS, grand, color="#00d4ff", linewidth=1.4)
        ax_e.fill_between(TIME_MS,
                           grand - grand_sd, grand + grand_sd,
                           color="#00d4ff", alpha=0.15)
        ax_e.axvline(0, color="gold", linestyle="--", linewidth=0.7)
        ax_e.axhline(0, color="white", linestyle=":", linewidth=0.5, alpha=0.3)
        _dark(ax_e, xlabel="Time (ms)")
        ax_e.tick_params(colors=GREY_TEXT, labelsize=5)
        if col == 0:
            ax_e.set_ylabel("Amplitude (σ)", color=GREY_TEXT, fontsize=7)
        else:
            ax_e.set_yticks([])

    _save(fig, save_dir / "5_multi_image_eeg_grid.png")


# ──────────────────────────────────────────────────────────────
# Figure 6 — Within-image ERP variability ribbon plot
# ──────────────────────────────────────────────────────────────

def fig_variability_ribbon(data: dict, image_idx: int, save_dir: Path):
    """
    Shows the full distribution of per-channel ERPs for one image.
    Ribbons = percentile bands (10th–90th, 25th–75th) + median.
    One sub-panel for each subject to highlight intra- vs inter-subject spread.
    """
    by_image  = data["by_image"]
    img_names = data["img_names"]
    labels    = data["labels"]

    trials   = by_image[image_idx]
    img_name = img_names[image_idx]
    synset   = labels[trials[0]["label"]]

    # Group by subject
    subj_eegs: dict = defaultdict(list)
    for t in trials:
        subj_eegs[t["subject"]].append(t["eeg"])

    subjects = sorted(subj_eegs)[:6]   # cap at 6 for readability
    n_s      = len(subjects)

    fig, axes = plt.subplots(
        1, n_s, figsize=(4.5 * n_s, 4.5), facecolor=DARK_BG,
        sharey=True, gridspec_kw={"wspace": 0.10},
    )
    if n_s == 1:
        axes = [axes]

    fig.suptitle(
        f"Per-Channel ERP Variability (percentile ribbons)\n"
        f"{img_name}  ({synset})",
        color="white", fontsize=12, y=1.03,
    )

    for i, s in enumerate(subjects):
        ax   = axes[i]
        eegs = np.stack(subj_eegs[s])    # [trials, 128, 440]
        # Collapse across trials → one value per (channel, time)
        ch_mean = eegs.mean(axis=0)       # [128, 440]

        # Percentile ribbons over channels
        p10 = np.percentile(ch_mean, 10, axis=0)
        p25 = np.percentile(ch_mean, 25, axis=0)
        p50 = np.percentile(ch_mean, 50, axis=0)
        p75 = np.percentile(ch_mean, 75, axis=0)
        p90 = np.percentile(ch_mean, 90, axis=0)

        ax.fill_between(TIME_MS, p10, p90, color="#00d4ff", alpha=0.12, label="10–90 %")
        ax.fill_between(TIME_MS, p25, p75, color="#00d4ff", alpha=0.28, label="25–75 %")
        ax.plot(TIME_MS, p50, color="#00d4ff", linewidth=1.8, label="Median")

        ax.axvline(0, color="gold", linestyle="--", linewidth=0.8)
        ax.axhline(0, color="white", linestyle=":", linewidth=0.5, alpha=0.3)

        _dark(ax, title=f"Subject {s}  ({len(subj_eegs[s])} trials)",
              xlabel="Time (ms)")
        if i == 0:
            ax.set_ylabel("Amplitude (σ)", color=GREY_TEXT, fontsize=8)
            ax.legend(fontsize=7, facecolor="#1e1e1e",
                      labelcolor="white", edgecolor=SPINE_COL)

    _save(fig, save_dir / f"6_variability_ribbon_img{image_idx:04d}.png")


# ──────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────

def _save(fig, path: Path, dpi: int = 150):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  OK  {path.name}")


def _best_image(by_image: dict) -> int:
    """Return the image index with the most unique subjects."""
    return max(
        by_image,
        key=lambda idx: len({t["subject"] for t in by_image[idx]}),
    )


def _diverse_images(data: dict, n: int = 8) -> list:
    """One image per class (sorted by label), up to n."""
    all_images = data["all_images"]
    all_labels = data["all_labels"]
    by_image   = data["by_image"]

    label_to_img = {}
    for img_idx, lbl in zip(all_images, all_labels):
        img_idx = int(img_idx)
        if lbl not in label_to_img and img_idx in by_image:
            label_to_img[int(lbl)] = img_idx

    return [label_to_img[lbl] for lbl in sorted(label_to_img)[:n]]


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  CATVis  —  EEG Time-Series Spatio-Temporal Analysis")
    print("=" * 60)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load
    data     = load_eeg_data()
    by_image = data["by_image"]

    # Pick a "spotlight" image (most subjects → richest inter-subject plot)
    spotlight = _best_image(by_image)
    n_subj    = len({t["subject"] for t in by_image[spotlight]})
    n_trials  = len(by_image[spotlight])
    img_name  = data["img_names"][spotlight]
    print(
        f"\nSpotlight -> image {spotlight}  ({img_name})"
        f"  -  {n_trials} trials  -  {n_subj} subjects\n"
    )

    print("[1/6] Single-image deep dive …")
    fig_deep_dive(data, spotlight, OUT_DIR)

    print("[2/6] Same image, per-subject heatmaps …")
    fig_multi_subject(data, spotlight, OUT_DIR, max_subjects=6)

    print("[3/6] Inter-subject correlation …")
    fig_intersubject(data, spotlight, OUT_DIR)

    print("[4/6] Class ERP gallery (first 16 classes) …")
    fig_class_erp_gallery(data, OUT_DIR, n_classes=16)

    print("[5/6] Multi-image EEG grid (8 diverse classes) …")
    diverse = _diverse_images(data, n=8)
    fig_multi_image_grid(data, diverse, OUT_DIR)

    print("[6/6] Per-channel variability ribbon plot …")
    fig_variability_ribbon(data, spotlight, OUT_DIR)

    print(f"\nAll 6 figures saved to:\n  {OUT_DIR}\n")


if __name__ == "__main__":
    main()
