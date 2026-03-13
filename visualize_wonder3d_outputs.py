"""
Visualization script for Wonder3D multi-view outputs.
Creates grid plots with ground truth image + 6 RGB views per row.
Each plot has 7 rows x 7 columns.
Includes captions for ground truth, predicted class, and retrieved text.
"""

import json
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import math
import textwrap


def load_metadata(metadata_path: Path) -> dict:
    """Load metadata from JSON file."""
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return {}


def get_wonder3d_outputs(wonder3d_output_dir: Path):
    """
    Get all Wonder3D output folders and their RGB images.

    Returns:
        List of tuples: (folder_name, dict of view_name -> image_path)
    """
    outputs = []

    # Wonder3D saves outputs in cropsize-192-cfg3.0 subfolder
    output_base = wonder3d_output_dir / 'cropsize-192-cfg3.0'

    if not output_base.exists():
        print(f"Warning: {output_base} does not exist")
        return outputs

    # RGB view names in desired order
    view_names = ['front', 'front_left', 'left', 'back', 'right', 'front_right']

    for folder in sorted(output_base.iterdir()):
        if folder.is_dir():
            rgb_images = {}
            for view in view_names:
                rgb_path = folder / f'rgb_000_{view}.png'
                if rgb_path.exists():
                    rgb_images[view] = rgb_path

            # Only include if we have all 6 views
            if len(rgb_images) == 6:
                outputs.append((folder.name, rgb_images))
            else:
                print(f"Warning: {folder.name} missing some views, skipping")

    return outputs


def find_ground_truth_image(sample_name: str, gt_image_dir: Path):
    """
    Find the actual ground truth image for a given sample.
    Ground truth images are saved with '_gt.JPEG' suffix.
    """
    # Look for ground truth image with _gt suffix
    gt_path = gt_image_dir / f'{sample_name}_gt.JPEG'
    if gt_path.exists():
        return gt_path

    # Try with different extensions
    for ext in ['.JPEG', '.jpeg', '.jpg', '.png']:
        gt_path = gt_image_dir / f'{sample_name}_gt{ext}'
        if gt_path.exists():
            return gt_path

    return None


def wrap_text(text: str, width: int = 25) -> str:
    """Wrap text to specified width."""
    return '\n'.join(textwrap.wrap(text, width=width))


def create_visualization_plots(
    wonder3d_output_dir: Path,
    gt_image_dir: Path,
    generated_image_dir: Path,
    save_dir: Path,
    metadata: dict,
    rows_per_plot: int = 5,
    cols_per_plot: int = 8  # 1 GT + 6 views + 1 text column
):
    """
    Create visualization plots with ground truth + 6 RGB views + text details per row.
    Includes captions showing GT class, predicted class, and retrieved text.

    Args:
        wonder3d_output_dir: Path to Wonder3D outputs folder
        gt_image_dir: Path to folder containing ground truth images
        generated_image_dir: Path to folder containing generated images
        save_dir: Path to save visualization plots
        metadata: Dictionary with sample metadata
        rows_per_plot: Number of rows per plot (samples)
        cols_per_plot: Number of columns (8: 1 GT + 6 views + 1 text)
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # Get all Wonder3D outputs
    outputs = get_wonder3d_outputs(wonder3d_output_dir)

    if not outputs:
        print("No Wonder3D outputs found!")
        return

    print(f"Found {len(outputs)} Wonder3D outputs")

    # View order for display
    view_order = ['front', 'front_left', 'left', 'back', 'right', 'front_right']
    column_labels = ['Ground Truth'] + [v.replace('_', ' ').title() for v in view_order] + ['Details']

    # Calculate number of plots needed
    num_plots = math.ceil(len(outputs) / rows_per_plot)

    for plot_idx in range(num_plots):
        start_idx = plot_idx * rows_per_plot
        end_idx = min(start_idx + rows_per_plot, len(outputs))
        current_outputs = outputs[start_idx:end_idx]

        # Create figure - wider to accommodate text column
        fig_width = cols_per_plot * 3
        fig_height = len(current_outputs) * 3.5
        fig, axes = plt.subplots(
            len(current_outputs), cols_per_plot,
            figsize=(fig_width, fig_height),
            gridspec_kw={'width_ratios': [1, 1, 1, 1, 1, 1, 1, 1.5]}  # Text column wider
        )

        # Handle single row case
        if len(current_outputs) == 1:
            axes = [axes]

        for row_idx, (sample_name, rgb_images) in enumerate(current_outputs):
            # Get metadata for this sample
            sample_meta = metadata.get(sample_name, {})
            gt_class = sample_meta.get('ground_truth_class', 'Unknown')
            pred_class = sample_meta.get('predicted_class', 'Unknown')
            retrieved_caption = sample_meta.get('retrieved_caption', 'No caption')

            # Find ground truth image
            gt_path = find_ground_truth_image(sample_name, gt_image_dir)

            # Plot ground truth (column 0)
            ax = axes[row_idx][0]
            if gt_path and gt_path.exists():
                gt_img = Image.open(gt_path).convert('RGB')
                ax.imshow(gt_img)
            else:
                ax.text(0.5, 0.5, 'GT Not Found', ha='center', va='center', fontsize=12)
            ax.axis('off')

            # Plot 6 RGB views (columns 1-6)
            for col_idx, view_name in enumerate(view_order):
                ax = axes[row_idx][col_idx + 1]
                img_path = rgb_images.get(view_name)

                if img_path and img_path.exists():
                    img = Image.open(img_path).convert('RGB')
                    ax.imshow(img)
                else:
                    ax.text(0.5, 0.5, 'Missing', ha='center', va='center', fontsize=12)
                ax.axis('off')

            # Text details column (column 7)
            ax_text = axes[row_idx][7]
            ax_text.axis('off')
            ax_text.set_facecolor('#f5f5f5')

            # Create formatted text with details
            wrapped_caption = wrap_text(retrieved_caption, width=30)
            details_text = (
                f"Ground Truth:\n{gt_class}\n\n"
                f"Predicted:\n{pred_class}\n\n"
                f"Retrieved Caption:\n{wrapped_caption}"
            )

            ax_text.text(0.05, 0.95, details_text,
                        transform=ax_text.transAxes,
                        fontsize=10, fontweight='normal',
                        ha='left', va='top',
                        family='monospace',
                        bbox=dict(boxstyle='round,pad=0.5',
                                 facecolor='white',
                                 edgecolor='gray',
                                 alpha=0.9))

        # Add column headers
        for col_idx, label in enumerate(column_labels):
            axes[0][col_idx].set_title(label, fontsize=12, fontweight='bold', pad=10)

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.15, wspace=0.05)

        # Save plot
        plot_path = save_dir / f'wonder3d_visualization_{plot_idx + 1:02d}.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"Saved: {plot_path}")

    print(f"\nCompleted! {num_plots} visualization plots saved to {save_dir}")


def main():
    # Paths
    project_root = Path(__file__).parent
    wonder3d_output_dir = project_root / 'Wonder3D' / 'outputs'
    gt_image_dir = project_root / 'outputs' / 'ground_truth_images'
    generated_image_dir = project_root / 'outputs' / 'eeg_multiview_batch'
    save_dir = project_root / 'outputs' / 'wonder3d_visualizations'
    metadata_path = generated_image_dir / 'metadata.json'

    print("Wonder3D Multi-View Visualization")
    print("=" * 50)
    print(f"Wonder3D outputs: {wonder3d_output_dir}")
    print(f"Ground truth images: {gt_image_dir}")
    print(f"Generated images: {generated_image_dir}")
    print(f"Metadata file: {metadata_path}")
    print(f"Save directory: {save_dir}")
    print("=" * 50)

    # Load metadata
    metadata = load_metadata(metadata_path)
    if metadata:
        print(f"Loaded metadata for {len(metadata)} samples")
    else:
        print("Warning: No metadata found. Captions will not be displayed.")

    create_visualization_plots(
        wonder3d_output_dir=wonder3d_output_dir,
        gt_image_dir=gt_image_dir,
        generated_image_dir=generated_image_dir,
        save_dir=save_dir,
        metadata=metadata,
        rows_per_plot=5,
        cols_per_plot=8
    )


if __name__ == '__main__':
    main()
