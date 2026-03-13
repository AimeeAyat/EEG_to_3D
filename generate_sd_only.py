import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import os
import json
import torch
import yaml
import subprocess
import shutil
import random
from PIL import Image

os.chdir(project_root)

from src.models.eeg_classifier import EEGClassifier
from src.models.contrastive_encoder import ContrastiveEncoder
from src.data.data_loader import CATVisDataLoader
from src.data.preprocessor import DataPreprocessor
from src.pipeline.retrieval import TextRetrieval

# Flux.1 imports
from diffusers import FluxPipeline

# Background removal
from rembg import remove

# def create_enhanced_prompt(predicted_class: str, caption: str) -> str:
#     """
#     Create an enhanced prompt optimized for generating clean, centered objects
#     suitable for 3D reconstruction with Wonder3D.
#     """
#     # Clean up the caption - extract key descriptive elements
#     # Remove phrases that might add unwanted background elements
#     caption_clean = caption.replace("sitting on", "").replace("in front of", "")
#     caption_clean = caption_clean.replace("on top of", "").replace("next to", "")

#     # Build enhanced prompt for clean object generation
#     prompt = (



#         f"A high-quality, detailed {predicted_class}, "
#         f"{caption_clean}, "
#         f"isolated object, centered composition, "
#         f"pure white background, studio photography, "
#         f"professional product shot, soft diffused lighting, "
#         f"sharp focus, high resolution, 8k, photorealistic"
#     )

#     return prompt


# def create_negative_prompt() -> str:
#     """Create negative prompt to avoid unwanted elements."""
#     return (
#         "background, environment, scene, multiple objects, "
#         "shadows on ground, floor, table, surface, "
#         "text, watermark, logo, signature, "
#         "blurry, low quality, distorted, deformed, "
#         "cropped, cut off, partial object"
#     )

def is_caption_relevant(predicted_class: str, caption: str) -> bool:
    """Check if caption is relevant to the predicted class."""
    # Extract key words from predicted class
    class_words = predicted_class.lower().replace("-", " ").replace("_", " ").split()
    caption_lower = caption.lower()

    # Check if any class word appears in caption
    for word in class_words:
        if len(word) > 3 and word in caption_lower:  # Skip short words
            return True

    return False


def create_enhanced_prompt(predicted_class: str, caption: str) -> tuple:
    """
    Create an enhanced prompt for single object with pose/action details.
    Suitable for 3D reconstruction with Wonder3D.

    Returns:
        tuple: (prompt, caption_used) - prompt string and whether caption was used
    """
    # Check if caption is relevant to predicted class
    use_caption = is_caption_relevant(predicted_class, caption)

    if use_caption:
        # Remove only multi-object phrases, keep action/pose words
        caption_clean = caption.replace("A group of", "A single")
        caption_clean = caption_clean.replace("group of", "single")
        caption_clean = caption_clean.replace("Several", "One").replace("several", "one")
        caption_clean = caption_clean.replace("Many", "One").replace("many", "one")
        caption_clean = caption_clean.replace("Two", "One").replace("two", "one")
        caption_clean = caption_clean.replace("Three", "One").replace("three", "one")
        caption_clean = caption_clean.replace("people", "").replace("persons", "")
        caption_clean = caption_clean.replace("men", "").replace("women", "")

        prompt = (
            f"ONE single {predicted_class}, exactly one object, "
            f"{caption_clean}, "
            # f"front view, facing the camera, frontal perspective, "
            f"full body visible, entire figure from head to feet, complete anatomy, whole body in frame, "
            f"solo subject only, no other objects, "
            f"isolated on pure white background, "
            f"centered, studio product photography, "
            f"professional lighting, sharp focus, 8k"
        )
    else:
        # Caption doesn't match predicted class - use only predicted class
        prompt = (
            f"ONE single {predicted_class}, exactly one object, "
            # f"front view, facing the camera, frontal perspective, "
            f"full body visible, entire figure from head to feet, complete anatomy, whole body in frame, "
            f"solo subject only, no other objects, "
            f"isolated on pure white background, "
            f"centered, studio product photography, "
            f"professional lighting, sharp focus, 8k, photorealistic"
        )

    return prompt, use_caption


def create_negative_prompt() -> str:
    """Create negative prompt to avoid multiple objects."""
    return (
        "multiple objects, two objects, three objects, many objects, "
        "group, collection, crowd, pair, couple, several, duplicate, "
        "second object, additional items, extra objects, "
        "people, person, human, hands, face, "
        "busy background, cluttered, complex scene, "
        "text, watermark, logo, signature, "
        "blurry, low quality, distorted, deformed, "
        "cropped, cut off, partial object"
    )


def main():
    with open('config/config.yaml') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    print("Loading EEG models...")
    classifier = EEGClassifier(config).to(device)
    classifier.model.load_state_dict(torch.load('checkpoints/eeg_classifier_best.pth', map_location=device))
    classifier.eval()

    contrastive = ContrastiveEncoder(config).to(device)
    contrastive.model.load_state_dict(torch.load('checkpoints/contrastive_model_best.pth', map_location=device))
    contrastive.eval()

    print("Loading data...")
    data_loader = CATVisDataLoader(config)
    df = data_loader.get_dataset_dataframe()
    _, _, test_df = data_loader.get_train_val_test_splits(df)

    preprocessor = DataPreprocessor(config)
    test_dataset = preprocessor.create_pipeline_dataset(test_df)

    text_retrieval = TextRetrieval(config, contrastive, device)
    text_retrieval.setup_retrieval_corpus(test_df)

    print("Loading Flux.1 model...")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch.bfloat16
    )
    pipe.enable_model_cpu_offload()  # Saves VRAM by offloading to CPU when not needed
    print("Flux.1 loaded successfully")

    output_dir = Path('outputs/eeg_multiview_batch')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create directory for ground truth images
    gt_output_dir = Path('outputs/ground_truth_images')
    gt_output_dir.mkdir(parents=True, exist_ok=True)

    # Metadata dictionary to store info for visualization
    metadata = {}

    wonder3d_dir = Path('Wonder3D')
    wonder3d_input = wonder3d_dir / 'example_images'
    wonder3d_input.mkdir(parents=True, exist_ok=True)

    indices = random.sample(range(len(test_dataset)), min(10, len(test_dataset)))

    for idx, sample_idx in enumerate(indices):
        print(f"\n{'='*60}")
        print(f"Sample {idx+1}/10 (index {sample_idx})")
        print(f"{'='*60}")

        sample = test_dataset[sample_idx]
        eeg_data = sample['eeg'].unsqueeze(0).to(device)
        ground_truth_idx = sample['label'].item() if torch.is_tensor(sample['label']) else sample['label']
        image_idx = sample['image'].item() if torch.is_tensor(sample['image']) else sample['image']

        with torch.no_grad():
            outputs, _ = classifier(eeg_data)
            pred_idx = outputs.argmax().item()

        # Use the actual label order from raw data (not config keys which have different order)
        labels_order = data_loader.raw_data['labels']
        predicted_class = config['class_prompts'][labels_order[pred_idx]]
        ground_truth_class = config['class_prompts'][labels_order[ground_truth_idx]]

        retrieved = text_retrieval.retrieve_top_k_from_eeg(eeg_data.squeeze(0), k=1)
        top_caption = retrieved[0][0]

        print(f"Ground Truth: {ground_truth_class}")
        print(f"Predicted: {predicted_class}")
        print(f"Caption: {top_caption}")

        # Create enhanced prompt for clean object generation
        prompt, caption_used = create_enhanced_prompt(predicted_class, top_caption)
        if caption_used:
            print(f"Using caption in prompt (matches predicted class)")
        else:
            print(f"Ignoring caption (doesn't match predicted class)")
        print(f"Enhanced prompt: {prompt[:100]}...")

        print("Generating image with Flux.1...")

        # Generate with Flux.1-schnell (optimized for 4 steps)
        result = pipe(
            prompt,
            guidance_scale=0.0,  # Flux.1-schnell doesn't use guidance
            num_inference_steps=4,
            max_sequence_length=256,
            generator=torch.Generator("cpu").manual_seed(42 + idx),
            height=1024,
            width=1024,
        )

        generated_image = result.images[0]

        # Remove background using rembg
        print("Removing background...")
        image_no_bg = remove(generated_image)

        # Save the image with transparent background
        # Sanitize class names for filename (replace spaces with underscores)
        gt_name = ground_truth_class.replace(" ", "_").replace("/", "-")
        pred_name = predicted_class.replace(" ", "_").replace("/", "-")
        img_name = f"{gt_name}_{pred_name}_{idx:03d}.png"
        img_path = output_dir / img_name
        image_no_bg.save(img_path)
        print(f"Saved: {img_path}")

        # Copy actual ground truth image from dataset
        gt_image_rel_path = data_loader.image_to_path[image_idx]
        gt_image_src = Path(config['data']['root_dir']) / config['data']['imagenet_images'] / gt_image_rel_path
        gt_image_dst = gt_output_dir / f"{gt_name}_{pred_name}_{idx:03d}_gt.JPEG"
        if gt_image_src.exists():
            shutil.copy(gt_image_src, gt_image_dst)
            print(f"Saved GT: {gt_image_dst}")
        else:
            print(f"Warning: GT image not found: {gt_image_src}")

        # Save metadata for this sample
        sample_key = f"{gt_name}_{pred_name}_{idx:03d}"
        metadata[sample_key] = {
            'ground_truth_class': ground_truth_class,
            'predicted_class': predicted_class,
            'retrieved_caption': top_caption,
            'caption_used_in_prompt': caption_used,
            'sample_idx': sample_idx
        }

        # Copy to Wonder3D input directory
        wonder3d_input_path = wonder3d_input / img_name
        shutil.copy(img_path, wonder3d_input_path)

        print("Running Wonder3D...")
        try:
            subprocess.run([
                'python', 'test_mvdiffusion_seq.py',
                '--config', 'configs/mvdiffusion-joint-ortho-6views.yaml',
                f'validation_dataset.filepaths=[{img_name}]'
            ], cwd=str(wonder3d_dir), check=True, capture_output=True)
            print("Wonder3D completed")
        except subprocess.CalledProcessError as e:
            print(f"Wonder3D failed: {e}")

    # Save metadata to JSON file
    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")

    print(f"\n{'='*60}")
    print(f"Completed! Results in: {output_dir}")
    print(f"Wonder3D outputs in: {wonder3d_dir / 'outputs'}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
