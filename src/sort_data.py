import os
import json
import random
from PIL import Image

# --------------------------------------------------------------------------- #
# Simulated forensic metadata
# --------------------------------------------------------------------------- #
# The metadata is fabricated, so its correlation with the label is a free
# parameter. Real-world forensics is a *weak* clue: genuine photos usually carry
# camera provenance but can be stripped (screenshots, re-uploads), and AI images
# usually lack it but can inherit/spoof EXIF. We model exactly one informative
# axis -- "carries camera provenance or not" -- and the knob controls how well
# that axis predicts the true label.
#
# Both classes draw their fields from the SAME vocabularies (only the *rate* of
# camera-vs-stripped differs), so there is no structural tell to bypass the knob.

REAL_CAMERAS = [
    {"Make": "Sony", "Model": "ILCE-7M4", "FocalLength": "35mm"},
    {"Make": "Canon", "Model": "EOS R5", "FocalLength": "50mm"},
    {"Make": "Apple", "Model": "iPhone 15 Pro", "FocalLength": "24mm"},
    {"Make": "Nikon", "Model": "Z6 III", "FocalLength": "85mm"},
    {"Make": "Fujifilm", "Model": "X-T5", "FocalLength": "23mm"},
]
CAMERA_SOFTWARE = [
    "Camera Internal Firmware", "Adobe Lightroom Classic 13.2",
    "Apple Camera 17.4", "Capture One 23",
]
CAMERA_COLOR = ["Display P3", "Adobe RGB", "sRGB"]
CAMERA_COMPRESSION = [
    "JPEG q=92 (4:2:0)", "JPEG q=88 (4:2:2)", "HEIC 10-bit", "Lossy->JPEG q=90",
]
STRIPPED_COMPRESSION = [
    "None / Raw Buffer", "PNG (re-encoded, lossless)", "JPEG q=95 (4:4:4) re-save",
]


def _camera_bearing_metadata(rng):
    """Metadata of a file that carries camera provenance."""
    return {
        "software_sig": rng.choice(CAMERA_SOFTWARE),
        "exif_profile": dict(rng.choice(REAL_CAMERAS)),
        "color_space": rng.choice(CAMERA_COLOR),
        "compression_ratio": rng.choice(CAMERA_COMPRESSION),
    }


def _stripped_metadata(rng):
    """Metadata of a file with no camera provenance (stripped real OR raw AI)."""
    return {
        "software_sig": "None Detected",
        "exif_profile": "None",
        "color_space": "sRGB",
        "compression_ratio": rng.choice(STRIPPED_COMPRESSION),
    }


def simulate_metadata(true_label, metadata_informativeness=0.5, rng=random):
    """Generate forensic metadata whose predictiveness is tunable.

    ``metadata_informativeness`` in [0, 1] sets how well metadata predicts the
    label -- equivalently, the accuracy a metadata-only classifier could reach:

        0.0 -> 50% (independent of the label; pure noise)
        0.5 -> 75% (weak, realistic clue)
        1.0 -> 100% (perfectly diagnostic, i.e. the old leaky behavior)

    Mechanism: real images (label 0) *typically* carry camera provenance and AI
    images (label 1) *typically* don't; with probability ``1 - p_consistent``
    that is flipped (a stripped real photo, or an AI image with spoofed EXIF),
    where ``p_consistent = 0.5 + 0.5 * informativeness``. All the label
    correlation flows through this single rate, so dialing the knob to 0 makes
    the two classes' metadata distributions identical.
    """
    info = max(0.0, min(1.0, metadata_informativeness))
    p_consistent = 0.5 + 0.5 * info

    presents_as_real = (true_label == 0)
    if rng.random() >= p_consistent:
        presents_as_real = not presents_as_real

    return _camera_bearing_metadata(rng) if presents_as_real else _stripped_metadata(rng)


def preprocess_and_standardize_image(input_path, output_path, max_dim=1024):
    """
    Standardizes a target image file:
    - Converts color space to strict RGB (stripping alpha channels)
    - Resizes while preserving aspect ratio if dimensions exceed max_dim
    - Saves as an uncompressed, lossless PNG
    """
    try:
        with Image.open(input_path) as img:
            # 1. Force strict RGB formatting (fixes grayscale or RGBA matrix crashes)
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            # 2. Resize maintaining aspect ratio if it exceeds our maximum profile limit
            w, h = img.size
            if max(w, h) > max_dim:
                if w > h:
                    new_w = max_dim
                    new_h = int(h * (max_dim / w))
                else:
                    new_h = max_dim
                    new_w = int(w * (max_dim / h))
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                
            # 3. Save as uncompressed standalone PNG
            img.save(output_path, "PNG", compress_level=0)
            return True, img.size
    except Exception as e:
        print(f"Error processing image {input_path}: {e}")
        return False, (0, 0)

def build_dataset_pipeline(real_dir, ai_dir, output_images_dir, manifest_path,
                           sample_limit=1500, metadata_informativeness=0.5):
    """
    Iterates through your raw real and fake image directories, standardizes the pixels,
    injects simulated forensic metadata, and produces the nested JSONL dataset blueprint.

    ``metadata_informativeness`` (0..1) controls how predictive the simulated
    metadata is of the label -- see :func:`simulate_metadata`. Lower it to push
    the policy toward image-only reasoning; 0.0 makes metadata pure noise.
    """
    os.makedirs(output_images_dir, exist_ok=True)

    # Gather all valid image assets from your target directories
    valid_exts = ('.png', '.jpg', '.jpeg', '.webp')
    real_files = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.lower().endswith(valid_exts)]
    ai_files = [os.path.join(ai_dir, f) for f in os.listdir(ai_dir) if f.lower().endswith(valid_exts)]

    # Balance target counts
    target_per_class = sample_limit // 2
    selected_real = random.sample(real_files, min(len(real_files), target_per_class))
    selected_ai = random.sample(ai_files, min(len(ai_files), target_per_class))

    total_processed = 0

    with open(manifest_path, "w") as f_manifest:
        # Process Real Images (Label: 0)
        print("Processing real control baseline images...")
        for filepath in selected_real:
            out_filename = f"real_{total_processed:04d}.png"
            out_path = os.path.join(output_images_dir, out_filename)
            
            success, final_dims = preprocess_and_standardize_image(filepath, out_path)
            if not success:
                continue
                
            metadata_payload = simulate_metadata(0, metadata_informativeness)

            record = {
                "id": f"sample_{total_processed:04d}",
                "file_name": out_path,
                "label": 0,
                "label_text": "Real",
                "metadata": metadata_payload
            }
            f_manifest.write(json.dumps(record) + "\n")
            total_processed += 1

        # Process Fake/AI Images (Label: 1)
        print("Processing synthetic AI generated images...")
        for filepath in selected_ai:
            out_filename = f"fake_{total_processed:04d}.png"
            out_path = os.path.join(output_images_dir, out_filename)
            
            success, final_dims = preprocess_and_standardize_image(filepath, out_path)
            if not success:
                continue
                
            metadata_payload = simulate_metadata(1, metadata_informativeness)

            record = {
                "id": f"sample_{total_processed:04d}",
                "file_name": out_path,
                "label": 1,
                "label_text": "AI",
                "metadata": metadata_payload
            }
            f_manifest.write(json.dumps(record) + "\n")
            total_processed += 1

    print(f"\nPipeline Successful! Preprocessed {total_processed} balanced images.")
    print(f"Manifest schema saved cleanly to: {manifest_path}")

if __name__ == "__main__":
    # Define your local path configurations
    RAW_REAL_DIR = "/Users/manasganti/VLM-RL-aicontent-detection/data/raw_data/real_images/real"  # Point to your downloaded Unsplash/COCO/Real folder
    RAW_AI_DIR = "/Users/manasganti/VLM-RL-aicontent-detection/data/raw_data/fake_images"      # Point to your existing AI image folder
    
    OUTPUT_IMG_DIR = "/Users/manasganti/VLM-RL-aicontent-detection/data/processed_images"
    MANIFEST_FILE = "/Users/manasganti/VLM-RL-aicontent-detection/data/metadata.jsonl"
    
    # How predictive the simulated metadata should be of the label:
    #   0.0 -> pure noise (forces image-only reasoning)
    #   0.5 -> weak, realistic forensic clue (metadata-only ~75% accuracy)
    #   1.0 -> perfectly diagnostic (the old leaky behavior)
    METADATA_INFORMATIVENESS = 0.5

    # Run the builder
    build_dataset_pipeline(
        real_dir=RAW_REAL_DIR,
        ai_dir=RAW_AI_DIR,
        output_images_dir=OUTPUT_IMG_DIR,
        manifest_path=MANIFEST_FILE,
        sample_limit=2400, # Optimal sizing for single-machine consumer RL iterations
        metadata_informativeness=METADATA_INFORMATIVENESS,
    )