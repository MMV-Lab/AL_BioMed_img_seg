import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
from cellSAM import segment_cellular_image
import cv2
from aicsimageio.writers import OmeTiffWriter
import argparse
import torch

###########################################################
def get_pseudo_GT(image_paths, output_folder):
    """
    For each image, use a sliding window (256×256) to crop patches, perform CellSAM segmentation on each patch,
    stitch the segmentation results, convert the stitched semantic mask into a binary mask (0 and 1),
    and save the result.
    """
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    patch_size = 4096

    for i, filepath in enumerate(tqdm(image_paths, desc="Processing images")):
        if i >= 500:
            break
        if filepath.suffix.lower() == '.png':
            print(f"Processing image: {filepath}")
            # Read the PNG image in color (BGR format)
            img = cv2.imread(str(filepath), cv2.IMREAD_COLOR)
            if img is None:
                print(f"Failed to load image: {filepath}")
                continue

            # Convert the BGR image to grayscale
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            original_h, original_w = img.shape
            print("Original image shape:", img.shape)
            print("Device:", DEVICE)

            # Pad image so its dimensions are multiples of patch_size
            new_h = ((original_h + patch_size - 1) // patch_size) * patch_size
            new_w = ((original_w + patch_size - 1) // patch_size) * patch_size
            padded_img = np.zeros((new_h, new_w), dtype=img.dtype)
            padded_img[:original_h, :original_w] = img

            # Create an empty semantic mask for the padded image
            full_mask = np.zeros((new_h, new_w), dtype=np.uint8)

            # Generate all patch coordinates
            coords = [(y, x) for y in range(0, new_h, patch_size) for x in range(0, new_w, patch_size)]
            # Process patches with tqdm progress bar (per image)
            for (y, x) in tqdm(coords, desc="Processing patches", leave=False):
                patch = padded_img[y:y+patch_size, x:x+patch_size]
                # Perform CellSAM segmentation on the patch
                try:
                    mask_patch, _, _ = segment_cellular_image(patch, device=DEVICE)
                except Exception as e:
                    print(f"❌ Error in patch segmentation at ({x}, {y}): {e}")
                    mask_patch = np.zeros((patch_size, patch_size), dtype=np.uint8)
                if mask_patch is None:
                    print(f"⚠️ Warning: No cells detected in patch at ({x}, {y}). Using zero mask.")
                    mask_patch = np.zeros((patch_size, patch_size), dtype=np.uint8)
                
                # If mask_patch is multi-channel, convert it to a single channel
                if mask_patch.ndim == 3:
                    # Assume channels-first (3,256,256); otherwise, take the first channel
                    if mask_patch.shape[0] == 3:
                        mask_patch = mask_patch[0]
                    else:
                        mask_patch = mask_patch[..., 0]
                
                mask_patch = mask_patch.astype(np.uint8)
                # Assign the patch result into the full mask
                full_mask[y:y+patch_size, x:x+patch_size] = mask_patch

            # Crop the full mask to the original image size
            full_mask = full_mask[:original_h, :original_w]
            # Convert the stitched semantic mask to binary (0 and 1)
            binary_mask = (full_mask > 0).astype(np.uint8)
            print("Stitched binary semantic mask shape:", binary_mask.shape)

            # Save the semantic mask (binary mask)
            dim_orders = "YX"
            out_path_GT = output_folder / f'{filepath.stem}_GT.tiff'
            OmeTiffWriter.save(binary_mask, out_path_GT, dim_order=dim_orders)
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate pseudo semantic labels using CellSAM with sliding window patch segmentation")
    parser.add_argument("--input_2D_raw_folder", type=str, required=True, help="Input folder for raw 2D images")
    parser.add_argument("--output_folder", type=str, required=True, help="Output folder for CellSAM pseudo labels")
    args = parser.parse_args()

    # Set the input folder (raw images in PNG format)
    folder_2D = Path(args.input_2D_raw_folder)
    image_paths = sorted(folder_2D.glob("*.png"))
    print("Number of images found:", len(image_paths))

    # Create output directory for semantic segmentation masks
    output_folder = Path(args.output_folder)
    output_folder_mask = output_folder / "EM30_R_2D_CellSAM_mask"
    os.makedirs(output_folder_mask, exist_ok=True)

    get_pseudo_GT(image_paths, output_folder_mask)
