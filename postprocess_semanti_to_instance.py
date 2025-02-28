import os
import argparse
import numpy as np
from tqdm import tqdm
from skimage.io import imread, imsave
from scipy import ndimage

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Postprocess semantic segmentation to instance segmentation using connected components.")
    parser.add_argument("--semantic_pred", required=True, help="Input folder containing semantic segmentation predictions (TIF images).")
    args = parser.parse_args()

    # Define input folder
    input_folder = args.semantic_pred

    # Create output folder in the same directory as the input folder, named 'instance_pred'
    parent_dir = os.path.dirname(os.path.abspath(input_folder))
    output_folder = os.path.join(parent_dir, "instance_pred")
    os.makedirs(output_folder, exist_ok=True)

    # Get a sorted list of TIF image files in the input folder
    # Get a sorted list of TIF image files in the input folder (supports .tif and .tiff)
    image_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith((".tif", ".tiff"))])

    if not image_files:
        print("No TIF files found in the input folder.")
        return

    # Read the first image to determine the image shape
    first_image_path = os.path.join(input_folder, image_files[0])
    first_image = imread(first_image_path)
    img_shape = first_image.shape
    num_images = len(image_files)

    # Allocate a numpy array to store the 3D volume (stack of images)
    volume = np.zeros((num_images, *img_shape), dtype=np.int64)

    # Read images from the input folder and stack them into a 3D array
    for idx, filename in tqdm(enumerate(image_files), total=num_images, desc="Reading images"):
        img_path = os.path.join(input_folder, filename)
        volume[idx] = imread(img_path)

    # Apply connected components to convert semantic segmentation to instance segmentation
    labeled_volume, num_objects = ndimage.label(volume)
    print(f"Number of objects found: {num_objects}")

    # Save each slice of the instance segmentation result to the output folder
    for idx in tqdm(range(labeled_volume.shape[0]), desc="Saving instance segmentation slices"):
        slice_instance = labeled_volume[idx]
        output_path = os.path.join(output_folder, f"instance_{idx:04d}.tif")
        imsave(output_path, slice_instance.astype(np.uint16))
    
    print(f"Instance segmentation results have been saved in: {output_folder}")

if __name__ == "__main__":
    main()
