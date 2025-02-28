import os
import numpy as np
import cv2
import tifffile as tiff
from tqdm import tqdm
import argparse

# Define arguments
parser = argparse.ArgumentParser()
parser.add_argument('--raw_inputdir', type=str, required=True, help='Directory of raw input 2D PNG images')
parser.add_argument('--raw_outputdir', type=str, required=True, help='Directory for cropped raw output TIFF images')
parser.add_argument('--label_inputdir_train', type=str, required=True, help='Directory of label training 2D TIFF images')
parser.add_argument('--label_inputdir_val', type=str, required=True, help='Directory of label validation 2D TIFF images')
parser.add_argument('--label_outputdir', type=str, required=True, help='Directory for cropped label output TIFF images')
args = parser.parse_args()

# Function to create or clear a directory
def prepare_directory(directory):
    if os.path.exists(directory):
        for file in os.listdir(directory):
            os.remove(os.path.join(directory, file))
    else:
        os.makedirs(directory)

# Prepare output directories
prepare_directory(args.raw_outputdir)
prepare_directory(args.label_outputdir)

# Function to load and process images
def load_images(directory, max_images=500, size=(4096, 4096), file_ext='.png'):
    files = sorted([f for f in os.listdir(directory) if f.endswith(file_ext)])[:max_images]
    images = []
    for file in tqdm(files, desc=f'Loading images from {directory}'):
        img = cv2.imread(os.path.join(directory, file), cv2.IMREAD_GRAYSCALE)
        img = img[:size[0], :size[1]]  # Crop to (4096, 4096)
        images.append(img)
    return np.array(images)

# Function to perform sliding window cropping
def sliding_window_crop(volume, output_dir, prefix, step_x=256, step_y=256, step_z=16, crop_size=(32, 512, 512)):
    z_max, y_max, x_max = volume.shape
    crop_z, crop_y, crop_x = crop_size
    
    for z in tqdm(range(0, z_max - crop_z + 1, step_z), desc=f'Cropping {prefix} images'):
        for y in range(0, y_max - crop_y + 1, step_y):
            for x in range(0, x_max - crop_x + 1, step_x):
                crop = volume[z:z + crop_z, y:y + crop_y, x:x + crop_x]
                filename = f'crop_{z}_{y}_{x}.tiff'
                tiff.imwrite(os.path.join(output_dir, filename), crop)

# Load raw images
raw_images = load_images(args.raw_inputdir, file_ext='.png')

# Pad raw images to (512, 4096, 4096)
padded_raw_images = np.zeros((512, 4096, 4096), dtype=np.uint8)
padded_raw_images[:raw_images.shape[0], :, :] = raw_images

# Perform sliding window cropping
sliding_window_crop(padded_raw_images, args.raw_outputdir, 'raw')

# Load label images
label_images_train = load_images(args.label_inputdir_train, max_images=400, file_ext='.tif')
label_images_val = load_images(args.label_inputdir_val, max_images=100, file_ext='.tif')

# Merge labels into one array of (500, 4096, 4096) and convert nonzero values to 1
label_images = np.zeros((500, 4096, 4096), dtype=np.uint8)
label_images[:400, :, :] = label_images_train
label_images[400:500, :, :] = label_images_val
label_images[label_images > 0] = 1  # Convert nonzero values to 1

# Pad label images to (512, 4096, 4096)
padded_label_images = np.zeros((512, 4096, 4096), dtype=np.uint8)
padded_label_images[:500, :, :] = label_images


# Perform sliding window cropping
sliding_window_crop(padded_label_images, args.label_outputdir, 'label')
