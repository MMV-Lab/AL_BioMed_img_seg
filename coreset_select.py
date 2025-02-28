import os
import torch
import numpy as np
from tqdm import tqdm
import shutil
import time
import torch.nn.functional as F
from pathlib import Path
import argparse
import yaml  # for saving the YAML file

# Function to compute the cosine distance matrix between two tensors
def cosine_distance_matrix(tensor1, tensor2):
    # Compute the norms of the tensors
    norm_tensor1 = torch.norm(tensor1, dim=1, keepdim=True)
    norm_tensor2 = torch.norm(tensor2, dim=1, keepdim=True)
    # Normalize the tensors
    norm_tensor1 = tensor1 / norm_tensor1
    norm_tensor2 = tensor2 / norm_tensor2
    # Compute the cosine distance (1 - cosine similarity)
    dot_product = 1 - torch.mm(norm_tensor1, norm_tensor2.t())
    return dot_product

# Function to select the most representative sample based on maximum minimum distance
def calc_F(distance, options, images_selected, keys):
    max_dis = float('-inf')  # Initialize max_dis to negative infinity
    best_option = None  # Initialize best_option to None
    # Iterate through each option
    for option in options:
        min_dis = float('inf')  # Initialize min_dis to positive infinity for each option
        # Compute the minimum distance between the option and all selected images
        for image_selected in images_selected:
            min_dis = min(min_dis, distance[keys[option], keys[image_selected]])
        # Update the best option if the current option's minimum distance is larger than max_dis
        if max_dis < min_dis:
            max_dis = min_dis
            best_option = option
    return best_option

def main():
    parser = argparse.ArgumentParser(description="Core set selection based on extracted features")
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Directory containing the original dataset tiff images')
    parser.add_argument('--dataset_feature_dir', type=str, required=True,
                        help='Directory containing the pre-extracted dataset features (embeddings, keys, etc.)')
    parser.add_argument('--core_set_select_ratio', type=float, required=True,
                        help='Ratio for core set selection (e.g., 0.5)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the core set selection results')
    args = parser.parse_args()

    # Create save directory inside output_dir using core_set_select_ratio as folder name
    save_dir = os.path.join(args.output_dir, str(args.core_set_select_ratio))
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # Get sorted list of all tiff images in dataset_dir
    img_list = sorted(Path(args.dataset_dir).rglob("*.tiff"))
    total_samples = len(img_list)
    # Create a mapping from image file name to its absolute path
    img_path_dict = {p.name: str(p.resolve()) for p in img_list}

    # Calculate the number of samples for the core set
    k = int(total_samples * args.core_set_select_ratio)
    print("Total samples:", total_samples)
    print("Number of samples to select (k):", k)

    # Load the feature tensor (embeddings)
    embeddings_path = os.path.join(args.dataset_feature_dir, 'embeddings_512.pth')
    embeddings = torch.load(embeddings_path)
    print("Original embeddings shape:", embeddings.shape)

    # If the embedding dimension is greater than 1000, downsample to 1000 using interpolation
    if embeddings.shape[1] > 1000:
        embeddings = embeddings.unsqueeze(1)  # add channel dimension
        scale_factor = embeddings.shape[2] / 1000
        embeddings = F.interpolate(embeddings, scale_factor=1/scale_factor, mode='linear')
        embeddings = embeddings.squeeze(1)
    print("Resampled embeddings shape:", embeddings.shape)
    print("Embeddings dtype:", embeddings.dtype)

    # Compute the cosine distance matrix between embeddings
    distance = cosine_distance_matrix(embeddings, embeddings)
    np_distance = distance.detach().cpu().numpy()
    # Save the distance matrix in the dataset_feature_dir
    np.save(os.path.join(args.dataset_feature_dir, 'distance_512_pool.npy'), np_distance)
    print("Saved distance matrix with shape:", np_distance.shape)

    # Load the distance matrix, keys, and options_available
    distance = np.load(os.path.join(args.dataset_feature_dir, 'distance_512_pool.npy'))
    keys = np.load(os.path.join(args.dataset_feature_dir, 'keys_512.npy'), allow_pickle=True).item()
    options_available = np.load(os.path.join(args.dataset_feature_dir, 'options_available_512.npy')).tolist()
    # Filter keys to include only those present in options_available
    keys = {key: keys[key] for key in options_available}

    # Initialize the selected images list using random initialization
    images_selected = []
    np.random.seed(4099)
    Random_Initialization = 3
    for i in range(Random_Initialization):
        init_id = np.random.randint(0, len(options_available))
        init_data = options_available[init_id]
        images_selected.append(init_data)
        options_available.remove(init_data)

    # Iteratively select samples using the maximum minimum distance strategy
    for i in tqdm(range(k - len(images_selected)), desc="Core set selection"):
        start = time.time()
        best_option = calc_F(distance, options_available, images_selected, keys)
        end = time.time()
        # print("Selected:", best_option, "Time taken:", end - start)
        options_available.remove(best_option)
        images_selected.append(best_option)

    # Convert selected images (filenames) to their absolute paths
    selected_absolute_paths = [img_path_dict[name] for name in images_selected if name in img_path_dict]

    # Save the selected core set list as a YAML file in the save_dir
    selected_yaml_path = os.path.join(save_dir, 'selected_core_set.yaml')
    with open(selected_yaml_path, 'w') as f:
        yaml.dump(selected_absolute_paths, f)
    print("Core set selection saved at:", selected_yaml_path)

if __name__ == '__main__':
    main()
