import os
import shutil
import torch
import numpy as np
from pathlib import Path
from torchvision import transforms
from tqdm import tqdm
import tifffile

# Please ensure that the 'models_mae' module is available in your environment.
import sys
sys.path.append('/mnt/eternus/users/Shuo/project/30_BIBM_conference/github_code/MAE')
import models_mae


def prepare_model(ckpt_dir, arch='mae_vit_base_patch16'):
    # Build the model with the specified architecture and image size
    model = getattr(models_mae, arch)(img_size=224)
    # Load the pretrained checkpoint
    # checkpoint = torch.load(ckpt_dir, map_location='cpu')
    checkpoint = torch.load(ckpt_dir, map_location='cpu', weights_only=False)
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print("Checkpoint loading message:", msg)
    return model

def transform_2d(img):
    # Convert grayscale image to RGB if necessary
    if len(img.shape) == 2:
        # Stack the grayscale image into three channels (H, W, 3)
        img = np.stack([img, img, img], axis=-1)
    # Convert the image to a PIL image
    img = transforms.ToPILImage()(img)
    # Apply a random resized crop with bicubic interpolation
    img = transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=transforms.InterpolationMode.BICUBIC)(img)
    # Apply a random horizontal flip
    img = transforms.RandomHorizontalFlip()(img)
    # Convert the PIL image to tensor
    img = transforms.ToTensor()(img)
    # Normalize the tensor with mean and std
    img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])(img)
    return img

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract features from 3D patches using a pretrained MAE model")
    parser.add_argument('--dataset_dir', type=str, required=True, 
                        help='Path to the dataset directory containing .tiff files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the extracted features')
    parser.add_argument('--ckpt_dir', type=str, required=True,
                        help='Path to the pretrained MAE checkpoint (e.g., checkpoint-399.pth)')
    args = parser.parse_args()

    # Create the save directory based on the core_set_select_ratio inside output_dir
    save_dir = os.path.join(args.output_dir, "dataset_feature")
    os.makedirs(save_dir, exist_ok=True)

    # Load the pretrained MAE model
    model = prepare_model(args.ckpt_dir, 'mae_vit_base_patch16')
    model.eval()
    model.cuda()

    # Get a sorted list of all .tiff files in the dataset directory
    img_list = sorted(Path(args.dataset_dir).rglob("*.tiff"))
    total_images = len(img_list)
    print("Total images:", total_images)


    embeddings = []
    options_available = []
    keys = {}

    # Process each image with a progress bar
    for i, img_path in enumerate(tqdm(img_list)):
        # Read the tiff image using tifffile
        img = tifffile.imread(str(img_path))
        
        # Determine if the image is 2D or 3D based on its shape
        if len(img.shape) == 2:
            # Process as a 2D image
            img_tensor = transform_2d(img)
            # Add batch dimension
            img_tensor = img_tensor.unsqueeze(0)
            with torch.no_grad():
                # Extract feature using the model's encoder
                embedding, _, _ = model.forward_encoder(img_tensor.float().cuda(), mask_ratio=0)
            # Append the flattened embedding
            embeddings.append(embedding.detach().cpu().flatten())
            options_available.append(str(img_path.name))
            keys[str(img_path.name)] = i
            torch.cuda.empty_cache()
        elif len(img.shape) == 3:
            # Process as a 3D image by iterating over each slice
            slice_embeddings = []
            for slice_idx in range(img.shape[0]):
                slice_img = img[slice_idx]
                slice_tensor = transform_2d(slice_img)
                slice_tensor = slice_tensor.unsqueeze(0)
                with torch.no_grad():
                    slice_embedding, _, _ = model.forward_encoder(slice_tensor.float().cuda(), mask_ratio=0)
                slice_embeddings.append(slice_embedding.detach().cpu().flatten())
                torch.cuda.empty_cache()
            # Concatenate embeddings from all slices to form the 3D image embedding
            embedding_3d = torch.cat(slice_embeddings, dim=0)
            embeddings.append(embedding_3d)
            options_available.append(str(img_path.name))
            keys[str(img_path.name)] = i
        else:
            print(f"Unsupported image shape: {img.shape}")

    # Stack all embeddings into a single tensor
    embeddings = torch.stack(embeddings, dim=0)
    
    # Save the keys and available options
    np.save(os.path.join(save_dir, 'keys_512.npy'), keys)
    np.save(os.path.join(save_dir, 'options_available_512.npy'), options_available)
    # Save the extracted features (embeddings)
    torch.save(embeddings, os.path.join(save_dir, 'embeddings_512.pth'))
    
    # Output the shape of the embeddings tensor
    print("Embeddings shape:", embeddings.shape)

if __name__ == '__main__':
    main()
