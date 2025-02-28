import os
import shutil
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def sliding_window_crop(image, crop_size=(512, 512), step_size=512):
    height, width = image.shape[:2]
    crop_height, crop_width = crop_size
    windows = []
    for top in range(0, height - crop_height + 1, step_size):
        for left in range(0, width - crop_width + 1, step_size):
            windows.append((top, left))
    return windows

def main(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # 如果输出文件夹存在则清空，否则创建
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有png格式的图片路径
    img_paths = sorted(input_dir.glob("*.png"))
    
    for filepath in tqdm(img_paths, desc="Processing images"):
        # 读取图像并转换为灰度图
        img = Image.open(filepath).convert("L")
        img_arr = np.array(img)
        
        # 利用滑动窗口裁剪成512x512
        windows = sliding_window_crop(img_arr, crop_size=(512, 512), step_size=512)
        for i, (top, left) in enumerate(windows):
            cropped_arr = img_arr[top:top + 512, left:left + 512]
            cropped_img = Image.fromarray(cropped_arr)
            out_path = output_dir / f"{filepath.stem}_crop_{i}.tiff"
            cropped_img.save(out_path, format="TIFF")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Crop 2D PNG images using a sliding window (512x512) and save as TIFF for MAE trainset."
    )
    parser.add_argument('--inputdir', type=str, required=True, help="Path to input directory containing PNG images")
    parser.add_argument('--outputdir', type=str, required=True, help="Path to output directory to save cropped TIFF images")
    args = parser.parse_args()
    
    main(args.inputdir, args.outputdir)
