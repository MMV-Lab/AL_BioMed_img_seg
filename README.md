# AL_BioMed_img_seg

## An Active Learning Pipeline Based on Data-Centric AI for Biomedical Image Instance Segmentation

This repository contains the code and instructions for the **AL_BioMed_img_seg** project, which has been accepted as a poster at the **BVM 2025 Conference**. More details about the conference can be found [here](https://www.bvm-conf.org/).

### Pipeline

![Pipeline](https://github.com/MMV-Lab/AL_BioMed_img_seg/raw/main/fig/pipline.png)

---

## Clone the Repository

```bash
git clone https://github.com/MMV-Lab/AL_BioMed_img_seg
cd AL_BioMed_img_seg
```

## Create and Activate Environment

```bash
conda create -n AL_BioMed_img_seg python=3.10 -y
conda activate AL_BioMed_img_seg
```

### Install PyTorch with CUDA

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Test GPU Availability

```bash
python -c "import torch; print('CUDA is available:', torch.cuda.is_available())"
```

### Install Additional Dependencies

```bash
pip install tqdm tensorboard monai timm==0.4.5 tifffile scikit-image opencv-python-headless matplotlib dask-image scikit-learn git+https://github.com/vanvalenlab/cellSAM.git aicsimageio
```

Alternatively, you can install via `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## Data Preparation

### 3D MitoEM Challenge: Large-scale 3D Mitochondria Instance Segmentation

Dataset: [3D MitoEM Challenge](https://mitoem.grand-challenge.org/)

```bash
mkdir -p data/MitoEM_3D
cd data
chmod +x download_and_unzip_MitoEM_3D.sh
./download_and_unzip_MitoEM_3D.sh /Your/path/to/MitoEM_3D
cd ..
```

---

## MAE Feature Extraction

### 2D Patches Training

```bash
cd MAE
bash pretrain_EM30_R_2D_512_0001.sh /path/to/your/dataset /path/to/output
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch \
--nproc_per_node=8 \
--master_port=25678 main_pretrain.py \
--data_path /path/to/input \
--batch_size 8 \
--model mae_vit_base_patch16 \
--norm_pix_loss \
--mask_ratio 0.75 \
--epochs 400 \
--warmup_epochs 40 \
--blr 1e-3 \
--weight_decay 0.05 \
--accum_iter 4 \
--input_size 512 \
--img_size 512 \
--output_dir /path/to/output > /path/to/log/pretrain_log.txt 2>&1
```

---

## 3D Patched Data Preparation

```bash
python ./data/crop_3D_tiff_32_512_512_for_trainset.py \
--raw_inputdir /path/to/raw/input \
--raw_outputdir /path/to/raw/output \
--label_inputdir_train /path/to/label/train \
--label_inputdir_val /path/to/label/val \
--label_outputdir /path/to/label/output
```

### 3D Patched Feature Extraction

```bash
CUDA_VISIBLE_DEVICES=7 python image_feature_extraction.py \
--dataset_dir /path/to/dataset \
--output_dir /path/to/output \
--ckpt_dir /path/to/checkpoint
```

---

## Coreset Selection

```bash
CUDA_VISIBLE_DEVICES=7 python coreset_select.py \
--dataset_dir /path/to/dataset \
--dataset_feature_dir /path/to/dataset_feature \
--core_set_select_ratio 0.5 \
--output_dir /path/to/output
```

---

## nnUNet

Refer to the [nnUNet repository](https://github.com/MIC-DKFZ/nnUNet) for automated data preparation, training, and prediction.

---

## Post-processing from Semantic to Instance Segmentation

```bash
python postprocess_semanti_to_instance.py --semantic_pred /path/to/semantic_pred
```

---

## Generate Pseudo Labels based on CellSAM

**cellSAM Repository:** [cellSAM](https://github.com/vanvalenlab/cellSAM.git)

```bash
CUDA_VISIBLE_DEVICES=6 python get_pseudo_label_CellSAM.py
```

```bash
conda activate zs_BIBM
screen -S ZS_BVM_2025
CUDA_VISIBLE_DEVICES=7 python ./CellSAM/get_pseudo_label_CellSAM.py \
--input_2D_raw_folder /path/to/raw \
--output_folder /path/to/output
```

---

## Citation

If you use this code or data in your research, please cite our work as presented at **BVM 2025**.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

