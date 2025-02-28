# AL_BioMed_img_seg

## An Active Learning Pipeline Based on Data-Centric AI for Biomedical Image Instance Segmentation

This repository presents the implementation of our paper **"AL_BioMed_img_seg: An Active Learning Pipeline Based on Data-Centric AI for Biomedical Image Instance Segmentation"**, which has been accepted as a poster at the **BVM 2025 Conference**.  
Conference link: [BVM 2025](https://www.bvm-conf.org/)

### Overview
This project aims to develop an active learning pipeline for biomedical image instance segmentation using a data-centric approach. It leverages **MAE (Masked Autoencoders)** for feature extraction and **nnUNet** for segmentation, enhanced with active learning and pseudo-label generation strategies.

### Pipeline
![Pipeline](https://github.com/MMV-Lab/AL_BioMed_img_seg/fig/pipline.png)

---

## Clone Repository

```bash
git clone https://github.com/MMV-Lab/AL_BioMed_img_seg
cd AL_BioMed_img_seg


## Create Conda Environment
```bash
conda create -n AL_BioMed_img_seg python=3.10 -y
conda activate AL_BioMed_img_seg

### Install Dependencies
#### Install PyTorch with CUDA Support
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

#### Test if GPU is Available
```bash
python -c "import torch; print('CUDA is available:', torch.cuda.is_available())"


