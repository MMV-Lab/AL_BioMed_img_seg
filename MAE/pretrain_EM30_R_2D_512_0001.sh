CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch \
--nproc_per_node=8 \
--master_port=25678 main_pretrain.py \
--data_path $1 \
--batch_size 16 \
--model mae_vit_base_patch16 \
--norm_pix_loss \
--mask_ratio 0.75 \
--epochs 400 \
--warmup_epochs 40 \
--blr 1e-3 \
--weight_decay 0.05 \
--accum_iter 4 \
--input_size 224 \
--img_size 224 \
# --output_dir /mnt/eternus/users/Shuo/project/30_BIBM_conference/github_code/MAE/EM30_R_2D_MAE_224_0001/ > /mnt/eternus/users/Shuo/project/30_BIBM_conference/github_code/MAE/EM30_R_2D_MAE_224_0001/pretrain_log.txt 2>&1
# --output_dir ./EM30_R_2D_MAE/ > ./EM30_R_2D_MAE/pretrain_log.txt 2>&1
--output_dir $2 > ${2}/pretrain_log.txt 2>&1