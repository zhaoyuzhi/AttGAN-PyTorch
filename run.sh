python train.py \
--baseroot "../step 4 resized data/train" \
--finetune_path "" \
--save_mode "epoch" \
--save_by_epoch 50 \
--epochs 400 \
--multi_gpu False \
--lr_g 0.0001 \
--lr_decrease_mode "epoch" \
--lr_decrease_epoch 100 \
--lr_decrease_factor 0.5 \
--batch_size 16 \
--num_workers 8 \
--lambda_gan 1 \
--lambda_recon 100 \
--lambda_class 10 \
--pad 'zero' \
--norm_g 'bn' \
--norm_d 'ln' \
--activ_g 'lrelu' \
--activ_d 'lrelu' \
--in_channels 3 \
--out_channels 3 \
--start_channels 64 \
--attr_channels 4 \
--init_type "normal" \
--init_gain 0.02 \