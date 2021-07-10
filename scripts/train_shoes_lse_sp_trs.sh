set -ex
python train.py  \
--dataroot ./datasets/shoes/ \
--name shoes2rgb_lse_attn_both \
--save_epoch_freq 2 \
--model sc \
--gpu_ids 0 \
--lambda_spatial 10 \
--lambda_gradient 0 \
--attn_layers 4,7,9 \
--loss_mode cos \
--gan_mode lsgan \
--display_port 8093 \
--learned_attn \
--attn_type both \
--augment \
--patch_size 64
