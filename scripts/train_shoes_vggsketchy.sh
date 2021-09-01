set -ex
python train.py  \
--dataset_mode shoes \
--dataroot ./datasets/ShoeV2/ \
--name shoes2rgb_fse_vggsketchy \
--save_epoch_freq 2 \
--model sc \
--gpu_ids 0 \
--lambda_spatial 10 \
--lambda_gradient 0 \
--attn_layers 4,7,9 \
--loss_mode cos \
--gan_mode lsgan \
--display_port 8093 \
--vggA_weights_file vgg-pretrained/vgg-sketchy-model.pt \
--patch_size 64 \
--visualize_ssim \
--continue_train \
--epoch_count 61
