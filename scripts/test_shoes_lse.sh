set -ex
python test.py \
--dataset_mode shoes \
--dataroot ./datasets/ShoeV2/ \
--name shoes2rgb_lse \
--model sc \
--epoch 60 \
--num_test 0
