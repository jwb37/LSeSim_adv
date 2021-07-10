set -ex
python test.py \
--dataroot datasets/shoes \
--name shoes2rgb_fse_vggsketchy \
--model sc \
--epoch 60 \
--num_test 0
