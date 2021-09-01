Branch of code produced by: [Chuanxia Zheng](http://www.chuanxiaz.com), [Tat-Jen Cham](http://www.ntu.edu.sg/home/astjcham/), [Jianfei Cai](https://research.monash.edu/en/persons/jianfei-cai) <br>
For their work <br>
[The Spatially-Correlative Loss for Various Image Translation Tasks](https://arxiv.org/abs/2104.00854) <br>
[Chuanxia Zheng](http://www.chuanxiaz.com), [Tat-Jen Cham](http://www.ntu.edu.sg/home/astjcham/), [Jianfei Cai](https://research.monash.edu/en/persons/jianfei-cai) <br>
NTU and Monash University <br>
In CVPR2021 <br>

<ul>
<li>Adapted to allow STN attention maps to be included instead/in addition to original convolutional maps (using the option '--attn_layer_types').</li>
<li>Attention maps may also now be applied locally to individual patches rather than to entire feature maps (using the option '--local_attn').</li>
<li>A 'shoes' dataset is provided to work directly with the ShoesV2 dataset available at http://sketchx.eecs.qmul.ac.uk/downloads/</li>
<li>'--vggA_weights_file' option is available to specify a file containing pre-trained weights for the vgg model used to extract features from the 'A' image</li>
<li>'--visualize_ssim' option is available - adds creation of ssim visualization images during training</li>
</ul>

Adaptations made by [Jack Bakes](https://www.github.com/jwb37/)

# Instructions
Download ShoeV2 dataset from http://sketchx.eecs.qmul.ac.uk/downloads/ and extract it into the datasets folder.<br>
From the main directory:<br>
run './scripts/train_shoes_fse.sh' to train a standard FSE model<br>
run './scripts/train_shoes_lse.sh' to train a standard LSE model<br>
run './scripts/train_shoes_lse_attn_sc.sh' to train an LSE model with an STN->Conv attention module<br>
run './scripts/test_shoes_lse.sh' to test the LSE model

# Requirements
<ul>
<li>Python 3 (tested on 3.9)</li>
<li>torchvision</li>
<li>PIL</li>
</ul>
