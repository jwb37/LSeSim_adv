import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

from ..init_net import init_net

from .STN import STN
from .patch_sim import PatchSim
from .conv_attn import ConvAttentionLayer

import math
import random

class BaseLoss(nn.Module):
    """
    learnable patch-based spatially-correlative loss with contrastive learning
    """
    def __init__(self, opt, gpu_ids=[]):
        super().__init__()
        self.patch_size = opt.patch_size
        self.patch_nums = opt.patch_nums
        self.norm = opt.use_norm
        self.patch_sim = PatchSim(patch_nums=self.patch_nums, patch_size=self.patch_size, norm=self.norm)
        self.use_attn = opt.learned_attn
        self.attn_layer_types = [t for t in opt.attn_layer_types.split(',')]
        self.init_type = 'normal'
        self.init_gain = 0.02
        self.gpu_ids = gpu_ids
        self.loss_mode = opt.loss_mode
        self.T = opt.T
        self.criterion = nn.L1Loss() if self.norm else nn.SmoothL1Loss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.conv_layers = dict()
        self.visuals = {'src': {}, 'tgt': {}}

    def create_attn(self, feat, layer):
        """
        create the attention mapping layer used to transform features before building similarity maps
        :param feat: extracted features from a pretrained VGG or encoder for the similarity and dissimilarity map
        :param layer: different layers use different filter
        :return:
        """
        attn_layers = []
        for layer_code in self.attn_layer_types:
            if layer_code == 'c':
                attn_layers.append( ConvAttentionLayer() )
            elif layer_code == 's':
                attn_layers.append( STN() )

        if not attn_layers:
            raise ValueError("Command line option attn_type must be a comma separated list of letters c or s" )

        for l in attn_layers:
            l.build_net(feat)
            feat = l(feat)
            l.init_params(self.init_type, self.init_gain, self.gpu_ids)

        # Extract the convolutional layers
        self.conv_layers[layer] = [l for l, char in zip(attn_layers, self.attn_layer_types) if char=='c']
        if self.conv_layers[layer]:
            self.conv_layers[layer] = nn.Sequential(*self.conv_layers[layer])

        attn_net = nn.Sequential(*attn_layers)
        setattr(self, 'attn_%d' % layer, attn_net)


#          norm_real_A = torch.cat([norm_real_A, norm_real_A], dim=0)
#          norm_fake_B = torch.cat([norm_fake_B, norm_aug_A], dim=0)
#          norm_real_B = torch.cat([norm_real_B, norm_aug_B], dim=0)

    def compare_sim(self, sim_src, sim_tgt, sim_other):
        """
        measure the shape distance between the same shape and different inputs
        :param sim_src: the shape similarity map from source input image
        :param sim_tgt: the shape similarity map from target output image
        :param sim_other: the shape similarity map from other input image
        :return:
        """
        B, Num, N = sim_src.size()
        if self.loss_mode == 'info' or sim_other is not None:
            sim_src = F.normalize(sim_src, dim=-1)
            sim_tgt = F.normalize(sim_tgt, dim=-1)
            sim_other = F.normalize(sim_other, dim=-1)
            sam_neg1 = (sim_src.bmm(sim_other.permute(0, 2, 1))).view(-1, Num) / self.T
            sam_neg2 = (sim_tgt.bmm(sim_other.permute(0, 2, 1))).view(-1, Num) / self.T
            sam_self = (sim_src.bmm(sim_tgt.permute(0, 2, 1))).view(-1, Num) / self.T
            sam_self = torch.cat([sam_self, sam_neg1, sam_neg2], dim=-1)
            loss = self.cross_entropy_loss(sam_self, torch.arange(0, sam_self.size(0), dtype=torch.long, device=sim_src.device) % (Num))
        else:
            tgt_sorted, _ = sim_tgt.sort(dim=-1, descending=True)
            num = int(N / 4)
            src = torch.where(sim_tgt < tgt_sorted[:, :, num:num + 1], 0 * sim_src, sim_src)
            tgt = torch.where(sim_tgt < tgt_sorted[:, :, num:num + 1], 0 * sim_tgt, sim_tgt)
            if self.loss_mode == 'l1':
                loss = self.criterion((N / num) * src, (N / num) * tgt)
            elif self.loss_mode == 'cos':
                sim_pos = F.cosine_similarity(src, tgt, dim=-1)
                loss = self.criterion(torch.ones_like(sim_pos), sim_pos)
            else:
                raise NotImplementedError('padding [%s] is not implemented' % self.loss_mode)

        return loss

    def loss(self, f_src, f_tgt, f_other=None, layer=0, visualize=False):
        """
        calculate the spatial similarity and dissimilarity loss for given features from source and target domain
        :param f_src: source domain features
        :param f_tgt: target domain features
        :param f_other: other random sampled features
        :param layer:
        :return:
        """
        sim_src, sim_tgt, sim_other = self.cal_sim(f_src, f_tgt, f_other, layer)
        # calculate the spatial similarity for source and target domain
        loss = self.compare_sim(sim_src, sim_tgt, sim_other)
        if visualize:
            batch_idx = random.randrange(0,sim_src.size(0))
            patch_idx = random.randrange(0,sim_src.size(1))
            for name, sim in (('src', sim_src), ('tgt', sim_tgt)):
                vis = sim[batch_idx,patch_idx,:]

                mean, std = vis.mean(), vis.std()
                vis = (vis - mean) / std

                side = int(round(math.sqrt(vis.size(0))))
                vis = vis.view((1, 1, side, -1))
                vis = vis.expand(-1, 3, -1, -1)
                vis = F.interpolate(vis, size=(256,256), mode='bilinear')
                self.visuals[name][layer] = vis.clone()
        return loss


    # Virtual function
    def cal_sim(self, f_src, f_tgt, f_other=None, layer=0, patch_ids=None):
        pass

