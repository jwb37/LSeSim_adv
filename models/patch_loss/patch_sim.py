import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PatchSim(nn.Module):
    """Calculate the similarity in selected patches"""
    def __init__(self, patch_nums=256, patch_size=None, norm=True):
        super(PatchSim, self).__init__()
        self.patch_nums = patch_nums
        self.patch_size = patch_size
        self.use_norm = norm

    def forward(self, feat, patch_ids=None):
        """
        Calculate the similarity for selected patches
        """
        query, key, patch_ids = self.select_patch(feat, patch_ids=patch_ids)
        return self.create_sim(query, key), patch_ids


    def create_sim(self, query, key):
        B = query.size(0)
        patch_sim = query.bmm(key) if self.use_norm else torch.tanh(query.bmm(key)/10)
        patch_sim = patch_sim.view(B, self.patch_nums, -1)
        return patch_sim


    def select_patch(self, feat, patch_ids=None):
        """
        Select the patches
        """
        B, C, W, H = feat.size()

        feat = feat - feat.mean(dim=[-2, -1], keepdim=True)
        feat = F.normalize(feat, dim=1) if self.use_norm else feat / np.sqrt(C)

        pw, ph = self.patch_size, self.patch_size
        feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2) # B*N*C
        if self.patch_nums > 0:
            if patch_ids is None:
                patch_ids = torch.randperm(feat_reshape.size(1), device=feat.device)
                patch_ids = patch_ids[:int(min(self.patch_nums, patch_ids.size(0)))]
            feat_query = feat_reshape[:, patch_ids, :]       # B*Num*C
            feat_key = []
            Num = feat_query.size(1)
            if pw < W and ph < H:
                pos_x, pos_y = patch_ids // W, patch_ids % W
                # patch should in the feature
                left, top = pos_x - int(pw / 2), pos_y - int(ph / 2)
                left, top = torch.where(left > 0, left, torch.zeros_like(left)), torch.where(top > 0, top, torch.zeros_like(top))
                start_x = torch.where(left > (W - pw), (W - pw) * torch.ones_like(left), left)
                start_y = torch.where(top > (H - ph), (H - ph) * torch.ones_like(top), top)
                for i in range(Num):
                    feat_key.append(feat[:, :, start_x[i]:start_x[i]+pw, start_y[i]:start_y[i]+ph]) # B*C*patch_w*patch_h
                feat_key = torch.stack(feat_key, dim=0).permute(1, 0, 2, 3, 4) # B*Num*C*patch_w*patch_h
                feat_key = feat_key.reshape(B * Num, C, pw * ph)  # Num * C * N
                feat_query = feat_query.reshape(B * Num, 1, C)  # Num * 1 * C
            else: # if patch larger than features size, use B * C * N (H * W)
                feat_key = feat.reshape(B, C, W*H)
        else:
            feat_query = feat.reshape(B, C, H*W).permute(0, 2, 1) # B * N (H * W) * C
            feat_key = feat.reshape(B, C, H*W)  # B * C * N (H * W)

        return feat_query, feat_key, patch_ids 
