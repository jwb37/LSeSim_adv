import torch

from .base_loss import BaseLoss


class LocalAttnLoss(BaseLoss):
    """
    LSE Sim implementation with attention layers applied locally to individual patches
    """
    def __init__(self, *args, **kwargs):
        super().__init__( *args, **kwargs )


    def cal_sim(self, f_src, f_tgt, f_other=None, layer=0, patch_ids=None):
        """
        calculate the similarity map using the fixed/learned query and key
        :param f_src: feature map from source domain
        :param f_tgt: feature map from target domain
        :param f_other: feature map from other image (only used for contrastive learning for spatial network)
        :return:
        """
        src_qry, src_key, patch_ids = self.patch_sim.select_patch(f_src, patch_ids)
        tgt_qry, tgt_key, patch_ids = self.patch_sim.select_patch(f_tgt, patch_ids)
        if f_other is not None:
            oth_qry, oth_key, _ = self.patch_sim.select_patch(f_other, patch_ids)
        else:
            oth_qry = oth_key = None

        if not hasattr(self, 'attn_%d' % layer):
            self.create_attn(src_key, layer)
        attn = getattr(self, 'attn_%d' % layer)

        pw = min(f_src.size(2), self.patch_size)
        ph = min(f_src.size(3), self.patch_size)

        qrys = [src_qry, tgt_qry, oth_qry]
        keys = [src_key, tgt_key, oth_key]
        sims = [None, None, None]

        for idx, (key, qry) in enumerate(zip(keys,qrys)):
            if key is None:
                continue

            Num, C, N = key.size()
            key = key.view(Num, C, pw, ph)
            key = attn(key).view(Num, C, pw*ph)

            # Query vector is 1x1, so cannot use STN on it.
            # However, it does need the channel attention transform
            if self.conv_layers[layer]:
                qry = qry.permute(0,2,1).unsqueeze(-1)
                qry = self.conv_layers[layer](qry)
                qry = qry.squeeze(-1).permute(0,2,1)

            sims[idx] = self.patch_sim.create_sim(qry, key)

        return sims
