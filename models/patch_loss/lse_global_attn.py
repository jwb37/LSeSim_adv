from .base_loss import BaseLoss


class GlobalAttnLoss(BaseLoss):
    """
    LSE Sim implementation with flexible attention layers applied globally to feature maps before patch extraction
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
        if not hasattr(self, 'attn_%d' % layer):
            self.create_attn(f_src, layer)
        attn = getattr(self, 'attn_%d' % layer)
        f_src, f_tgt = attn(f_src), attn(f_tgt)
        f_other = attn(f_other) if f_other is not None else None

        sim_src, patch_ids = self.patch_sim(f_src, patch_ids)
        sim_tgt, patch_ids = self.patch_sim(f_tgt, patch_ids)
        if f_other is not None:
            sim_other, _ = self.patch_sim(f_other, patch_ids)
        else:
            sim_other = None

        return sim_src, sim_tgt, sim_other
