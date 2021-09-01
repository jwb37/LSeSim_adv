from .base_loss import BaseLoss


class FSeLoss(BaseLoss):
    """
    FSE Sim implementation
    """
    def __init__(self, *args, **kwargs):
        super().__init__( *args, **kwargs )

    def cal_sim(self, f_src, f_tgt, f_other=None, layer=0, patch_ids=None):
        """
        calculate the similarity map using the fixed query and key
        :param f_src: feature map from source domain
        :param f_tgt: feature map from target domain
        :param f_other: feature map from other image (only used for contrastive learning for spatial network)
        :return:
        """
        sim_src, patch_ids = self.patch_sim(f_src, patch_ids)
        sim_tgt, patch_ids = self.patch_sim(f_tgt, patch_ids)
        if f_other is not None:
            sim_other, _ = self.patch_sim(f_other, patch_ids)
        else:
            sim_other = None

        return sim_src, sim_tgt, sim_other
