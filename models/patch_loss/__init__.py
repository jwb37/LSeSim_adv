from .lse_local_attn import LocalAttnLoss
from .lse_global_attn import GlobalAttnLoss
from .fse import FSeLoss


def SpatialCorrelativeLoss(opt, gpu_ids=[]):
    if not opt.learned_attn:
        return FSeLoss(opt, gpu_ids=gpu_ids)
    if opt.local_attn:
        return LocalAttnLoss(opt, gpu_ids=gpu_ids)

    return GlobalAttnLoss(opt, gpu_ids=gpu_ids)
