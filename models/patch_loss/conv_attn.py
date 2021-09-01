import torch.nn as nn

from ..init_net import init_net


class ConvAttentionLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def build_net(self, feat):
        input_nc = feat.size(1)
        output_nc = max(32, input_nc // 4)
        self.net = nn.Sequential(
            nn.Conv2d(input_nc, output_nc, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(output_nc, output_nc, kernel_size=1)
        )
        self.net.to(feat.device)

    def init_params(self, init_type, init_gain, gpu_ids):
        init_net(self, init_type, init_gain, gpu_ids)

    def forward(self, x):
        return self.net(x) 
