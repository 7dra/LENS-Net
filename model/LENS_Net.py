import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone.sdtv3 import Spiking_vit_MetaFormerv2, MultiSpike_norm
from model.SpikeMAD import SpikeMAD
import  random


def set_seed(seed=42):
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(3407)  # 一行调用


class LENS_Net(nn.Module):
    def __init__(self, channels=[320, 128, 64, 32], num_classes=1, kernel_sizes=[1,3,5], expansion_factor=2, dw_parallel=True, add=True, lgag_ks=3, activation='relu', encoder='pvt_v2_b2', pretrain=True, pretrained_dir='./pretrained_pth/pvt/'):
        super(LENS_Net, self).__init__()

        self.backbone = Spiking_vit_MetaFormerv2(T=4)  # [64, 128, 320, 512]

        self.decoder = SpikeMAD(channels=channels, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, lgag_ks=lgag_ks, activation=activation)

        # prediction heads
        self.out_head4 = nn.Conv2d(channels[0], num_classes, 1)
        self.out_head3 = nn.Conv2d(channels[1], num_classes, 1)
        self.out_head2 = nn.Conv2d(channels[2], num_classes, 1)
        self.out_head1 = nn.Conv2d(channels[3], num_classes, 1)

        self.spike1 = MultiSpike_norm()
        self.spike2 = MultiSpike_norm()
        self.spike3 = MultiSpike_norm()
        self.spike4 = MultiSpike_norm()

    def forward(self, x):

        # backbone
        x1,x2,x3,x4 = self.backbone(x)


        dec_outs = self.decoder(x4, [x3, x2, x1])


        p4 = self.out_head4(self.spike1(dec_outs[0]))
        p3 = self.out_head3(self.spike2(dec_outs[1]))
        p2 = self.out_head2(self.spike3(dec_outs[2]))
        p1 = self.out_head1(self.spike4(dec_outs[3]))

        p4 = F.interpolate(p4, scale_factor=16, mode='bilinear')
        p3 = F.interpolate(p3, scale_factor=8, mode='bilinear')
        p2 = F.interpolate(p2, scale_factor=4, mode='bilinear')
        p1 = F.interpolate(p1, scale_factor=2, mode='bilinear')

        return p1,p2,p3,p4



def count_parameters_and_size(model):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 假设参数是 float32（4字节/参数）
    param_size = num_params * 4  # 总字节数
    size_mb = param_size / (1024 ** 2)  # 转换为 MB

    return num_params, size_mb
from thop import profile


def get_model_flops(model, input_size=(1, 3, 352, 352), device="cpu"):

    dummy_input = torch.randn(input_size).to(device)
    model = model.to(device)

    # 使用thop计算FLOPs
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    # print(f"************{flops}, {params}************")

    return flops


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.rand(1, 3, 352, 352).to(device)
    print("input", x.shape)
    model = LENS_Net().to(device)

    num_params, size_mb = count_parameters_and_size(model)
    print(f"参数数量: {num_params:,}")
    print(f"模型大小: {size_mb:.2f} MB")

    # 计算FLOPs
    flops, params = profile(model, inputs=(x,))
    print(f"FLOPs: {flops:,}")
    print(f"FLOPs: {flops/1e9:.2f} G")  # 以G为单位显示
    print(f"params:{params}")

    y = model.forward(x)


