# from visualizer import get_local
import torch
import torch.nn as nn
import numpy as np
from timm.models.layers import trunc_normal_, DropPath
import torch.nn.functional as F
from collections import OrderedDict
from mmengine.model import BaseModule
from mmengine.logging import print_log
from mmengine.runner import CheckpointLoader


class Quant(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, i, min_value=0, max_value=4):
        ctx.min = min_value
        ctx.max = max_value
        ctx.save_for_backward(i)
        return torch.round(torch.clamp(i, min=min_value, max=max_value))

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        i, = ctx.saved_tensors

        # 使用Sigmoid平滑过渡
        alpha = 10  # 控制平滑程度的超参数
        grad_mask = torch.sigmoid(alpha * (i - ctx.min)) * torch.sigmoid(alpha * (ctx.max - i))
        grad_input = grad_input * grad_mask

        return grad_input, None, None
    # @staticmethod
    # @torch.cuda.amp.custom_fwd
    # def backward(ctx, grad_output):
    #     grad_input = grad_output.clone()
    #     i, = ctx.saved_tensors
    #     grad_input[i < ctx.min] = 0
    #     grad_input[i > ctx.max] = 0
    #     return grad_input, None, None

class MultiSpike_norm(nn.Module):
    def __init__(
            self,
            Norm=4,
    ):
        super().__init__()
        self.spike = Quant()
        self.Norm = Norm
        # if Vth_learnable == False:
        #     self.Vth = Vth
        # else:
        #     self.register_parameter("Vth", nn.Parameter(torch.tensor([1.0])))

    def __repr__(self):
        return f"MultiSpike_norm(Norm={self.Norm})"

    def forward(self, x):  # B C H W
        y = (self.spike.apply(x) / (self.Norm))
        # print(f"查看MultiSpike_norm的中间结果*************{y}*************")
        return y
        # return self.spike.apply(x)



class MultiSpike_norm_channel(nn.Module):
    def __init__(
            self,
            Vth=1.0,
            coefficient_shape=None,
            Norm_learnable=True,
    ):
        super().__init__()
        self.spike = Quant()
        self.Norm_learnable = Norm_learnable
        self.Vth = Vth
        self.Norm_type = 'Channel_Norm'

        self.register_parameter("Norm", nn.Parameter(torch.ones(coefficient_shape) * 4))

        # if Vth_learnable == False:
        #     self.Vth = Vth
        # else:
        #     self.register_parameter("Vth", nn.Parameter(torch.tensor([1.0])))

    def __repr__(self):
        return f"MultiSpike_norm_channel(Vth={self.Vth}, Norm_learnable={self.Norm_learnable}, Norm_type={self.Norm_type})"

    def forward(self, x):
        return self.spike.apply(x) / (self.Norm)


class BNAndPadLayer(nn.Module):
    def __init__(
            self,
            pad_pixels,
            num_features,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
    ):
        super(BNAndPadLayer, self).__init__()
        self.bn = nn.BatchNorm2d(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.pad_pixels = pad_pixels

    def forward(self, input):
        output = self.bn(input)
        if self.pad_pixels > 0:
            if self.bn.affine:
                pad_values = (
                        self.bn.bias.detach()
                        - self.bn.running_mean
                        * self.bn.weight.detach()
                        / torch.sqrt(self.bn.running_var + self.bn.eps)
                )
            else:
                pad_values = -self.bn.running_mean / torch.sqrt(
                    self.bn.running_var + self.bn.eps
                )
            output = F.pad(output, [self.pad_pixels] * 4)
            pad_values = pad_values.view(1, -1, 1, 1)
            output[:, :, 0: self.pad_pixels, :] = pad_values
            output[:, :, -self.pad_pixels:, :] = pad_values
            output[:, :, :, 0: self.pad_pixels] = pad_values
            output[:, :, :, -self.pad_pixels:] = pad_values
        return output

    @property
    def weight(self):
        return self.bn.weight

    @property
    def bias(self):
        return self.bn.bias

    @property
    def running_mean(self):
        return self.bn.running_mean

    @property
    def running_var(self):
        return self.bn.running_var

    @property
    def eps(self):
        return self.bn.eps

class SepConv_Spike(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """

    def __init__(
            self,
            dim,
            expansion_ratio=2,
            act2_layer=nn.Identity,
            bias=False,
            kernel_size=7,
            padding=3,
    ):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.spike1 = MultiSpike_norm()
        self.pwconv1 = nn.Sequential(
            nn.Conv2d(dim, med_channels, kernel_size=1, stride=1, bias=bias),
            nn.BatchNorm2d(med_channels)
        )


        self.spike2 = MultiSpike_norm()
        self.dwconv = nn.Sequential(
            nn.Conv2d(med_channels, med_channels, kernel_size=kernel_size, padding=padding, groups=med_channels,
                      bias=bias),
            nn.BatchNorm2d(med_channels)
        )

        self.spike3 = MultiSpike_norm()
        self.pwconv2 = nn.Sequential(
            nn.Conv2d(med_channels, dim, kernel_size=1, stride=1, bias=bias),
            nn.BatchNorm2d(dim)
        )
        # self.k = kernel_size

    def forward(self, x):
        x = self.spike1(x)



        x = self.pwconv1(x)

        x = self.spike2(x)


        x = self.dwconv(x)

        x = self.spike3(x)

        x = self.pwconv2(x)
        return x



class MS_ConvBlock_spike_SepConv(nn.Module):
    def __init__(
            self,
            dim,
            mlp_ratio=4.0,
    ):
        super().__init__()

        self.Conv = SepConv_Spike(dim=dim)

        self.mlp_ratio = mlp_ratio

        self.spike1 = MultiSpike_norm()
        self.conv1 = nn.Conv2d(
            dim, dim * mlp_ratio, kernel_size=3, padding=1, groups=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(dim * mlp_ratio)  # 这里可以进行改进
        self.spike2 = MultiSpike_norm()
        self.conv2 = nn.Conv2d(
            dim * mlp_ratio, dim, kernel_size=3, padding=1, groups=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(dim)  # 这里可以进行改进

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.Conv(x) + x

        x_feat = x
        x = self.spike1(x)



        x = self.bn1(self.conv1(x)).reshape(B, self.mlp_ratio * C, H, W)
        x = self.spike2(x)

        x = self.bn2(self.conv2(x)).reshape(B, C, H, W)
        x = x_feat + x

        return x


class MS_MLP(nn.Module):
    def __init__(
            self, in_features, hidden_features=None, out_features=None, drop=0.0, layer=0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.a = in_features
        self.b = hidden_features
        self.c = out_features

        self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_spike = MultiSpike_norm()

        self.fc2_conv = nn.Conv1d(
            hidden_features, out_features, kernel_size=1, stride=1
        )
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_spike = MultiSpike_norm()

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        x = x.flatten(2)
        # print("a b c",self.a, self.b, self.c)
        x = self.fc1_spike(x)



        x = self.fc1_conv(x)
        # print("fc1",self.fc1_conv.shape)
        x = self.fc1_bn(x).reshape(B, self.c_hidden, N).contiguous()
        x = self.fc2_spike(x)



        x = self.fc2_conv(x)
        x = self.fc2_bn(x).reshape(B, C, H, W).contiguous()

        return x


class MS_Attention_linear(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            sr_ratio=1,
            lamda_ratio=1,
    ):
        super().__init__()
        assert (
                dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.lamda_ratio = lamda_ratio

        self.head_spike = MultiSpike_norm()

        self.q_conv = nn.Sequential(nn.Conv2d(dim, dim, 1, 1, bias=False), nn.BatchNorm2d(dim))

        self.q_spike = MultiSpike_norm()

        self.k_conv = nn.Sequential(nn.Conv2d(dim, dim, 1, 1, bias=False), nn.BatchNorm2d(dim))

        self.k_spike = MultiSpike_norm()

        self.v_conv = nn.Sequential(nn.Conv2d(dim, int(dim * lamda_ratio), 1, 1, bias=False),
                                    nn.BatchNorm2d(int(dim * lamda_ratio)))

        self.v_spike = MultiSpike_norm()

        self.attn_spike = MultiSpike_norm()

        # self.proj_conv = nn.Sequential(
        #     RepConv(dim*lamda_ratio, dim, bias=False), nn.BatchNorm2d(dim)
        # )

        self.proj_conv = nn.Sequential(
            nn.Conv2d(dim * lamda_ratio, dim, 1, 1, bias=False), nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        C_v = int(C * self.lamda_ratio)

        x = self.head_spike(x)
        # print(x.shape)
        ## 打印结果 ##

        q = self.q_conv(x)
        # print(q.shape,"**************************************q***************")
        k = self.k_conv(x)
        v = self.v_conv(x)
        # print("5555",v.shape)


        q = self.q_spike(q)
        ## 打印结果 ##

        q = q.flatten(2)
        q = (
            q.transpose(-1, -2)
            .reshape(B, N, self.num_heads, C // self.num_heads)  # D = C
            .permute(0, 2, 1, 3)
            .contiguous()
        )

        k = self.k_spike(k)
        ## 打印结果 ##

        k = k.flatten(2)
        k = (
            k.transpose(-1, -2)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

        v = self.v_spike(v)
        ## 打印结果 ##


        # print(f"***************{q}****************")
        # print(f"***************{k}****************")
        # print(f"***************{v}****************")

        v = v.flatten(2)
        v = (
            v.transpose(-1, -2)
            .reshape(B, N, self.num_heads, C_v // self.num_heads)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

        x = q @ k.transpose(-2, -1)
        # print("q",q)
        # print("k",k)
        # print("v",v)
        # print(x)
        x = (x @ v) * (self.scale * 2)

        x = x.transpose(2, 3).reshape(B, C_v, N).contiguous()
        x = self.attn_spike(x)


        x = x.reshape(B, C_v, H, W)
        x = self.proj_conv(x).reshape(B, C, H, W)
        # print("proj_conv",x.shape)

        return x


class MS_Block_Spike_SepConv(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.0,
            qkv_bias=False,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            norm_layer=nn.LayerNorm,
            sr_ratio=1,
            init_values=1e-6
    ):
        super().__init__()

        self.conv = SepConv_Spike(dim=dim, kernel_size=3, padding=1)

        self.attn = MS_Attention_linear(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
            lamda_ratio=4,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MS_MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

        self.layer_scale1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        self.layer_scale2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        self.layer_scale3 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.conv(x) * self.layer_scale1.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        x = x + self.attn(x) * self.layer_scale2.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        x = x + self.mlp(x) * self.layer_scale3.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        return x


class MS_DownSampling(nn.Module):
    def __init__(
            self,
            in_channels=2,
            embed_dims=256,
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=True,
            T=None,
    ):
        super().__init__()

        self.encode_conv = nn.Conv2d(
            in_channels,
            embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.encode_bn = nn.BatchNorm2d(embed_dims)
        self.first_layer = first_layer
        if not first_layer:
            self.encode_spike = MultiSpike_norm()

        self.k = kernel_size


    def forward(self, x):
        # print(self.k)


        if hasattr(self, "encode_spike"):
            x = self.encode_spike(x)



        x = self.encode_conv(x)
        x = self.encode_bn(x)

        return x

class Spiking_vit_MetaFormerv2(BaseModule):
    def __init__(
            self,
            in_channels=3,
            num_classes=0,
            embed_dim=[64, 128, 256, 320],
            num_heads=8,
            mlp_ratios=4,
            qkv_bias=False,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            norm_layer=nn.LayerNorm,
            depths=8,
            sr_ratios=1,
            T=4,
            decode_mode='QTrick',
            init_cfg=None,
            norm_cfg=dict(type='BN', requires_grad=True),
            pretrained=None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.depths = depths
        # embed_dim = [64, 128, 256, 512]
        self.T = T
        self.decode_mode = decode_mode

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depths)
        ]  # stochastic depth decay rule

        self.downsample1_1 = MS_DownSampling(
            in_channels=in_channels,
            embed_dims=embed_dim[0] // 2,
            kernel_size=7,
            stride=2,
            padding=3,
            first_layer=True,

        )

        self.ConvBlock1_1 = nn.ModuleList(
            [MS_ConvBlock_spike_SepConv(dim=embed_dim[0] // 2, mlp_ratio=mlp_ratios)]
        )

        self.downsample1_2 = MS_DownSampling(
            in_channels=embed_dim[0] // 2,
            embed_dims=embed_dim[0],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
        )

        self.ConvBlock1_2 = nn.ModuleList(
            [MS_ConvBlock_spike_SepConv(dim=embed_dim[0], mlp_ratio=mlp_ratios)]
        )

        self.downsample2 = MS_DownSampling(
            in_channels=embed_dim[0],
            embed_dims=embed_dim[1],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
        )

        self.ConvBlock2_1 = nn.ModuleList(
            [MS_ConvBlock_spike_SepConv(dim=embed_dim[1], mlp_ratio=mlp_ratios)]
        )

        self.ConvBlock2_2 = nn.ModuleList(
            [MS_ConvBlock_spike_SepConv(dim=embed_dim[1], mlp_ratio=mlp_ratios)]
        )

        self.downsample3 = MS_DownSampling(
            in_channels=embed_dim[1],
            embed_dims=embed_dim[2],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
        )

        self.block3 = nn.ModuleList(
            [
                MS_Block_Spike_SepConv(
                    dim=embed_dim[2],
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratios,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios,

                )
                for j in range(6)
            ]
        )

        self.downsample4 = MS_DownSampling(
            in_channels=embed_dim[2],
            embed_dims=embed_dim[3],
            kernel_size=3,
            stride=1,
            padding=1,
            first_layer=False,

        )

        self.block4 = nn.ModuleList(
            [
                MS_Block_Spike_SepConv(
                    dim=embed_dim[3],
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratios,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios,

                )
                for j in range(2)
            ]
        )

        # self.head = (
        #     nn.Linear(embed_dim[3], num_classes) if num_classes > 0 else nn.Identity()
        # )
        # self.spike = MultiSpike_norm()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self):
        # logger = MMlogger.get_current_instance()
        if self.init_cfg is None:
            print_log(f'No pre-trained weights for '
                      f'{self.__class__.__name__}, '
                      f'training start from scratch')
            # self.apply(self._init_weights)

            print_log("init_weighting.....")
            self.apply(self._init_weights)
            print_log("Time step: {:}".format(self.T))
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            ckpt = CheckpointLoader.load_checkpoint(
                self.init_cfg['checkpoint'], logger=None, map_location='cpu')
            # import pdb; pdb.set_trace()
            # state_dict = self.state_dict()
            # import pdb; pdb.set_trace()
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt
            # import pdb; pdb.set_trace()
            state_dict = OrderedDict()
            for k, v in _state_dict.items():
                # 使用mmseg保存的checkpoint中包含backbone, neck, decode_head三个部分
                if k.startswith('backbone.'):
                    state_dict[k[9:]] = v
                else:
                    state_dict[k] = v
            # import pdb; pdb.set_trace()
            self.load_state_dict(state_dict, strict=False)
            print_log("--------------Successfully load checkpoint for BACKNONE------------")
            print_log("Time step: {:}".format(self.T))

    def forward_features(self, x):
        x = self.downsample1_1(x)
        # print(f"downsample1_1:{x.shape}")
        for blk in self.ConvBlock1_1:
            x = blk(x)
        x1 = x



        x = self.downsample1_2(x)
        # print(f"downsample1_2:{x.shape}")

        for blk in self.ConvBlock1_2:
            x = blk(x)

        x2 = x


        x = self.downsample2(x)
        # print(f"downsample2:{x.shape}")

        for blk in self.ConvBlock2_1:
            x = blk(x)
        for blk in self.ConvBlock2_2:
            x = blk(x)
        x3 = x

        x = self.downsample3(x)
        # print(f"downsample3:{x.shape}")
        for blk in self.block3:
            x = blk(x)

        x = self.downsample4(x)
        # print(f"downsample4:{x.shape}")
        for blk in self.block4:
            x = blk(x)
        x4 = x



        if self.decode_mode == 'QTrick':

            return [x1, x2, x3, x4]
        if self.decode_mode=="frequency":

            return [x4, x3, x2, x1]
        return x  # T,B,C,N

    def forward(self, x):
        x = self.forward_features(x)  # B,C,H,W
        return x


def count_parameters_and_size(model):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 假设参数是 float32（4字节/参数）
    param_size = num_params * 4  # 总字节数
    size_mb = param_size / (1024 ** 2)  # 转换为 MB

    return num_params, size_mb


if __name__=="__main__":
    x = torch.rand(1,3,352,352)
    # print("输入size",x.shape)
    model = Spiking_vit_MetaFormerv2(T=4)
    model.train()
    model.eval()
    y = model(x)
    print(y[0].shape, y[1].shape, y[2].shape, y[3].shape)
    # print(y)
    num_params, size_mb = count_parameters_and_size(model)
    print(f"参数数量: {num_params:,}")
    print(f"模型大小: {size_mb:.2f} MB")
    from thop import profile
    flops, params = profile(model, inputs=(x,))
    print(f"FLOPs: {flops:,}")
    print(f"FLOPs: {flops/1e9:.2f} G")  # 以G为单位显示
