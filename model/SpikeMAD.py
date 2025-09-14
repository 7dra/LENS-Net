import torch
import torch.nn as nn
from functools import partial
import math
from timm.models.layers import trunc_normal_tf_
from timm.models.helpers import named_apply

from backbone.sdtv3 import MultiSpike_norm


def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


# Other types of layers can go here (e.g., nn.Linear, etc.)
def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            # efficientnet like
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x


#   Spike-driven Multi-scale Depth-wise convolution (SMDC)
class SMDC(nn.Module):
    def __init__(self, in_channels, kernel_sizes, stride, activation='relu6', dw_parallel=True):
        super(SMDC, self).__init__()

        self.in_channels = in_channels
        self.kernel_sizes = kernel_sizes
        self.activation = activation
        self.dw_parallel = dw_parallel
        # dilation = 2
        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                MultiSpike_norm(),  # 开始忘记加了
                nn.Conv2d(self.in_channels, self.in_channels, kernel_size, stride, (kernel_size) // 2,
                          groups=self.in_channels, dilation=1, bias=False),

                nn.BatchNorm2d(self.in_channels),
                # act_layer(self.activation, inplace=True)
            )
            for kernel_size in self.kernel_sizes
        ])

        self.init_weights('kaiming_normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        # Apply the convolution layers in a loop
        outputs = []
        # print("shape",self.in_channels)
        for dwconv in self.dwconvs:


            dw_out = dwconv(x)
            outputs.append(dw_out)
            if self.dw_parallel == False:
                x = x + dw_out
        # You can return outputs based on what you intend to do with them
        return outputs


class SMCB(nn.Module):
    """
    Spike-driven Multi-scale Convolution Block(SMCB)
    """

    def __init__(self, in_channels, out_channels, stride, kernel_sizes=[1, 3, 5], expansion_factor=2, dw_parallel=True,
                 add=True, activation='relu6'):
        super(SMCB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_sizes = kernel_sizes
        self.expansion_factor = expansion_factor
        self.dw_parallel = dw_parallel
        self.add = add
        self.activation = activation
        self.n_scales = len(self.kernel_sizes)
        # check stride value
        assert self.stride in [1, 2]
        # Skip connection if stride is 1
        self.use_skip_connection = True if self.stride == 1 else False

        # expansion factor
        self.ex_channels = int(self.in_channels * self.expansion_factor)
        # ex_channels = [320,128,64,32] * 2
        self.pconv1 = nn.Sequential(
            # pointwise convolution
            MultiSpike_norm(),
            nn.Conv2d(self.in_channels, self.ex_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ex_channels)
        )
        self.smdc = SMDC(self.ex_channels, self.kernel_sizes, self.stride, self.activation,
                         dw_parallel=self.dw_parallel)
        if self.add == True:
            self.combined_channels = self.ex_channels * 1
        else:
            self.combined_channels = self.ex_channels * self.n_scales
        self.pconv2 = nn.Sequential(
            # pointwise convolution
            MultiSpike_norm(),
            nn.Conv2d(self.combined_channels, self.out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.out_channels),
        )
        if self.use_skip_connection and (self.in_channels != self.out_channels):
            self.conv1x1 = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0, bias=False)
        self.init_weights('kaiming_normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):


        pout1 = self.pconv1(x)
        msdc_outs = self.smdc(pout1)
        if self.add == True:
            dout = 0
            for dwout in msdc_outs:
                dout = dout + dwout
        else:
            dout = torch.cat(msdc_outs, dim=1)
        dout = channel_shuffle(dout, gcd(self.combined_channels, self.out_channels))

        out = self.pconv2(dout)
        # print("$$$",out.shape )


        if self.use_skip_connection:
            if self.in_channels != self.out_channels:
                x = self.conv1x1(x)
            return x + out
        else:
            return out


# Spike-driven Multi-scale convolution block
def SMCBLayer(in_channels, out_channels, n=1, stride=1, kernel_sizes=[1, 3, 5], expansion_factor=2, dw_parallel=True,
              add=True, activation='relu6'):
    """
    create a series of multi-scale convolution blocks.
    """
    convs = []
    smcb = SMCB(in_channels, out_channels, stride, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor,
                dw_parallel=dw_parallel, add=add, activation=activation)
    convs.append(smcb)
    if n > 1:
        for i in range(1, n):
            smcb = SMCB(out_channels, out_channels, 1, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor,
                        dw_parallel=dw_parallel, add=add, activation=activation)
            convs.append(smcb)
    conv = nn.Sequential(*convs)
    return conv


# Spike-driven up-convolution block (SUCB)
class SUCB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation='relu'):
        super(SUCB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        # dilation = 2
        self.up_dwc = nn.Sequential(
            nn.Upsample(scale_factor=2),
            MultiSpike_norm(),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=kernel_size, dilation=1, stride=stride,
                      padding=kernel_size // 2, groups=self.in_channels, bias=False),

            nn.BatchNorm2d(self.in_channels),
            MultiSpike_norm()

        )
        self.pwc = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(self.out_channels)
        )
        self.init_weights('kaiming_normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        # 确保按顺序执行：up_dwc[0] → up_dwc[1] → up_dwc[2] → up_dwc[3] → up_dwc[4]
        x_processed = x
        for i, layer in enumerate(self.up_dwc):
            x_processed = layer(x_processed)  # 顺序执行每一层


        x = self.up_dwc(x)


        x = channel_shuffle(x, self.in_channels)
        x = self.pwc(x)
        return x

#   Spike-driven Channel attention block (SCAB)
class SCAB(nn.Module):
    def __init__(self, in_channels, out_channels=None, ratio=16, activation='relu'):
        super(SCAB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        if self.in_channels < ratio:
            ratio = self.in_channels
        self.reduced_channels = self.in_channels // ratio
        if self.out_channels == None:
            self.out_channels = in_channels

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # self.activation = act_layer(activation, inplace=True)
        self.activation = MultiSpike_norm()
        self.activation2 = MultiSpike_norm()
        # norm_layer = lambda channels: nn.GroupNorm(1, channels)  # Equivalent to LayerNorm for 2D

        self.fc1 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.reduced_channels, 1, bias=False),
            # norm_layer(self.reduced_channels)
        )

        self.fc2 = nn.Sequential(
            nn.Conv2d(self.reduced_channels, self.out_channels, 1, bias=False),
            # norm_layer(self.out_channels)
        )

        self.fc3 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.reduced_channels, 1, bias=False),
            # norm_layer(self.reduced_channels)
        )

        self.fc4 = nn.Sequential(
            nn.Conv2d(self.reduced_channels, self.out_channels, 1, bias=False),
            # norm_layer(self.out_channels)
        )

        self.sigmoid = nn.Sigmoid()
        # self.sfa = MultiSpike_norm()

        self.init_weights('kaiming_normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        # print("self.reduced_channels",self.reduced_channels)

        avg_pool_out = self.avg_pool(x)

        temp = self.fc1(self.activation(avg_pool_out))


        temp = self.activation2(temp)  # 此处需要修改
        avg_out = self.fc2(temp)


        max_pool_out = self.max_pool(x)
        temp = self.fc3(self.activation(max_pool_out))


        temp = self.activation2(temp)

        max_out = self.fc4(temp)


        # out = self.activation2(avg_out) + self.activation2(max_out)
        out = avg_out + max_out

        return self.sigmoid(out)


class SpikeMAD(nn.Module):
    def __init__(self, channels=[320, 128, 64, 32], kernel_sizes=[1, 3, 5], expansion_factor=6, dw_parallel=True,
                 add=True, lgag_ks=3, activation='relu6'):
        super(SpikeMAD, self).__init__()
        sucb_ks = 3  # kernel size for sucb
        self.smcb4 = SMCBLayer(channels[0], channels[0], n=1, stride=1, kernel_sizes=kernel_sizes,
                               expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add,
                               activation=activation)

        self.sucb3 = SUCB(in_channels=channels[0], out_channels=channels[1], kernel_size=sucb_ks, stride=sucb_ks // 2)
        self.smcb3 = SMCBLayer(channels[1], channels[1], n=1, stride=1, kernel_sizes=kernel_sizes,
                               expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add,
                               activation=activation)

        self.sucb2 = SUCB(in_channels=channels[1], out_channels=channels[2], kernel_size=sucb_ks, stride=sucb_ks // 2)
        self.smcb2 = SMCBLayer(channels[2], channels[2], n=1, stride=1, kernel_sizes=kernel_sizes,
                               expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add,
                               activation=activation)

        self.sucb1 = SUCB(in_channels=channels[2], out_channels=channels[3], kernel_size=sucb_ks, stride=sucb_ks // 2)
        self.smcb1 = SMCBLayer(channels[3], channels[3], n=1, stride=1, kernel_sizes=kernel_sizes,
                               expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add,
                               activation=activation)

        self.scab4 = SCAB(channels[0])
        self.scab3 = SCAB(channels[1])
        self.scab2 = SCAB(channels[2])
        self.scab1 = SCAB(channels[3])

        self.sfa = MultiSpike_norm()

    def forward(self, x, skips):
        # MSCAM4
        d4 = self.scab4(x) * self.sfa(x) + x
        d4 = self.smcb4(d4)
        d3 = self.sucb3(d4)

        x3 = skips[0]

        # Additive aggregation 3
        d3 = d3 + x3

        # MSCAM3
        d3 = self.scab3(d3) * self.sfa(d3) + d3
        d3 = self.smcb3(d3)

        d2 = self.sucb2(d3)

        x2 = skips[1]

        # Additive aggregation 2
        d2 = d2 + x2

        # MSCAM2
        d2 = self.scab2(d2) * self.sfa(d2) + d2
        d2 = self.smcb2(d2)

        d1 = self.sucb1(d2)

        x1 = skips[2]

        # Additive aggregation 1
        d1 = d1 + x1

        # # MSCAM1
        d1 = self.scab1(d1) * self.sfa(d1) + d1
        d1 = self.smcb1(d1)

        return [d4, d3, d2, d1]

