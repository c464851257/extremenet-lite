import torch
import torch.nn as nn
import torch.nn.functional as F

use_dsc = True # use depthwise separable convolution

class convolution(nn.Module):
    if use_dsc:
        def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
            super(convolution, self).__init__()

            pad = (k - 1) // 2
            self.conv = conv_bn_k(inp_dim, 32, k, stride, pad)
            self.conv_1x1_bn = conv_1x1_bn(32, out_dim, stride)


        def forward(self, x):
            conv = self.conv(x)
            conv_1x1_bn = self.conv_1x1_bn(conv)
            return conv_1x1_bn
    else:
        def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
            super(convolution, self).__init__()

            pad = (k - 1) // 2
            self.conv = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(pad, pad), stride=(stride, stride),
                                  bias=not with_bn)
            self.bn = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            conv = self.conv(x)
            bn = self.bn(conv)
            relu = self.relu(bn)
            return relu

class fully_connected(nn.Module):
    def __init__(self, inp_dim, out_dim, with_bn=True):
        super(fully_connected, self).__init__()
        self.with_bn = with_bn

        self.linear = nn.Linear(inp_dim, out_dim)
        if self.with_bn:
            self.bn = nn.BatchNorm1d(out_dim)
        self.relu   = nn.ReLU(inplace=True)

    def forward(self, x):
        linear = self.linear(x)
        bn     = self.bn(linear) if self.with_bn else linear
        relu   = self.relu(bn)
        return relu


class residual(nn.Module):
    if use_dsc:
        def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
            super(residual, self).__init__()

            self.conv_bn = conv_bn(inp_dim, 64, stride)
            self.conv_bn2 = conv_bn_same(64, 64, 1)
            self.conv_1x1_2 = conv_1x1(64, out_dim, 1)

            self.skip = nn.Sequential(conv_bn_k(inp_dim, 64, 1, stride, pad=0),
                                      conv_1x1(64, out_dim, stride)) \
                if stride != 1 or inp_dim != out_dim else nn.Sequential()
            self.relu = Hswish()

        def forward(self, x):
            conv1 = self.conv_bn(x)
            conv_bn2 = self.conv_bn2(conv1)
            conv_1x1_2 = self.conv_1x1_2(conv_bn2)
            skip = self.skip(x)
            return self.relu(conv_1x1_2 + skip)
    else:
        def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
            super(residual, self).__init__()

            self.conv1 = nn.Conv2d(inp_dim, out_dim, (3, 3), padding=(1, 1), stride=(stride, stride), bias=False)
            self.bn1 = nn.BatchNorm2d(out_dim)
            self.relu1 = nn.ReLU(inplace=True)

            self.conv2 = nn.Conv2d(out_dim, out_dim, (3, 3), padding=(1, 1), bias=False)
            self.bn2 = nn.BatchNorm2d(out_dim)

            self.skip = nn.Sequential(
                nn.Conv2d(inp_dim, out_dim, (1, 1), stride=(stride, stride), bias=False),
                nn.BatchNorm2d(out_dim)
            ) if stride != 1 or inp_dim != out_dim else nn.Sequential()
            self.relu = nn.ReLU(inplace=True)
        def forward(self, x):
            conv1 = self.conv1(x)
            bn1   = self.bn1(conv1)
            relu1 = self.relu1(bn1)

            conv2 = self.conv2(relu1)
            bn2   = self.bn2(conv2)

            skip  = self.skip(x)
            return self.relu(bn2 + skip)


def make_layer(k, inp_dim, out_dim, modules, layer=convolution, **kwargs):
    layers = [layer(k, inp_dim, out_dim, **kwargs)]
    for _ in range(1, modules):
        layers.append(layer(k, out_dim, out_dim, **kwargs))
    return nn.Sequential(*layers)

class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


def make_layer_revr(k, inp_dim, out_dim, modules, layer=convolution, **kwargs):
    layers = []
    for _ in range(modules - 1):
        layers.append(layer(k, inp_dim, inp_dim, **kwargs))
    layers.append(layer(k, inp_dim, out_dim, **kwargs))
    return nn.Sequential(*layers)

def conv_bn_k(inp, oup, k, stride, pad, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=Hswish):
    return nn.Sequential(
        conv_layer(inp, oup, (k, k), padding=(pad, pad), stride=(stride, stride), bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )

def conv_1x1_bn(inp, oup, stride, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=Hswish):
    return nn.Sequential(
        conv_layer(inp, oup, 1, 1, 0, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )

def conv_1x1(inp, oup, stride,conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 1, 1, 0, bias=False),
        norm_layer(oup)
    )

def conv_bn(inp, oup, stride, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=Hswish):
    return nn.Sequential(
        conv_layer(inp, oup, 3, stride, 1, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )
def conv_bn_same(inp, oup, stride, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=Hswish):
    return nn.Sequential(
        conv_layer(inp, oup, 3, stride, padding=1, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )
