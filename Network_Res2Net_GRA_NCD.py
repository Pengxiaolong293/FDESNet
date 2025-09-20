import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log
from lib.Res2Net_v1b import res2net50_v1b_26w_4s

from lib.pvtv2 import pvt_v2_b2

from torch.autograd import Variable
import math
import numpy as np
from options import opt


def cus_sample(feat, **kwargs):

    assert len(kwargs.keys()) == 1 and list(kwargs.keys())[0] in ["size", "scale_factor"]
    return F.interpolate(feat, **kwargs, mode="bilinear", align_corners=False)


class Conv1x1(nn.Module):
    def __init__(self, inplanes, planes):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, relu=False, bn=True):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 最大池化

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 平均池化
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 最大池化
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM_Block(nn.Module):
    def __init__(self, channel, ratio=16, kernel_size=7):
        super(CBAM_Block, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)
    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)

        return x

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        # 如果下面这个原论文代码用不了的话，可以换成另一个试试
        out = identity * a_w * a_h
        # out = a_h.expand_as(x) * a_w.expand_as(x) * identity

        return out

class DeformConv2D(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, bias=None):
        super(DeformConv2D, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv_kernel = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

    def forward(self, x, offset):
        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        # Change offset's order from [x1, x2, ..., y1, y2, ...] to [x1, y1, x2, y2, ...]
        # Codes below are written to make sure same results of MXNet implementation.
        # You can remove them, and it won't influence the module's performance.
        offsets_index = Variable(torch.cat([torch.arange(0, 2*N, 2), torch.arange(1, 2*N+1, 2)]), requires_grad=False).type_as(x).long()
        offsets_index = offsets_index.unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1).expand(*offset.size())
        offset = torch.gather(offset, dim=1, index=offsets_index)
        # ------------------------------------------------------------------------

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = Variable(p.data, requires_grad=False).floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], -1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], -1)

        # (b, h, w, N)
        mask = torch.cat([p[..., :N].lt(self.padding)+p[..., :N].gt(x.size(2)-1-self.padding),
                          p[..., N:].lt(self.padding)+p[..., N:].gt(x.size(3)-1-self.padding)], dim=-1).type_as(p)
        mask = mask.detach()
        floor_p = p - (p - torch.floor(p))
        p = p*(1-mask) + floor_p*mask
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv_kernel(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = np.meshgrid(range(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
                          range(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1), indexing='ij')
        # (2N, 1)
        p_n = np.concatenate((p_n_x.flatten(), p_n_y.flatten()))
        p_n = np.reshape(p_n, (1, 2*N, 1, 1))
        p_n = Variable(torch.from_numpy(p_n).type(dtype), requires_grad=False)

        return p_n

    @staticmethod
    def _get_p_0(h, w, N, dtype):
        p_0_x, p_0_y = np.meshgrid(range(1, h+1), range(1, w+1), indexing='ij')
        p_0_x = p_0_x.flatten().reshape(1, 1, h, w).repeat(N, axis=1)
        p_0_y = p_0_y.flatten().reshape(1, 1, h, w).repeat(N, axis=1)
        p_0 = np.concatenate((p_0_x, p_0_y), axis=1)
        p_0 = Variable(torch.from_numpy(p_0).type(dtype), requires_grad=False)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

        return x_offset


class SEM(nn.Module):
    def __init__(self, hchannel, channel):
        super(SEM, self).__init__()
        self.conv1_1 = Conv1x1(hchannel, channel // 4)

        self.deform_conv = DeformConv2D(channel // 4, channel // 4, 3, padding=1)

        self.conv = BasicConv2d(hchannel//2, channel//4, 3, padding=1)
        self.conv1_3 = BasicConv2d(channel // 4, channel // 4, kernel_size=(1, 3), padding=(0, 1))
        self.conv3_1 = BasicConv2d(channel // 4, channel // 4, kernel_size=(3, 1), padding=(1, 0))
        self.conv3_3_1 = BasicConv2d(channel // 4, channel // 4, 3, padding=1)
        self.conv1_5 = BasicConv2d(channel // 4, channel // 4, kernel_size=(1, 5), padding=(0, 2))
        self.dconv5_1 = BasicConv2d(channel // 4, channel // 4, kernel_size=(5, 1), padding=(2, 0))
        self.conv3_3_2 = BasicConv2d(channel // 4, channel // 4, 3, dilation=2, padding=2)
        self.conv1_7 = BasicConv2d(channel // 4, channel // 4, kernel_size=(1, 7), padding=(0, 3))
        self.dconv7_1 = BasicConv2d(channel // 4, channel // 4, kernel_size=(7, 1), padding=(3, 0))
        self.conv3_3_3 = BasicConv2d(channel // 4, channel // 4, 3, dilation=3, padding=3)
        self.conv3_3_4 = BasicConv2d(channel // 4, channel // 4, 3, dilation=4, padding=4)
        self.conv1_2 = Conv1x1(channel, channel)
        self.conv3_3 = BasicConv2d(channel, channel, 3, padding=1)

        self.offset_conv = nn.Conv2d(channel // 4, 2 * 3 * 3, kernel_size=3, padding=1)
        self.CBAM = CBAM_Block(channel)
        self.conv1 = BasicConv2d(16, 64, 3, padding=1)
        self.conv2 = BasicConv2d(64, 16, 3, padding=1)

    def forward(self, lf, hf):
        if lf.size()[2:] != hf.size()[2:]:
            hf = F.interpolate(hf, size=lf.size()[2:], mode='bilinear', align_corners=False)
        hf = self.conv1_1(hf)  # 16
        xc = torch.chunk(lf, 4, dim=1)

        x_0 = torch.cat((xc[0], hf), 1)  # 32
        x_0 = self.conv(x_0)    # x0  32-16

        offset = self.offset_conv(x_0 + xc[1])
        x_0 = self.deform_conv(x_0 + xc[1],offset)
        x0_1 = self.conv3_3_1(x_0)

        x0_1 = self.conv1(x0_1)
        x0_1 = self.CBAM(x0_1)
        x0_1 = self.conv2(x0_1)

        # x_0 = self.conv1_3(x_0 + xc[1])
        # x0_1 = self.conv3_1(x_0)  # x0+x1
        # x0_1 = self.conv3_3_1(x0_1)

        x_1 = torch.cat((x0_1, hf), 1)
        x_1 = self.conv(x_1)  # x1

        offset = self.offset_conv(x_1 + x0_1 + xc[2])
        x1_2 = self.deform_conv(x_1 + x0_1 + xc[2],offset)
        x1_2 = self.conv3_3_2(x1_2)

        x1_2 = self.conv1(x1_2)
        x1_2 = self.CBAM(x1_2)
        x1_2 = self.conv2(x1_2)

        # x1_2 = self.conv1_5(x_1 + x0_1 + xc[2])
        # x1_2 = self.dconv5_1(x1_2)
        # x1_2 = self.conv3_3_2(x1_2)

        x_2 = torch.cat((x1_2, hf), 1)
        x_2 = self.conv(x_2)

        offset = self.offset_conv(x_2 + x1_2 + xc[3])
        x2_3 = self.deform_conv(x_2 + x1_2 + xc[3],offset)
        x2_3 = self.conv3_3_3(x2_3)

        x2_3 = self.conv1(x2_3)
        x2_3 = self.CBAM(x2_3)
        x2_3 = self.conv2(x2_3)

        # x2_3 = self.conv1_7(x_2 + x1_2 + xc[3])
        # x2_3 = self.dconv7_1(x2_3)
        # x2_3 = self.conv3_3_3(x2_3)

        x_3 = torch.cat((x2_3, hf), 1)
        x_3 = self.conv(x_3)

        offset = self.offset_conv(x_3 + x2_3)
        x3_4 = self.deform_conv(x_3 + x2_3,offset)
        x3_4 = self.conv3_3_4(x3_4)

        x3_4 = self.conv1(x3_4)
        x3_4 = self.CBAM(x3_4)
        x3_4 = self.conv2(x3_4)

        # x3_4 = self.conv1_3(x_3 + x2_3)
        # x3_4 = self.conv3_1(x3_4)
        # x3_4 = self.conv3_3_4(x3_4)

        x2 = x0_1 + x1_2
        x3 = x2 + x2_3
        x4 = x3 + x3_4
        xx = self.conv1_2(torch.cat((x0_1, x2, x3, x4), dim=1))
        x = self.conv3_3(xx)

        return x


# class DWT(nn.Module):
#     def __init__(self):
#         super(DWT, self).__init__()
#         self.requires_grad = False
#
#     def forward(self, x):
#         x01 = x[:, :, 0::2, :] / 2
#         x02 = x[:, :, 1::2, :] / 2
#         x1 = x01[:, :, :, 0::2]
#         x2 = x02[:, :, :, 0::2]
#         x3 = x01[:, :, :, 1::2]
#         x4 = x02[:, :, :, 1::2]
#         ll = x1 + x2 + x3 + x4
#         lh = -x1 + x2 - x3 + x4
#         hl = -x1 - x2 + x3 + x4
#         hh = x1 - x2 - x3 + x4
#
#         return ll, lh, hl, hh


class BoxFilter(nn.Module):
    def __init__(self, r):
        super(BoxFilter, self).__init__()

        self.r = r

    def diff_x(self, input, r):
        assert input.dim() == 4

        left = input[:, :, r:2 * r + 1]
        middle = input[:, :, 2 * r + 1:] - input[:, :, :-2 * r - 1]
        right = input[:, :, -1:] - input[:, :, -2 * r - 1:    -r - 1]

        output = torch.cat([left, middle, right], dim=2)

        return output

    def diff_y(self, input, r):
        assert input.dim() == 4

        left = input[:, :, :, r:2 * r + 1]
        middle = input[:, :, :, 2 * r + 1:] - input[:, :, :, :-2 * r - 1]
        right = input[:, :, :, -1:] - input[:, :, :, -2 * r - 1:    -r - 1]

        output = torch.cat([left, middle, right], dim=3)

        return output

    def forward(self, x):
        assert x.dim() == 4
        return self.diff_y(self.diff_x(x.cumsum(dim=2), self.r).cumsum(dim=3), self.r)


class GF(nn.Module):
    def __init__(self, r, eps=1e-8):
        super(GF, self).__init__()

        self.r = r
        self.eps = eps
        self.boxfilter = BoxFilter(r)
        self.epss = 1e-12

    def forward(self, lr_x, lr_y, hr_x, l_a):
        n_lrx, c_lrx, h_lrx, w_lrx = lr_x.size()
        n_lry, c_lry, h_lry, w_lry = lr_y.size()
        n_hrx, c_hrx, h_hrx, w_hrx = hr_x.size()

        lr_x = lr_x.double()
        lr_y = lr_y.double()
        hr_x = hr_x.double()
        l_a = l_a.double()

        assert n_lrx == n_lry and n_lry == n_hrx
        assert c_lrx == c_hrx and (c_lrx == 1 or c_lrx == c_lry)
        assert h_lrx == h_lry and w_lrx == w_lry
        assert h_lrx > 2 * self.r + 1 and w_lrx > 2 * self.r + 1

        ## N
        N = self.boxfilter(Variable(lr_x.data.new().resize_((1, 1, h_lrx, w_lrx)).fill_(1.0)))

        # l_a = torch.abs(l_a)
        l_a = torch.abs(l_a) + self.epss

        t_all = torch.sum(l_a)
        l_t = l_a / t_all

        ## mean_attention
        mean_a = self.boxfilter(l_a) / N
        ## mean_a^2xy
        mean_a2xy = self.boxfilter(l_a * l_a * lr_x * lr_y) / N
        ## mean_tax
        mean_tax = self.boxfilter(l_t * l_a * lr_x) / N
        ## mean_ay
        mean_ay = self.boxfilter(l_a * lr_y) / N
        ## mean_a^2x^2
        mean_a2x2 = self.boxfilter(l_a * l_a * lr_x * lr_x) / N
        ## mean_ax
        mean_ax = self.boxfilter(l_a * lr_x) / N

        ## A
        temp = torch.abs(mean_a2x2 - N * mean_tax * mean_ax)
        A = (mean_a2xy - N * mean_tax * mean_ay) / (temp + self.eps)
        ## b
        b = (mean_ay - A * mean_ax) / (mean_a)

        # --------------------------------
        # Mean
        # --------------------------------
        A = self.boxfilter(A) / N
        b = self.boxfilter(b) / N

        ## mean_A; mean_b
        mean_A = F.interpolate(A, (h_hrx, w_hrx), mode='bilinear', align_corners=True)
        mean_b = F.interpolate(b, (h_hrx, w_hrx), mode='bilinear', align_corners=True)

        return (mean_A * hr_x + mean_b).float()


class getAlpha(nn.Module):
    def __init__(self, in_channels):
        super(getAlpha, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels, 1, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


# class ESM(nn.Module):
#     def __init__(self):
#         super(ESM, self).__init__()
#         self.conv_cat = BasicConv2d(64*2, 64, 3, padding=1)
#         self.CA = ChannelAttention(64)
#         self.SA = SpatialAttention()
#         self.GA = getAlpha(64)
#         self.DWT = DWT()
#         self.GF = GF(r=2, eps=1e-2)
#         #self.up = torch.nn.PixelShuffle(2)
#         self.up = cus_sample
#         self.conv = Conv1x1(64, 64)
#         self.conv_ll = BasicConv2d(64, 64, kernel_size=3, padding=1)
#         self.conv_lh = BasicConv2d(64, 64, kernel_size=3, padding=1)
#         self.conv_hl = BasicConv2d(64, 64, kernel_size=3, padding=1)
#         self.conv_hh = BasicConv2d(64, 64, kernel_size=3, padding=1)
#         self.once_conv_1 = BasicConv2d(64 + 64, 64, kernel_size=1)
#         self.once_conv_2 = BasicConv2d(64*2, 64, kernel_size=1)
#
#         self.block = nn.Sequential(
#             BasicConv2d(64, 64, 3, padding=1),
#             BasicConv2d(64, 64, 3, padding=1),
#             nn.Conv2d(64, 1, 3, padding=1))
#
#     def forward(self, x):
#         x1, x2, x3, x4 = self.DWT(x)
#
#         x_ll = self.GF(x1, x1, x1, x1)
#         x_lh = self.GF(x2, x2, x2, x2)
#         x_hl = self.GF(x3, x3, x3, x3)
#         x_hh = self.GF(x4, x4, x4, x4)
#         x_ll_w = torch.nn.functional.softmax(x_ll, dim=1)
#         x_lh_w = torch.nn.functional.softmax(x_lh, dim=1)
#         x_hl_w = torch.nn.functional.softmax(x_hl, dim=1)
#         x_hh_w = torch.nn.functional.softmax(x_hh, dim=1)
#
#         x_ll = self.conv_ll(torch.matmul(x1, x_ll_w))
#         x_lh = self.conv_lh(torch.matmul(x2, x_lh_w))
#         x_hl = self.conv_hl(torch.matmul(x3, x_hl_w))
#         x_hh = self.conv_hh(torch.matmul(x4, x_hh_w))
#
#         x_c = torch.cat((x_lh, x_hl), dim=1)
#         x_c = self.once_conv_1(x_c)
#
#         x_c_w = self.GA(x_c)
#         x_lh = x_lh * x_c_w
#         x_hl = x_hl * (1 - x_c_w)
#
#         x_ll = self.SA(x_ll) * x_ll
#
#         x_l = x_ll + x_lh
#
#         x_hh = self.conv(x_hh)
#         x_hh_w = self.CA(x_hh)
#         x_hh_1 = x_hh * x_hh_w
#         x_h = x_hh_1 + x_hl
#
#         f1 = self.once_conv_2(torch.cat((x_h, x_l), dim=1))
#         out1 = self.conv(x_h + f1)
#         out1 = self.up(out1, scale_factor=2)
#         out = x + out1
#         out = self.block(out)
#
#         return out

class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1):
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        batch, c, h, w = x.size()

        # (batch, c, h, w/2+1, 2)
        ffted = torch.fft.rfft2(x, norm='ortho')
        x_fft_real = torch.unsqueeze(torch.real(ffted), dim=-1)
        x_fft_imag = torch.unsqueeze(torch.imag(ffted), dim=-1)
        ffted = torch.cat((x_fft_real, x_fft_imag), dim=-1)
        # (batch, c, 2, h, w/2+1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(self.bn(ffted))

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.view_as_complex(ffted)

        output = torch.fft.irfft2(ffted, s=(h, w), norm='ortho')

        return output

class Freq_Fusion(nn.Module):
    def __init__(
            self,
            dim,
            kernel_size=[1,3,5,7],
            se_ratio=4,
            local_size=8,
            scale_ratio=2,
            spilt_num=4
    ):
        super(Freq_Fusion, self).__init__()
        self.dim = dim
        self.c_down_ratio = se_ratio
        self.size = local_size
        self.dim_sp = dim*scale_ratio//spilt_num
        self.conv_init_1 = nn.Sequential(  # PW
            nn.Conv2d(dim, dim, 1),
            nn.GELU()
        )
        self.conv_init_2 = nn.Sequential(  # DW
            nn.Conv2d(dim, dim, 1),
            nn.GELU()
        )
        self.conv_mid = nn.Sequential(
            nn.Conv2d(dim*2, dim, 1),
            nn.GELU()
        )
        self.FFC = FourierUnit(self.dim*2, self.dim*2)

        self.bn = torch.nn.BatchNorm2d(dim*2)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x_1, x_2 = torch.split(x, self.dim, dim=1)
        x_1 = self.conv_init_1(x_1)
        x_2 = self.conv_init_2(x_2)
        x0 = torch.cat([x_1, x_2], dim=1)
        x = self.FFC(x0) + x0
        x = self.relu(self.bn(x))

        return x

class Fused_Fourier_Conv_Mixer(nn.Module):
    def __init__(
            self,
            dim,
            token_mixer_for_gloal=Freq_Fusion,
            mixer_kernel_size=[1,3,5,7],
            local_size=8
    ):
        super(Fused_Fourier_Conv_Mixer, self).__init__()
        self.dim = dim
        self.mixer_gloal = token_mixer_for_gloal(dim=self.dim, kernel_size=mixer_kernel_size,
                                 se_ratio=8, local_size=local_size)

        self.ca_conv = nn.Sequential(
            nn.Conv2d(2*dim, dim, 1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, padding_mode='reflect'),
            nn.GELU()
        )
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.conv_init = nn.Sequential(  # PW->DW->
            nn.Conv2d(dim, dim * 2, 1),
            nn.GELU()
        )
        self.dw_conv_1 = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=3 // 2,
                      groups=self.dim, padding_mode='reflect'),
            nn.GELU()
        )
        self.dw_conv_2 = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, kernel_size=5, padding=5 // 2,
                      groups=self.dim, padding_mode='reflect'),
            nn.GELU()
        )

        self.block = nn.Sequential(
                    BasicConv2d(64, 64, 3, padding=1),
                    BasicConv2d(64, 64, 3, padding=1),
                    nn.Conv2d(64, 1, 3, padding=1))


    def forward(self, x):
        x = self.conv_init(x)
        x = list(torch.split(x, self.dim, dim=1))
        x_local_1 = self.dw_conv_1(x[0])
        x_local_2 = self.dw_conv_2(x[0])
        x_gloal = self.mixer_gloal(torch.cat([x_local_1, x_local_2], dim=1))
        x = self.ca_conv(x_gloal)
        x = self.ca(x) * x

        out = self.block(x)

        return out


# without BN version
class ASPP(nn.Module):
    def __init__(self, in_channel=64, out_channel=64):
        super(ASPP, self).__init__()
        self.mean = nn.AdaptiveAvgPool2d((1, 1))  # (1,1)means ouput_dim
        self.conv = nn.Conv2d(in_channel,out_channel, 1, 1)
        self.atrous_block1 = nn.Conv2d(in_channel, out_channel, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, out_channel, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, out_channel, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, out_channel, 3, 1, padding=18, dilation=18)
        self.conv_1x1_output = nn.Conv2d(out_channel * 5, out_channel, 1, 1)

    def forward(self, x):
        size = x.shape[2:]

        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.upsample(image_features, size=size, mode='bilinear')

        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)

        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        return net


class AP_MP(nn.Module):
    def __init__(self, stride=8):
        super(AP_MP, self).__init__()
        self.sz=stride
        self.gapLayer=nn.AvgPool2d(kernel_size=self.sz, stride=8)
        self.gmpLayer=nn.MaxPool2d(kernel_size=self.sz, stride=8)

    def forward(self, x1, x2):
        apimg=self.gapLayer(x1)
        mpimg=self.gmpLayer(x2)
        byimg=torch.norm(abs(apimg-mpimg), p=2, dim=1, keepdim=True)
        return byimg


class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.aspp = ASPP(64)
        self.conv1x1 = Conv1x1(channel*2+1, 64)
        self.conv = BasicConv2d(64, 64, 3, padding=1)
        self.glbamp = AP_MP()

        self.coordatt = CoordAtt(channel,channel)

        # self.CBAM = CBAM_Block(channel)
        self.channel = channel

        self.conv1 = BasicConv2d(64, 1, 3, padding=1)

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(1, channel, kernel_size=3, padding=1),
            nn.ReLU(True)
        )
        self.mlp_gamma = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(channel, channel, kernel_size=3, padding=1)

    def forward(self, x_i, x_e, x_ii):

        x1_i = self.aspp(x_i)
        # x2_i = self.CBAM(x_i)
        x2_i = self.coordatt(x_i)

        # xw_ii = self.CBAM(x_ii)
        xw_ii = self.coordatt(x_ii)

        x1 = x1_i * xw_ii + x_i
        x2 = self.conv(x2_i * x_i)

        fe = self.glbamp(x1, x2)
        fe = fe/math.sqrt(self.channel)

        if fe.size()[2:] != x_e.size()[2:]:
            fe = F.interpolate(fe, size=x_e.size()[2:], mode='bilinear', align_corners=False)
        # edge = fe * x_e + x_e

        actv = self.mlp_shared(x_e)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        edge = fe * (1 + gamma) + beta
        edge = self.conv1(edge)

        out1 = self.conv1x1(torch.cat([x1, edge, x2], dim=1))
        out = self.conv(out1 + x_i)

        return out

class Network(nn.Module):
    def __init__(self, imagenet_pretrained=True):
        super(Network, self).__init__()
        # self.resnet = res2net50_v1b_26w_4s(pretrained=imagenet_pretrained)

        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = '/mnt/harddisk3/Pengxiaolong/codespace/EPFDNet-main/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.rfb1_1 = RFB_modified(64, 64)
        self.rfb2_1 = RFB_modified(128, 64)
        self.rfb3_1 = RFB_modified(320, 64)
        self.rfb4_1 = RFB_modified(512, 64)
        self.conv = Conv1x1(512, 64)

        # self.rfb1_1 = RFB_modified(256, 64)
        # self.rfb2_1 = RFB_modified(512, 64)
        # self.rfb3_1 = RFB_modified(1024, 64)
        # self.rfb4_1 = RFB_modified(2048, 64)
        # self.conv = Conv1x1(2048, 64)
        self.block = nn.Sequential(
            BasicConv2d(64, 64, 3, padding=1),
            BasicConv2d(64, 64, 3, padding=1),
            nn.Conv2d(64, 64, 3, padding=1))

        self.upsample = cus_sample
        # self.edge = ESM()
        self.edge = Fused_Fourier_Conv_Mixer(64)

        # self.mfem1 = MFEM(64,64)
        # self.mfem2 = MFEM(64, 64)
        # self.mfem3 = MFEM(64, 64)
        self.sem1 = SEM(64, 64)
        self.sem2 = SEM(64, 64)
        self.sem3 = SEM(64, 64)
        self.sem4 = SEM(64, 64)

        self.aspp = ASPP(64)

        self.fam1 = FAM(64)
        self.fam2 = FAM(64)
        self.fam3 = FAM(64)

        self.reduce2 = Conv1x1(64, 128)
        self.reduce3 = Conv1x1(64, 256)

        # self.reduce0 = Conv1x1(64, 1)

        self.predictor1 = nn.Conv2d(64, 1, 3, padding=1)
        self.predictor2 = nn.Conv2d(128, 1, 3, padding=1)
        self.predictor3 = nn.Conv2d(256, 1, 3, padding=1)
        self.predictor4 = nn.Conv2d(1, 1, 3, padding=1)

    def forward(self, x):

        pvt = self.backbone(x)

        x1 = pvt[0]
        x2 = pvt[1]
        x3 = pvt[2]
        x4 = pvt[3]

        x1_rfb = self.rfb1_1(x1)  # channel -> 64
        x2_rfb = self.rfb2_1(x2)  # channel -> 64
        x3_rfb = self.rfb3_1(x3)  # channel -> 64
        x4_rfb = self.rfb4_1(x4)  # channel -> 64

        x2_r = -1 * (torch.sigmoid(x2_rfb)) + 1
        x2_r = self.block(x2_r)

        x3_r = -1 * (torch.sigmoid(x3_rfb)) + 1
        x3_r = self.block(x3_r)

        x4_r = -1 * (torch.sigmoid(x4_rfb)) + 1
        x4_r = self.block(x4_r)

        x2a = self.sem2(x2_rfb, x2_r)

        x2a = self.upsample(x2a, scale_factor=2)
        x3a = self.sem3(x3_rfb, x3_r)

        x3a = self.upsample(x3a, scale_factor=4)
        x4a = self.sem4(x4_rfb, x4_r)

        x4a = self.upsample(x4a, scale_factor=8)
        x41 = self.conv(x4)
        x41 = self.upsample(x41, scale_factor=8)
        x4 = self.aspp(x41)
        #
        edge = self.edge(x1_rfb)
        edge_att = torch.sigmoid(edge)

        #
        x34 = self.fam1(x4a, edge_att, x4)
        x234 = self.fam2(x3a, edge_att, x34)
        x1234 = self.fam3(x2a, edge_att, x234)

        x34 = self.reduce3(x34)
        x234 = self.reduce2(x234)

        # x1_rfb = self.reduce0(x1_rfb)
        # x2_rfb = F.interpolate(x2_rfb, scale_factor=2, mode='bilinear', align_corners=False)
        # x3_rfb = F.interpolate(x3_rfb, scale_factor=4, mode='bilinear', align_corners=False)
        # x4_rfb = F.interpolate(x4_rfb, scale_factor=8, mode='bilinear', align_corners=False)
        # x4 = F.interpolate(x4, scale_factor=8, mode='bilinear', align_corners=False)

        # x34 = self.fam1(x4a, x1_rfb, x4)
        # x234 = self.fam2(x3a, x1_rfb, x34)
        # x1234 = self.fam3(x2a, x1_rfb, x234)
        #
        # x34 = self.reduce3(x34)
        # x234 = self.reduce2(x234)

        o3 = self.predictor3(x34)
        o3 = F.interpolate(o3, scale_factor=4, mode='bilinear', align_corners=False)
        o2 = self.predictor2(x234)
        o2 = F.interpolate(o2, scale_factor=4, mode='bilinear', align_corners=False)
        o1 = self.predictor1(x1234)
        o1 = F.interpolate(o1, scale_factor=4, mode='bilinear', align_corners=False)
        #oe = self.predictor4(edge)
        oe = F.interpolate(edge_att, scale_factor=4, mode='bilinear', align_corners=False)

        return o3, o2, o1, oe


if __name__ == '__main__':
    import numpy as np
    from time import time
    net = Network(imagenet_pretrained=False)
    net.eval()

    dump_x = torch.randn(1, 3, 352, 352)
    frame_rate = np.zeros((1000, 1))
    for i in range(1000):
        start = time()
        y = net(dump_x)
        end = time()
        running_frame_rate = 1 * float(1 / (end - start))
        print(i, '->', running_frame_rate)
        frame_rate[i] = running_frame_rate
    print(np.mean(frame_rate))
    print(y.shape)