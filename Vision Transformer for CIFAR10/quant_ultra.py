import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def uniform_quantize(k):  # 基础量化操作
    class qfn(torch.autograd.Function):

        @staticmethod
        def forward(ctx, input):
            if k == 32:
                out = input  # 32-bit保持原值
            elif k == 1:
                out = torch.sign(input)  # 1-bit二值化，sign为符号函数sgn
            else:
                n = float(2 ** k - 1)  # 量化级别数（公式2中的分母）
                out = torch.round(input * n) / n  # 公式1-3 核心量化操作 out = w_hat
            return out

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()  # STE直通估计（公式4的替代方案）
            return grad_input

    return qfn().apply


class weight_quantize_fn(nn.Module):  # 权重量化
    def __init__(self, w_bit):
        super(weight_quantize_fn, self).__init__()
        assert w_bit <= 8 or w_bit == 32
        self.w_bit = w_bit
        # 符号位 占一位
        self.uniform_q = uniform_quantize(k=w_bit - 1)

    def forward(self, x):
        # print('===================')
        if self.w_bit == 32:
            weight_q = x
            # weight = torch.tanh(x)
            # weight_q = weight / torch.max(torch.abs(weight))
        elif self.w_bit == 1:
            E = torch.mean(torch.abs(x)).detach()  # 计算缩放因子
            weight_q = (self.uniform_q(x / E) + 1) / 2 * E  # 二值特殊处理
        else:
            # clip(v,l,h) 把大于h的值设置为h，把小于l的值设置为l
            weight = torch.tanh(
                x)  # 用tanh约束权重范围（公式1的clip替代） soft_clip(x,a,b)=b⋅ tanh(ax) (公式5)  这里固定a=b=1(原为可学习参数)，a，b使soft_clip(x,a,b)逼近clip(v,l,h)
            # weight = weight / 2 / torch.max(torch.abs(weight)) + 0.5
            # weight_q = 2 * self.uniform_q(weight) - 1
            weight = weight / torch.max(torch.abs(weight))  # 归一化到[-1,1]
            # 想量化到带符号的 k bit
            weight_q = self.uniform_q(weight)  # 应用公式1-3的量化操作
        return weight_q


class QuantizedLinear(nn.Module):
    """
    (新增)量化后线性层
    """

    def __init__(self, in_features, out_features, bias=True,
                 w_bit=4, in_bit=4, out_bit=4, l_shift=8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.in_bit = in_bit
        self.out_bit = out_bit
        self.l_shift = l_shift
        # Actual weights and bias
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        # Quantize input
        x_int = torch.clamp(x * (2 ** self.in_bit - 1), 0, 2 ** self.in_bit - 1)
        x_int = torch.round(x_int)
        # Quantize weights
        w_scale = (2 ** (self.w_bit - 1) - 1)
        w_int = torch.clamp(self.weight * w_scale, -w_scale, w_scale - 1)
        w_int = torch.round(w_int)
        # Compute output
        out = F.linear(x_int, w_int)
        if self.bias is not None:
            out = out + self.bias * (2 ** self.l_shift)
        # Scale output
        out = out / (2 ** self.l_shift)
        out = torch.clamp(out, 0, 2 ** self.out_bit - 1)
        return out


class activation_quantize_fn(nn.Module):  # 激活量化
    def __init__(self, a_bit):
        super(activation_quantize_fn, self).__init__()
        assert a_bit <= 8 or a_bit == 32
        self.a_bit = a_bit
        self.uniform_q = uniform_quantize(k=a_bit)

    def forward(self, x):
        if self.a_bit == 32:
            activation_q = x
            # activation_q = torch.clamp(x, 0, 6)  # ==ReLU6 = min(6,max(0,x)) = clip(x,0,6)
        else:
            activation_q = self.uniform_q(torch.clamp(x, 0, 1))  # clamp(x,l,h) == 公式1中clip(x,l,h)，把大于h的值设置为h，把小于l的值设置为l
            # print(np.unique(activation_q.detach().numpy()))
        return activation_q


def conv2d_Q_fn(w_bit):  # 量化卷积层
    class Conv2d_Q(nn.Conv2d):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                     padding=0, dilation=1, groups=1, bias=True):
            super(Conv2d_Q, self).__init__(in_channels, out_channels, kernel_size, stride,
                                           padding, dilation, groups, bias)
            self.w_bit = w_bit
            self.quantize_fn = weight_quantize_fn(w_bit=w_bit)  # 权重量化器

        def forward(self, input, order=None):
            weight_q = self.quantize_fn(self.weight)  # 量化的权重
            # print(np.unique(weight_q.detach().numpy()))
            return F.conv2d(input, weight_q, self.bias, self.stride,  # 标准卷积计算
                            self.padding, self.dilation, self.groups)

    return Conv2d_Q