import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from quant_ultra import *

def create_grids(self, img_size=416, ng=(13, 13), device='cpu', type=torch.float32):
    nx, ny = ng  # x and y grid size
    self.img_size = max(img_size)
    self.stride = self.img_size / max(ng)

    # build xy offsets
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    self.grid_xy = torch.stack((xv, yv), 2).to(device).type(type).view((1, 1, ny, nx, 2))

    # build wh gains
    self.anchor_vec = self.anchors.to(device) / self.stride
    self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2).to(device).type(type)
    self.ng = torch.Tensor(ng).to(device)
    self.nx = nx
    self.ny = ny

class YOLOLayer(nn.Module):
    def __init__(self, anchors):
        super(YOLOLayer, self).__init__()

        self.anchors = torch.Tensor(anchors)    # 预设锚框尺寸
        self.na = len(anchors)  # number of anchors (3)  锚框数量
        self.no = 6  # number of outputs  输出维度 (x,y,w,h,obj,cls)
        self.nx = 0  # initialize number of x gridpoints  网格尺寸初始化
        self.ny = 0  # initialize number of y gridpoints
    def forward(self, p, img_size):
        
        bs, _, ny, nx = p.shape  # bs, 255, 13, 13
        if (self.nx, self.ny) != (nx, ny):
            create_grids(self, img_size, (nx, ny), p.device, p.dtype)  # 创建网格坐标

        # 重整张量维度: (bs, na, no, ny, nx)
        # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
        p = p.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        if self.training:
            return p  # 训练时直接返回原始预测

        else:  # inference
            # s = 1.5  # scale_xy  (pxy = pxy * s - (s - 1) / 2)
            # 推理时解码预测 (对应论文III-B节)
            io = p.clone()  # inference output
            # 中心坐标解码: σ(tx) + cx (公式未显式给出，标准YOLO解码)
            io[..., :2] = torch.sigmoid(io[..., :2]) + self.grid_xy  # xy
            # 宽高解码: exp(tw) * anchor_w (公式未显式给出)
            io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh  # wh yolo method
            # io[..., 2:4] = ((torch.sigmoid(io[..., 2:4]) * 2) ** 3) * self.anchor_wh  # wh power method
            io[..., :4] *= self.stride  # 原始像素尺度  缩放到原图尺寸

            
            torch.sigmoid_(io[..., 4:])     # 对象置信度和分类概率
            
            
            return io.view(bs, -1, self.no), p      # 重整输出维度

class UltraNetQua(nn.Module):  # 量化网络实现
    def __init__(self):
        super(UltraNetQua, self).__init__()
        W_BIT = 4
        A_BIT = 4  # 4-bit量化
        # 量化卷积函数 (对应公式1)
        conv2d_q = conv2d_Q_fn(W_BIT)
        # act_q = activation_quantize_fn(4)

        self.layers = nn.Sequential(
            # 量化卷积层: 公式1 q_w = round(clip(w,-c,c)/s)
            conv2d_q(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            # 激活量化: 使用可学习裁剪参数 (对应PACT方法)
            activation_quantize_fn(A_BIT),
            nn.MaxPool2d(2, stride=2),

            conv2d_q(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            activation_quantize_fn(A_BIT),

            nn.MaxPool2d(2, stride=2),

            conv2d_q(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            activation_quantize_fn(A_BIT),

            nn.MaxPool2d(2, stride=2),

            conv2d_q(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            activation_quantize_fn(A_BIT),

            nn.MaxPool2d(2, stride=2),

            conv2d_q(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            activation_quantize_fn(A_BIT),

            conv2d_q(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            activation_quantize_fn(A_BIT),

            conv2d_q(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            activation_quantize_fn(A_BIT),

            # conv2d_q(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(64),
            # activation_quantize_fn(A_BIT),

            # conv2d_q(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(64),
            # activation_quantize_fn(A_BIT),

            conv2d_q(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            activation_quantize_fn(A_BIT),


            # nn.Conv2d(256, 18, kernel_size=1, stride=1, padding=0)
            conv2d_q(64, 36, kernel_size=1, stride=1, padding=0)    # 输出层用8-bit
            
        )
        # 锚框尺寸根据数据集优化
        self.yololayer = YOLOLayer([[20,20], [20,20], [20,20], [20,20], [20,20], [20,20]])
        self.yolo_layers = [self.yololayer]

    def forward(self, x):
        img_size = x.shape[-2:]
        yolo_out, out = [], []

        x = self.layers(x)
        x = self.yololayer(x, img_size)

        yolo_out.append(x)

        if self.training:  # train
            return yolo_out
        else:  # test
            io, p = zip(*yolo_out)  # inference output, training output
            return torch.cat(io, 1), p
        return x  