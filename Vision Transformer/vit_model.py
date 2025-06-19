"""
original code from rwightman:
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import quant


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class QuantizedPatchEmbed(nn.Module):
    """
    （新增）量化后PatchEmbed
    通过卷积，将2D图片转成Patch序列的嵌入表示
    2D Image to Patch Embedding
    """

    # img_size: 输入图像大小(224) patch_size: 每个patch大小(16)
    # in_C: 图像通道数(3) embed_dim: 每个patch的嵌入维度(768 = 16x16x3)
    # norm_layer: 对嵌入向量进行归一化的层，可以是任意归一化层的类型，默认为None，表示不进行归一化
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None,
                 w_bit=4, in_bit=4, out_bit=4, l_shift=8):
        super().__init__()
        # 转化为元组的形式
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size  # 原始图像大小为224*224*3（H*W*C）
        self.patch_size = patch_size  # 每个Patch的尺寸P设为16，则每个Patch下图片的大小为：16*16*3
        # Patch共有 (224/16) x (224/16) = 14 x 14=196个
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])  # 图像被划分为多少个网格
        self.num_patches = self.grid_size[0] * self.grid_size[1]  # 表示总共有多少个patch
        """
            每个Patch对应着一个token，将每个Patch展平，则得到输入矩阵X，其大小为(196, 768)，其中16*16*3=768，
        也就是每个token是768维
            通过这样的方式，我们成功将图像数据处理成自然语言的向量表达方式。
        """
        # Quantization parameters
        self.w_bit = w_bit
        self.in_bit = in_bit
        self.out_bit = out_bit
        self.l_shift = l_shift
        self.uniform_q = quant.uniform_quantize(k=self.w_bit - 1)  # 符号位 占一位
        self.quantize_fn = quant.weight_quantize_fn(w_bit=self.w_bit)  # 权重量化器  weight_quantize_fn中 k = w_bit-1

        '''Patch Embedding'''
        # Aim: 将每一个Patch的矩阵拉伸成为一个1*768维度向量，从而获得近似词向量堆叠的效果
        # nn.Conv2d层，用于将输入图像进行卷积操作，将每个patch编码为嵌入向量
        # 采用768个16*16*3尺寸的卷积核，stride=16，padding=0。这样我们就能得到14*14*768大小的特征图
        # 特征图中每一个1*1*768大小的子特征图，都是由卷积核对第一块patch做处理而来，因此它就能表示第一块patch的token向量
        # 卷积核、步幅都是patch_size，可以确保每个卷积操作只处理一个patch,不重复
        # 输出通道数为embed_dim(768)，表示每个patch的嵌入向量维度
        # Quantized conv2d projector
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)

        # 归一化层，用于对嵌入向量进行归一化
        # 提供了norm_layer，则使用该类型创建norm层，否则使用nn.Identity()作为默认的归一化层。
        # Optional norm layer
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):  # x是一个图像张量
        B, C, H, W = x.shape  # [批量大小,通道数,高度,宽度]
        # 检查输入图像是否与模型期望的图像大小一致
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # Quantize input
        x_int = self.uniform_q(torch.clamp(x, 0, 1))
        # x_int = torch.clamp(x * (2 ** self.in_bit - 1), 0, 2 ** self.in_bit - 1)
        # x_int = torch.round(x_int)

        # Quantize weights
        w = self.proj.weight
        w_int = self.quantize_fn(w)
        # w_scale = (2 ** (self.w_bit - 1) - 1)
        # w_int = torch.clamp(w * w_scale, -w_scale, w_scale - 1)
        # w_int = torch.round(w_int)

        # Conv projection with quantized weights
        x = F.conv2d(x_int, w_int, self.proj.bias, stride=self.patch_size)

        # Scale output
        x = x / (2 ** self.l_shift)
        x = torch.round(torch.clamp(x, 0, 2 ** self.out_bit - 1))

        # Reshape and normalize
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class PatchEmbed(nn.Module):
    """
    通过卷积，将2D图片转成Patch序列的嵌入表示
    2D Image to Patch Embedding
    """

    # img_size: 输入图像大小(224) patch_size: 每个patch大小(16)
    # in_C: 图像通道数(3) embed_dim: 每个patch的嵌入维度(768 = 16x16x3)
    # norm_layer: 对嵌入向量进行归一化的层，可以是任意归一化层的类型，默认为None，表示不进行归一化
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        # 转化为元组的形式
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size  # 原始图像大小为224*224*3（H*W*C）
        self.patch_size = patch_size  # 每个Patch的尺寸P设为16，则每个Patch下图片的大小为：16*16*3
        # Patch共有 (224/16) x (224/16) = 14 x 14=196个
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])  # 图像被划分为多少个网格
        self.num_patches = self.grid_size[0] * self.grid_size[1]  # 表示总共有多少个patch
        """
            每个Patch对应着一个token，将每个Patch展平，则得到输入矩阵X，其大小为(196, 768)，其中16*16*3=768，
        也就是每个token是768维
            通过这样的方式，我们成功将图像数据处理成自然语言的向量表达方式。
        """

        '''Patch Embedding'''
        # Aim: 将每一个Patch的矩阵拉伸成为一个1*768维度向量，从而获得近似词向量堆叠的效果
        # nn.Conv2d层，用于将输入图像进行卷积操作，将每个patch编码为嵌入向量
        # 采用768个16*16*3尺寸的卷积核，stride=16，padding=0。这样我们就能得到14*14*768大小的特征图
        # 特征图中每一个1*1*768大小的子特征图，都是由卷积核对第一块patch做处理而来，因此它就能表示第一块patch的token向量
        # 卷积核、步幅都是patch_size，可以确保每个卷积操作只处理一个patch,不重复
        # 输出通道数为embed_dim(768)，表示每个patch的嵌入向量维度
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)

        # 归一化层，用于对嵌入向量进行归一化
        # 提供了norm_layer，则使用该类型创建norm层，否则使用nn.Identity()作为默认的归一化层。
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):  # x是一个图像张量
        B, C, H, W = x.shape  # [批量大小,通道数,高度,宽度]
        # 检查输入图像是否与模型期望的图像大小一致
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # 将输入图像x传递给proj层进行卷积操作，将图像划分为一系列的patch，
        # 并将每个patch编码为一个嵌入向量，输出的形状为[B, embed_dim, grid_size[0], grid_size[1]]
        # flatten: [B, C, H, W] -> [B, C, HW]   将每个patch展平为一个向量
        # transpose: [B, C, HW] -> [B, HW, C]   将嵌入向量的维度放在第二个维度上
        x = self.proj(x).flatten(2).transpose(1, 2)
        # 将嵌入向量传递给归一化层norm进行归一化处理
        x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,  # 输入token的dim
                 num_heads=8,  # 注意力头的数量，默认为8
                 qkv_bias=False,  # 控制是否在查询、键、值投影中使用偏置，默认为False
                 qk_scale=None,  # 缩放因子，用于缩放查询和键的点积，默认为None，若为None，则设置为head_dim的倒数平方
                 attn_drop_ratio=0.,  # 注意力权重的dropout比例，默认为0，表示不使用dropout
                 proj_drop_ratio=0.):  # 注意力输出的投影层的dropout比例，默认为0
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads  # 每个注意力头的维度head_dim
        self.scale = qk_scale or head_dim ** -0.5  # 缩放因子
        # q:query(to match others)   key(to be matched)   v(information to be extracted)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # 将输入进行查询、键、值的投影
        self.attn_drop = nn.Dropout(attn_drop_ratio)  # dropout层,在注意力权重上应用dropout
        self.proj = nn.Linear(dim, dim)  # 将多头注意力的输出进行投影
        self.proj_drop = nn.Dropout(proj_drop_ratio)  # 在投影输出上应用dropout

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]    (196+1, 768)
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3(qkv三个参数), num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        ''' Attention(Q,K,V)=softmax( QK^{T}/sqrt(d_{k}) )V'''
        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply矩阵乘法(只对最后两个维度操作) -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale  # QK^{T}/sqrt(d_{k})
        attn = attn.softmax(dim=-1)  # -1代表最后一个维度，即对每一行进行softmax处理  softmax( QK^{T}/sqrt(d_{k}) )
        attn = self.attn_drop(attn)  # 在注意力权重上应用dropout
        ''' MultiHead(Q,K,V)=Concat(head1,...,headh)W^{O}, where headi = Attention(QW^{Q}i,KW^{K}i,VW^{V}i})'''
        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]    Concat拼接
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)  # 将多头注意力的输出进行投影  W^{O}
        x = self.proj_drop(x)
        return x


class QuantizedAttention(nn.Module):
    """
    (新增)量化后注意力机制
    """

    def __init__(self,
                 dim,  # 输入token的dim
                 num_heads=8,  # 注意力头的数量，默认为8
                 qkv_bias=False,  # 控制是否在查询、键、值投影中使用偏置，默认为False
                 qk_scale=None,  # 缩放因子，用于缩放查询和键的点积，默认为None，若为None，则设置为head_dim的倒数平方
                 attn_drop_ratio=0.,  # 注意力权重的dropout比例，默认为0，表示不使用dropout
                 proj_drop_ratio=0.,  # 注意力输出的投影层的dropout比例，默认为0
                 w_bit=4, in_bit=4, out_bit=4):
        super(QuantizedAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads  # 每个注意力头的维度head_dim
        self.scale = qk_scale or head_dim ** -0.5
        # q:query(to match others)   key(to be matched)   v(information to be extracted)
        # 量化的QKV投影, 将输入进行查询、键、值的投影
        self.qkv = quant.QuantizedLinear(dim, dim * 3, bias=qkv_bias,
                                               w_bit=w_bit, in_bit=in_bit, out_bit=out_bit)
        # dropout层,在注意力权重上应用dropout
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        # 量化的, 将多头注意力的输出进行投影
        self.proj = quant.QuantizedLinear(dim, dim,
                                                w_bit=w_bit, in_bit=in_bit, out_bit=out_bit)
        self.proj_drop = nn.Dropout(proj_drop_ratio)  # 在投影输出上应用dropout
        # 注意力分数的量化
        self.uniform_q = quant.uniform_quantize(k=w_bit - 1)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]    (196+1, 768)
        B, N, C = x.shape

        # Q/K/V计算并量化
        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3(qkv三个参数), num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        ''' Attention(Q,K,V)=softmax( QK^{T}/sqrt(d_{k}) )V'''
        # 注意力分数计算并量化
        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply矩阵乘法(只对最后两个维度操作) -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale  # QK^{T}/sqrt(d_{k})
        attn = attn.softmax(dim=-1)  # -1代表最后一个维度，即对每一行进行softmax处理  softmax( QK^{T}/sqrt(d_{k}) )
        attn = self.uniform_q(attn)  # 量化注意力权重
        attn = self.attn_drop(attn)  # 在注意力权重上应用dropout

        ''' MultiHead(Q,K,V)=Concat(head1,...,headh)W^{O}, where headi = Attention(QW^{Q}i,KW^{K}i,VW^{V}i})'''
        # 多头注意力输出
        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]    Concat拼接
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)  # 将多头注意力的输出进行投影  W^{O}
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # MLP Block:  Linear -> GELU -> Dropout -> Linear -> Dropout
        x = self.fc1(x)  # [197,768] -> [197,786*4=3072]
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)  # [197,3072] -> [198, 768]
        x = self.drop(x)
        return x


class QuantizedMlp(nn.Module):
    """
    （新增）量化后MLP
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 w_bit=4, in_bit=4, out_bit=4):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # 量化的全连接层
        self.fc1 = quant.QuantizedLinear(in_features, hidden_features,
                                               w_bit=w_bit, in_bit=in_bit, out_bit=out_bit)
        self.fc2 = quant.QuantizedLinear(hidden_features, out_features,
                                               w_bit=w_bit, in_bit=in_bit, out_bit=out_bit)

        # 激活函数量化
        self.act = act_layer()
        self.act_quant = quant.activation_quantize_fn(a_bit=w_bit)
        self.drop = nn.Dropout(drop)

        self.uniform_q = quant.uniform_quantize(k=w_bit - 1)

    def forward(self, x):
        # MLP Block:  Linear -> GELU -> Dropout -> Linear -> Dropout
        x = self.fc1(x)  # [197,768] -> [197,786*4=3072]
        x = self.act(x)
        x = self.act_quant(x)  # Quantize activation
        x = self.drop(x)
        x = self.fc2(x)  # [197,3072] -> [198, 768]
        x = self.uniform_q(x)  # Quantize output
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        # Encoder Block:
        # Layer Norm -> MultiHead Attention -> DropPath/Dropout ->捷径分支相加 ->
        # Layer Norm -> MLP Block -> DropPath/Dropout ->捷径分支相加
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class QuantizedBlock(nn.Module):
    """
    (新增) 量化后Encoder Block
    """

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 w_bit=4, in_bit=4, out_bit=4):
        super(QuantizedBlock, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = QuantizedAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                       attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio,
                                       w_bit=w_bit, in_bit=in_bit, out_bit=out_bit)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = QuantizedMlp(in_features=dim, hidden_features=mlp_hidden_dim,
                                act_layer=act_layer, drop=drop_ratio,
                                w_bit=w_bit, in_bit=in_bit, out_bit=out_bit)

    def forward(self, x):
        # Encoder Block:
        # Layer Norm -> MultiHead Attention -> DropPath/Dropout ->捷径分支相加 ->
        # Layer Norm -> MLP Block -> DropPath/Dropout ->捷径分支相加
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer  在Transformer Encoder中重复堆叠Encoder Block的次数
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models        VIT中为False
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        # patch embedding
        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches  # patch的个数
        # class embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # class token(1,1,768) 第一个1对应batch维度，为了方便后续拼接
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None  # VIT中dist_token为None
        # position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))  # (1,196+1,768)
        self.pos_drop = nn.Dropout(p=drop_ratio)
        # Encoder Block中 每个传入DropPath的drop_path_ratio为等差序列
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        # Transformer Encoder
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Representation layer     MLP Head 中 是否构建Pre_Logits
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s)     MLP Head中的Linear
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        """输入 -> Patch Embedding"""
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 196, 768]
        # cls_token由(1, 1, 768)->(B, 1, 768), B是batch_size
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        """Patch Embedding -> Concat(Class token)"""
        if self.dist_token is None:
            # dist_token是None,DeiT models才会用到dist_token
            x = torch.cat((cls_token, x), dim=1)  # [B, 196+1, 768] # Concat拼接
        else:
            # x由(B, N, E)->(B, 2+N, E)
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        """Concat(Class token) -> +Position Embedding -> Dropout"""
        # +pos_embed:(1, 1+N, E)，再加一个dropout层
        x = self.pos_drop(x + self.pos_embed)
        """Dropout -> Transformer Encoder (堆叠12次Encoder Block) """
        x = self.blocks(x)
        """Transformer Encoder -> Layer Norm """
        x = self.norm(x)  # [197, 768]
        """Layer Norm -> Extract Class Token -> MLP Head Pre_Logits"""
        if self.dist_token is None:  # VIT为None
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x = self.forward_features(x)
        """MLP Head Pre_Logits -> MLP Head Linear"""
        if self.head_dist is not None:  # VIT为None
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x


class QuantizedVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=QuantizedPatchEmbed, norm_layer=None,
                 act_layer=None,
                 w_bit=4, in_bit=4, out_bit=4):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer  在Transformer Encoder中重复堆叠Encoder Block的次数
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models        VIT中为False
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(QuantizedVisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1  # 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        # patch embedding
        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim,
                                       w_bit=w_bit, in_bit=in_bit, out_bit=out_bit)
        num_patches = self.patch_embed.num_patches  # patch的个数
        # class embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # class token(1,1,768) 第一个1对应batch维度，为了方便后续拼接
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None  # VIT中dist_token为None
        # position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))  # (1,196+1,768)
        self.pos_drop = nn.Dropout(p=drop_ratio)
        # Encoder Block中 每个传入DropPath的drop_path_ratio为等差序列
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        # Transformer Encoder
        self.blocks = nn.Sequential(*[
            QuantizedBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                           qk_scale=qk_scale,
                           drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                           norm_layer=norm_layer, act_layer=act_layer,
                           w_bit=w_bit, in_bit=in_bit, out_bit=out_bit
                           )
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Representation layer     MLP Head 中 是否构建Pre_Logits
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s)     MLP Head中的Linear
        self.head = quant.QuantizedLinear(self.num_features, num_classes, w_bit=w_bit, in_bit=in_bit,
                                                out_bit=out_bit) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = quant.QuantizedLinear(self.embed_dim, self.num_classes, w_bit=w_bit, in_bit=in_bit,
                                                         out_bit=out_bit) if num_classes > 0 else nn.Identity()

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        """输入 -> Patch Embedding"""
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 196, 768]
        # cls_token由(1, 1, 768)->(B, 1, 768), B是batch_size
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        """Patch Embedding -> Concat(Class token)"""
        if self.dist_token is None:
            # dist_token是None,DeiT models才会用到dist_token
            x = torch.cat((cls_token, x), dim=1)  # [B, 196+1, 768] # Concat拼接
        else:
            # x由(B, N, E)->(B, 2+N, E)
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        """Concat(Class token) -> +Position Embedding -> Dropout"""
        # +pos_embed:(1, 1+N, E)，再加一个dropout层
        x = self.pos_drop(x + self.pos_embed)
        """Dropout -> Transformer Encoder (堆叠12次Encoder Block) """
        x = self.blocks(x)
        """Transformer Encoder -> Layer Norm """
        x = self.norm(x)  # [197, 768]
        """Layer Norm -> Extract Class Token -> MLP Head Pre_Logits"""
        if self.dist_token is None:  # VIT为None
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x = self.forward_features(x)
        """MLP Head Pre_Logits -> MLP Head Linear"""
        if self.head_dist is not None:  # VIT为None
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


# patch_size越大，token维度越大，序列长度（即token个数）越小，计算量越小
# VIT模型需要在非常大的数据集上训练后才有很好的效果，故要使用预训练权重
def vit_base_patch16_224(num_classes: int = 1000):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1zqb08naP0RPqqfSXfkB2EA  密码: eu9f
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_base_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_base_patch16_224_in21k_Qua(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth
    """
    model = QuantizedVisionTransformer(img_size=224,
                                       patch_size=16,
                                       embed_dim=768,
                                       depth=12,
                                       num_heads=12,
                                       representation_size=768 if has_logits else None,
                                       num_classes=num_classes,
                                       w_bit=4, in_bit=4, out_bit=4)
    return model


def vit_base_patch32_224(num_classes: int = 1000):
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1hCv0U8pQomwAtHBYc4hmZg  密码: s5hl
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_base_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch32_224_in21k-8db57226.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_large_patch16_224(num_classes: int = 1000):
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1cxBgZJJ6qUWPSBNcE4TdRQ  密码: qqt8
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_large_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch16_224_in21k-606da67d.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_large_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_huge_patch14_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: converted weights not currently available, too large for github release hosting.
    """
    model = VisionTransformer(img_size=224,
                              patch_size=14,
                              embed_dim=1280,
                              depth=32,
                              num_heads=16,
                              representation_size=1280 if has_logits else None,
                              num_classes=num_classes)
    return model
