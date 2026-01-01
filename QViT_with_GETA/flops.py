import torch
from fvcore.nn import FlopCountAnalysis
from vit_model import vit_base_patch16_224_in21k as create_model
from vit_model import Attention


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建ViT Base模型 (embed_dim=768, depth=12, num_heads=12)
    model = create_model(num_classes=15, has_logits=False).to(device)

    # 打印模型基本信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"ViT Base模型总参数数量: {total_params:,}")

    # Self-Attention (单头注意力，用于对比)
    a1 = Attention(dim=768, num_heads=1)  # 使用ViT Base的embed_dim=768
    a1.proj = torch.nn.Identity()  # remove Wo

    # Multi-Head Attention (ViT Base使用12个头)
    a2 = Attention(dim=768, num_heads=12)  # ViT Base的配置

    # 输入张量 - 标准ViT输入
    # ViT Base: 224x224图像，patch_size=16，所以有(224/16)^2 = 196个patches，加上1个cls token = 197个tokens
    t = (torch.rand(1, 197, 768),)  # [batch_size, num_tokens, embed_dim]
    tt = (torch.rand(1, 3, 224, 224, device=device),)  # 标准输入图像

    # 计算各部分FLOPs
    flops1 = FlopCountAnalysis(a1, t)
    print("Single-Head Attention FLOPs:", f"{flops1.total():,}")

    flops2 = FlopCountAnalysis(a2, t)
    print("Multi-Head Attention (12 heads) FLOPs:", f"{flops2.total():,}")

    # 计算整个ViT Base模型的FLOPs
    flops3 = FlopCountAnalysis(model, tt)
    total_flops = flops3.total()
    print("ViT Base模型总FLOPs:", f"{total_flops:,}")
    print("ViT Base模型总FLOPs (GFLOPs):", f"{total_flops / 1e9:.2f}")

if __name__ == '__main__':
    main()
