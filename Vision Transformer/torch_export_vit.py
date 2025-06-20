import torch
import numpy as np
import json
import os
import sys
import quant
from vit_model import vit_base_patch16_224_in21k_Qua, vit_base_patch16_224_in21k

MODEL_PATH = r"D:\python\PycharmProjects\VIT_pretrained_weights\vit_base_patch16_224_in21k.pth"
NUM_CLASSES = 21843  # 根据你的需求调整类别数
USE_QUANTIZED = False  # 是否使用量化版本

# 导出PyTorch模型的参数和配置信息，以便在硬件上部署

# 加载量化模型（小型量化网络）
# model = mymodel.UltraNetQua()
# model = mymodel.TempNetQua()
# print(model)
def load_vit_model():
    """加载Vision Transformer模型"""
    if USE_QUANTIZED:
        # 加载量化版本的ViT模型
        model = vit_base_patch16_224_in21k_Qua(num_classes=NUM_CLASSES, has_logits=False)
        print("已创建量化版本的ViT模型")
    else:
        # 加载普通版本的ViT模型
        model = vit_base_patch16_224_in21k(num_classes=NUM_CLASSES, has_logits=False)
        print("已创建普通版本的ViT模型")


    # 加载预训练权重（4-bit权重量化、4-bit激活量化）
    # model.load_state_dict(torch.load('ultranet_4w4a.pt', map_location='cpu', weights_only=False)['model'])
    # model.load_state_dict(torch.load('model.pkl', map_location='cpu'))
    if os.path.exists(MODEL_PATH):
        try:
            # 尝试不同的加载方式
            checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)

            # 检查checkpoint的格式
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
               state_dict = checkpoint['model']
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            # 处理键名不匹配的问题
            model_dict = model.state_dict()
            filtered_state_dict = {}

            for k, v in state_dict.items():
                # 去掉可能的前缀
                key = k.replace('module.', '')
                if key in model_dict:
                    if model_dict[key].shape == v.shape:
                        filtered_state_dict[key] = v
                    else:
                        print(f"跳过形状不匹配的参数: {key}, 模型: {model_dict[key].shape}, 权重: {v.shape}")
                else:
                    print(f"跳过未找到的参数: {key}")

            # 加载匹配的权重
            model.load_state_dict(filtered_state_dict, strict=False)
            print(f"成功加载预训练权重，匹配参数数量: {len(filtered_state_dict)}")

        except Exception as e:
            print(f"加载预训练权重时出错: {e}")
            print("使用随机初始化的权重")
    else:
        print(f"未找到预训练权重文件: {MODEL_PATH}")
        print("使用随机初始化的权重")

    return model


def generate_vit_config(model, img_size=224):
    """生成ViT模型配置信息"""
    config = {}

    # 基本配置
    config['model_type'] = 'vision_transformer'
    config['img_size'] = img_size
    config['quantized'] = USE_QUANTIZED

    # 从模型中提取配置
    if hasattr(model, 'patch_embed'):
        config['patch_size'] = model.patch_embed.patch_size[0] if hasattr(model.patch_embed, 'patch_size') else 16
        config['embed_dim'] = model.embed_dim
        config['num_patches'] = model.patch_embed.num_patches

    if hasattr(model, 'blocks'):
        config['depth'] = len(model.blocks)
        # 获取第一个block的配置
        first_block = model.blocks[0]
        if hasattr(first_block, 'attn'):
            config['num_heads'] = first_block.attn.num_heads
        if hasattr(first_block, 'mlp'):
            config['mlp_ratio'] = 4.0  # 默认值

    config['num_classes'] = model.num_classes if hasattr(model, 'num_classes') else NUM_CLASSES

    return config

#   config.json：
'''
{
    "conv_0": {
        "in_shape": [3, 160, 320],
        "out_shape": [16, 158, 318],
        "k": 3,
        "s": 1,
        "p": 0
    },
    "pool_0": {
        "in_shape": [16, 158, 318],
        "out_shape": [16, 79, 159],
        "p": 2
    },
    "linear_0": {
        "in_len": 1024,
        "out_len": 10
    }
}
'''


def generate_vit_params(model):
    """导出ViT模型参数"""
    params_dict = {}
    cnt = 0

    print("开始导出模型参数...")

    for name, module in model.named_modules():
        # Patch Embedding层
        if hasattr(module, 'proj') and 'patch_embed' in name:
            if hasattr(module.proj, 'weight'):
                w = module.proj.weight.detach().numpy()
                params_dict[f'arr_{cnt}'] = w
                print(f"arr_{cnt}: {name}.proj.weight, shape: {w.shape}")
                cnt += 1

                if hasattr(module.proj, 'bias') and module.proj.bias is not None:
                    b = module.proj.bias.detach().numpy()
                    params_dict[f'arr_{cnt}'] = b
                    print(f"arr_{cnt}: {name}.proj.bias, shape: {b.shape}")
                    cnt += 1

        # Linear层 (包括量化的Linear层)
        elif isinstance(module, (torch.nn.Linear, quant.linear_Q_fn(4))) or \
                (hasattr(module, '__class__') and 'Linear' in str(type(module))):
            if hasattr(module, 'weight'):
                w = module.weight.detach().numpy()
                params_dict[f'arr_{cnt}'] = w
                print(f"arr_{cnt}: {name}.weight, shape: {w.shape}")
                cnt += 1

                if hasattr(module, 'bias') and module.bias is not None:
                    b = module.bias.detach().numpy()
                    params_dict[f'arr_{cnt}'] = b
                    print(f"arr_{cnt}: {name}.bias, shape: {b.shape}")
                    cnt += 1

        # LayerNorm层
        elif isinstance(module, torch.nn.LayerNorm):
            if hasattr(module, 'weight'):
                gamma = module.weight.detach().numpy()
                params_dict[f'arr_{cnt}'] = gamma
                print(f"arr_{cnt}: {name}.weight (gamma), shape: {gamma.shape}")
                cnt += 1

                beta = module.bias.detach().numpy()
                params_dict[f'arr_{cnt}'] = beta
                print(f"arr_{cnt}: {name}.bias (beta), shape: {beta.shape}")
                cnt += 1

    # 导出特殊参数 (cls_token, pos_embed)
    if hasattr(model, 'cls_token'):
        cls_token = model.cls_token.detach().numpy()
        params_dict[f'arr_{cnt}'] = cls_token
        print(f"arr_{cnt}: cls_token, shape: {cls_token.shape}")
        cnt += 1

    if hasattr(model, 'pos_embed'):
        pos_embed = model.pos_embed.detach().numpy()
        params_dict[f'arr_{cnt}'] = pos_embed
        print(f"arr_{cnt}: pos_embed, shape: {pos_embed.shape}")
        cnt += 1

    print(f"总共导出了 {cnt} 个参数数组")
    return params_dict

#    ultranet_4w4a.npz：
'''
arr_0: 卷积层0权重
arr_1: BN层0的gamma
arr_2: BN层0的beta
arr_3: BN层0的mean
arr_4: BN层0的var
arr_5: BN层0的eps
arr_6: 卷积层1权重
...
'''


def main():
    """主函数"""
    # 1. 加载模型
    model = load_vit_model()
    model.eval()  # 设置为评估模式

    # 2. 导出模型参数
    print("\n开始导出模型参数...")
    params_dict = generate_vit_params(model)

    # 3. 保存参数到NPZ文件
    output_npz = 'vit_quantized_params.npz' if USE_QUANTIZED else 'vit_params.npz'
    np.savez(output_npz, **params_dict)
    print(f"参数已保存到: {output_npz}")

    # 4. 生成配置信息
    print("\n生成模型配置信息...")
    config = generate_vit_config(model)
    print("模型配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # 5. 保存配置到JSON文件
    output_json = 'vit_config.json'
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    print(f"配置已保存到: {output_json}")

    # 6. 生成参数摘要
    print("\n参数摘要:")
    total_params = 0
    for key, value in params_dict.items():
        param_count = np.prod(value.shape)
        total_params += param_count
        print(f"  {key}: {value.shape} ({param_count} 参数)")

    print(f"\n总参数数量: {total_params:,}")
    print(f"模型大小估计: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
    if USE_QUANTIZED:
        print(f"量化后大小估计: {total_params * 0.5 / 1024 / 1024:.2f} MB (4-bit)")

    print("\n导出完成!")


if __name__ == "__main__":
    main()
