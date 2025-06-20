import torch
import torch.nn as nn
from vit_model import vit_base_patch16_224_in21k_Qua, vit_base_patch16_224_in21k


def test_model_quantization():
    """测试量化模型的功能"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建模型
    original_model = vit_base_patch16_224_in21k(num_classes=1000).to(device)
    quantized_model = vit_base_patch16_224_in21k_Qua(num_classes=1000).to(device)

    # 创建测试输入
    test_input = torch.randn(2, 3, 224, 224).to(device)

    # 测试前向传播
    print("Testing forward pass...")

    with torch.no_grad():
        original_output = original_model(test_input)
        quantized_output = quantized_model(test_input)

    print(f"Original model output shape: {original_output.shape}")
    print(f"Quantized model output shape: {quantized_output.shape}")

    # 比较模型参数数量
    original_params = sum(p.numel() for p in original_model.parameters())
    quantized_params = sum(p.numel() for p in quantized_model.parameters())

    print(f"Original model parameters: {original_params:,}")
    print(f"Quantized model parameters: {quantized_params:,}")

    # 测试梯度
    print("Testing backward pass...")
    quantized_model.train()
    loss_fn = nn.CrossEntropyLoss()
    target = torch.randint(0, 1000, (2,)).to(device)

    output = quantized_model(test_input)
    loss = loss_fn(output, target)
    loss.backward()

    print("Backward pass successful!")
    print(f"Loss: {loss.item():.4f}")


if __name__ == "__main__":
    test_model_quantization()