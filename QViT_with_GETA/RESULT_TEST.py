import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from vit_model import vit_base_patch16_224_in21k as create_model


def build_cifar100_test_loader(root: str, batch_size: int):
    """Build CIFAR-100 test dataloader with CIFAR-100 stats."""
    normalize = transforms.Normalize((0.5071, 0.4865, 0.4409),
                                     (0.2673, 0.2564, 0.2762))
    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    test_dataset = datasets.CIFAR100(
        root=root, train=False, download=False, transform=test_transform)

    # Windows 的 DataLoader 多进程容易在启动时反复导入主模块；按需限制 workers。
    max_workers = min([os.cpu_count() or 0, batch_size if batch_size > 1 else 0, 8])
    workers = 0 if os.name == "nt" else max_workers
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max_workers,
        pin_memory=True
    )
    return test_loader, test_dataset.classes


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Path where CIFAR-100 is stored (set download=True in train.py if missing).
    test_data_root = r"/root/autodl-tmp/data"
    assert os.path.exists(test_data_root), (
        f"Test data path: '{test_data_root}' does not exist. "
        "Download CIFAR-100 to this folder or update the path."
    )

    batch_size = 64
    test_loader, class_names = build_cifar100_test_loader(test_data_root, batch_size)
    print(f"Loaded CIFAR-100 test set with {len(test_loader.dataset)} images.")

    # Load model trained for CIFAR-100 (num_classes=100).
    model_weight_path = "/root/autodl-tmp/geta_vit/geta_output/save/VisionTransformer_compressed.pt"
    assert os.path.exists(model_weight_path), f"Model weight path: '{model_weight_path}' does not exist."

    model = torch.load(model_weight_path).to(device)
    model.eval()
    print("Model loaded successfully!")
    # fc1 = model.blocks[0].mlp.fc1
    # fc2 = model.blocks[0].mlp.fc2
    # print(fc1.weight.shape, fc2.weight.shape)

    top1_correct = 0
    top5_correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            top1_preds = probs.argmax(dim=1)
            top1_correct += (top1_preds == labels).sum().item()

            top5_preds = torch.topk(probs, k=min(5, probs.shape[1]), dim=1).indices
            top5_correct += (top5_preds == labels.unsqueeze(1)).any(dim=1).sum().item()

            total += labels.size(0)

    if total == 0:
        print("No images were processed successfully!")
        return

    top1_accuracy = 100 * top1_correct / total
    top5_accuracy = 100 * top5_correct / total

    print("\n" + "=" * 50)
    print("Evaluation Results:")
    print(f"Total images: {total}")
    print(f"Top-1 Accuracy: {top1_accuracy:.2f}% ({top1_correct}/{total})")
    print(f"Top-5 Accuracy: {top5_accuracy:.2f}% ({top5_correct}/{total})")
    print("Classes order:", class_names)
    print("=" * 50)

    result_path = "测试精度.txt"
    with open(result_path, 'w') as f:
        f.write("Evaluation Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Test data path: {test_data_root}\n")
        f.write(f"Model weight path: {model_weight_path}\n")
        f.write(f"Total images: {total}\n")
        f.write(f"Top-1 Accuracy: {top1_accuracy:.2f}% ({top1_correct}/{total})\n")
        f.write(f"Top-5 Accuracy: {top5_accuracy:.2f}% ({top5_correct}/{total})\n")
        f.write(f"Classes order: {class_names}\n")
        f.write("=" * 50 + "\n")

    print(f"\nResults saved to {result_path}")


if __name__ == '__main__':

    main()
