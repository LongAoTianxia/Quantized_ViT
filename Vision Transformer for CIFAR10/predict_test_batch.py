import os                          # 文件和目录操作
import json                        # JSON文件读写，用于类别索引
import pandas as pd                # 数据处理和CSV文件操作
import torch                       # PyTorch深度学习框架
from PIL import Image              # 图像处理库
from torchvision import transforms # 图像预处理变换
from torch.utils.data import Dataset, DataLoader  # 数据集和数据加载器
import numpy as np                 # 数值计算
from tqdm import tqdm              # 进度条显示
import glob                        # 文件路径模式匹配

from vit_model import vit_base_patch16_224_in21k as create_model


class TestDataset(Dataset):

    def __init__(self, test_dir, transform=None):
        """
        初始化测试数据集
            est_dir: 测试图片目录路径
            transform: 图像预处理变换
        """
        self.transform = transform

        print("正在扫描测试图片...")
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
        self.image_paths = []

        # 遍历所有图像扩展名，收集测试图片路径
        for ext in image_extensions:
            self.image_paths.extend(glob.glob(os.path.join(test_dir, ext)))

        # 按文件名排序,确保预测结果的可重现性
        self.image_paths.sort()
        print(f"找到 {len(self.image_paths)} 张测试图片")

    def __len__(self):
        """返回数据集大小"""
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]  # 获取图片路径

        # 确保ID为字符串格式
        filename = os.path.basename(image_path)  # 提取文件名
        image_id = os.path.splitext(filename)[0]    # 去掉文件扩展名得到ID
        # 强制转换为字符串，避免任何张量或数字类型
        image_id = str(image_id)

        try:
            image = Image.open(image_path)
            # 确保图片是RGB格式
            if image.mode != 'RGB':
                image = image.convert('RGB')

            if self.transform:
                image = self.transform(image)

            return image, image_id

        except Exception as e:
            print(f"处理图片 {image_path} 时出错: {e}")
            # 创建一个黑色的替代图片
            dummy_image = Image.new('RGB', (32, 32), (0, 0, 0))
            if self.transform:
                dummy_image = self.transform(dummy_image)
            return dummy_image, image_id


def predict_and_create_submission():
    """预测函数"""

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 数据预处理
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 测试数据路径
    test_dir = "D:/python/PycharmProjects/data/CIFAR-10/test"

    if not os.path.exists(test_dir):
        print(f"错误：测试目录不存在 {test_dir}")
        return

    # 创建测试数据集
    test_dataset = TestDataset(test_dir, transform=data_transform)
    # batch_size=32，不打乱顺序，使用4个工作进程
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # 加载类别索引
    json_path = "D:/python/PycharmProjects/vision_transformer_for_CIFAR10/class_indices.json"
    if not os.path.exists(json_path):
        print("错误：找不到 class_indices.json 文件")
        return

    with open(json_path, "r") as f:
        class_indices = json.load(f)

    # 创建索引到类别名的映射，确保是字符串
    idx_to_class = {}
    for k, v in class_indices.items():
        idx_to_class[int(k)] = str(v)  # 强制转换为字符串

    # print("类别映射:")
    # for idx, name in idx_to_class.items():
    #     print(f"  {idx}: {name}")

    # 加载模型
    model = create_model(num_classes=10, has_logits=False).to(device)   # num_classes对应不同的数据集

    # 加载训练好的权重
    model_weight_path = "D:/python/PycharmProjects/vision_transformer_for_CIFAR10/weights/model-best-cifar10.pth"
    if not os.path.exists(model_weight_path):
        print(f" 错误：找不到模型权重文件 {model_weight_path}")
        return

    # 加载权重到模型中
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()    # 设置为评估模式（关闭dropout等）

    # 存储预测结果 - 使用列表存储，确保数据类型正确
    predictions = []    # 预测的类别名
    image_ids = []  # 图片ID

    print("开始预测...")
    with torch.no_grad():
        for batch_idx, (images, ids) in enumerate(tqdm(test_loader, desc="预测进度")):
            images = images.to(device)

            # 模型预测
            outputs = model(images)  # 获取模型输出logits
            probabilities = torch.softmax(outputs, dim=1)   # 转换为概率分布
            predicted_classes = torch.argmax(probabilities, dim=1)  # 获取预测类别索引

            # 正确转换预测结果
            for i, pred_class in enumerate(predicted_classes):
                # 将张量转换为Python整数，再转换为类别名字符串
                pred_idx = int(pred_class.item())  # 先转为Python int
                class_name = idx_to_class[pred_idx]  # 获取类别名

                # 确保都是字符串格式
                predictions.append(str(class_name))
                image_ids.append(str(ids[i]))  # ids[i] 已经是字符串，但再次确保

            # 每100批显示进度
            if (batch_idx + 1) % 100 == 0:
                processed = (batch_idx + 1) * test_loader.batch_size
                print(f"已处理 {processed:,} 张图片")

    print(f"\n 预测完成，共处理 {len(predictions):,} 张图片")

    # 验证数据类型
    print("\n 数据类型验证:")
    print(f"   image_ids 样例: {image_ids[:5]} (类型: {type(image_ids[0])})")
    print(f"   predictions 样例: {predictions[:5]} (类型: {type(predictions[0])})")

    # 确保所有数据都是字符串
    image_ids = [str(id_val) for id_val in image_ids]
    predictions = [str(pred) for pred in predictions]

    # 创建DataFrame
    submission_df = pd.DataFrame({
        'id': image_ids,
        'label': predictions
    })

    print(f"\n DataFrame 信息:")
    print(f"   形状: {submission_df.shape}")
    print(f"   列名: {list(submission_df.columns)}")
    print(f"   ID列数据类型: {submission_df['id'].dtype}")
    print(f"   Label列数据类型: {submission_df['label'].dtype}")

    # 检查是否有异常值
    print(f"\n 数据完整性检查:")

    # 检查缺失值
    missing_count = submission_df.isnull().sum().sum()
    print(f"   缺失值: {missing_count}")

    # 检查标签有效性
    valid_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                    'dog', 'frog', 'horse', 'ship', 'truck']
    invalid_labels = submission_df[~submission_df['label'].isin(valid_labels)]
    if len(invalid_labels) > 0:
        print(f"     发现无效标签: {invalid_labels['label'].unique()}")
    else:
        print(f"   所有标签都有效")

    # 按ID排序，确保顺序正确
    try:
        # 尝试按数字排序
        submission_df['id_numeric'] = pd.to_numeric(submission_df['id'], errors='coerce')
        if submission_df['id_numeric'].notna().all():
            submission_df = submission_df.sort_values('id_numeric').drop('id_numeric', axis=1)
            print(f"    按数字ID排序")
        else:
            submission_df = submission_df.sort_values('id')
            print(f"    按字符串ID排序")
    except Exception as e:
        print(f"     排序出错，使用默认顺序: {e}")

    # 重置索引
    submission_df = submission_df.reset_index(drop=True)

    # 保存提交文件
    submission_path = "./submission_cifar10.csv"

    # 使用明确的参数保存CSV
    submission_df.to_csv(
        submission_path,
        index=False,  # 不保存行索引
        encoding='utf-8',  # 使用UTF-8编码
        lineterminator='\n'  # 使用Unix换行符
    )

    print(f"\n 提交文件已保存: {submission_path}")

    return submission_df


if __name__ == '__main__':
    submission_df = predict_and_create_submission()