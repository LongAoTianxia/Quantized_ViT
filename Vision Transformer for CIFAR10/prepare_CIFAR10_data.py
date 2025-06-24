import os
import pandas as pd
import shutil
from pathlib import Path
import zipfile
import py7zr

# 数据集预处理，解压，重新组织数据集结构

def extract_7z_files(data_path):
    """解压7z文件"""
    train_7z = os.path.join(data_path, 'train.7z')
    test_7z = os.path.join(data_path, 'test.7z')

    # 解压训练数据
    if os.path.exists(train_7z):
        with py7zr.SevenZipFile(train_7z, mode='r') as z:
            z.extractall(path=data_path)
        print("训练数据解压完成")

    # 解压测试数据
    if os.path.exists(test_7z):
        with py7zr.SevenZipFile(test_7z, mode='r') as z:
            z.extractall(path=data_path)
        print("测试数据解压完成")


def organize_cifar10_data(data_path):
    """重新组织CIFAR-10数据集结构"""

    # CIFAR-10类别名称列表（按字母顺序排列，确保与标签文件一致）
    class_names = [
        'airplane',
        'automobile',
        'bird',
        'cat',
        'deer',
        'dog',
        'frog',
        'horse',
        'ship',
        'truck'
    ]

    # 创建训练集目录结构
    train_organized_path = os.path.join(data_path, 'train_organized')
    os.makedirs(train_organized_path, exist_ok=True)

    for class_name in class_names:
        os.makedirs(os.path.join(train_organized_path, class_name), exist_ok=True)

    # 读取训练标签
    train_labels_path = os.path.join(data_path, 'trainLabels.csv')
    if os.path.exists(train_labels_path):
        train_labels = pd.read_csv(train_labels_path)

        # 查看标签文件的前几行，了解数据格式
        print("标签文件前5行:")
        print(train_labels.head())
        print(f"标签列的数据类型: {train_labels['label'].dtype}")
        print(f"唯一标签: {sorted(train_labels['label'].unique())}")

        train_images_path = os.path.join(data_path, 'train')

        print("正在组织训练数据...")

        # 统计每个类别的数量
        class_counts = {}
        for class_name in class_names:
            class_counts[class_name] = 0

        # 处理每个图片
        for idx, row in train_labels.iterrows():
            image_id = row['id']
            label = row['label']

            # 如果label是字符串，直接使用；如果是数字，需要映射
            if isinstance(label, str):
                class_name = label
            else:
                # 如果是数字标签，使用索引映射
                class_name = class_names[int(label)]

            # 检查类别名称是否有效
            if class_name not in class_names:
                print(f"警告: 未知类别 '{class_name}', 跳过图片 {image_id}")
                continue

            src_path = os.path.join(train_images_path, f'{image_id}.png')
            dst_path = os.path.join(train_organized_path, class_name, f'{image_id}.png')

            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
                class_counts[class_name] += 1
            else:
                print(f"警告: 图片文件不存在 {src_path}")

            # 显示进度
            if (idx + 1) % 5000 == 0:
                print(f"已处理 {idx + 1} 张图片")

        print(f"训练数据组织完成，保存在: {train_organized_path}")

        # 统计每个类别的图片数量
        print("\n各类别图片数量统计:")
        total_images = 0
        for class_name in class_names:
            count = class_counts[class_name]
            total_images += count
            print(f"{class_name}: {count} 张图片")

        print(f"\n总计: {total_images} 张图片")

    else:
        print(f"错误: 找不到标签文件 {train_labels_path}")
        return None

    return train_organized_path


def check_data_structure(data_path):
    """检查数据集的结构"""
    print(f"检查数据集目录: {data_path}")

    if not os.path.exists(data_path):
        print(f"错误: 数据集目录不存在 {data_path}")
        return

    files = os.listdir(data_path)
    print(f"目录中的文件/文件夹: {files}")

    # 检查标签文件
    train_labels_path = os.path.join(data_path, 'trainLabels.csv')
    if os.path.exists(train_labels_path):
        print(f"找到标签文件: {train_labels_path}")
        df = pd.read_csv(train_labels_path)
        print(f"标签文件形状: {df.shape}")
        print(f"列名: {df.columns.tolist()}")

        if 'label' in df.columns:
            unique_labels = sorted(df['label'].unique())
            print(f"唯一标签: {unique_labels}")
            print(f"标签数量: {len(unique_labels)}")
    else:
        print(f"未找到标签文件: {train_labels_path}")


if __name__ == '__main__':
    data_path = "D:/python/PycharmProjects/data/CIFAR-10"

    # 首先检查数据结构
    check_data_structure(data_path)

    # 解压7z文件
    try:
        extract_7z_files(data_path)
    except Exception as e:
        print(f"解压文件时出错: {e}")

    # 重新组织数据结构
    try:
        train_organized_path = organize_cifar10_data(data_path)
        if train_organized_path:
            print("数据预处理完成！")
            print(f"训练数据路径: {train_organized_path}")
        else:
            print("数据预处理失败！")
    except Exception as e:
        print(f"组织数据时出错: {e}")
        import traceback

        traceback.print_exc()
