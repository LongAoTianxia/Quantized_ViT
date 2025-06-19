import pandas as pd
import os


def check_cifar10_data():
    data_path = "D:/python/PycharmProjects/data/CIFAR-10"

    print("=== CIFAR-10 数据集结构检查 ===")

    # 检查目录是否存在
    if not os.path.exists(data_path):
        print(f"错误: 数据集目录不存在 {data_path}")
        return

    print(f"数据集目录: {data_path}")
    files = os.listdir(data_path)
    print(f"目录内容: {files}")

    # 检查标签文件
    labels_file = os.path.join(data_path, 'trainLabels.csv')
    if os.path.exists(labels_file):
        print(f"\n找到标签文件: {labels_file}")

        # 读取标签文件
        df = pd.read_csv(labels_file)
        print(f"标签文件形状: {df.shape}")
        print(f"列名: {list(df.columns)}")

        # 显示前几行
        print("\n前5行数据:")
        print(df.head())

        # 检查标签列
        if 'label' in df.columns:
            print(f"\n标签列数据类型: {df['label'].dtype}")
            unique_labels = sorted(df['label'].unique())
            print(f"唯一标签: {unique_labels}")
            print(f"标签数量: {len(unique_labels)}")

            # 统计每个标签的数量
            print("\n各标签数量统计:")
            label_counts = df['label'].value_counts().sort_index()
            for label, count in label_counts.items():
                print(f"  {label}: {count}")

    else:
        print(f"未找到标签文件: {labels_file}")

    # 检查训练图片目录
    train_dir = os.path.join(data_path, 'train')
    if os.path.exists(train_dir):
        print(f"\n找到训练图片目录: {train_dir}")
        train_files = os.listdir(train_dir)
        print(f"训练图片数量: {len(train_files)}")
        if train_files:
            print(f"示例文件名: {train_files[:5]}")
    else:
        print(f"未找到训练图片目录: {train_dir}")


if __name__ == '__main__':
    check_cifar10_data()