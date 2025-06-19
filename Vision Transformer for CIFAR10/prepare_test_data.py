import os
import pandas as pd
from PIL import Image
import numpy as np


def prepare_test_data(data_path):
    """准备测试数据，生成测试图片列表"""

    test_dir = os.path.join(data_path, 'test')

    if not os.path.exists(test_dir):
        print(f"错误: 测试目录不存在 {test_dir}")
        return None

    # 获取所有测试图片
    test_files = []
    for file in os.listdir(test_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            test_files.append(file)

    # 按文件名排序，确保结果可重现
    test_files.sort()

    print(f"找到 {len(test_files)} 个测试图片")
    # 检查数量是否正确
    if len(test_files) != 300000:
        print(f"警告: 测试图片数量为 {len(test_files)}，期望 300,000")

    # 创建测试图片信息DataFrame
    test_data = []
    for i, filename in enumerate(test_files):
        # 提取图片ID（去掉扩展名）
        image_id = os.path.splitext(filename)[0]
        test_data.append({
            'id': image_id,
            'filename': filename,
            'path': os.path.join(test_dir, filename)
        })

        if (i + 1) % 10000 == 0:
            print(f"已处理 {i + 1} 个文件")

    test_df = pd.DataFrame(test_data)

    # 保存测试文件列表
    test_list_path = os.path.join(data_path, 'test_files_list.csv')
    test_df.to_csv(test_list_path, index=False)
    print(f"测试文件列表已保存到: {test_list_path}")

    return test_df


if __name__ == '__main__':
    data_path = "D:/python/PycharmProjects/data/CIFAR-10"
    test_df = prepare_test_data(data_path)

    if test_df is not None:
        print(f"测试数据准备完成，共 {len(test_df)} 个图片")
        print("前5个文件:")
        print(test_df.head())