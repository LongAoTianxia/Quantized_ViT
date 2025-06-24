import pandas as pd
import os

# 检查提交文件格式csv是否正确

def check_submission_format(submission_path):
    """检查提交文件格式是否正确"""

    if not os.path.exists(submission_path):
        print(f"错误: 提交文件不存在 {submission_path}")
        return False

    print("=== 提交文件格式检查 ===")

    # 读取提交文件
    try:
        df = pd.read_csv(submission_path)
        print("文件读取成功")
    except Exception as e:
        print(f"文件读取失败: {e}")
        return False

    # 检查行数
    expected_rows = 300000
    actual_rows = len(df)
    if actual_rows == expected_rows:
        print(f"行数正确: {actual_rows}")
    else:
        print(f"行数不匹配: 实际 {actual_rows}, 期望 {expected_rows}")

    # 检查列数和列名
    expected_cols = ['id', 'label']
    actual_cols = list(df.columns)
    if actual_cols == expected_cols:
        print(f"列名正确: {actual_cols}")
    else:
        print(f"列名不匹配: 实际 {actual_cols}, 期望 {expected_cols}")

    # 检查是否有缺失值
    missing_count = df.isnull().sum().sum()
    if missing_count == 0:
        print("无缺失值")
    else:
        print(f"发现 {missing_count} 个缺失值")

    # 检查标签是否有效
    valid_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                    'dog', 'frog', 'horse', 'ship', 'truck']

    invalid_labels = df[~df['label'].isin(valid_labels)]['label'].unique()
    if len(invalid_labels) == 0:
        print("所有标签都有效")
    else:
        print(f" 发现无效标签: {invalid_labels}")

    # 统计标签分布
    print(f"\n=== 标签分布 ===")
    label_counts = df['label'].value_counts().sort_index()
    for label, count in label_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {label}: {count:,} ({percentage:.1f}%)")

    # 显示文件大小
    file_size = os.path.getsize(submission_path) / (1024 * 1024)  # MB
    print(f"\n=== 文件信息 ===")
    print(f"文件大小: {file_size:.2f} MB")

    print(f"\n前5行预览:")
    print(df.head())

    return True


if __name__ == '__main__':
    submission_path = "./submission_cifar10_fixed.csv"
    check_submission_format(submission_path)
