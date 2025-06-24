import os
import sys
import json
import pickle
import random
import torch.nn.functional as F
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt


def qkd_loss(student_logits, targets, teacher_logits, temperature=2.0, alpha=0.5):
    """    计算知识蒸馏损失函数   """
    # alpha: 权重系数, 0.5-0.9, 越大，越注重真实标签;越小，越注重teacher输出
    # temperature: 温度参数,2-4,平滑teacher和student输出分布，放大soft label中的小概率信息,越大，概率分布越平滑，teacher信息更丰富
    ce_loss = F.cross_entropy(student_logits, targets)  # 交叉熵损失
    assert teacher_logits.shape == student_logits.shape, \
        f"teacher_logits shape {teacher_logits.shape} != student_logits shape {student_logits.shape}"
    # 计算KL散度损失
    kd_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),  # 学生模型的softmax输出
        F.softmax(teacher_logits / temperature, dim=1),  # 教师模型的softmax输出
        reduction='batchmean'   # 批量平均
    ) * (temperature * temperature)  # 缩放KL散度损失, temperature: 温度参数
    return alpha * ce_loss + (1 - alpha) * kd_loss  # 加权损失函数

def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证各平台顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 排序，保证各平台顺序一致
        images.sort()
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."

    plot_image = False
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(flower_class)), flower_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def train_one_epoch(model, teacher_model, optimizer, data_loader, device, epoch, use_mixed_precision=False,
                    alpha=0.5, temperature=2.0):
    """量化模型的训练函数，支持混合精度训练和知识蒸馏"""
    model.train()
    teacher_model.eval()
    # loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    optimizer.zero_grad()

    # 添加混合精度支持
    if use_mixed_precision:
        scaler = torch.amp.GradScaler('cuda')

    sample_num = 0  # 累计样本数
    # 使用tqdm显示进度条
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        # 获取数据
        images, labels = data
        # 累计样本数
        sample_num += images.shape[0]

        with torch.no_grad():  # 不需要梯度计算
            with torch.amp.autocast('cuda') if use_mixed_precision else torch.no_grad():    # 使用混合精度
                teacher_logits = teacher_model(images.to(device))   # 教师模型的输出

        if use_mixed_precision:
            # 使用混合精度训练 ,with 指令符可以自动处理前向和反向传播的精度
            with torch.amp.autocast('cuda'):
                student_logits = model(images.to(device))   # 学生模型的输出
                # 检查学生模型和教师模型的输出形状是否一致
                assert student_logits.shape == teacher_logits.shape, \
                    f"Shape mismatch: student {student_logits.shape}, teacher {teacher_logits.shape}"
                loss = qkd_loss(student_logits, labels.to(device), teacher_logits, temperature, alpha)  # 计算知识蒸馏损失
            scaler.scale(loss).backward()   # 反向传播和优化
            scaler.step(optimizer)  # 更新参数
            scaler.update()  # 更新缩放器
        else:
            # 不使用混合精度训练
            student_logits = model(images.to(device))

            assert student_logits.shape == teacher_logits.shape, \
                f"Shape mismatch: student {student_logits.shape}, teacher {teacher_logits.shape}"
            loss = qkd_loss(student_logits, labels.to(device), teacher_logits, temperature, alpha)  # 计算知识蒸馏损失
            loss.backward()
            optimizer.step()    # 更新参数

        # 计算预测类别
        pred_classes = torch.max(student_logits, dim=1)[1]
        # 累计预测正确的样本数和损失
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        # 累计损失
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        # 检查损失是否为非有限值
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        # 清除梯度
        optimizer.zero_grad()

    # 返回平均损失和准确率
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()
    # 将模型设置为评估模式
    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


