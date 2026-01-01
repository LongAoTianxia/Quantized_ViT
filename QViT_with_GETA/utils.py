import os
import sys
import json
import pickle
import random

import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F

import matplotlib.pyplot as plt
from only_train_once.transform import TensorTransform, tensor_transformation_param_group

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

    json_path = '../../Desktop/ViT_GETA/class_indices.json'
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


def one_hot(y: torch.Tensor, num_classes: int, smoothing_eps: float = None) -> torch.Tensor:
    """
    y: [N]，类别索引；返回 one-hot 或带 label smoothing 的 one-hot。
    """
    one_hot_y = F.one_hot(y, num_classes).float()
    if smoothing_eps is None:
        return one_hot_y
    v1 = 1 - smoothing_eps + smoothing_eps / float(num_classes)
    v0 = smoothing_eps / float(num_classes)
    new_y = one_hot_y * (v1 - v0) + v0
    return new_y


def cross_entropy_onehot_target(logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Qtest_ViT 中使用的 one-hot 交叉熵形式：target 必须是 one-hot。:contentReference[oaicite:3]{index=3}
    """
    prob_logit = F.log_softmax(logit, dim=1)
    loss = -(target * prob_logit).sum(dim=1).mean()
    return loss


def mixup_func(inputs: torch.Tensor,
               targets: torch.Tensor,
               alpha: float = 0.2):
    """
    Qtest_ViT 里使用的 mixup 实现：对输入和 one-hot target 做线性插值。:contentReference[oaicite:4]{index=4}
    """
    gamma = np.random.beta(alpha, alpha)
    perm = torch.randperm(inputs.size(0), device=inputs.device)
    perm_input = inputs[perm]
    perm_target = targets[perm]
    mixed_inputs = inputs.mul_(gamma).add_(1 - gamma, perm_input)
    mixed_targets = targets.mul_(gamma).add_(1 - gamma, perm_target)
    return mixed_inputs, mixed_targets


def compute_group_lasso_loss_from_geta(optimizer, device):
    """
    使用 GETA 优化器的 param_groups 和 tensor_transformation_param_group
    计算 Group Lasso 正则项
    """
    gl_loss = torch.zeros(1, device=device)

    # 遍历所有参数组
    for group in optimizer.param_groups:
        # 只对可剪枝、非辅助组施加正则
        if not group.get("is_prunable", False) or group.get("is_auxiliary", False):
            continue

        num_groups = group["num_groups"]

        for param, p_transform in zip(group["params"], group["p_transform"]):
            if p_transform == TensorTransform.NO_PRUNE:
                continue

            # 通过 only_train_once 的工具，把 param 按 group 维度重排
            # param_t: [num_groups, group_size, ...]
            param_t = tensor_transformation_param_group(param, p_transform, group)
            # 把后面的维度展平为 group_size
            param_t = param_t.view(num_groups, -1)
            if "gl_scale" in group:
                scale = group["gl_scale"].to(device)
                if scale.numel() != num_groups:
                    # 尺寸不匹配时，给个 warning，并退回无权重 Lasso，避免直接炸 nan
                    print(f"[WARN][GL] gl_scale shape {scale.shape} != num_groups {num_groups}, "
                          f"fallback to unscaled group lasso.")
                    gl_loss = gl_loss + torch.norm(param_t, dim=1).sum()
                else:
                    gl_loss = gl_loss + (torch.norm(param_t, dim=1) * scale).sum()
            else:
                # 对每个 group 计算 L2 范数，然后求和
                gl_loss = gl_loss + torch.norm(param_t, dim=1).sum()

    return gl_loss


def train_one_epoch(model, optimizer, data_loader, device, epoch,
                    mix_up: bool = False,
                    label_smooth: bool = False,
                    inferred_num_classes: int = None,
                    teacher_model=None,
                    kd_alpha: float = 0.5,
                    kd_temperature: float = 4.0,
                    use_group_lasso: bool = False,
                    group_lasso_lambda: float = 1e-4,
                    gl_start_epoch: int = 0,
                    log_interval: int = 100
                    ):
    model.train()
    if label_smooth:
        criterion = cross_entropy_onehot_target
    else:
        criterion = torch.nn.CrossEntropyLoss()

    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数

    ce_loss_epoch = torch.zeros(1).to(device)
    kd_loss_epoch = torch.zeros(1).to(device)
    gl_loss_epoch = torch.zeros(1).to(device)

    optimizer.zero_grad()

    sample_num = 0
    total_steps = len(data_loader)
    # tqdm 只在交互式终端下显示，nohup/重定向时自动关闭，避免刷屏
    data_loader = tqdm(
        data_loader,
        file=sys.stdout,
        total=total_steps,
        dynamic_ncols=True,
        disable=not sys.stdout.isatty()
    )

    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        hard_labels = labels.to(device)
        # 构造用于 loss 的 target（可能是 one-hot / mixup）
        if mix_up or label_smooth:
            with torch.no_grad():
                if inferred_num_classes is None:
                    inferred_num_classes = int(labels.max().item()) + 1

                if label_smooth and not mix_up:
                    targets = one_hot(labels, num_classes=inferred_num_classes, smoothing_eps=0.1)
                elif mix_up and not label_smooth:
                    targets = one_hot(labels, num_classes=inferred_num_classes)
                    images, targets = mixup_func(images, targets)
                else:  # mix_up and label_smooth
                    targets = one_hot(labels, num_classes=inferred_num_classes, smoothing_eps=0.1)
                    images, targets = mixup_func(images, targets)
        else:
            targets = labels

        images = images.to(device)
        targets = targets.to(device)

        pred = model(images)
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        ce_loss = criterion(pred, targets)
        ce_loss_epoch += ce_loss.detach()

        # 知识蒸馏 KD loss
        if (teacher_model is not None) and (kd_alpha > 0.0):
            with torch.no_grad():
                teacher_logits = teacher_model(images)
            T = kd_temperature
            log_p_student = F.log_softmax(pred / T, dim=1)
            p_teacher = F.softmax(teacher_logits / T, dim=1)
            kd_loss = F.kl_div(log_p_student, p_teacher, reduction='batchmean') * (T * T)
            kd_loss_epoch += kd_loss.detach()

            # 总 loss： (1-alpha)*CE + alpha*KD
            loss = (1.0 - kd_alpha) * ce_loss + kd_alpha * kd_loss
        else:
            loss = ce_loss

        # Group Lasso loss
        gl_enabled = use_group_lasso and (epoch >= gl_start_epoch) and (group_lasso_lambda > 0.0)
        if gl_enabled:
            gl_loss = compute_group_lasso_loss_from_geta(optimizer, device)
            gl_loss_epoch += gl_loss.detach()
            loss = loss + group_lasso_lambda * gl_loss
        else:
            gl_loss = None

        loss.backward()
        optimizer.grad_clipping()

        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        """
        if ((step + 1) % log_interval == 0) or (step + 1 == total_steps):
            data_loader.set_description(
                "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(
                    epoch,
                    accu_loss.item() / (step + 1),
                    accu_num.item() / sample_num
                )
            )
        """

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    num_batches = step + 1
    avg_total_loss = accu_loss.item() / num_batches
    avg_ce_loss = ce_loss_epoch.item() / num_batches
    # 如果未启用 KD / GL，对应平均值就设为 0，方便打印
    avg_kd_loss = kd_loss_epoch.item() / num_batches if ((teacher_model is not None) and (kd_alpha > 0.0)) else 0.0
    avg_gl_loss = gl_loss_epoch.item() / num_batches if gl_enabled else 0.0
    avg_acc = accu_num.item() / sample_num
    return avg_total_loss, avg_acc, avg_ce_loss, avg_kd_loss, avg_gl_loss


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, log_interval: int = 100):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    total_steps = len(data_loader)
    data_loader = tqdm(
        data_loader,
        file=sys.stdout,
        total=total_steps,
        dynamic_ncols=True,
        disable=not sys.stdout.isatty()
    )
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        if ((step + 1) % log_interval == 0) or (step + 1 == total_steps):
            data_loader.set_description(
                "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(
                    epoch,
                    accu_loss.item() / (step + 1),
                    accu_num.item() / sample_num
                )
            )

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num
