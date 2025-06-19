import os
import math
import argparse

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from my_dataset import MyDataSet
from vit_model import vit_base_patch16_224_in21k as create_model
from utils import read_split_data, train_one_epoch, evaluate


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # print(f"使用设备: {device}")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter()

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    # CIFAR-10图片预处理 (32x32 -> 224x224)
    data_transform = {
        "train": transforms.Compose([
            transforms.Resize((224, 224)),  # CIFAR-10是32x32，需要resize到224x224
            transforms.RandomHorizontalFlip(p=0.5),     # 0.5概率随机水平翻转，增加数据多样性，提高模型泛化能力
            transforms.RandomRotation(10),      # 随机旋转±10度，模拟现实中拍摄角度的微小变化，增强模型对旋转的鲁棒性
            # 0.8-1.2调整图片的亮度、对比度、饱和度，-0.1 - 0.1调整色调，模拟不同光照条件和相机设置
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),  # 将PIL图片或numpy数组转换为PyTorch张量
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet标准化， 数据来源ImageNet数据集的实际统计值
        ]),
        "val": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    # 创建模型
    model = create_model(num_classes=args.num_classes, has_logits=False).to(device)     # 不需要Pre_Logits

    # 加载预训练权重
    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        # 删除不需要的权重
        del_keys = ['head.weight', 'head.bias'] if model.has_logits \
            else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
        for k in del_keys:
            if k in weights_dict:
                del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    # 冻结特征提取层
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head, pre_logits(不需要)外，其他权重全部冻结  只训练MLP Head中的权重
            if "head" not in name and "pre_logits" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # Cosine学习率调度
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    best_acc = 0.0
    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        scheduler.step()

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        # 保存最好的模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "./weights/model-best-cifar10.pth")

        # 保存当前epoch的模型
        torch.save(model.state_dict(), "./weights/model-{}-cifar10.pth".format(epoch))

        print(f"Epoch {epoch}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Best Acc: {best_acc:.4f}")

    print(f"训练完成！最佳验证准确率: {best_acc:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=10)  # CIFAR-10有10个类别
    parser.add_argument('--epochs', type=int, default=10)  # 10轮
    parser.add_argument('--batch-size', type=int, default=16)  # 适合CIFAR-10的batch size
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)

    # CIFAR-10数据集路径
    parser.add_argument('--data-path', type=str,
                        default="D:/python/PycharmProjects/data/CIFAR-10/train_organized")

    # 预训练权重路径
    parser.add_argument('--weights', type=str,
                        default='D:/python/PycharmProjects/VIT_pretrained_weights/vit_base_patch16_224_in21k.pth',
                        help='initial weights path')

    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=True)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()
    main(opt)