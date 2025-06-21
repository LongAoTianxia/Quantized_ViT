import os
import math
import argparse
import tarfile  # 新增导入
import tempfile  # 新增导入

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from my_dataset import MyDataSet
from vit_model import vit_base_patch16_224_in21k_Qua as create_model
from utils import read_split_data, train_one_epoch, evaluate

def extract_tar_gz(tar_path, extract_dir):
    """解压 .tar.gz 文件到指定目录"""
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(path=extract_dir)
    print(f"Extracted dataset to: {extract_dir}")

def main(args):
    # 定义分阶段QAT的bit宽度设置
    bit_stages = [(8, 8, 8), (6, 6, 6), (4, 4, 4)]
    stage_epochs = [args.epochs // 3, args.epochs // 3, args.epochs - 2 * (args.epochs // 3)]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    """
    # 新增：处理 .tar.gz 数据集 ---------------------------------
    original_data_path = args.data_path
    temp_extract_dir = None

    if args.data_path.endswith('.tar.gz'):
        # 创建临时解压目录
        temp_extract_dir = tempfile.mkdtemp()
        print(f"Created temp directory: {temp_extract_dir}")

        # 解压数据集
        extract_tar_gz(args.data_path, temp_extract_dir)
        args.data_path = temp_extract_dir  # 使用解压后的路径
    # ---------------------------------------------------------
    """
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter()

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    # 图片预处理 （224x224x3大小，并进行其他预处理）
    """
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}
    """
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet标准化
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
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
    model = create_model(num_classes=args.num_classes, has_logits=False).to(device)  # 不需要Pre_Logits

    # 加载预训练权重
    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)

        # 如果是原始VIT权重，需要适配量化模型
        if 'model_state_dict' not in weights_dict:
            # 处理原始VIT权重
            del_keys = ['head.weight', 'head.bias'] if model.has_logits \
                else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
            for k in del_keys:
                if k in weights_dict:
                    del weights_dict[k]
            print(model.load_state_dict(weights_dict, strict=False))
        else:
            # 加载量化模型权重
            model.load_state_dict(weights_dict['model_state_dict'])

    # 冻结特征提取层
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head, pre_logits外，其他权重全部冻结  只训练MLP Head中的权重
            if "head" not in name and "pre_logits" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]

    # 量化模型可能需要更小的学习率
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)

    # 分阶段QAT主循环
    start_epoch = 0
    best_acc = 0.0
    for stage, (w_bit, in_bit, out_bit) in enumerate(bit_stages):
        print(f"\n=== QAT Stage {stage + 1}: {w_bit}bit ===")
        model.set_quant_bit(w_bit, in_bit, out_bit)
        # 可选：每阶段重置学习率调度器
        lf = lambda x: ((1 + math.cos(x * math.pi / stage_epochs[stage])) / 2) * (1 - args.lrf) + args.lrf  # cosine
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
        # 每阶段保存best权重
        stage_best_acc = 0.0
        for epoch in range(stage_epochs[stage]):
            global_epoch = start_epoch + epoch
            # 训练
            train_loss, train_acc = train_one_epoch(model=model, optimizer=optimizer, data_loader=train_loader,
                                                    device=device, epoch=global_epoch,
                                                    use_mixed_precision=args.mixed_precision)
            scheduler.step()
            # 验证
            val_loss, val_acc = evaluate(model=model, data_loader=val_loader, device=device, epoch=global_epoch)
            # 记录日志
            tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
            tb_writer.add_scalar(tags[0], train_loss, global_epoch)
            tb_writer.add_scalar(tags[1], train_acc, global_epoch)
            tb_writer.add_scalar(tags[2], val_loss, global_epoch)
            tb_writer.add_scalar(tags[3], val_acc, global_epoch)
            tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], global_epoch)
            # 每阶段定期保存检查点
            if (epoch + 1) % 5 == 0:
                torch.save(model.state_dict(), f"./weights/model-{global_epoch}.pth")
            # 保存最佳
            if val_acc > stage_best_acc:
                stage_best_acc = val_acc
                torch.save(model.state_dict(), f"./weights/best_model_stage{stage + 1}.pth")
                print(f"Best model in stage {stage + 1} saved with accuracy: {stage_best_acc:.4f}")
            if val_acc > best_acc:
                best_acc = val_acc
        print(f"Stage {stage + 1} finished. Best acc: {stage_best_acc:.4f}")
        start_epoch += stage_epochs[stage]

    print(f"Progressive QAT completed. Best validation accuracy: {best_acc:.4f}")
    """
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    best_acc = 0.0
    for epoch in range(args.epochs):
        # 训练
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch,
                                                use_mixed_precision=args.mixed_precision)

        scheduler.step()

        # 验证
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        # 记录日志
        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        # 定期保存检查点
        if (epoch + 1) % 5 ==0:
            torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "./weights/best_model.pth")
            print(f"Best model saved with accuracy: {best_acc:.4f}")

    print(f"Training completed. Best validation accuracy: {best_acc:.4f}")
    """



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=15)     # 修改，种类数num_classes
    parser.add_argument('--epochs', type=int, default=50)       # 量化模型可能需要更多轮次 10
    parser.add_argument('--batch-size', type=int, default=16)    # 8
    parser.add_argument('--lr', type=float, default=0.00005)  # 更小的学习率 0.001
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--weight-decay', type=float, default=0.05)
        parser.add_argument('--mixed-precision', type=bool, default=False)

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default="D:/python/PycharmProjects/data/ball_15/train")
    parser.add_argument('--model-name', default='', help='create model name')

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default="D:/python/PycharmProjects/VIT_pretrained_weights/vit_base_patch16_224_in21k.pth",
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
