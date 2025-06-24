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
from vit_model import vit_base_patch16_224_in21k as create_teacher_model  # teacher模型
from utils import read_split_data, train_one_epoch, evaluate

def extract_tar_gz(tar_path, extract_dir):
    """解压 .tar.gz 文件到指定目录"""
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(path=extract_dir)
    print(f"Extracted dataset to: {extract_dir}")

def main(args):
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
    # ========== 创建教师模型 ==========
    teacher_model = create_teacher_model(num_classes=args.num_classes, has_logits=False).to(device)
    teacher_model.eval()
    assert os.path.exists(args.teacher_weights), "teacher weights file not found!"
    teacher_weights = torch.load(args.teacher_weights, map_location=device)
    if 'model_state_dict' in teacher_weights:
        # 如果是原始VIT权重，需要适配量化模型
        teacher_model.load_state_dict(teacher_weights['model_state_dict'], strict=False)
    else:
        # 处理原始VIT权重
        teacher_model.load_state_dict(teacher_weights, strict=False)
    for p in teacher_model.parameters():
        # 冻结教师模型参数
        p.requires_grad = False
    # ================================
    if os.path.exists("./weights") is False:
        # 如果不存在weights目录，则创建
        os.makedirs("./weights")
    # tensorboard日志目录
    tb_writer = SummaryWriter()
    # 读取数据集
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    # 图片预处理 （224x224x3大小，并进行其他预处理） 数据集图像应大于224x224

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),  # 随机裁剪一块面积和宽高比随机的区域，并将其缩放到224x224,增加数据多样性，防止模型过拟合
                                     transforms.RandomHorizontalFlip(),  # 随机水平翻转
                                     transforms.ToTensor(),  # 转换为Tensor
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val": transforms.Compose([transforms.Resize(256),  # 调整图片大小为256x256
                                   transforms.CenterCrop(224),  # 中心裁剪224x224大小
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

    """

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
    optimizer = torch.optim.AdamW(pg, lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)

    n_blocks = len(model.blocks) # 12
    mlp_blocks = list(range(n_blocks))

    # ========== Warmup：先FP32训练 ==========
    print("==== Warmup FP32 Training (no quantization) ====")
    model.set_quant_bit(32, 32, 32, quantize_head=True, quantize_patch_embed=True,
                        quantize_attn_blocks=list(range(n_blocks)),
                        quantize_mlp_blocks=mlp_blocks)
    warmup_epochs = 3  # 可根据实际情况设为10~20
    warmup_optimizer = torch.optim.AdamW(pg, lr=0.002, weight_decay=args.weight_decay)
    for epoch in range(warmup_epochs):
        train_loss, train_acc = train_one_epoch(
            model=model,
            teacher_model=teacher_model,  # FP32模型
            optimizer=warmup_optimizer,
            data_loader=train_loader,
            device=device,
            epoch=epoch,
            use_mixed_precision=args.mixed_precision,
            alpha=0.7,  temperature=2.0     # 超参数。需要调整
        )
        val_loss, val_acc = evaluate(model=model, data_loader=val_loader, device=device, epoch=epoch)
        print(f"[FP32 warmup] epoch {epoch}: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")
    torch.save(model.state_dict(), "./weights/FP32_warmup.pth")  # 保存warmup后的模型

    # ========== 量化配置Sweep ==========
    sweep_configs = [
        # 量化范围配置: (描述, quantize_head, quantize_patch_embed, quantize_attn_blocks, quantize_mlp_blocks)
        ("head+MLP", True, False, [], mlp_blocks),
        ("head+MLP+last1Attn", True, False, [n_blocks - 1], mlp_blocks),
        ("head+MLP+last2Attn", True, False, list(range(n_blocks - 2, n_blocks)), mlp_blocks),
        ("head+MLP+last3Attn", True, False, list(range(n_blocks - 3, n_blocks)), mlp_blocks),
        ("head+MLP+last4Attn", True, False, list(range(n_blocks - 4, n_blocks)), mlp_blocks),
        ("head+MLP+last5Attn", True, False, list(range(n_blocks - 5, n_blocks)), mlp_blocks),
        ("head+MLP+last6Attn", True, False, list(range(n_blocks - 6, n_blocks)), mlp_blocks),
        ("head+MLP+last7Attn", True, False, list(range(n_blocks - 7, n_blocks)), mlp_blocks),
        ("head+MLP+last8Attn", True, False, list(range(n_blocks - 8, n_blocks)), mlp_blocks),
        ("head+MLP+last9Attn", True, False, list(range(n_blocks - 9, n_blocks)), mlp_blocks),
        ("head+MLP+last10Attn", True, False, list(range(n_blocks - 10, n_blocks)), mlp_blocks),
        ("head+MLP+last11Attn", True, False, list(range(n_blocks - 11, n_blocks)), mlp_blocks),
        ("head+MLP+allAttn", True, False, list(range(n_blocks)), mlp_blocks),
        ("all-main", True, True, list(range(n_blocks)), mlp_blocks),  # 全主干量化
    ]

    # 定义分阶段QAT的bit宽度设置
    # bit_stages = [(8, 8, 8), (6, 6, 6), (4, 4, 4)]
    # stage_epochs = [args.epochs // 3, args.epochs // 3, args.epochs - 2 * (args.epochs // 3)]
    # stage_epochs = [9, 12, 15]

    # 针对每个量化配置，定制训练轮数和初始学习率
    sweep_hyperparams = {
        # 每个key下加 "bit_epochs" 和 "bit_lrs"，分别为每个stage设置epochs/lr
        "head+MLP": {
            "epochs": 8, "lr": 9e-4,
            "bit_epochs": [6, 10, 8],  # 8bit, 6bit, 4bit
            "bit_lrs": [0.001, 0.0005, 0.0005]
        },
        "head+MLP+last1Attn": {
            "epochs": 10, "lr": 8e-4,
            "bit_epochs": [7, 9, 12],
            "bit_lrs": [0.002, 0.00045, 0.0005]
        },
        "head+MLP+last2Attn": {
            "epochs": 12, "lr": 7e-4,
            "bit_epochs": [10, 12, 10],
            "bit_lrs": [0.0015, 0.00035, 0.0005]
        },
        "head+MLP+last3Attn": {
            "epochs": 14, "lr": 6e-4,
            "bit_epochs": [9, 12, 10],
            "bit_lrs": [0.002, 0.0004, 0.00045]
        },
        "head+MLP+last4Attn": {
            "epochs": 16, "lr": 5e-4,
            "bit_epochs": [9, 12, 12],
            "bit_lrs": [0.001, 0.00045, 0.0005]
        },
        "head+MLP+last5Attn": {
            "epochs": 18, "lr": 4e-4,
            "bit_epochs": [10, 10, 12],
            "bit_lrs": [0.002, 0.0004, 0.00045]
        },
        "head+MLP+last6Attn": {     # 下降明显
            "epochs": 20, "lr": 3e-4,
            "bit_epochs": [10, 12, 12],
            "bit_lrs": [0.002, 0.00045, 0.00055]
        },
        "head+MLP+last7Attn": {     # 掉落严重
            "epochs": 22, "lr": 2.5e-4,
            "bit_epochs": [10, 10, 12],
            "bit_lrs": [0.0025, 0.0005, 0.00055]
        },
        "head+MLP+last8Attn": {     # 大幅掉落
            "epochs": 24, "lr": 2e-4,
            "bit_epochs": [10, 12, 14],
            "bit_lrs": [0.0025, 0.0005, 0.00045]
        },
        "head+MLP+last9Attn": {     # 大幅掉落
            "epochs": 26, "lr": 1.5e-4,
            "bit_epochs": [10, 12, 12],
            "bit_lrs": [0.0025, 0.0005, 0.00045]
        },
        "head+MLP+last10Attn": {
            "epochs": 28, "lr": 1e-4,
            "bit_epochs": [10, 12, 12],
            "bit_lrs": [0.0025, 0.00045, 0.00045]
        },
        "head+MLP+last11Attn": {
            "epochs": 30, "lr": 1e-4,
            "bit_epochs": [10, 12, 12],
            "bit_lrs": [0.0025, 0.00045, 0.0004]
        },
        "head+MLP+allAttn": {
            "epochs": 32, "lr": 1e-4,
            "bit_epochs": [11, 11, 12],
            "bit_lrs": [0.0025, 0.00045, 0.0005]
        },
        "all-main": {
            "epochs": 36, "lr": 8e-5,
            "bit_epochs": [12, 12, 14],
            "bit_lrs": [0.0025, 0.0004, 0.00045]
        }
    }

    # 定义分阶段QAT的bit宽度设置
    bit_stages = [(8, 8, 8), (6, 6, 6), (4, 4, 4)]
    # stage_epochs = [args.epochs // 3, args.epochs // 3, args.epochs - 2 * (args.epochs // 3)]
    # stage_epochs = [8, 10, 12]

    for sweep_idx, (desc, quantize_head, quantize_patch_embed, attn_blocks, mlp_blocks_) in enumerate(sweep_configs):

        print(f"\n==== Sweep {sweep_idx}: {desc} ====")
        # 每轮建议重新加载warmup后参数
        # torch.save(model.state_dict(), "./weights/warmup_model.pth")
        # model.load_state_dict(torch.load("./weights/warmup_model.pth"))
        best_acc = 0.0
        # 获取当前配置的训练轮数和学习率
        # cur_epochs = sweep_hyperparams.get(desc, {}).get("epochs", 12)
        # cur_lr = sweep_hyperparams.get(desc, {}).get("lr", 3e-4)
        bit_epochs = sweep_hyperparams.get(desc, {}).get("bit_epochs", [4, 4, 4])
        bit_lrs = sweep_hyperparams.get(desc, {}).get("bit_lrs", [0.0009, 0.00045, 0.00035])
        # 分阶段QAT主循环
        start_epoch = 0
        for stage, (w_bit, in_bit, out_bit) in enumerate(bit_stages):
            # 共3层 8bit, 6bit, 4bit
            print(f"\n=== QAT Stage {stage+1} ({w_bit}bit, {desc}) ===")
            model.set_quant_bit(
                w_bit=w_bit, in_bit=in_bit, out_bit=out_bit,
                quantize_head=quantize_head,
                quantize_patch_embed=quantize_patch_embed,
                quantize_attn_blocks=attn_blocks,
                quantize_mlp_blocks=mlp_blocks_
            )
            qat_optimizer = torch.optim.AdamW(pg, lr=bit_lrs[stage], weight_decay=args.weight_decay)
            # 可选：每阶段重置学习率调度器    Scheduler https://arxiv.org/pdf/1812.01187.pdf
            lf = lambda x: ((1 + math.cos(x * math.pi / bit_epochs[stage])) / 2) * (1 - args.lrf) + args.lrf  # Cosine学习率调度
            scheduler = lr_scheduler.LambdaLR(qat_optimizer, lr_lambda=lf)
            # 每阶段保存best权重
            stage_best_acc = 0.0
            for epoch in range(bit_epochs[stage]):
                global_epoch = start_epoch + epoch
                # 训练
                # 先粗调alpha，再微调temperature
                # 精度提升很小，适当减少alpha; 过拟合/精度不稳定,大alpha
                train_loss, train_acc = train_one_epoch(model=model, teacher_model=teacher_model,
                                                        optimizer=qat_optimizer, data_loader=train_loader,
                                                        device=device, epoch=global_epoch,
                                                        use_mixed_precision=args.mixed_precision,
                                                        alpha=0.7, temperature=2.0)
                scheduler.step()
                # 验证
                val_loss, val_acc = evaluate(model=model, data_loader=val_loader, device=device, epoch=global_epoch)
                # 记录日志
                tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
                tb_writer.add_scalar(tags[0], train_loss, global_epoch)
                tb_writer.add_scalar(tags[1], train_acc, global_epoch)
                tb_writer.add_scalar(tags[2], val_loss, global_epoch)
                tb_writer.add_scalar(tags[3], val_acc, global_epoch)
                tb_writer.add_scalar(tags[4], qat_optimizer.param_groups[0]["lr"], global_epoch)
                # 每阶段定期保存检查点
                # if (epoch + 1) % 5 == 0:
                #     torch.save(model.state_dict(), f"./weights/model-{desc}-stage{stage+1}-epoch{global_epoch}.pth")
                # 保存最佳
                if val_acc > stage_best_acc:
                    stage_best_acc = val_acc
                    torch.save(model.state_dict(), f"./weights/best_model-{desc}-stage{stage + 1}.pth")
                    print(f"Best model ({desc} Stage {stage + 1}) saved with accuracy: {stage_best_acc:.4f}")
                if val_acc > best_acc:
                    best_acc = val_acc

            print(f"Stage {stage + 1} ({desc}) finished. Best acc: {stage_best_acc:.4f}")
            start_epoch += bit_epochs[stage]

        print(f"Progressive QAT completed. Best validation accuracy: {best_acc:.4f}")





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=15)     # 修改，种类数num_classes
    parser.add_argument('--epochs', type=int, default=50)       # 量化模型可能需要更多轮次(模型内具体修改)
    parser.add_argument('--batch-size', type=int, default=16)    # 8
    parser.add_argument('--lr', type=float, default=0.0003)  # 更小的学习率 0.001 (模型内具体修改)
    parser.add_argument('--lrf', type=float, default=0.01)  # 学习率衰减系数
    parser.add_argument('--weight-decay', type=float, default=0.05)  # 权重衰减
    parser.add_argument('--mixed-precision', type=bool, default=True)

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default="D:/python/PycharmProjects/data/ball_15/train")
    parser.add_argument('--model-name', default='', help='create model name')

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default="D:/python/PycharmProjects/VIT_pretrained_weights/vit_base_patch16_224_in21k.pth",
                        help='initial weights path')
    # 教师模型权重路径
    parser.add_argument('--teacher-weights', type=str,
                        default="D:/python/PycharmProjects/vision_transformer/weights/weightsSave/model-9_ball.pth",
                        help='teacher weights path (FP32)')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
