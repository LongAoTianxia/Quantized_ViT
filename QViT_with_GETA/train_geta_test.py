from tqdm import tqdm
import torch
import os
import math
import argparse
import random
import json
import sys
import torch.optim.lr_scheduler as lr_scheduler
import tempfile
import tarfile
import warnings
import logging
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import torch.nn.functional as F
from my_dataset import MyDataSet
from vit_model import vit_base_patch16_224_in21k as create_model
from utils import read_split_data, train_one_epoch, evaluate
from torchvision import transforms, datasets  # 新增 datasets 引入

from only_train_once import OTO
from only_train_once.optimizer.utils import (
    load_checkpoint,
    save_checkpoint,
    scan_checkpoint,
)
from only_train_once.quantization import (
    model_to_quantize_model,
    QuantizationType,
    QuantizationMode
)

# Ignore warnings
warnings.filterwarnings("ignore")


def accuracy_topk(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    _ = target.size(0)  # batch_size

    # 获取前maxk个最高概率的预测结果
    # output.topk(maxk, 1, True, True)的参数含义：
    # - maxk: 返回前maxk个最大值
    # - 1: 沿着第1个维度（类别维度）
    # - True: 返回最大值（而不是最小值）
    # - True: 返回排序后的结果
    _, pred = output.topk(maxk, 1, True, True)

    pred = pred.t()  # 从[batch_size, maxk] 转为 [maxk, batch_size]

    # 比较预测结果和真实标签
    # target.view(1, -1): 将target从[batch_size]变为[1, batch_size]
    # expand_as(pred): 扩展为与pred相同的形状[maxk, batch_size]
    # pred.eq(...): 逐元素比较，相等为True，不等为False
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        # 取前k行，表示在前k个预测中是否有正确答案
        correct_k = correct[:k].reshape(-1).view(-1).float().sum(0, keepdim=True)
        res.append(correct_k)
    return res


def check_accuracy(model, testloader, two_input=False):
    """检查模型准确率"""
    correct1 = 0
    correct5 = 0
    total = 0
    model = model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for X, y in testloader:
            X = X.to(device)
            y = y.to(device)
            if two_input:
                y_pred = model.forward(X, X)
            else:
                y_pred = model.forward(X)
            total += y.size(0)

            prec1, prec5 = accuracy_topk(y_pred.data, y, topk=(1, 5))

            correct1 += prec1.item()
            correct5 += prec5.item()

    model = model.train()
    accuracy1 = correct1 / total
    accuracy5 = correct5 / total
    return accuracy1, accuracy5

def get_quant_param_dict(model):
    # Access quantization parameter information
    param_dict = {}
    for name, param in model.named_parameters():
        if "d_quant" in name or "t_quant" in name or "q_m" in name:
            layer_name = ".".join(name.split(".")[:-1])
            param_name = name.split(".")[-1]
            if layer_name in param_dict:
                param_dict[layer_name][param_name] = param.item()
            else:
                param_dict[layer_name] = {}
                param_dict[layer_name][param_name] = param.item()
    return param_dict

def get_bitwidth_dict(param_dict):
    bit_dict = {}

    for key in param_dict.keys():
        bit_dict[key] = {}

        d_quant_wt = param_dict[key]["d_quant_wt"]
        q_m_wt = abs(param_dict[key]["q_m_wt"])
        if "t_quant_wt" in param_dict[key]:
            t_quant_wt = param_dict[key]["t_quant_wt"]
        else:
            t_quant_wt = 1.0
        bit_width_wt = (
            math.log2(math.exp(t_quant_wt * math.log(q_m_wt)) / abs(d_quant_wt) + 1) + 1
        )
        bit_dict[key]["weight"] = bit_width_wt

        if "d_quant_act" in param_dict[key]:
            d_quant_act = param_dict[key]["d_quant_act"]
            q_m_act = abs(param_dict[key]["q_m_act"])
            if "t_quant_act" in param_dict[key]:
                t_quant_act = param_dict[key]["t_quant_act"]
            else:
                t_quant_act = 1.0
            bit_width_act = (
                math.log2(
                    math.exp(t_quant_act * math.log(q_m_act)) / abs(d_quant_act) + 1
                )
                + 1
            )
            bit_dict[key]["activation"] = bit_width_act

    return bit_dict

"""
def transforms_train():
    return transforms.Compose(
        [transforms.RandomResizedCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    )


def transforms_test():
    return transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    )
"""
def transforms_train():
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))
    return transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(224, padding=4),  # 或 RandomResizedCrop(224)
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

def transforms_test():
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))
    return transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

def build_cifar10_loaders(root, batch_size, nw):
    train_ds = datasets.CIFAR10(root=root, train=True, download=False,
                                transform=transforms_train())
    val_ds = datasets.CIFAR10(root=root, train=False, download=False,
                              transform=transforms_test())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=nw, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=nw, pin_memory=True)
    return train_loader, val_loader

def Train_loader(train_images_path, train_images_label, batch_size, nw):
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=transforms_train())

    return DataLoader(train_dataset,
                      batch_size=batch_size,
                      shuffle=True,
                      pin_memory=True,
                      num_workers=nw,
                      collate_fn=train_dataset.collate_fn)



def Test_loader(val_images_path, val_images_label, batch_size, nw):
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=transforms_test())
    return DataLoader(val_dataset,
                      batch_size=batch_size,
                      shuffle=False,
                      pin_memory=True,
                      num_workers=nw,
                      collate_fn=val_dataset.collate_fn)


def Model(device, num_classes, weights):
    base_model = create_model(num_classes=num_classes, has_logits=False).to(device)
    # 加载预训练权重
    if weights != "":
        assert os.path.exists(weights), f"weights file: '{weights}' not exist."
        weights_dict = torch.load(weights, map_location=device)

        # 删除不需要的权重
        del_keys = ['head.weight', 'head.bias'] if base_model.has_logits \
            else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
        for k in del_keys:
            if k in weights_dict:
                del weights_dict[k]
        print(base_model.load_state_dict(weights_dict, strict=False))

    model = model_to_quantize_model(base_model,
                                    num_bits=32,
                                    quant_type=QuantizationType.SYMMETRIC_NONLINEAR,
                                    quant_mode=QuantizationMode.WEIGHT_AND_ACTIVATION)
    #model = base_model
    return model.to(device)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main(args):
    # ====== 初始化 ======
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建必要的目录
    if not os.path.exists("./weights"):
        os.makedirs("./weights")

    tb_writer = SummaryWriter() if args.tensorboard else None

    # 读取数据集
    # train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0])
    print(f'Using {nw} dataloader workers every process')

    # train_loader = Train_loader(train_images_path, train_images_label, batch_size, nw)
    # val_loader = Test_loader(val_images_path, val_images_label, batch_size, nw)
    train_loader, val_loader = build_cifar10_loaders(args.data_path, batch_size, nw)

    # 创建模型
    print("Creating model...")
    model = Model(device, args.num_classes, args.weights)

    # 创建教师模型（全精度 ViT），仅用于知识蒸馏
    teacher_model = None
    if args.use_kd:
        print("Creating teacher model for KD...")
        # teacher 用全精度 ViT，不经过 GETA 量化
        teacher_model = create_model(num_classes=args.num_classes, has_logits=False).to(device)

        if args.teacher_weights != "":
            # 如果你已经有在 ball_15 上训练好的全精度 ViT，就在这里加载它
            assert os.path.exists(args.teacher_weights), \
                f"teacher weights file: '{args.teacher_weights}' not exist."
            teacher_state = torch.load(args.teacher_weights, map_location=device)
            print(teacher_model.load_state_dict(teacher_state, strict=False))
        else:
            # 否则退回到 ImageNet21k 的预训练权重
            assert os.path.exists(args.weights), f"weights file: '{args.weights}' not exist."
            weights_dict = torch.load(args.weights, map_location=device)
            del_keys = ['head.weight', 'head.bias'] if teacher_model.has_logits \
                else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
            for k in del_keys:
                if k in weights_dict:
                    del weights_dict[k]
            print(teacher_model.load_state_dict(weights_dict, strict=False))

        # teacher 只前向，不反传
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad_(False)

    # 验证量化参数是否被添加
    quant_param_names = [name for name, _ in model.named_parameters() if 'quant' in name]
    # 量化参数为.d_quant_wt,.t_quant_wt,.d_quant_act,.t_quant_act
    # patch_embed:1x4, blocks:12x16（qkv 4个，proj 4个，mlp.fc1 4个，mlp.fc2 4个）, head:1x4, 总共4+4+192=200个量化参数
    print(f"  新增控制量化的参数数量:{len(quant_param_names)}")
    print(f"  参数示例: {quant_param_names[:5]}")

    # 冻结特征提取层（可选）
    if args.freeze_layers:
        for name, para in model.named_parameters():
            if "head" not in name and "pre_logits" not in name:
                para.requires_grad_(False)
            else:
                print(f"Training {name}")

    dummy_input = torch.rand(1, 3, 224, 224).to(device)
    oto = OTO(model=model, dummy_input=dummy_input)

    # 标记不可剪枝的参数（参考 test_vit.py）
    oto.mark_unprunable_by_param_names(
        ['patch_embed.proj.weight',
        'pos_embed',
         'head.weight', 'head.bias']
    )

    # A ResNet_zig.gv.pdf will be generated to display the depandancy graph.
    # oto.visualize(view=False, out_dir='../cache')

    # 参数信息
    print(f"[GETA] 训练参数信息：")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - 总参数量: {total_params:,}")
    print(f"  - 可训练参数量: {trainable_params:,}")
    print(f"  - 模型层数: {len(list(model.named_modules()))}")

    # 输出所有 param_group 的列表
    """
    pp = oto._graph.get_param_groups()
    print("==== Param Groups p_names (BEGIN) ====")
    for i, group in enumerate(pp):
        print(f"参数组 {i}: is_prunable={group.get('is_prunable', False)}, "
              f"is_auxiliary={group.get('is_auxiliary', False)}, "
              f"num_groups={group.get('num_groups', 0)}，\n"
              f"op_names={group.get('op_names', [])}, \n"
              f"p_names={group.get('p_names', [])}, \n"
              f"p_transform_type={group.get('p_transform')}, ")
    print("==== Param Groups p_names (END) ====")
    """
    # 创建 GETA 优化器
    print("Creating GETA optimizer...")
    # 设置前1轮与后3轮不进行量化和剪枝
    steps_per_epoch = len(train_loader)

    # 量化参数设置：从第5轮后开始量化，量化步长为总训练轮数的3/5
    start_projection_step = 5 * steps_per_epoch
    projection_steps = ((args.epochs - 7) * 3 // 5) * steps_per_epoch
    # 从32bit降到8bit，每周期减少4bit，需要(32-8)/4=6个周期
    projection_periods = (32 - 8) // 4  # 6个周期
    # 从32bit降到6bit，每周期减少2bit，需要(32-6)/2=13个周期
    #projection_periods = (32 - 6) // 2
    quantization_end_step = start_projection_step + projection_steps  # 量化结束步数

    # 剪枝设置
    start_pruning_step = quantization_end_step + 1 * len(train_loader)  # 在量化位宽到达4-8bit后开始剪枝
    # 剪枝到训练结束前2轮停止，增加缓冲期
    pruning_end_step = (args.epochs - 2) * steps_per_epoch
    pruning_steps = pruning_end_step - start_pruning_step
    # 调整剪枝周期数，避免过于频繁的剪枝
    pruning_periods = max(1, pruning_steps // 1000)  # 约每800步一个周期，减少剪枝频率
    optimizer = oto.geta(
        variant="adam",
        lr=args.lr,
        lr_quant=args.lr_quant,
        first_momentum=0.9,
        weight_decay=args.weight_decay,
        target_group_sparsity=args.target_group_sparsity,

        start_projection_step=start_projection_step,
        projection_steps=projection_steps,
        projection_periods=projection_periods,

        start_pruning_step=start_pruning_step,
        pruning_steps=pruning_steps,
        pruning_periods=pruning_periods,

        bit_reduction=4,
        #bit_reduction=2,
        min_bit_wt=4,
        max_bit_wt=32,
        min_bit_act=4,
        max_bit_act=32,
    )
    print(f"  学习率: {args.lr}")
    print(f"  目标组稀疏度: {args.target_group_sparsity}")
    print(f"  权重位宽: {optimizer.min_bit_wt}-{optimizer.max_bit_wt} bits")
    print(f"  激活位宽: {optimizer.min_bit_act}-{optimizer.max_bit_act} bits")
    print(f"  - 总轮数: {args.epochs}")
    print(f"  - 每轮步数: {steps_per_epoch}")
    print(f"  - 量化开始步数: {start_projection_step} (第{start_projection_step // steps_per_epoch + 1}轮)")
    print(f"  - 量化总步数: {projection_steps}")
    print(f"  - 量化结束步数: {quantization_end_step} (第{quantization_end_step // steps_per_epoch + 1}轮)")
    print(f"  - 剪枝开始步数: {start_pruning_step} (第{start_pruning_step // steps_per_epoch + 1}轮)")
    print(f"  - 剪枝结束步数: {pruning_end_step} (第{pruning_end_step // steps_per_epoch + 1}轮)")
    print(f"  - 剪枝总步数: {pruning_steps}")

    # Get full/original floating-point model MACs, BOPs, and number of parameters
    full_macs = oto.compute_macs(in_million=True, layerwise=True)
    full_bops = oto.compute_bops(in_million=True, layerwise=True)
    full_num_params = oto.compute_num_params(in_million=True)
    full_weight_size = oto.compute_weight_size(in_million=True)

    # hard fix for full_bops calculation
    full_bops["total"] = full_bops["total"] * 32 / 32


    # 学习率调度器
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    start_epoch = 0
    # 训练循环
    print("Starting training...")
    best_acc = 0.0
    model.train()

    # ========== 训练循环 ==========
    print(f"[GETA] 开始训练")
    for epoch in range(start_epoch, args.epochs):
        # 训练
        train_loss, train_acc, train_ce_loss, train_kd_loss, train_gl_loss = train_one_epoch(model=model,
                                                                            optimizer=optimizer,
                                                                            data_loader=train_loader,
                                                                            device=device,
                                                                            epoch=epoch,
                                                                            mix_up=bool(args.mix_up),
                                                                            label_smooth=bool(args.label_smooth),
                                                                            inferred_num_classes=args.num_classes,
                                                                            teacher_model=teacher_model,
                                                                            kd_alpha=args.kd_alpha,
                                                                            kd_temperature=args.kd_temperature,
                                                                            use_group_lasso=args.use_group_lasso,
                                                                            group_lasso_lambda=args.group_lasso_lambda,
                                                                            gl_start_epoch=args.group_lasso_warmup_epochs
        )
        scheduler.step()

        # 验证
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        # 计算详细准确率
        val_acc1, val_acc5 = check_accuracy(model, val_loader)
        print(f"[Epoch {epoch}] "
              f"Train total={train_loss:.4f}, "
              f"CE={train_ce_loss:.4f}, "
              f"KD={train_kd_loss:.4f}, "
              f"GL={train_gl_loss:.4f}, "
              f"Acc={train_acc:.4f}\n")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Val Top-1/Top-5: {val_acc1:.4f}/{val_acc5:.4f}")

        # 计算压缩指标
        opt_metrics = optimizer.compute_metrics()
        avg_wt_bit = oto.compute_average_bit_width()
        print("Compression metrics computed.")
        print(
            "Ep: {ep}, norm_all:{param_norm:.2f}, grp_sparsity: {gs:.2f}, "
            "norm_import: {norm_import:.2f}, norm_redund: {norm_redund:.2f}, "
            "num_grp_import: {num_grps_import}, num_grp_redund: {num_grps_redund}, "
            "avg_wt_bit_width: {avg_wt_bit:.2f}".format(
                ep=epoch,
                param_norm=opt_metrics.norm_params,
                gs=opt_metrics.group_sparsity,
                norm_import=opt_metrics.norm_important_groups,
                norm_redund=opt_metrics.norm_redundant_groups,
                num_grps_import=opt_metrics.num_important_groups,
                num_grps_redund=opt_metrics.num_redundant_groups,
                avg_wt_bit=avg_wt_bit,
            )
        )

        # TensorBoard 记录
        if tb_writer:
            tags = ["train_loss", "train_acc", "val_loss", "val_acc",
                    "learning_rate", "val_acc1", "val_acc5", "avg_wt_bit"]
            tb_writer.add_scalar(tags[0], train_loss, epoch)
            tb_writer.add_scalar(tags[1], train_acc, epoch)
            tb_writer.add_scalar(tags[2], val_loss, epoch)
            tb_writer.add_scalar(tags[3], val_acc, epoch)
            tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
            tb_writer.add_scalar(tags[5], val_acc1, epoch)
            tb_writer.add_scalar(tags[6], val_acc5, epoch)
            tb_writer.add_scalar(tags[7], avg_wt_bit, epoch)

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"./weights/best_model.pth")
            print(f"Best model saved with accuracy: {best_acc:.4f}")

        # 保存最终模型
        if epoch == args.epochs - 1:
            torch.save(model.state_dict(), "./weights/final_model-{}.pth".format(epoch))
            print(f"Final model saved with accuracy: {best_acc:.4f}")
            checkpoint = {
                "model": model.state_dict(),           # 量化后的模型
                "optimizer": optimizer.state_dict(),   # GETA 的 state_dict，包含 pruned_idxes/param_groups/num_steps 等剪枝信息
                "args": vars(args),                    # 可选，方便复现超参
            }
            torch.save(checkpoint, "ckpt_final.pt")

    # 训练完成后，保存压缩模型
    print("\nTraining completed.")

    # 保存原始和压缩模型用于比较
    print("[GETA] 构建量化子网...")
    os.makedirs('./geta_output', exist_ok=True)

    oto.construct_subnet(
        export_huggingface_format=False,
        export_float16=False,
        full_group_sparse_model_dir='./geta_output',
        compressed_model_dir='./geta_output'
    )

    print(f"[GETA] 量化模型已保存:")
    print(f"  - 完整稀疏模型: {oto.full_group_sparse_model_path}")
    print(f"  - 压缩模型: {oto.compressed_model_path}")

    compressed_model = torch.load(oto.compressed_model_path)
    oto_compressed = OTO(compressed_model, dummy_input)

    model_name = args.model_name
    print(f"Full MACs for {model_name}: {full_macs['total']} M MACs")
    print(f"Full BOPs for {model_name}: {full_bops['total']} M BOPs")
    print(f"Full num params for {model_name}: {full_num_params} M params")
    print(f"Full weight size for {model_name}: {full_weight_size['total']} MB")
    if "layer_info" in full_macs and "layer_info" in full_bops:
        print("Layer-by-layer breakdown for full model:")
        print(f"{'Layer':<30} {'Type':<15} {'MACs (M)':<15} {'BOPs (M)':<15}")
        print("-" * 75)
        for mac_info, bop_info in zip(full_macs["layer_info"], full_bops["layer_info"]):
            print(
                f"{mac_info['name']:<30} {mac_info['type']:<15} {mac_info['macs']:<15.2f} {bop_info['bops']:<15.2f}"
            )

    # Get compressed model MACs, BOPs, and number of parameters
    compressed_macs = oto_compressed.compute_macs(in_million=True, layerwise=True)
    compressed_bops = oto_compressed.compute_bops(in_million=True, layerwise=True)
    compressed_num_params = oto_compressed.compute_num_params(in_million=True)
    compressed_weight_size = oto_compressed.compute_weight_size(in_million=True)

    print(f"Compressed MACs for Q{model_name}: {compressed_macs['total']} M MACs")
    print(f"Compressed BOPs for Q{model_name}: {compressed_bops['total']} M BOPs")
    print(
        f"Compressed num params for Q{model_name}: {compressed_num_params} M params"
    )
    print(
        f"Compressed weight size for Q{model_name}: {compressed_weight_size['total']} MB"
    )
    if "layer_info" in compressed_macs and "layer_info" in compressed_bops:
        print("Layer-by-layer breakdown for compressed model:")
        print(f"{'Layer':<30} {'Type':<15} {'MACs (M)':<15} {'BOPs (M)':<15}")
        print("-" * 75)
        for mac_info, bop_info in zip(
                compressed_macs["layer_info"], compressed_bops["layer_info"]
        ):
            print(
                f"{mac_info['name']:<30} {mac_info['type']:<15} {mac_info['macs']:<15.2f} {bop_info['bops']:<15.2f}"
            )

    print(
        f"MAC reduction    : {(1.0 - compressed_macs['total'] / full_macs['total']) * 100}%"
    )
    print(
        f"BOP reduction    : {(1.0 - compressed_bops['total'] / full_bops['total']) * 100}%"
    )
    print(
        f"Param reduction  : {(1.0 - compressed_num_params / full_num_params) * 100}%"
    )
    print(f"MAC ratio: {full_macs['total'] / compressed_macs['total']}")
    print(
        f"BOP compresion ratio: {full_bops['total'] / compressed_bops['total']}"
    )

    full_model_size = os.path.getsize(oto.full_group_sparse_model_path) / (1024 ** 3)
    compressed_model_size = os.path.getsize(oto.compressed_model_path) / (1024 ** 3)
    print(f"Size of full/ model: {full_model_size:.4f} GB")
    print(f"Size of compressed model: {compressed_model_size:.4f} GB")

    # Print and visualize each layer bit width info
    param_dict = get_quant_param_dict(model)
    bit_dict = get_bitwidth_dict(param_dict)
    print("=========================")
    print(json.dumps(bit_dict))

    print("[GETA] 训练完成！")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # 模型参数
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')

    # 优化器参数
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--lr_quant', type=float, default=0.001, help='Quantization learning rate')
    parser.add_argument('--lrf', type=float, default=0.01, help='Learning rate factor')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--lr_step', type=int, default=50, help='LR scheduler step size (epochs)')
    parser.add_argument('--lr_gamma', type=float, default=0.1, help='LR scheduler gamma')

    # GETA 特定参数
    parser.add_argument('--target_group_sparsity', type=float, default=0.4,  # 目标组稀疏度
                        help='Target group sparsity (0.0-1.0)')
    # 量化投影
    parser.add_argument('--start_projection_step', type=int, default=0,
                        help='Step to start projection')
    parser.add_argument('--projection_steps', type=int, default=20,
                        help='Number of projection steps')
    parser.add_argument('--projection_periods', type=int, default=4,
                        help='Projection periods')
    # 剪枝
    parser.add_argument('--start_pruning_step', type=int, default=30,
                        help='Step to start pruning')
    parser.add_argument('--pruning_steps', type=int, default=50,
                        help='Number of pruning steps')
    parser.add_argument('--pruning_periods', type=int, default=10,
                        help='Pruning periods')

    # 数据和模型路径
    #parser.add_argument('--data-path', type=str,
    #                    default="D:/python/PycharmProjects/data/cifar10",
    #                    help='Dataset path')
    parser.add_argument('--data-path', type=str,
                        default="/root/autodl-tmp/data/cifar10")
    #parser.add_argument('--weights', type=str,
    #                    default='D:/python/PycharmProjects/VIT_pretrained_weights/model-1.pth',
    #                    help='Initial weights path')
    parser.add_argument('--weights', type=str,
                        default='/root/autodl-tmp/VIT_pretrained_weights/vit_base_patch16_224_in21k.pth',
                        help='initial weights path')
    parser.add_argument('--model-name', default='ViT', help='Model name for saving')

    # 训练设置
    parser.add_argument('--freeze-layers', type=bool, default=False,
                        help='Freeze feature extraction layers')
    parser.add_argument('--device', default='cuda:0', help='Device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--save_freq', type=int, default=5, help='Checkpoint save frequency')
    parser.add_argument('--tensorboard', type=bool, default=True, help='Use tensorboard logging')
    parser.add_argument('--mix_up', type=int, default=1, help='use mixup? 1=yes, 0=no')
    parser.add_argument('--label_smooth', type=int, default=1, help='use label smoothing? 1=yes, 0=no')

    # 知识蒸馏参数
    parser.add_argument('--use_kd', type=int, default=1,
                        help='use knowledge distillation? 1=yes, 0=no')
    #parser.add_argument('--teacher-weights', type=str, default="D:/python/PycharmProjects/VIT_pretrained_weights/model-1.pth",
    #                    help='teacher model weights path (full-precision ViT on target dataset, optional)')
    parser.add_argument('--teacher-weights', type=str, default="/root/autodl-tmp/VIT_pretrained_weights/model-1-10.pth",
                        help='teacher model weights path (full-precision ViT on target dataset, optional)')
    parser.add_argument('--kd-alpha', type=float, default=0.4,
                        help='KD loss weight (0-1, 0 means no KD effect)')
    parser.add_argument('--kd-temperature', type=float, default=5.0,
                        help='KD temperature T')

    # Lasso 正则化参数
    parser.add_argument('--use_group_lasso', type=bool, default=True,
                        help='whether to add group lasso regularization')
    parser.add_argument('--group_lasso_lambda', type=float, default=8e-6,
                        help='lambda for group lasso penalty')
    parser.add_argument('--group_lasso_warmup_epochs', type=int, default=17,
                        help='start GL from this epoch')
    # 随机种子
    parser.add_argument('--seed', type=int, default=0, help='Random seed')

    opt = parser.parse_args()
    main(opt)
