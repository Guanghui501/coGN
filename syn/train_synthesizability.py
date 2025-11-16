#!/usr/bin/env python
"""
晶体可合成性二分类训练脚本

这个脚本用于训练可合成性二分类模型：
- 输入：晶体结构 + 文本描述
- 输出：可合成 (1) 或 不可合成 (0)

使用方法:
    python train_synthesizability.py --data_dir ./synthesizability_data
"""

import os
import sys
import csv
import time
import json
import argparse
import numpy as np

import torch
import torch.nn as nn

from jarvis.core.atoms import Atoms

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'crysmmnet-main/src'))

from data import get_train_val_loaders
from train import train_dgl
from config import TrainingConfig
from models.alignn import ALIGNN, ALIGNNConfig


def get_parser():
    """命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description='晶体可合成性二分类训练',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 数据参数
    parser.add_argument('--data_dir', type=str, default='./synthesizability_data',
                        help='数据目录（包含description.csv和cif/文件夹）')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='权重衰减')

    # 模型参数
    parser.add_argument('--alignn_layers', type=int, default=4,
                        help='ALIGNN层数')
    parser.add_argument('--gcn_layers', type=int, default=4,
                        help='GCN层数')
    parser.add_argument('--hidden_features', type=int, default=256,
                        help='隐藏层维度')

    # 跨模态注意力参数
    parser.add_argument('--use_cross_modal', action='store_true', default=False,
                        help='是否使用跨模态注意力（指定此标志则启用）')
    parser.add_argument('--cross_modal_hidden_dim', type=int, default=256,
                        help='跨模态注意力隐藏维度')
    parser.add_argument('--cross_modal_num_heads', type=int, default=8,
                        help='跨模态注意力头数')
    parser.add_argument('--cross_modal_dropout', type=float, default=0.1,
                        help='跨模态注意力dropout')

    # 中期融合参数
    parser.add_argument('--use_middle_fusion', action='store_true', default=False,
                        help='是否使用中期融合（指定此标志则启用）')
    parser.add_argument('--middle_fusion_layers', type=str, default='2',
                        help='中期融合层索引')

    # 对比学习参数
    parser.add_argument('--use_contrastive', action='store_true', default=False,
                        help='是否使用对比学习（指定此标志则启用）')
    parser.add_argument('--contrastive_weight', type=float, default=0.1,
                        help='对比学习损失权重')
    parser.add_argument('--contrastive_temperature', type=float, default=0.1,
                        help='对比学习温度参数')

    # 其他参数
    parser.add_argument('--output_dir', type=str, default='./output_synthesizability/',
                        help='输出目录')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='数据加载workers数量')

    return parser


def load_synthesizability_dataset(data_dir):
    """
    加载可合成性分类数据集

    Args:
        data_dir: 数据目录

    Returns:
        train_data, val_data, test_data: 三个数据集列表
    """
    print("\n" + "="*80)
    print("加载可合成性分类数据集")
    print("="*80)

    csv_file = os.path.join(data_dir, 'description.csv')
    cif_dir = os.path.join(data_dir, 'cif')

    # 检查文件是否存在
    if not os.path.exists(csv_file):
        raise FileNotFoundError(
            f"找不到数据文件: {csv_file}\n"
            f"请先运行: python prepare_synthesizability_data.py"
        )

    if not os.path.exists(cif_dir):
        raise FileNotFoundError(f"找不到CIF目录: {cif_dir}")

    print(f"数据文件: {csv_file}")
    print(f"CIF目录: {cif_dir}")

    # 读取CSV文件
    train_data = []
    val_data = []
    test_data = []

    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            id = row['id']
            composition = row['composition']
            label = int(row['label'])  # 0 or 1
            text = row['text']
            split = row['split']  # 'train', 'val', or 'test'

            # 加载CIF文件
            cif_file = os.path.join(cif_dir, f"{id}.cif")
            if not os.path.exists(cif_file):
                print(f"⚠️  跳过: 找不到CIF文件 {cif_file}")
                continue

            try:
                atoms = Atoms.from_cif(cif_file)

                info = {
                    "atoms": atoms.to_dict(),
                    "jid": id,
                    "text": text,
                    "target": label,  # 0 or 1
                }

                # 根据split划分数据
                if split == 'train':
                    train_data.append(info)
                elif split == 'val':
                    val_data.append(info)
                elif split == 'test':
                    test_data.append(info)

            except Exception as e:
                print(f"⚠️  跳过: 无法加载 {cif_file}: {e}")
                continue

    print(f"\n数据集大小:")
    print(f"  训练集: {len(train_data)}")
    print(f"    - 可合成 (label=1): {sum(1 for d in train_data if d['target']==1)}")
    print(f"    - 不可合成 (label=0): {sum(1 for d in train_data if d['target']==0)}")
    print(f"  验证集: {len(val_data)}")
    print(f"    - 可合成 (label=1): {sum(1 for d in val_data if d['target']==1)}")
    print(f"    - 不可合成 (label=0): {sum(1 for d in val_data if d['target']==0)}")
    print(f"  测试集: {len(test_data)}")
    print(f"    - 可合成 (label=1): {sum(1 for d in test_data if d['target']==1)}")
    print(f"    - 不可合成 (label=0): {sum(1 for d in test_data if d['target']==0)}")

    return train_data, val_data, test_data


def main():
    """主训练流程"""

    parser = get_parser()
    args = parser.parse_args()

    print("\n" + "="*80)
    print("晶体可合成性二分类训练")
    print("="*80)

    print(f"\n数据配置:")
    print(f"  数据目录: {args.data_dir}")

    print(f"\n训练配置:")
    print(f"  批次大小: {args.batch_size}")
    print(f"  训练轮数: {args.epochs}")
    print(f"  学习率: {args.learning_rate}")

    print(f"\n模型配置:")
    print(f"  ALIGNN层数: {args.alignn_layers}")
    print(f"  隐藏层维度: {args.hidden_features}")
    print(f"  ✅ 二分类任务 (output_features=2)")

    print(f"\n跨模态注意力:")
    print(f"  启用: {args.use_cross_modal}")
    if args.use_cross_modal:
        print(f"  注意力头数: {args.cross_modal_num_heads}")

    print(f"\n对比学习:")
    print(f"  启用: {args.use_contrastive}")
    if args.use_contrastive:
        print(f"  损失权重: {args.contrastive_weight}")

    print(f"\n输出目录: {args.output_dir}")
    print("="*80 + "\n")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载数据集
    train_data, val_data, test_data = load_synthesizability_dataset(args.data_dir)

    # 合并数据集（保持train->val->test顺序）
    dataset_array = train_data + val_data + test_data
    n_train = len(train_data)
    n_val = len(val_data)
    n_test = len(test_data)

    # 创建模型配置 - 二分类任务
    model_config = ALIGNNConfig(
        name="alignn",
        alignn_layers=args.alignn_layers,
        gcn_layers=args.gcn_layers,
        atom_input_features=92,
        edge_input_features=80,
        triplet_input_features=40,
        embedding_features=64,
        hidden_features=args.hidden_features,
        output_features=2,  # ✅ 二分类：输出2个类别
        classification=True,  # ✅ 分类任务
        # 跨模态注意力
        use_cross_modal_attention=args.use_cross_modal,
        cross_modal_hidden_dim=args.cross_modal_hidden_dim,
        cross_modal_num_heads=args.cross_modal_num_heads,
        cross_modal_dropout=args.cross_modal_dropout,
        # 中期融合
        use_middle_fusion=args.use_middle_fusion,
        middle_fusion_layers=args.middle_fusion_layers if args.use_middle_fusion else None,
        # 对比学习
        use_contrastive_loss=args.use_contrastive,
        contrastive_loss_weight=args.contrastive_weight,
        contrastive_temperature=args.contrastive_temperature,
        link="identity",
        zero_inflated=False,
    )

    # 创建训练配置
    config = TrainingConfig(
        random_seed=args.random_seed,
        criterion='mse',  # 会自动转换为CrossEntropy
        target='target',
        epochs=args.epochs,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        filename='synthesizability',
        warmup_steps=2000,
        num_workers=args.num_workers,
        model=model_config,
        output_dir=args.output_dir,
        save_dataloader=False,
        write_checkpoint=True,
        write_predictions=True,
        store_outputs=True,
        progress=True,
        log_tensorboard=False,
        standard_scalar_and_pca=False,
        scheduler="onecycle",
        pin_memory=False,
        keep_data_order=True,  # 保持预定义的train/val/test划分
    )

    # 保存配置
    config_file = os.path.join(args.output_dir, 'config.json')
    config_dict = config.dict() if hasattr(config, 'dict') else config.__dict__
    with open(config_file, 'w') as f:
        json.dump(config_dict, f, indent=4, default=str)
    print(f"配置已保存到: {config_file}\n")

    # 创建数据加载器
    print("创建数据加载器...")
    train_loader, val_loader, test_loader, prepare_batch = get_train_val_loaders(
        dataset_array=dataset_array,
        target=config.target,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        keep_data_order=True,  # 保持我们预定义的train/val/test划分
        batch_size=config.batch_size,
        atom_features=config.atom_features,
        neighbor_strategy=config.neighbor_strategy,
        id_tag=config.id_tag,
        pin_memory=config.pin_memory,
        workers=config.num_workers,
        save_dataloader=config.save_dataloader,
        use_canonize=config.use_canonize,
        filename=config.filename,
        cutoff=config.cutoff,
        max_neighbors=config.max_neighbors,
        output_dir=config.output_dir
    )

    print(f"\n数据加载器创建完成:")
    print(f"  训练集: {len(train_loader.dataset)} 样本")
    print(f"  验证集: {len(val_loader.dataset)} 样本")
    print(f"  测试集: {len(test_loader.dataset)} 样本")
    print()

    # 开始训练
    print("="*80)
    print("开始训练...")
    print("="*80 + "\n")

    start_time = time.time()

    train_dgl(
        config=config,
        train_val_test_loaders=[train_loader, val_loader, test_loader, prepare_batch],
        resume=0
    )

    end_time = time.time()
    elapsed_time = end_time - start_time

    print("\n" + "="*80)
    print(f"训练完成！")
    print(f"总用时: {elapsed_time/3600:.2f} 小时 ({elapsed_time/60:.2f} 分钟)")
    print(f"结果保存在: {args.output_dir}")
    print("="*80 + "\n")

    # 显示预测结果路径
    pred_file = os.path.join(args.output_dir, 'prediction_results_test_set.csv')
    if os.path.exists(pred_file):
        print(f"📊 预测结果: {pred_file}")
        print(f"\n查看预测结果:")
        print(f"  head -20 {pred_file}")
        print()


if __name__ == "__main__":
    main()
