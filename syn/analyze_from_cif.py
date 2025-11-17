#!/usr/bin/env python
"""
端到端可解释性分析脚本

从 .pt 模型文件和 .cif 文件直接进行完整分析

用法:
    python analyze_from_cif.py --model model.pt --cif structure.cif --text "Description"

    # 或者批量分析
    python analyze_from_cif.py --model model.pt --cif_dir ./cif_files/ --text_file descriptions.csv
"""

import os
import sys
import torch
import argparse
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from jarvis.core.atoms import Atoms
    from jarvis.core.graphs import Graph
except ImportError:
    print("❌ 错误: 未安装 jarvis-tools")
    print("请运行: pip install jarvis-tools")
    sys.exit(1)

try:
    import dgl
except ImportError:
    print("❌ 错误: 未安装 dgl")
    print("请运行: pip install dgl")
    sys.exit(1)

from interpretability_enhanced import EnhancedInterpretabilityAnalyzer
from interpretability_graph_structure import GraphStructureAnalyzer
from demo_complete_analysis import complete_interpretability_analysis


def infer_model_config_from_state_dict(state_dict):
    """
    从 state_dict 的键推断模型配置

    Args:
        state_dict: 模型权重字典

    Returns:
        config_kwargs: 推断出的配置参数
    """
    config_kwargs = {
        'name': 'alignn',
        'alignn_layers': 4,
        'gcn_layers': 4,
        'hidden_features': 256,
        'output_features': 1,
        'use_cross_modal_attention': False,
        'use_middle_fusion': False,
        'classification': False
    }

    # 推断 atom_input_features（从 atom_embedding 层的权重形状）
    if 'atom_embedding.layer.0.weight' in state_dict:
        # weight shape is [out_features, in_features]
        weight_shape = state_dict['atom_embedding.layer.0.weight'].shape
        atom_input_features = weight_shape[1]  # 输入维度
        config_kwargs['atom_input_features'] = atom_input_features
        print(f"  🔍 检测到 atom_input_features: {atom_input_features}")

    # 推断 hidden_features（从 ALIGNN 层的权重，比 atom_embedding 更准确）
    # ALIGNNConv 使用 EdgeGatedGraphConv，检查 node_update 的权重
    if 'alignn_layers.0.node_update.src_gate.weight' in state_dict:
        # EdgeGatedGraphConv 的 src_gate 输入维度就是 hidden_features
        hidden_features = state_dict['alignn_layers.0.node_update.src_gate.weight'].shape[1]
        config_kwargs['hidden_features'] = hidden_features
        print(f"  🔍 检测到 hidden_features (from ALIGNN layer): {hidden_features}")
    elif 'atom_embedding.layer.0.weight' in state_dict:
        # 回退到从 atom_embedding 推断
        hidden_features = state_dict['atom_embedding.layer.0.weight'].shape[0]
        config_kwargs['hidden_features'] = hidden_features
        print(f"  🔍 检测到 hidden_features (from atom_embedding): {hidden_features}")

    # 检测跨模态注意力
    has_cross_modal = any('cross_modal_attention' in key for key in state_dict.keys())
    if has_cross_modal:
        config_kwargs['use_cross_modal_attention'] = True
        print(f"  🔍 检测到跨模态注意力模块")

    # 检测中期融合
    middle_fusion_keys = [key for key in state_dict.keys() if 'middle_fusion_modules' in key]
    if middle_fusion_keys:
        config_kwargs['use_middle_fusion'] = True

        # 推断融合层
        fusion_layers = set()
        for key in middle_fusion_keys:
            if 'layer_' in key:
                layer_num = key.split('layer_')[1].split('.')[0]
                fusion_layers.add(layer_num)

        if fusion_layers:
            config_kwargs['middle_fusion_layers'] = ','.join(sorted(fusion_layers))
            print(f"  🔍 检测到中期融合模块，融合层: {config_kwargs['middle_fusion_layers']}")

    # 检测是否为分类任务（通过输出层大小）
    if 'fc.weight' in state_dict:
        output_size = state_dict['fc.weight'].shape[0]
        if output_size > 1:
            config_kwargs['output_features'] = output_size
            # 如果输出是2或更多，可能是分类任务
            if output_size == 2:
                print(f"  🔍 检测到二分类输出层")
                # 注意：这里不自动设置 classification=True，因为可能是多输出回归
        else:
            config_kwargs['output_features'] = 1

    # 推断层数
    alignn_layer_keys = [key for key in state_dict.keys() if 'alignn_layers.' in key]
    if alignn_layer_keys:
        max_layer = max([int(key.split('alignn_layers.')[1].split('.')[0]) for key in alignn_layer_keys])
        config_kwargs['alignn_layers'] = max_layer + 1
        print(f"  🔍 检测到 ALIGNN 层数: {config_kwargs['alignn_layers']}")

    gcn_layer_keys = [key for key in state_dict.keys() if 'gcn_layers.' in key]
    if gcn_layer_keys:
        max_layer = max([int(key.split('gcn_layers.')[1].split('.')[0]) for key in gcn_layer_keys])
        config_kwargs['gcn_layers'] = max_layer + 1
        print(f"  🔍 检测到 GCN 层数: {config_kwargs['gcn_layers']}")

    return config_kwargs


def load_model(model_path, device='cuda', auto_infer_config=True):
    """
    加载训练好的模型（支持自动配置推断）

    Args:
        model_path: .pt 模型文件路径
        device: 计算设备
        auto_infer_config: 是否自动从 state_dict 推断配置

    Returns:
        model: 加载的模型
        config: 模型配置
    """
    print(f"\n{'='*80}")
    print("🔄 加载模型...")
    print(f"{'='*80}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    # 加载 checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    print(f"✅ Checkpoint 加载成功")
    print(f"   文件: {model_path}")
    print(f"   包含的键: {list(checkpoint.keys())}")

    # 获取 state_dict
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # 尝试提取配置
    config = None
    if 'config' in checkpoint:
        config = checkpoint['config']
        print(f"✅ 找到配置信息")

    # 创建模型
    from models.alignn import ALIGNN, ALIGNNConfig

    if config is None:
        if auto_infer_config:
            print(f"\n🔍 自动推断模型配置...")
            config_kwargs = infer_model_config_from_state_dict(state_dict)
            model_config = ALIGNNConfig(**config_kwargs)
            print(f"✅ 配置推断完成\n")
        else:
            # 使用默认配置（保留以防万一）
            print(f"⚠️  使用默认配置（可能不准确，建议使用 auto_infer_config=True）")
            model_config = ALIGNNConfig(
                name="alignn",
                alignn_layers=4,
                gcn_layers=4,
                hidden_features=256,
                output_features=1,
                use_cross_modal_attention=True,
                use_middle_fusion=False,
                classification=False
            )
    else:
        # 从 checkpoint 提取配置
        if hasattr(config, 'model'):
            model_config = config.model
        else:
            model_config = config

    # 打印配置摘要
    print(f"📋 模型配置:")
    print(f"   ALIGNN 层数: {model_config.alignn_layers}")
    print(f"   GCN 层数: {model_config.gcn_layers}")
    print(f"   隐藏层维度: {model_config.hidden_features}")
    print(f"   输出维度: {model_config.output_features}")
    print(f"   跨模态注意力: {'✅' if model_config.use_cross_modal_attention else '❌'}")
    print(f"   中期融合: {'✅' if model_config.use_middle_fusion else '❌'}")
    if model_config.use_middle_fusion:
        print(f"   融合层: {model_config.middle_fusion_layers}")
    print(f"   分类任务: {'✅' if model_config.classification else '❌'}")

    # 创建模型
    model = ALIGNN(model_config)

    # 加载权重
    try:
        model.load_state_dict(state_dict)
        print(f"\n✅ 模型权重加载成功")
    except RuntimeError as e:
        print(f"\n❌ 权重加载失败: {e}")
        print(f"\n💡 尝试使用 strict=False 加载...")
        model.load_state_dict(state_dict, strict=False)
        print(f"⚠️  部分权重未加载，但继续运行")

    model = model.to(device)
    model.eval()

    print(f"\n✅ 模型加载完成")
    print(f"   设备: {device}")
    print(f"   参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'='*80}\n")

    return model, model_config


def cif_to_graph(cif_path, cutoff=8.0, max_neighbors=12):
    """
    将 CIF 文件转换为 DGL 图

    Args:
        cif_path: CIF 文件路径
        cutoff: 截断半径（埃）
        max_neighbors: 最大邻居数

    Returns:
        g: DGL graph
        lg: Line graph
        atoms: Atoms 对象
    """
    print(f"\n🔄 处理 CIF 文件: {cif_path}")

    if not os.path.exists(cif_path):
        raise FileNotFoundError(f"CIF 文件不存在: {cif_path}")

    # 读取 CIF
    atoms = Atoms.from_cif(cif_path)

    print(f"✅ CIF 读取成功")
    print(f"   化学式: {atoms.composition.reduced_formula}")
    print(f"   原子数: {atoms.num_atoms}")
    print(f"   元素: {set(atoms.elements)}")

    # 构建图（使用静态方法直接生成 DGL 图）
    # 注意：使用 atomic_number 特征以匹配训练时的配置
    g, lg = Graph.atom_dgl_multigraph(
        atoms=atoms,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        atom_features="atomic_number",
        compute_line_graph=True,
        use_canonize=True
    )

    print(f"✅ 图构建成功")
    print(f"   节点数: {g.num_nodes()}")
    print(f"   边数: {g.num_edges()}")
    print(f"   Line graph 节点数: {lg.num_nodes()}")
    print(f"   Line graph 边数: {lg.num_edges()}")

    # 转换原子特征：atomic_number -> cgcnn features
    # 这一步匹配训练时 StructureDataset 的特征转换
    from jarvis.core.specie import chem_data, get_node_attributes
    import numpy as np

    # 构建特征查找表（与 StructureDataset._get_attribute_lookup 相同）
    max_z = max(v["Z"] for v in chem_data.values())
    template = get_node_attributes("C", atom_features="cgcnn")
    features = np.zeros((1 + max_z, len(template)))

    for element, v in chem_data.items():
        z = v["Z"]
        x = get_node_attributes(element, atom_features="cgcnn")
        if x is not None:
            features[z, :] = x

    # 转换特征（与 StructureDataset.__init__ 中的代码相同）
    z = g.ndata.pop("atom_features")
    g.ndata["atomic_number"] = z
    z = z.type(torch.LongTensor).squeeze()  # 使用 LongTensor 进行索引
    f = torch.tensor(features[z], dtype=torch.float32)  # 明确指定 float32
    if g.num_nodes() == 1:
        f = f.unsqueeze(0)
    g.ndata["atom_features"] = f

    print(f"✅ 特征转换完成")
    print(f"   原子序数: {z[:5].tolist() if len(z) > 5 else z.tolist()}")
    print(f"   特征维度: {g.ndata['atom_features'].shape}")
    print(f"   特征数据类型: {g.ndata['atom_features'].dtype}")

    return g, lg, atoms


def predict(model, g, lg, text, device='cuda'):
    """
    使用模型进行预测

    Args:
        model: 模型
        g: DGL graph
        lg: Line graph
        text: 文本描述（列表）
        device: 设备

    Returns:
        prediction: 预测值
        output: 完整输出（包含特征等）
    """
    model.eval()

    # 检查图移动前的特征
    print(f"\n🔍 图移动到设备前:")
    print(f"   atom_features 形状: {g.ndata['atom_features'].shape}")
    print(f"   atom_features 数据类型: {g.ndata['atom_features'].dtype}")
    print(f"   atom_features 设备: {g.ndata['atom_features'].device}")
    if 'atomic_number' in g.ndata:
        print(f"   atomic_number 存在: {g.ndata['atomic_number'].shape}")

    g = g.to(device)
    lg = lg.to(device)

    # 检查图移动后的特征
    print(f"\n🔍 图移动到设备后:")
    print(f"   atom_features 形状: {g.ndata['atom_features'].shape}")
    print(f"   atom_features 数据类型: {g.ndata['atom_features'].dtype}")
    print(f"   atom_features 设备: {g.ndata['atom_features'].device}")
    if 'atomic_number' in g.ndata:
        print(f"   atomic_number 存在: {g.ndata['atomic_number'].shape}")

    with torch.no_grad():
        output = model([g, lg, text], return_features=True, return_attention=True)

    if isinstance(output, dict):
        prediction = output['predictions'].cpu().item()
    else:
        prediction = output.cpu().item()

    return prediction, output


def analyze_single_cif(
    model_path,
    cif_path,
    text_description,
    true_value=None,
    save_dir='./analysis_results',
    device='cuda',
    cutoff=8.0,
    max_neighbors=12
):
    """
    分析单个 CIF 文件

    Args:
        model_path: 模型文件路径
        cif_path: CIF 文件路径
        text_description: 文本描述
        true_value: 真实值（可选）
        save_dir: 保存目录
        device: 计算设备
        cutoff: 图构建截断半径
        max_neighbors: 最大邻居数

    Returns:
        report: 分析报告
    """
    print("\n" + "="*80)
    print("🔬 单个 CIF 文件可解释性分析")
    print("="*80)

    # 1. 加载模型
    model, config = load_model(model_path, device=device)

    # 2. 处理 CIF
    g, lg, atoms = cif_to_graph(cif_path, cutoff=cutoff, max_neighbors=max_neighbors)

    # 3. 预测
    print(f"\n{'='*80}")
    print("🎯 进行预测...")
    print(f"{'='*80}")

    text = [text_description]
    prediction, output = predict(model, g, lg, text, device=device)

    print(f"\n📊 预测结果:")
    print(f"   预测值: {prediction:.4f}")
    if true_value is not None:
        error = abs(prediction - true_value)
        rel_error = 100 * error / abs(true_value) if true_value != 0 else 0
        print(f"   真实值: {true_value:.4f}")
        print(f"   绝对误差: {error:.4f}")
        print(f"   相对误差: {rel_error:.2f}%")

    # 4. 完整可解释性分析
    print(f"\n{'='*80}")
    print("🔍 开始完整可解释性分析...")
    print(f"{'='*80}\n")

    # 获取样本 ID
    sample_id = Path(cif_path).stem

    # 执行分析
    report = complete_interpretability_analysis(
        model, g, lg, text,
        atoms_object=atoms,
        true_value=true_value,
        save_dir=save_dir,
        sample_id=sample_id
    )

    # 添加预测信息
    report['cif_file'] = cif_path
    report['text_description'] = text_description
    report['prediction'] = prediction

    return report


def analyze_batch_cifs(
    model_path,
    cif_dir,
    text_file=None,
    save_dir='./batch_analysis',
    device='cuda',
    max_samples=None
):
    """
    批量分析 CIF 文件

    Args:
        model_path: 模型文件路径
        cif_dir: CIF 文件目录
        text_file: 文本描述文件（CSV格式: filename,description,true_value）
        save_dir: 保存目录
        device: 计算设备
        max_samples: 最大分析样本数

    Returns:
        all_reports: 所有样本的分析报告列表
    """
    print("\n" + "="*80)
    print("🔬 批量 CIF 文件可解释性分析")
    print("="*80)

    # 加载模型
    model, config = load_model(model_path, device=device)

    # 获取所有 CIF 文件
    cif_dir = Path(cif_dir)
    cif_files = sorted(cif_dir.glob("*.cif"))

    if max_samples is not None:
        cif_files = cif_files[:max_samples]

    print(f"\n找到 {len(cif_files)} 个 CIF 文件")

    # 读取文本描述
    text_dict = {}
    true_value_dict = {}

    if text_file and os.path.exists(text_file):
        import csv
        with open(text_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row.get('filename', row.get('id', ''))
                text_dict[filename] = row.get('description', row.get('text', ''))
                if 'true_value' in row or 'target' in row:
                    try:
                        true_value_dict[filename] = float(row.get('true_value', row.get('target', 0)))
                    except:
                        pass
        print(f"✅ 加载了 {len(text_dict)} 个文本描述")

    # 批量分析
    all_reports = []

    for i, cif_path in enumerate(cif_files, 1):
        print(f"\n{'='*80}")
        print(f"处理 {i}/{len(cif_files)}: {cif_path.name}")
        print(f"{'='*80}")

        try:
            # 获取文本描述
            sample_id = cif_path.stem
            text = text_dict.get(sample_id, text_dict.get(cif_path.name, "No description provided"))
            true_value = true_value_dict.get(sample_id, true_value_dict.get(cif_path.name, None))

            # 分析
            sample_save_dir = Path(save_dir) / sample_id
            report = analyze_single_cif(
                model_path,
                str(cif_path),
                text,
                true_value=true_value,
                save_dir=str(sample_save_dir),
                device=device
            )

            all_reports.append(report)

        except Exception as e:
            print(f"❌ 分析失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 生成批量摘要
    print(f"\n{'='*80}")
    print("📊 批量分析摘要")
    print(f"{'='*80}")

    summary = {
        'total_samples': len(cif_files),
        'successful': len(all_reports),
        'failed': len(cif_files) - len(all_reports)
    }

    if all_reports:
        predictions = [r['prediction'] for r in all_reports]
        summary['avg_prediction'] = float(sum(predictions) / len(predictions))

        # 如果有真实值，计算统计
        true_values = [r['true_value'] for r in all_reports if r['true_value'] is not None]
        if true_values:
            errors = [abs(r['prediction'] - r['true_value']) for r in all_reports if r['true_value'] is not None]
            summary['avg_error'] = float(sum(errors) / len(errors))
            summary['mae'] = summary['avg_error']

    # 保存摘要
    summary_path = Path(save_dir) / 'batch_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n批量分析完成:")
    print(f"  成功: {summary['successful']}")
    print(f"  失败: {summary['failed']}")
    if 'avg_prediction' in summary:
        print(f"  平均预测: {summary['avg_prediction']:.4f}")
    if 'mae' in summary:
        print(f"  平均误差 (MAE): {summary['mae']:.4f}")
    print(f"\n摘要已保存: {summary_path}")
    print(f"{'='*80}\n")

    return all_reports


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='CIF 文件可解释性分析工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单个文件分析
  python analyze_from_cif.py --model model.pt --cif structure.cif \\
      --text "Perovskite oxide with high bandgap" --true_value 2.5

  # 批量分析
  python analyze_from_cif.py --model model.pt --cif_dir ./cif_files/ \\
      --text_file descriptions.csv --batch

  # 使用 CPU
  python analyze_from_cif.py --model model.pt --cif structure.cif \\
      --text "Description" --device cpu
        """
    )

    # 模型参数
    parser.add_argument('--model', type=str, required=True,
                       help='训练好的模型文件路径 (.pt)')

    # 输入参数
    parser.add_argument('--cif', type=str,
                       help='CIF 文件路径（单文件模式）')
    parser.add_argument('--cif_dir', type=str,
                       help='CIF 文件目录（批量模式）')
    parser.add_argument('--text', type=str,
                       help='文本描述（单文件模式）')
    parser.add_argument('--text_file', type=str,
                       help='文本描述文件 CSV（批量模式）')
    parser.add_argument('--true_value', type=float,
                       help='真实值（可选）')

    # 模式选择
    parser.add_argument('--batch', action='store_true',
                       help='批量分析模式')

    # 输出参数
    parser.add_argument('--save_dir', type=str, default='./analysis_results',
                       help='结果保存目录')

    # 计算参数
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='计算设备')
    parser.add_argument('--cutoff', type=float, default=8.0,
                       help='图构建截断半径（埃）')
    parser.add_argument('--max_neighbors', type=int, default=12,
                       help='最大邻居数')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='批量模式下最大分析样本数')

    args = parser.parse_args()

    # 检查设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA 不可用，切换到 CPU")
        args.device = 'cpu'

    # 执行分析
    try:
        if args.batch:
            # 批量分析
            if not args.cif_dir:
                parser.error("批量模式需要 --cif_dir 参数")

            reports = analyze_batch_cifs(
                model_path=args.model,
                cif_dir=args.cif_dir,
                text_file=args.text_file,
                save_dir=args.save_dir,
                device=args.device,
                max_samples=args.max_samples
            )

        else:
            # 单文件分析
            if not args.cif:
                parser.error("单文件模式需要 --cif 参数")
            if not args.text:
                parser.error("单文件模式需要 --text 参数")

            report = analyze_single_cif(
                model_path=args.model,
                cif_path=args.cif,
                text_description=args.text,
                true_value=args.true_value,
                save_dir=args.save_dir,
                device=args.device,
                cutoff=args.cutoff,
                max_neighbors=args.max_neighbors
            )

            print(f"\n{'='*80}")
            print("🎉 分析完成！")
            print(f"{'='*80}")
            print(f"\n生成的文件:")
            for file in report['files_generated']:
                print(f"  ✅ {file}")
            print(f"\n保存目录: {args.save_dir}")
            print(f"{'='*80}\n")

    except Exception as e:
        print(f"\n❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
