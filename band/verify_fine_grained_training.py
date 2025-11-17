#!/usr/bin/env python
"""
验证训练是否启用了细粒度注意力

用法:
    python verify_fine_grained_training.py /path/to/checkpoint.pt
"""

import sys
import torch

def verify_checkpoint(checkpoint_path):
    """验证checkpoint是否包含细粒度注意力"""

    print(f"\n{'='*80}")
    print(f"验证 Checkpoint: {checkpoint_path}")
    print(f"{'='*80}\n")

    # 加载checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        print(f"❌ 无法加载checkpoint: {e}")
        return

    # 获取模型状态字典
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # 检查细粒度注意力相关的参数
    fg_keys = [k for k in state_dict.keys() if 'fine_grained_attention' in k]

    print("🔍 检查结果:\n")

    if fg_keys:
        print("✅ **细粒度注意力已启用**\n")
        print(f"找到 {len(fg_keys)} 个相关参数:")
        for key in fg_keys[:10]:  # 只显示前10个
            shape = state_dict[key].shape
            print(f"  - {key}: {shape}")
        if len(fg_keys) > 10:
            print(f"  ... 还有 {len(fg_keys) - 10} 个参数")
    else:
        print("❌ **细粒度注意力未启用**")
        print("\n提示：训练时需要添加参数：")
        print("  --use_fine_grained_attention True")

    # 统计参数量
    total_params = sum(p.numel() for p in state_dict.values())
    print(f"\n📊 模型参数统计:")
    print(f"  总参数量: {total_params:,}")

    if fg_keys:
        fg_params = sum(state_dict[k].numel() for k in fg_keys)
        print(f"  细粒度注意力参数: {fg_params:,} ({fg_params/total_params*100:.1f}%)")
        print(f"  预期参数量: ~5,200,000")

        if total_params < 4_800_000:
            print("\n⚠️  参数量偏少，可能没有正确启用")
        elif total_params > 5_000_000:
            print("\n✅ 参数量正常")
    else:
        print(f"  预期参数量（无细粒度注意力）: ~4,400,000")
        if total_params > 5_000_000:
            print("\n⚠️  参数量异常，可能配置有误")

    # 检查其他注意力机制
    print(f"\n🔍 其他注意力机制:")

    cross_modal_keys = [k for k in state_dict.keys() if 'cross_modal_attention' in k and 'fine_grained' not in k]
    if cross_modal_keys:
        print(f"  ✅ 全局跨模态注意力: 启用 ({len(cross_modal_keys)} 参数)")
    else:
        print(f"  ❌ 全局跨模态注意力: 未启用")

    middle_fusion_keys = [k for k in state_dict.keys() if 'middle_fusion' in k]
    if middle_fusion_keys:
        print(f"  ✅ 中期融合: 启用 ({len(middle_fusion_keys)} 参数)")
    else:
        print(f"  ❌ 中期融合: 未启用")

    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python verify_fine_grained_training.py /path/to/checkpoint.pt")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    verify_checkpoint(checkpoint_path)
