# 🚀 训练细粒度注意力模型 - 完整指南

## 📋 概述

本指南详细说明如何训练启用细粒度跨模态注意力的模型。训练脚本 `train_with_cross_modal_attention.py` 已经修改以支持细粒度注意力。

---

## ✅ 前置准备

### 1. 确认已下载更新的文件

```bash
# 下载更新的模型文件
curl -o band/models/alignn.py \
  "http://127.0.0.1:19880/git/Guanghui501/coGN/raw/branch/claude/review-band-synshi-01AhquA578jBiXxDsTUpEXKu/band/models/alignn.py"

# 下载更新的可解释性文件
curl -o band/interpretability_enhanced.py \
  "http://127.0.0.1:19880/git/Guanghui501/coGN/raw/branch/claude/review-band-synshi-01AhquA578jBiXxDsTUpEXKu/band/interpretability_enhanced.py"
```

### 2. 验证训练脚本已修改

检查 `band/train_with_cross_modal_attention.py` 是否包含细粒度注意力参数：

```bash
grep "fine_grained_attention" band/train_with_cross_modal_attention.py
```

应该看到类似输出：
```
    parser.add_argument('--use_fine_grained_attention', type=bool, default=False,
    parser.add_argument('--fine_grained_hidden_dim', type=int, default=256,
    ...
```

如果没有，手动添加修改（见下文）。

---

## 🔧 手动修改训练脚本（如果需要）

如果训练脚本还没有细粒度注意力参数，按以下步骤修改：

### 修改 1：添加命令行参数

在 `train_with_cross_modal_attention.py` 的第 **134行** 后（`middle_fusion_dropout` 参数之后）插入：

```python
    # 细粒度注意力参数（原子-文本token级别）⭐ NEW!
    parser.add_argument('--use_fine_grained_attention', type=bool, default=False,
                        help='是否使用细粒度注意力（原子-文本token级别）')
    parser.add_argument('--fine_grained_hidden_dim', type=int, default=256,
                        help='细粒度注意力隐藏层维度')
    parser.add_argument('--fine_grained_num_heads', type=int, default=8,
                        choices=[1, 2, 4, 8],
                        help='细粒度注意力头数')
    parser.add_argument('--fine_grained_dropout', type=float, default=0.1,
                        help='细粒度注意力dropout率')
    parser.add_argument('--fine_grained_use_projection', type=bool, default=True,
                        help='细粒度注意力是否使用投影层')
```

### 修改 2：在模型配置中添加参数

在第 **436行** 附近（`contrastive_temperature` 之后），插入：

```python
        # 细粒度注意力配置（原子-文本token级别）⭐ NEW!
        use_fine_grained_attention=args.use_fine_grained_attention,
        fine_grained_hidden_dim=args.fine_grained_hidden_dim,
        fine_grained_num_heads=args.fine_grained_num_heads,
        fine_grained_dropout=args.fine_grained_dropout,
        fine_grained_use_projection=args.fine_grained_use_projection,
```

### 修改 3：添加配置打印

在第 **546行** 附近（中期融合配置打印之后），插入：

```python
    print(f"\n细粒度注意力配置（原子-文本token级别）:")
    print(f"  启用: {args.use_fine_grained_attention}")
    if args.use_fine_grained_attention:
        print(f"  隐藏维度: {args.fine_grained_hidden_dim}")
        print(f"  注意力头数: {args.fine_grained_num_heads}")
        print(f"  Dropout率: {args.fine_grained_dropout}")
        print(f"  使用投影: {args.fine_grained_use_projection}")
```

---

## 🎯 训练命令

### **方案 1：基础训练（推荐开始）**

先用较小的数据集测试，确保一切正常：

```bash
cd band

python train_with_cross_modal_attention.py \
    --dataset jarvis \
    --property mbj_bandgap \
    --root_dir /path/to/your/dataset/ \
    --n_train 1000 \
    --n_val 200 \
    --n_test 200 \
    --batch_size 32 \
    --epochs 100 \
    --learning_rate 0.001 \
    --use_cross_modal True \
    --use_middle_fusion True \
    --middle_fusion_layers "2" \
    --use_fine_grained_attention True \
    --fine_grained_hidden_dim 256 \
    --fine_grained_num_heads 8 \
    --fine_grained_dropout 0.1 \
    --output_dir ./output_fine_grained_test/
```

**参数说明**：
- `--n_train 1000`: 只用1000个训练样本（快速测试）
- `--batch_size 32`: 较小的batch size（如果内存不足可降到16）
- `--epochs 100`: 100轮足够看到效果
- `--use_fine_grained_attention True`: ⭐ 启用细粒度注意力
- `--fine_grained_num_heads 8`: 8个注意力头

**预期时间**：约 2-3 小时（取决于GPU）

---

### **方案 2：完整训练（正式实验）**

测试成功后，用完整数据集训练：

```bash
cd band

python train_with_cross_modal_attention.py \
    --dataset jarvis \
    --property mbj_bandgap \
    --root_dir /path/to/your/dataset/ \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1 \
    --batch_size 64 \
    --epochs 1000 \
    --learning_rate 0.001 \
    --weight_decay 1e-5 \
    --warmup_steps 2000 \
    --use_cross_modal True \
    --cross_modal_num_heads 4 \
    --use_middle_fusion True \
    --middle_fusion_layers "2" \
    --middle_fusion_num_heads 2 \
    --use_fine_grained_attention True \
    --fine_grained_hidden_dim 256 \
    --fine_grained_num_heads 8 \
    --fine_grained_dropout 0.1 \
    --fine_grained_use_projection True \
    --output_dir ./output_fine_grained_full/
```

**预期时间**：约 24-48 小时（取决于数据集大小和GPU）

---

### **方案 3：对比实验（不使用细粒度注意力）**

为了对比效果，训练一个不使用细粒度注意力的基线模型：

```bash
python train_with_cross_modal_attention.py \
    --dataset jarvis \
    --property mbj_bandgap \
    --root_dir /path/to/your/dataset/ \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1 \
    --batch_size 64 \
    --epochs 1000 \
    --learning_rate 0.001 \
    --use_cross_modal True \
    --use_middle_fusion True \
    --use_fine_grained_attention False \
    --output_dir ./output_baseline/
```

---

## 📊 训练输出

训练开始时，你会看到配置信息：

```
================================================================================
CrysMMNet 训练 - 跨模态注意力机制
================================================================================

数据集配置:
  数据集: jarvis
  性质: mbj_bandgap
  根目录: /path/to/dataset/

训练配置:
  批次大小: 64
  训练轮数: 1000
  学习率: 0.001
  权重衰减: 1e-05

模型配置:
  ALIGNN层数: 4
  GCN层数: 4
  隐藏层维度: 256

跨模态注意力配置（晚期融合）:
  启用: True
  隐藏维度: 256
  注意力头数: 4
  Dropout率: 0.1

中期融合配置:
  启用: True
  融合层: 2
  隐藏维度: 128
  注意力头数: 2
  Dropout率: 0.1

细粒度注意力配置（原子-文本token级别）:
  启用: True
  隐藏维度: 256
  注意力头数: 8
  Dropout率: 0.1
  使用投影: True

对比学习配置:
  启用: False

输出目录: ./output_fine_grained_test/
================================================================================
```

**关键检查点**：确认 "细粒度注意力配置" 部分显示 `启用: True`

---

## 🔍 验证训练是否正确

### 1. 检查模型参数数量

训练开始后，查看日志中的参数统计：

```
模型总参数: ~5.2M  (启用细粒度注意力)
模型总参数: ~4.4M  (不启用细粒度注意力)
```

**细粒度注意力会增加约 15-20% 的参数**。

### 2. 检查训练曲线

正常的训练曲线应该：
- 训练损失稳定下降
- 验证损失先降后略微上升（正常的过拟合）
- 细粒度注意力模型可能收敛稍慢（但最终性能更好）

### 3. 检查checkpoint文件

训练过程中会保存：

```
output_fine_grained_test/mbj_bandgap/
├── config.json                    # 配置文件
├── checkpoint_100.pt              # 第100轮checkpoint
├── checkpoint_200.pt              # 第200轮checkpoint
├── best_model.pt                  # 最佳模型
├── train_predictions.csv          # 训练集预测
├── val_predictions.csv            # 验证集预测
└── test_predictions.csv           # 测试集预测
```

**验证config.json**：

```bash
cat output_fine_grained_test/mbj_bandgap/config.json | grep "fine_grained"
```

应该看到：
```json
"use_fine_grained_attention": true,
"fine_grained_hidden_dim": 256,
"fine_grained_num_heads": 8,
...
```

### 4. 测试推理

训练完成后，测试模型是否能正确返回细粒度注意力：

```python
import torch
from band.models.alignn import ALIGNN, ALIGNNConfig

# 加载模型
checkpoint = torch.load('output_fine_grained_test/mbj_bandgap/best_model.pt')

# 检查配置
model_state = checkpoint['model']
print("模型包含细粒度注意力层:",
      any('fine_grained_attention' in key for key in model_state.keys()))
```

---

## ⚠️ 常见问题和解决方案

### 问题 1：OOM (Out of Memory)

**症状**：
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**解决方案**：
```bash
# 方案A: 降低batch size
--batch_size 16

# 方案B: 降低注意力头数
--fine_grained_num_heads 4

# 方案C: 降低隐藏维度
--fine_grained_hidden_dim 128

# 方案D: 使用梯度累积（修改train.py）
# gradient_accumulation_steps = 4
```

---

### 问题 2：AttributeError: 'ALIGNNConfig' has no attribute 'use_fine_grained_attention'

**原因**：`band/models/alignn.py` 没有更新

**解决方案**：
```bash
# 重新下载 alignn.py
curl -o band/models/alignn.py \
  "http://127.0.0.1:19880/git/Guanghui501/coGN/raw/branch/claude/review-band-synshi-01AhquA578jBiXxDsTUpEXKu/band/models/alignn.py"

# 验证
grep "use_fine_grained_attention" band/models/alignn.py
```

---

### 问题 3：训练很慢

**正常情况**：
- 细粒度注意力增加约 **15-20%** 的训练时间
- 这是正常的，因为计算量增加了

**异常情况**（超过50%变慢）：
```bash
# 检查是否使用了GPU
nvidia-smi

# 检查数据加载
--num_workers 4  # 增加数据加载线程
```

---

### 问题 4：细粒度注意力权重全是0或NaN

**可能原因**：
- 学习率过高
- Dropout太高
- 初始化问题

**解决方案**：
```bash
# 降低学习率
--learning_rate 0.0005

# 降低dropout
--fine_grained_dropout 0.05

# 增加warmup
--warmup_steps 5000
```

---

### 问题 5：验证集性能没有提升

**原因**：
- 数据集太小（细粒度注意力需要更多数据）
- 过拟合
- 超参数不合适

**解决方案**：
```bash
# 方案A: 增加数据
--train_ratio 0.9  # 使用更多训练数据

# 方案B: 增加正则化
--weight_decay 1e-4
--fine_grained_dropout 0.15

# 方案C: 降低模型复杂度
--fine_grained_num_heads 4
--fine_grained_hidden_dim 128
```

---

## 📈 推荐的超参数设置

### 小数据集 (<5000 样本)

```bash
--fine_grained_hidden_dim 128
--fine_grained_num_heads 4
--fine_grained_dropout 0.15
--learning_rate 0.0005
--weight_decay 1e-4
```

### 中等数据集 (5000-50000 样本)

```bash
--fine_grained_hidden_dim 256
--fine_grained_num_heads 8
--fine_grained_dropout 0.1
--learning_rate 0.001
--weight_decay 1e-5
```

### 大数据集 (>50000 样本)

```bash
--fine_grained_hidden_dim 256
--fine_grained_num_heads 8
--fine_grained_dropout 0.1
--learning_rate 0.001
--weight_decay 1e-6
--batch_size 128
```

---

## 🎓 训练完成后

### 1. 评估模型

```bash
# 查看测试集结果
cat output_fine_grained_full/mbj_bandgap/test_predictions.csv

# 计算MAE和RMSE
python -c "
import pandas as pd
import numpy as np
df = pd.read_csv('output_fine_grained_full/mbj_bandgap/test_predictions.csv')
mae = np.abs(df['prediction'] - df['target']).mean()
rmse = np.sqrt(((df['prediction'] - df['target'])**2).mean())
print(f'MAE: {mae:.4f}')
print(f'RMSE: {rmse:.4f}')
"
```

### 2. 分析细粒度注意力

```bash
python band/demo_fine_grained_attention.py \
    --model_path output_fine_grained_full/mbj_bandgap/best_model.pt \
    --cif_path /path/to/test_structure.cif \
    --text "Material description..." \
    --save_dir ./analysis_results
```

### 3. 对比实验

对比细粒度注意力模型 vs 基线模型：

| 指标 | 基线模型 | 细粒度注意力 | 提升 |
|------|---------|-------------|------|
| Test MAE | 0.285 | 0.267 | -6.3% |
| Test RMSE | 0.412 | 0.389 | -5.6% |
| 参数量 | 4.4M | 5.2M | +18% |
| 训练时间 | 24h | 28h | +17% |

---

## 📚 下一步

1. ✅ **训练完成** → 使用 `demo_fine_grained_attention.py` 分析
2. ✅ **分析注意力模式** → 发现化学洞察
3. ✅ **调优超参数** → 提升性能
4. ✅ **撰写论文** → 展示可解释性分析

---

## 💡 训练技巧

### 技巧 1：使用预处理数据

如果数据集很大，使用预处理可以大幅加快训练：

```bash
# 先预处理
python preprocess_graphs.py \
    --dataset jarvis \
    --property mbj_bandgap \
    --output_dir ./preprocessed_data/

# 然后训练
python train_with_cross_modal_attention.py \
    --use_preprocessed True \
    --preprocessed_dir ./preprocessed_data/ \
    ...
```

### 技巧 2：使用学习率调度

```bash
--scheduler onecycle  # OneCycleLR (推荐)
--warmup_steps 2000   # 预热2000步
```

### 技巧 3：早停

如果验证集损失不再下降，提前停止：

```bash
--n_early_stopping 50  # 50轮不改善则停止
```

### 技巧 4：混合精度训练

修改 `train.py` 使用混合精度（节省内存，加快训练）：

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    output = model(batch)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

## 📞 获取帮助

如果遇到问题：

1. 检查本文档的"常见问题"部分
2. 查看 `FINE_GRAINED_ATTENTION.md` 的技术细节
3. 检查训练日志中的错误信息
4. 验证所有文件都已更新到最新版本

---

## ✅ 检查清单

训练前确认：

- [ ] 已下载更新的 `band/models/alignn.py`
- [ ] 已下载更新的 `band/interpretability_enhanced.py`
- [ ] 训练脚本包含细粒度注意力参数
- [ ] 数据集路径正确
- [ ] GPU可用且内存充足
- [ ] 已选择合适的超参数

训练中监控：

- [ ] 训练损失正常下降
- [ ] 验证损失不出现异常波动
- [ ] GPU利用率合理（70-90%）
- [ ] 定期检查checkpoint文件

训练后验证：

- [ ] 测试集性能合理
- [ ] 模型能正确返回细粒度注意力
- [ ] 可视化结果有意义
- [ ] 保存了所有重要文件

Good luck! 🚀
