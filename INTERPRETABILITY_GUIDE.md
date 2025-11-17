# 🔍 可解释性分析完整指南

## 📖 目录
1. [快速开始](#快速开始)
2. [功能概览](#功能概览)
3. [详细使用方法](#详细使用方法)
4. [可视化示例](#可视化示例)
5. [进阶用法](#进阶用法)

---

## 🚀 快速开始

### 最简单的使用方式

```python
import torch
from band.interpretability_enhanced import EnhancedInterpretabilityAnalyzer
from band.models.alignn import ALIGNN

# 1. 加载训练好的模型
checkpoint = torch.load('path/to/checkpoint.pt')
model = ALIGNN(config.model)
model.load_state_dict(checkpoint['model'])
model.eval()

# 2. 创建分析器
analyzer = EnhancedInterpretabilityAnalyzer(model, device='cuda')

# 3. 分析单个样本
result = analyzer.extract_attention_weights(g, lg, text)

# 4. 可视化注意力
analyzer.visualize_cross_modal_attention(
    result['attention_weights'],
    save_path='attention_weights.png'
)

# 5. 计算原子重要性
importance = analyzer.compute_atom_importance(g, lg, text, method='gradient')

# 6. 可视化原子重要性
analyzer.visualize_atom_importance(
    atoms_object,
    importance,
    save_path='atom_importance.png'
)
```

---

## 🎯 功能概览

### ✅ 已实现的可解释性分析

| 功能 | 方法 | 输出 |
|------|------|------|
| **跨模态注意力** | `extract_attention_weights()` | 注意力权重字典 |
| **注意力可视化** | `visualize_cross_modal_attention()` | 热图 |
| **多头注意力** | `visualize_attention_by_heads()` | 分头热图 |
| **原子重要性（梯度）** | `compute_atom_importance(method='gradient')` | 重要性分数 |
| **原子重要性（积分梯度）** | `compute_atom_importance(method='integrated_gradients')` | 重要性分数 |
| **原子重要性可视化** | `visualize_atom_importance()` | 3张图：分布/元素/空间 |
| **特征空间可视化** | `visualize_feature_space()` | t-SNE/PCA图 |
| **单样本完整解释** | `explain_single_prediction()` | 完整报告 |

---

## 📝 详细使用方法

### 1. 提取跨模态注意力权重

```python
# 确保模型启用了跨模态注意力
config.model.use_cross_modal_attention = True

# 前向传播时返回注意力
result = analyzer.extract_attention_weights(g, lg, text)

# 结果包含:
# - result['prediction']: 预测值
# - result['attention_weights']: 注意力字典
#   - 'graph_to_text': [batch, heads, 1, 1]
#   - 'text_to_graph': [batch, heads, 1, 1]
# - result['graph_features']: 图特征
# - result['text_features']: 文本特征
```

### 2. 可视化跨模态注意力

#### 基础版本
```python
analyzer.visualize_cross_modal_attention(
    attention_weights=result['attention_weights'],
    save_path='attention.png'
)
```

#### 多头分析
```python
analyzer.visualize_attention_by_heads(
    attention_weights=result['attention_weights'],
    save_path='attention_heads.png'
)
```

**输出解释**：
- **Graph-to-Text**: 图特征关注文本特征的强度（数值越大表示图越依赖该文本信息）
- **Text-to-Graph**: 文本特征关注图特征的强度（数值越大表示文本越依赖该图信息）
- **多头**: 每个头学习不同类型的关联模式

### 3. 计算原子重要性

#### 梯度法（快速）
```python
importance = analyzer.compute_atom_importance(
    g, lg, text,
    method='gradient',
    target_class=None  # 回归任务设为None，分类任务设为目标类别
)
# 返回: [num_atoms] 数组，每个原子的重要性分数
```

#### 积分梯度法（更准确）
```python
importance = analyzer.compute_atom_importance(
    g, lg, text,
    method='integrated_gradients',
    target_class=None
)
# 更准确但计算较慢（默认50步）
```

### 4. 可视化原子重要性

```python
from jarvis.core.atoms import Atoms

# 从字典创建Atoms对象
atoms = Atoms.from_dict(sample_data['atoms'])

# 可视化
df = analyzer.visualize_atom_importance(
    atoms_object=atoms,
    importance_scores=importance,
    save_path='atom_importance.png',
    top_k=10  # 显示top-10重要原子
)

# 返回DataFrame，包含每个原子的信息和重要性
print(df.head())
#    Index Element  Importance
# 0     5       O    0.982
# 1    12       Ti   0.876
# 2     3       Ca   0.734
# ...
```

**生成3张子图**：
1. **左图**: 所有原子的重要性分布（柱状图）
2. **中图**: 按元素类型的平均重要性（横向柱状图）
3. **右图**: 原子在晶体结构中的空间分布（散点图，颜色表示重要性）

### 5. 特征空间可视化

```python
# 收集多个样本的特征
all_graph_features = []
all_text_features = []
all_labels = []

for batch in test_loader:
    g, lg, text, labels = batch
    result = analyzer.extract_attention_weights(g, lg, text)

    all_graph_features.append(result['graph_features'])
    all_text_features.append(result['text_features'])
    all_labels.append(labels)

# 合并
graph_features = torch.cat(all_graph_features, dim=0)
text_features = torch.cat(all_text_features, dim=0)
labels = torch.cat(all_labels, dim=0)

# 可视化
analyzer.visualize_feature_space(
    graph_features, text_features,
    labels=labels,
    method='tsne',  # 或 'pca'
    save_path='feature_space.png'
)
```

**输出解释**：
- **左图**: 图特征（蓝色圆点）和文本特征（红色三角）在降维空间的分布，灰线连接配对样本
- **右图**: 按目标值着色，观察特征空间与目标的关系

### 6. 单样本完整解释

```python
explanation = analyzer.explain_single_prediction(
    g, lg, text,
    atoms_object=atoms,
    true_value=target,
    save_dir='./results/sample_001',
    sample_id='JVASP-1234'
)

# 自动生成:
# - JVASP-1234_attention.png (跨模态注意力)
# - JVASP-1234_attention_heads.png (多头注意力)
# - JVASP-1234_atom_importance.png (原子重要性)
# - JVASP-1234_explanation.json (完整解释JSON)
```

---

## 🎨 可视化示例

### 跨模态注意力热图

```
Graph-to-Text Attention
┌──────────────────────┐
│ Graph │ 0.856 │ Text │  ← 高权重：图强烈关注文本
└──────────────────────┘

Text-to-Graph Attention
┌──────────────────────┐
│ Text  │ 0.723 │ Graph│  ← 中等权重：文本中等关注图
└──────────────────────┘
```

### 多头注意力分析

```
Head 1    Head 2    Head 3    Head 4
0.92      0.45      0.78      0.61
  ↑        ↑         ↑         ↑
 结构     电子      几何      化学键
 特征     性质      信息      信息
```

### 原子重要性可视化

```
Top 10 Most Important Atoms:
════════════════════════════════════════
Index  Element  Importance
  5      O        0.982      ← 最重要：氧原子
 12      Ti       0.876
  3      Ca       0.734
 ...
════════════════════════════════════════
```

---

## 🚀 进阶用法

### 1. 批量分析

```python
from band.interpretability_enhanced import batch_interpretability_analysis

summary = batch_interpretability_analysis(
    analyzer=analyzer,
    test_loader=test_loader,
    save_dir='./batch_results',
    num_samples=100,
    analyze_feature_space=True
)

# 自动生成:
# - feature_space_tsne.png
# - feature_space_pca.png
# - 每个样本的单独分析
```

### 2. 对比不同融合策略

```python
# 模型1: 仅后期融合
model1 = ALIGNN(config_late_only)
analyzer1 = EnhancedInterpretabilityAnalyzer(model1)

# 模型2: 中期+后期融合
model2 = ALIGNN(config_both)
analyzer2 = EnhancedInterpretabilityAnalyzer(model2)

# 对比注意力模式
result1 = analyzer1.extract_attention_weights(g, lg, text)
result2 = analyzer2.extract_attention_weights(g, lg, text)

# 可视化对比
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# ... 自定义对比可视化
```

### 3. 案例研究：理解预测错误

```python
# 找到预测误差最大的样本
errors = []
for batch in test_loader:
    g, lg, text, labels = batch
    result = analyzer.extract_attention_weights(g, lg, text)
    pred = result['prediction']
    error = abs(pred - labels.numpy())
    errors.append((error, g, lg, text, labels))

# 分析误差最大的样本
errors.sort(reverse=True, key=lambda x: x[0])
worst_case = errors[0]

# 完整分析
analyzer.explain_single_prediction(
    worst_case[1], worst_case[2], worst_case[3],
    atoms_object=atoms,
    true_value=worst_case[4].item(),
    save_dir='./worst_case_analysis'
)

# 问题诊断:
# - 注意力权重是否异常？
# - 哪些原子被错误地高估/低估？
# - 图和文本特征是否对齐？
```

### 4. 领域知识验证

```python
# 假设：钙钛矿材料中，Ti原子应该很重要

# 分析多个钙钛矿样本
perovskite_samples = [...]

ti_importance_scores = []
for sample in perovskite_samples:
    importance = analyzer.compute_atom_importance(...)

    # 找到Ti原子的索引
    ti_indices = [i for i, elem in enumerate(atoms.elements) if elem == 'Ti']

    # 记录Ti的平均重要性
    ti_importance = importance[ti_indices].mean()
    ti_importance_scores.append(ti_importance)

# 统计分析
print(f"Ti 平均重要性: {np.mean(ti_importance_scores):.3f}")
print(f"Ti 重要性标准差: {np.std(ti_importance_scores):.3f}")

# 验证：Ti是否在前20%重要原子中？
```

---

## 🔧 常见问题

### Q1: 注意力权重为None？
**A**: 确保模型配置中启用了跨模态注意力：
```python
config.model.use_cross_modal_attention = True
```

### Q2: 如何理解注意力权重的数值？
**A**:
- **0.0-0.3**: 低关注度，该模态对另一模态的依赖较小
- **0.3-0.7**: 中等关注度，有一定的跨模态交互
- **0.7-1.0**: 高关注度，强烈依赖另一模态的信息

### Q3: 原子重要性为负数？
**A**: 使用梯度法时不会出现负数（取L2范数）。如果需要符号信息，修改代码直接使用梯度而非范数。

### Q4: 特征空间可视化没有明显聚类？
**A**: 这可能表明：
- 任务本身的数据分布比较均匀
- 特征维度需要调整
- 尝试不同的降维方法（t-SNE vs PCA）
- 增加样本数量

### Q5: 如何在训练过程中实时监控注意力？
**A**: 在训练循环中添加：
```python
if epoch % 10 == 0:  # 每10个epoch
    with torch.no_grad():
        result = model(sample, return_attention=True)
        # 记录注意力权重的统计信息
        attn_mean = result['attention_weights']['graph_to_text'].mean()
        logger.add_scalar('attention/g2t_mean', attn_mean, epoch)
```

---

## 📚 相关文献

1. **Attention Mechanisms**: "Attention Is All You Need" (Vaswani et al., 2017)
2. **Integrated Gradients**: "Axiomatic Attribution for Deep Networks" (Sundararajan et al., 2017)
3. **Multi-Modal Learning**: "Multimodal Machine Learning: A Survey and Taxonomy" (Baltrušaitis et al., 2019)
4. **Graph Neural Networks Interpretability**: "Explainability in Graph Neural Networks: A Taxonomic Survey" (Yuan et al., 2021)

---

## 🤝 贡献

欢迎提交Issue和PR来改进可解释性分析工具！

## 📧 联系

如有问题，请在GitHub上提Issue。
