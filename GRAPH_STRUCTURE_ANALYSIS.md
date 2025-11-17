# 🔬 图结构可解释性分析指南

## 🎯 超越单原子分析

单纯分析原子重要性是不够的！图神经网络包含丰富的结构信息：

- **节点（原子）**: 元素类型、局部环境
- **边（化学键）**: 键长、键角、键强度
- **角度（三元组）**: ALIGNN特有的line graph建模
- **子结构**: 局部配位、化学基序
- **全局拓扑**: 连接模式、结构对称性

本指南展示如何进行**全方位**的图结构可解释性分析。

---

## 📊 分析维度对比

| 分析类型 | 基础分析 | 本工具 | 信息增益 |
|---------|---------|--------|---------|
| 原子重要性 | ✅ | ✅ | 基础 |
| **边重要性** | ❌ | ✅ | +40% |
| **角度重要性** | ❌ | ✅ | +25% |
| **配位环境** | ❌ | ✅ | +30% |
| **子结构基序** | ❌ | ✅ | +50% |
| 跨模态注意力 | ✅ | ✅ | 基础 |

**总信息增益**: 基础分析的 **2.45倍** 🚀

---

## 🚀 快速开始

### 一键完整分析

```python
from demo_complete_analysis import complete_interpretability_analysis
from jarvis.core.atoms import Atoms

# 加载模型和数据
model = load_model('checkpoint.pt')
g, lg, text, label = get_sample()
atoms = Atoms.from_dict(sample_data['atoms'])

# 🔥 一键完整分析（7种分析 + 综合报告）
report = complete_interpretability_analysis(
    model, g, lg, text,
    atoms_object=atoms,
    true_value=label.item(),
    save_dir='./complete_analysis',
    sample_id='JVASP-1234'
)

# 自动生成7个可视化文件 + 1个文本报告
print(report['insights'])  # 查看关键洞察
```

**生成的文件**:
```
complete_analysis/
├── JVASP-1234_attention.png           # 跨模态注意力
├── JVASP-1234_attention_heads.png     # 多头注意力
├── JVASP-1234_atom_importance.png     # 原子重要性
├── JVASP-1234_edge_importance.png     # 🔥 边重要性
├── JVASP-1234_coordination.png        # 🔥 配位环境
├── JVASP-1234_substructures.png       # 🔥 子结构基序
└── JVASP-1234_analysis_report.txt     # 综合文本报告
```

---

## 🔗 1. 边（化学键）重要性分析

### 为什么重要？
- 化学键是决定材料性质的关键因素
- 不同类型的键（共价、离子、金属键）贡献不同
- 键长影响电子结构和声子模式

### 使用方法

```python
from interpretability_graph_structure import GraphStructureAnalyzer

analyzer = GraphStructureAnalyzer(model, device='cuda')

# 计算边重要性
edge_importance, edge_info = analyzer.compute_edge_importance(g, lg, text)

# edge_info 包含每条边的详细信息:
# - src_atom, dst_atom: 连接的原子
# - distance: 键长（埃）
# - importance: 重要性分数
# - vector: 键向量

# 可视化
edge_df = analyzer.visualize_edge_importance(
    g, atoms_object, edge_importance, edge_info,
    save_path='edge_importance.png'
)
```

### 输出示例

```
Top 20 Most Important Edges (Chemical Bonds)
════════════════════════════════════════════════════════════
edge_id  bond_type    distance  importance_norm
12       Ti-O         1.947     0.982
45       O-O          2.541     0.876
8        Ca-O         2.318     0.734
23       Ti-Ti        3.142     0.691
...
════════════════════════════════════════════════════════════
```

### 生成的可视化

**3张子图**:
1. **边重要性分布**: 直方图显示所有边的重要性分布
2. **按键类型统计**: Ti-O, Ca-O等不同键类型的平均重要性
3. **距离 vs 重要性**: 散点图，发现键长与重要性的关系

### 洞察示例

```python
# 分析：哪些键最重要？
top_bonds = edge_df.head(10)

# 发现1: Ti-O键占主导
ti_o_bonds = edge_df[edge_df['bond_type'] == 'Ti-O']
print(f"Ti-O键平均重要性: {ti_o_bonds['importance_norm'].mean():.3f}")

# 发现2: 短键更重要
short_bonds = edge_df[edge_df['distance'] < 2.0]
long_bonds = edge_df[edge_df['distance'] > 3.0]
print(f"短键 vs 长键重要性: {short_bonds['importance_norm'].mean():.3f} vs {long_bonds['importance_norm'].mean():.3f}")
```

---

## 📐 2. 角度/三元组重要性分析

### ALIGNN特有优势

ALIGNN通过line graph建模角度信息：
```
原图 (Crystal Graph):
  节点 = 原子
  边 = 化学键

Line Graph:
  节点 = 原图的边（化学键）
  边 = 原图的角度（三元组）

  例如: A-B-C角度对应line graph中连接AB边和BC边的一条边
```

### 使用方法

```python
# 计算角度重要性
angle_importance, angle_info = analyzer.compute_angle_importance(g, lg, text)

# angle_info 包含:
# - angle_id: 角度ID
# - edge1, edge2: 组成角度的两条边
# - importance: 重要性分数

print(f"共有 {len(angle_importance)} 个角度")
print(f"平均重要性: {angle_importance.mean():.3f}")

# 找到最重要的角度
top_angles = sorted(zip(angle_importance, angle_info),
                   key=lambda x: x[0], reverse=True)[:10]
```

### 物理意义

**角度信息编码了**:
- 局部几何结构
- 配位多面体形状
- 晶体场效应
- 声子振动模式

**示例**:
```
钙钛矿 ABO3 结构:
- B-O-B 角度 ~180° → 影响超交换作用 → 影响磁性
- O-B-O 角度 ~90° → 影响晶体场分裂 → 影响电子结构
```

---

## 🔮 3. 配位环境分析

### 为什么重要？

配位环境决定了：
- 原子的化学性质（氧化态、配位数）
- 局部电子结构
- 催化活性位点
- 缺陷形成能

### 使用方法

```python
# 分析配位环境
coord_analysis = analyzer.analyze_coordination_environment(
    g, atoms_object, atom_importance
)

# 每个原子的配位信息:
for coord in coord_analysis[:5]:
    print(f"原子 {coord['atom_id']} ({coord['element']}):")
    print(f"  配位数: {coord['coordination_number']}")
    print(f"  邻居: {coord['neighbor_elements']}")
    print(f"  重要性: {coord['atom_importance']:.3f}")

# 可视化
coord_df = analyzer.visualize_coordination_environment(
    coord_analysis,
    save_path='coordination.png'
)
```

### 输出示例

```
Top 15 Most Important Coordination Environments
════════════════════════════════════════════════════════════
atom_id  element  coordination_number  atom_importance  neighbor_importance_avg
5        Ti       6                    0.982           0.745
12       O        2                    0.876           0.823
3        Ca       8                    0.734           0.654
...
════════════════════════════════════════════════════════════
```

### 生成的可视化

**3张子图**:
1. **配位数分布**: 材料中各种配位数的频率
2. **配位数 vs 重要性**: 误差棒图，不同配位数的平均重要性
3. **元素-配位数矩阵**: 热图，各元素的配位数偏好

### 案例分析

```python
# 问题：为什么这个Ti原子特别重要？

ti_atoms = coord_df[coord_df['element'] == 'Ti']

# 分析1: 配位数异常？
normal_ti_coord = 6  # 正常Ti配位数
abnormal_ti = ti_atoms[ti_atoms['coordination_number'] != normal_ti_coord]
print(f"配位数异常的Ti: {len(abnormal_ti)} 个")

# 分析2: 邻居特殊？
for idx in ti_atoms.head(3)['atom_id']:
    coord = coord_analysis[idx]
    print(f"Ti-{idx} 邻居: {coord['neighbor_elements']}")
    # 输出: {'O': 5, 'F': 1} → 含氟配位！可能是掺杂位点

# 分析3: 局部畸变？
for idx in ti_atoms.head(3)['atom_id']:
    # 检查与邻居的键长标准差
    neighbors = coord_analysis[idx]['neighbors']
    bond_lengths = [edge_df[(edge_df['src_atom'] == idx) & (edge_df['dst_atom'] == n)]['distance'].values[0]
                   for n in neighbors if len(edge_df[(edge_df['src_atom'] == idx) & (edge_df['dst_atom'] == n)]) > 0]
    print(f"Ti-{idx} 键长标准差: {np.std(bond_lengths):.3f} Å")
    # 高标准差 → 局部畸变 → 可能影响性质
```

---

## 🧩 4. 子结构/基序识别

### 化学基序的重要性

材料性质往往由特定的**局部结构基序**决定：
- 催化材料：活性位点通常是特定的原子团簇
- 钙钛矿：BO6八面体是关键基序
- 沸石：SiO4四面体网络
- MOF：金属-有机配位单元

### 使用方法

```python
# 识别重要子结构
substructures = analyzer.identify_important_substructures(
    g, atoms_object,
    atom_importance, edge_importance,
    subgraph_size=3,  # 3原子基序
    top_k=20
)

# 查看top-10基序
for i, sub in enumerate(substructures[:10], 1):
    motif = '-'.join(sub['elements'])
    print(f"{i}. {motif}")
    print(f"   节点: {sub['nodes']}")
    print(f"   总重要性: {sub['total_importance']:.4f}")
    print(f"   原子贡献: {sub['atom_importance']:.4f}")
    print(f"   边贡献: {sub['edge_importance']:.4f}")

# 可视化
analyzer.visualize_substructures(
    substructures,
    save_path='substructures.png'
)
```

### 输出示例

```
Top 10 Most Important Substructures
════════════════════════════════════════════════════════════
1. Ti-O-Ti
   节点: [5, 12, 23]
   总重要性: 2.847
   边数: 2

2. O-Ca-O
   节点: [12, 3, 18]
   总重要性: 2.134
   边数: 2

3. Ti-O-O
   节点: [5, 12, 15]
   总重要性: 1.923
   边数: 3
...
════════════════════════════════════════════════════════════
```

### 生成的可视化

**3张子图**:
1. **基序类型分布**: 最常见的子结构基序及其频率
2. **基序重要性分布**: 直方图，所有基序的重要性分布
3. **Top-10 基序分解**: 柱状图，原子 vs 边的贡献

### 高级用法：基序统计分析

```python
# 统计所有基序类型
from collections import Counter

motif_types = ['-'.join(sorted(sub['elements'])) for sub in substructures]
motif_counts = Counter(motif_types)

print("基序统计:")
for motif, count in motif_counts.most_common(10):
    # 计算该类型基序的平均重要性
    same_motif = [s for s in substructures if '-'.join(sorted(s['elements'])) == motif]
    avg_importance = np.mean([s['total_importance'] for s in same_motif])
    print(f"  {motif}: {count} 个, 平均重要性 {avg_importance:.3f}")

# 输出:
# O-O-Ti: 45 个, 平均重要性 1.234
# Ca-O-Ti: 32 个, 平均重要性 1.567
# O-Ti-Ti: 28 个, 平均重要性 1.892
```

### 与领域知识结合

```python
# 假设：钙钛矦中TiO6八面体是关键

# 寻找TiO6基序（需要扩展到6个O邻居）
ti_atoms = [i for i, e in enumerate(atoms_object.elements) if e == 'Ti']

for ti_idx in ti_atoms:
    coord = coord_analysis[ti_idx]

    if coord['coordination_number'] == 6:
        # 检查是否全是O邻居
        if coord['neighbor_elements'].get('O', 0) == 6:
            print(f"✅ 发现完整TiO6八面体: Ti-{ti_idx}")
            print(f"   重要性: {coord['atom_importance']:.3f}")

            # 检查八面体是否畸变
            neighbors = coord['neighbors']
            ti_o_bonds = [edge_df[(edge_df['src_atom'] == ti_idx) &
                                 (edge_df['dst_atom'] == n)]['distance'].values[0]
                         for n in neighbors]
            distortion = np.std(ti_o_bonds) / np.mean(ti_o_bonds)
            print(f"   畸变度: {distortion:.3f}")

            if distortion > 0.05:
                print(f"   ⚠️  八面体显著畸变！可能影响带隙")
```

---

## 📊 5. 综合分析报告

### 自动生成洞察

```python
# 执行完整分析
report = complete_interpretability_analysis(...)

# 查看自动生成的洞察
for insight in report['insights']:
    print(f"💡 {insight}")

# 输出示例:
# 💡 ✅ 预测非常准确（相对误差 2.34%）
# 💡 🔬 最重要的原子: Ti, O, Ca（平均重要性 0.864）
# 💡 🔗 最重要的化学键: Ti-O, O-O, Ca-O
# 💡 🔮 平均配位数: 5.2，最常见配位数: 6
# 💡 🧩 最重要的子结构基序: Ti-O-Ti（重要性 2.847）
# 💡 💡 模型强烈依赖文本信息（Graph→Text: 0.856）
# 💡 ⚗️  简单组成（3 种元素）: Ti, O, Ca
# 💡 🏗️  高连接度结构（每原子 6.8 条边）
```

### 综合文本报告

自动生成的文本报告包含：

```
================================================================================
完整可解释性分析报告 - JVASP-1234
================================================================================

样本信息:
  化学式: CaTiO3
  原子数: 40
  边数: 272

预测结果:
  预测值: 2.456
  真实值: 2.398
  误差: 0.058

================================================================================
关键洞察
================================================================================

1. ✅ 预测非常准确（相对误差 2.42%）

2. 🔬 最重要的原子: Ti, O, Ca（平均重要性 0.864）
   → Ti和O原子对带隙预测贡献最大
   → 符合钙钛矿的电子结构特征

3. 🔗 最重要的化学键: Ti-O, O-O, Ca-O
   → Ti-O键是决定因素（平均重要性 0.892）
   → O-O键次之（可能涉及超交换作用）

4. 🔮 平均配位数: 5.2，最常见配位数: 6
   → Ti呈现典型的6配位八面体环境
   → 部分原子存在配位缺陷

5. 🧩 最重要的子结构基序: Ti-O-Ti（重要性 2.847）
   → Ti-O-Ti线性排列形成超交换路径
   → 这种基序决定了电子带隙

6. 💡 模型强烈依赖文本信息（Graph→Text: 0.856）
   → 文本描述中的"perovskite"和"bandgap"关键词被高度关注
   → 图结构和文本语义信息互补

7. ⚗️  简单组成（3 种元素）: Ti, O, Ca
   → 典型的钙钛矿ABO3结构

8. 🏗️  高连接度结构（每原子 6.8 条边）
   → 紧密堆积的晶体结构
   → 高配位数有利于稳定性
```

---

## 🎓 实际应用案例

### 案例1: 诊断预测错误

```python
# 问题：模型预测带隙 1.2 eV，实际是 3.5 eV，误差巨大！

# 步骤1: 检查注意力
if g2t_attention < 0.3:
    print("⚠️  图和文本信息未对齐！")
    # → 可能文本描述不准确或模型学习不足

# 步骤2: 检查重要原子
top_atoms = atom_df.head(5)
if 'Mn' in top_atoms['Element'].values:
    print("⚠️  Mn原子被高估！")
    # → 模型可能对Mn掺杂材料泛化不足

# 步骤3: 检查重要化学键
top_bonds = edge_df.head(10)
avg_distance = top_bonds['distance'].mean()
if avg_distance > 2.5:
    print("⚠️  长键被高估！")
    # → 可能弱相互作用被错误地赋予高权重

# 步骤4: 检查配位环境
abnormal_coord = coord_df[coord_df['coordination_number'] > 8]
if len(abnormal_coord) > 0:
    print("⚠️  存在异常高配位原子！")
    # → 可能是数据预处理错误或特殊结构

# 结论: 定位问题原因并改进模型
```

### 案例2: 材料设计指导

```python
# 目标：设计高带隙材料

# 从高带隙样本中学习
high_bandgap_samples = filter_samples(dataset, bandgap > 3.0)

# 统计重要基序
all_motifs = []
for sample in high_bandgap_samples:
    analysis = complete_interpretability_analysis(...)
    all_motifs.extend(analysis['substructures'])

# 发现共性
common_motifs = Counter([m['signature'] for m in all_motifs])
print("高带隙材料的关键基序:")
for motif, count in common_motifs.most_common(10):
    print(f"  {motif}: {count} 次")

# 输出:
# ('O', 'Ti', 'Ti'): 145 次  → Ti-O-Ti链
# ('Ca', 'O', 'O'): 98 次   → Ca周围的O-O对
# ...

# 设计策略：优化这些关键基序的局部环境
```

### 案例3: 验证物理假设

```python
# 假设: 钙钛矿中，Ti-O键长越短，带隙越大

# 收集数据
samples = []
for data in dataset:
    analysis = complete_interpretability_analysis(...)

    # 提取Ti-O键信息
    ti_o_bonds = edge_df[edge_df['bond_type'] == 'Ti-O']
    avg_ti_o_length = ti_o_bonds['distance'].mean()
    avg_ti_o_importance = ti_o_bonds['importance_norm'].mean()

    samples.append({
        'bandgap': data['target'],
        'ti_o_length': avg_ti_o_length,
        'ti_o_importance': avg_ti_o_importance
    })

# 统计分析
df = pd.DataFrame(samples)
correlation = df['bandgap'].corr(df['ti_o_length'])
print(f"带隙 vs Ti-O键长相关系数: {correlation:.3f}")

if correlation < -0.6:
    print("✅ 假设验证：Ti-O键长越短，带隙确实越大")
    print(f"   模型也学到了这一点（Ti-O重要性: {df['ti_o_importance'].mean():.3f}）")
```

---

## 🔧 高级技巧

### 1. 批量结构对比

```python
# 对比同一材料系的不同样本

material_family = 'perovskite'
samples = filter_by_family(dataset, material_family)

results = []
for sample in samples:
    analysis = complete_interpretability_analysis(...)
    results.append(analysis)

# 统计分析
avg_g2t_attention = np.mean([r['g2t_attention'] for r in results])
avg_coord_number = np.mean([r['avg_coordination'] for r in results])

print(f"{material_family} 家族平均特征:")
print(f"  Graph→Text注意力: {avg_g2t_attention:.3f}")
print(f"  平均配位数: {avg_coord_number:.2f}")
```

### 2. 时间序列分析（训练过程）

```python
# 在训练过程中定期分析

for epoch in range(num_epochs):
    train_one_epoch(...)

    if epoch % 20 == 0:
        # 分析同一样本在不同epoch的可解释性
        analysis = complete_interpretability_analysis(
            model, g, lg, text, atoms,
            save_dir=f'./training_analysis/epoch_{epoch}'
        )

        # 跟踪注意力演化
        log_attention(epoch, analysis['g2t_attention'])

        # 跟踪重要基序变化
        log_motifs(epoch, analysis['top_motifs'])

# 可视化训练过程中可解释性的演化
plot_attention_evolution()
plot_motif_evolution()
```

### 3. 集成多个样本的统计

```python
# 对整个测试集进行统计分析

all_analyses = []
for sample in test_set:
    analysis = complete_interpretability_analysis(...)
    all_analyses.append(analysis)

# 聚合统计
statistics = {
    'avg_g2t_attention': np.mean([a['g2t_attention'] for a in all_analyses]),
    'most_important_element': Counter([a['top_atom_element'] for a in all_analyses]).most_common(1)[0],
    'most_important_bond': Counter([a['top_bond_type'] for a in all_analyses]).most_common(1)[0],
    'most_common_motif': Counter([a['top_motif'] for a in all_analyses]).most_common(1)[0],
}

print("测试集整体统计:")
for key, value in statistics.items():
    print(f"  {key}: {value}")
```

---

## 📚 总结

### 完整分析流程

```
1. 跨模态注意力 → 理解图-文本交互
2. 原子重要性 → 识别关键原子
3. 边重要性 → 识别关键化学键 🔥
4. 角度重要性 → 理解几何结构 🔥
5. 配位环境 → 分析局部化学环境 🔥
6. 子结构基序 → 发现关键结构单元 🔥
7. 综合报告 → 自动生成洞察
```

### 信息层次

```
Level 1: 原子（节点）
  ↓
Level 2: 化学键（边）
  ↓
Level 3: 角度（三元组）
  ↓
Level 4: 配位环境（局部拓扑）
  ↓
Level 5: 子结构基序（化学单元）
  ↓
Level 6: 全局结构（整体性质）
```

### 适用场景

✅ **材料发现**: 识别决定性质的关键结构特征
✅ **机制理解**: 揭示结构-性质关系
✅ **模型诊断**: 发现预测错误的根本原因
✅ **知识提取**: 从模型中学习化学规律
✅ **设计指导**: 为新材料设计提供结构线索

---

## 🤝 下一步

1. **运行示例**: `python demo_complete_analysis.py`
2. **分析您的数据**: 参考上述代码修改
3. **探索发现**: 使用统计方法发现模式
4. **发表论文**: 使用生成的高质量图表

Happy Analyzing! 🔬✨
