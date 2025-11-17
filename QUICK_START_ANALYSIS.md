# 🚀 快速开始：从 CIF 文件进行可解释性分析

## 📋 前提条件

您需要准备：
- ✅ 训练好的模型文件（`.pt` 格式）
- ✅ 待分析的晶体结构文件（`.cif` 格式）
- ✅ 可选：文本描述

---

## ⚡ 方法1: 单个文件分析（最简单）

### 步骤1: 准备文件

```bash
# 检查文件
ls -lh model.pt structure.cif
```

### 步骤2: 运行分析

```bash
python band/analyze_from_cif.py \
    --model model.pt \
    --cif structure.cif \
    --text "CaTiO3 perovskite with cubic structure and high dielectric constant" \
    --true_value 2.5
```

### 步骤3: 查看结果

```bash
# 结果保存在 ./analysis_results/ 目录
cd analysis_results/structure/

# 查看生成的文件
ls -lh
# structure_attention.png           # 跨模态注意力
# structure_attention_heads.png     # 多头注意力
# structure_atom_importance.png     # 原子重要性
# structure_edge_importance.png     # 边重要性
# structure_coordination.png        # 配位环境
# structure_substructures.png       # 子结构
# structure_analysis_report.txt     # 文本报告

# 查看文本报告
cat structure_analysis_report.txt
```

---

## 📦 方法2: 批量分析

### 步骤1: 准备文件结构

```bash
project/
├── model.pt                    # 训练好的模型
├── cif_files/                  # CIF 文件目录
│   ├── sample_001.cif
│   ├── sample_002.cif
│   ├── sample_003.cif
│   └── ...
└── descriptions.csv            # 文本描述文件
```

### 步骤2: 创建描述文件

创建 `descriptions.csv`:

```csv
filename,description,true_value
sample_001,CaTiO3 perovskite cubic structure,2.45
sample_002,SrTiO3 perovskite with oxygen vacancies,3.25
sample_003,BaTiO3 tetragonal ferroelectric phase,3.15
```

**CSV 格式说明**:
- `filename`: CIF 文件名（不含 .cif 后缀）
- `description`: 材料的文本描述
- `true_value`: 真实值（可选，用于计算误差）

### 步骤3: 运行批量分析

```bash
python band/analyze_from_cif.py \
    --model model.pt \
    --cif_dir cif_files/ \
    --text_file descriptions.csv \
    --save_dir batch_analysis \
    --batch
```

### 步骤4: 查看批量结果

```bash
cd batch_analysis/

# 查看摘要
cat batch_summary.json

# 查看各个样本的结果
ls -d sample_*/

# 示例：查看 sample_001 的结果
cd sample_001/
ls -lh
```

---

## 🎯 常用参数说明

### 必需参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--model` | 模型文件路径 | `--model model.pt` |
| `--cif` | CIF 文件路径（单文件） | `--cif structure.cif` |
| `--cif_dir` | CIF 目录（批量） | `--cif_dir ./cifs/` |

### 可选参数

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `--text` | 文本描述 | - | `--text "Description"` |
| `--text_file` | 文本文件（批量） | - | `--text_file desc.csv` |
| `--true_value` | 真实值 | None | `--true_value 2.5` |
| `--save_dir` | 保存目录 | `./analysis_results` | `--save_dir ./results` |
| `--device` | 计算设备 | `cuda` | `--device cpu` |
| `--cutoff` | 图构建截断半径 | 8.0 | `--cutoff 10.0` |
| `--max_neighbors` | 最大邻居数 | 12 | `--max_neighbors 16` |
| `--max_samples` | 最大分析数（批量） | None | `--max_samples 100` |

---

## 📝 实际示例

### 示例1: 分析钙钛矿材料

```bash
# 下载示例 CIF（或使用您自己的）
# 假设您有 CaTiO3.cif

python band/analyze_from_cif.py \
    --model trained_model.pt \
    --cif CaTiO3.cif \
    --text "Calcium titanate perovskite, cubic structure, Pm-3m space group, shows high dielectric constant and potential for capacitor applications" \
    --true_value 3.2 \
    --save_dir analysis_CaTiO3
```

**生成的分析**:
- 预测值: 3.15 eV
- 误差: 0.05 eV (1.56%)
- 最重要原子: Ti, O
- 最重要化学键: Ti-O
- 关键子结构: Ti-O-Ti 链

### 示例2: 批量分析一系列掺杂材料

```bash
# 准备文件
mkdir doped_materials
# 放入: CaTiO3.cif, Sr_doped_CaTiO3.cif, La_doped_CaTiO3.cif, ...

# 创建 descriptions.csv
cat > descriptions.csv << EOF
filename,description,true_value
CaTiO3,Pure calcium titanate perovskite,3.20
Sr_doped_CaTiO3,10% Sr-doped calcium titanate,3.35
La_doped_CaTiO3,5% La-doped calcium titanate,3.45
EOF

# 运行批量分析
python band/analyze_from_cif.py \
    --model model.pt \
    --cif_dir doped_materials/ \
    --text_file descriptions.csv \
    --save_dir doping_study \
    --batch

# 对比分析
cd doping_study/
python compare_results.py  # 需要自己编写对比脚本
```

### 示例3: 使用 CPU 分析（无 GPU）

```bash
python band/analyze_from_cif.py \
    --model model.pt \
    --cif structure.cif \
    --text "Material description" \
    --device cpu
```

### 示例4: 仅进行预测（不生成所有可视化）

如果只想快速得到预测值：

```python
# 创建简单预测脚本 quick_predict.py
import torch
from jarvis.core.atoms import Atoms
from jarvis.core.graphs import Graph
from models.alignn import ALIGNN

# 加载模型
model = ALIGNN.load_from_checkpoint('model.pt')
model.eval()

# 处理 CIF
atoms = Atoms.from_cif('structure.cif')
graph = Graph(atoms, cutoff=8.0, max_neighbors=12)
g, lg = graph.to_dgl_graph()

# 预测
text = ["Material description"]
with torch.no_grad():
    prediction = model([g, lg, text])

print(f"预测值: {prediction.item():.4f}")
```

---

## 🔧 高级用法

### 自定义图构建参数

```bash
# 使用更大的截断半径（适合低密度材料）
python band/analyze_from_cif.py \
    --model model.pt \
    --cif MOF_structure.cif \
    --text "Metal-organic framework" \
    --cutoff 12.0 \
    --max_neighbors 20
```

### 批量分析部分样本

```bash
# 只分析前100个样本
python band/analyze_from_cif.py \
    --model model.pt \
    --cif_dir large_dataset/ \
    --text_file descriptions.csv \
    --max_samples 100 \
    --batch
```

### 从 POSCAR 文件分析

```bash
# 先转换 POSCAR 到 CIF
python -c "
from jarvis.core.atoms import Atoms
atoms = Atoms.from_poscar('POSCAR')
atoms.write_cif('structure.cif')
"

# 然后分析
python band/analyze_from_cif.py --model model.pt --cif structure.cif --text "..."
```

---

## 📊 输出文件说明

### 单样本分析输出

```
analysis_results/sample_001/
├── sample_001_attention.png              # 跨模态注意力热图
├── sample_001_attention_heads.png        # 多头注意力分析
├── sample_001_atom_importance.png        # 原子重要性（3张子图）
├── sample_001_edge_importance.png        # 边重要性（3张子图）
├── sample_001_coordination.png           # 配位环境（3张子图）
├── sample_001_substructures.png          # 子结构基序（3张子图）
├── sample_001_structure_analysis.json    # 结构化数据（JSON）
└── sample_001_analysis_report.txt        # 综合文本报告
```

### 批量分析输出

```
batch_analysis/
├── batch_summary.json                     # 批量分析摘要
├── sample_001/                           # 样本1的完整分析
│   ├── sample_001_attention.png
│   ├── ...
├── sample_002/                           # 样本2的完整分析
│   ├── sample_002_attention.png
│   ├── ...
└── ...
```

### 摘要文件格式 (batch_summary.json)

```json
{
  "total_samples": 100,
  "successful": 98,
  "failed": 2,
  "avg_prediction": 2.847,
  "mae": 0.134
}
```

---

## ⚠️ 常见问题

### Q1: 找不到模型配置？

```bash
# 如果 checkpoint 中没有保存配置，会使用默认配置
# 您可以手动指定：修改 analyze_from_cif.py 中的默认配置

# 在 load_model() 函数中修改:
model_config = ALIGNNConfig(
    name="alignn",
    alignn_layers=4,        # 改为您的层数
    gcn_layers=4,           # 改为您的层数
    hidden_features=256,    # 改为您的隐藏层大小
    use_cross_modal_attention=True,  # 根据您的训练设置
    use_middle_fusion=False,         # 根据您的训练设置
    classification=False    # 回归设为 False，分类设为 True
)
```

### Q2: CIF 文件格式错误？

```bash
# 检查 CIF 文件
head structure.cif

# 确保文件格式正确（JARVIS 兼容）
# 如果有问题，可以用其他工具转换：
# - VESTA
# - ASE: ase convert POSCAR structure.cif
# - pymatgen
```

### Q3: 内存不足？

```bash
# 批量分析时限制样本数
python band/analyze_from_cif.py \
    --model model.pt \
    --cif_dir large_dataset/ \
    --max_samples 50 \
    --batch

# 或使用 CPU（慢但节省显存）
python band/analyze_from_cif.py \
    --model model.pt \
    --cif structure.cif \
    --text "..." \
    --device cpu
```

### Q4: 没有文本描述怎么办？

```bash
# 可以使用简单描述
python band/analyze_from_cif.py \
    --model model.pt \
    --cif structure.cif \
    --text "Crystal structure"

# 或者根据化学式自动生成
# 修改脚本添加自动生成功能（基于组成、空间群等）
```

### Q5: 如何只生成部分分析？

如果您只想要某些分析（比如只要原子重要性），可以修改脚本或使用：

```python
# 创建自定义脚本 custom_analysis.py
from interpretability_enhanced import EnhancedInterpretabilityAnalyzer

analyzer = EnhancedInterpretabilityAnalyzer(model)

# 只计算原子重要性
atom_importance = analyzer.compute_atom_importance(g, lg, text)
analyzer.visualize_atom_importance(atoms, atom_importance, save_path='atoms.png')

# 只提取注意力
result = analyzer.extract_attention_weights(g, lg, text)
analyzer.visualize_cross_modal_attention(result['attention_weights'], save_path='attn.png')
```

---

## 🎓 最佳实践

### 1. 文本描述编写建议

**好的描述**:
```
"BaTiO3 tetragonal perovskite structure with P4mm space group,
showing ferroelectric behavior with spontaneous polarization
along c-axis, used in multilayer ceramic capacitors"
```

**包含的信息**:
- ✅ 化学式
- ✅ 晶体结构类型
- ✅ 空间群
- ✅ 关键性质
- ✅ 应用领域

**不好的描述**:
```
"crystal"  # 太简单
```

### 2. 批量分析工作流

```bash
# Step 1: 小规模测试
python band/analyze_from_cif.py \
    --model model.pt \
    --cif test.cif \
    --text "Test" \
    --save_dir test_run

# Step 2: 检查结果
ls test_run/

# Step 3: 如果成功，批量运行
python band/analyze_from_cif.py \
    --model model.pt \
    --cif_dir all_cifs/ \
    --text_file descriptions.csv \
    --batch

# Step 4: 统计分析
python analyze_batch_results.py  # 需要自己编写
```

### 3. 性能优化

```bash
# 并行处理（需要修改脚本）
# 将大批量拆分成多个小批次，并行运行

# 示例：4个进程并行
for i in {0..3}; do
    python band/analyze_from_cif.py \
        --model model.pt \
        --cif_dir batch_$i/ \
        --text_file desc_$i.csv \
        --save_dir results_$i \
        --batch &
done
wait
```

---

## 📚 下一步

1. **运行第一个示例**: 使用您的模型和 CIF 文件
2. **查看生成的可视化**: 理解每种分析的含义
3. **阅读详细指南**: `GRAPH_STRUCTURE_ANALYSIS.md`
4. **自定义分析**: 根据需求修改脚本

祝分析顺利！🚀
