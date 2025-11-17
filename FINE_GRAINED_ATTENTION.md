# Fine-Grained Cross-Modal Attention

## 概述

细粒度跨模态注意力机制使模型能够在**原子级别**和**文本token级别**进行交互，提供比全局注意力更详细的可解释性分析。

### 关键特性

- **原子→文本注意力**：每个原子关注文本中的哪些词语
- **文本→原子注意力**：每个词语影响哪些原子
- **多头注意力**：不同的头可能学到不同的语义对应关系
- **详细可视化**：热力图展示原子-词语的交互矩阵

---

## 架构设计

### 信息流

```
文本描述 → BERT → 所有tokens [batch, seq_len, 768]
                         ↓
                    (保留所有tokens)
                         ↓
晶体结构 → ALIGNN → 节点特征 [num_atoms, 256]
                         ↓
                 转换为batch格式
                         ↓
            [batch, num_atoms, 256] ⟷ [batch, seq_len, 768]
                         ↓
              FineGrainedCrossModalAttention
                         ↓
        Enhanced Atoms [batch, num_atoms, 256]
                         ↓
                   转回DGL格式
                         ↓
                     Readout
                         ↓
                      预测
```

### 关键组件

#### 1. `FineGrainedCrossModalAttention` 模块

位于 `band/models/alignn.py`

**输入**：
- `node_feat`: 原子特征 `[batch, num_atoms, node_dim]`
- `token_feat`: 文本tokens `[batch, seq_len, token_dim]`
- `node_mask`: 原子mask（处理padding）
- `token_mask`: 文本mask（处理padding）

**输出**：
- `enhanced_nodes`: 增强的原子特征 `[batch, num_atoms, node_dim]`
- `enhanced_tokens`: 增强的文本特征 `[batch, seq_len, token_dim]`
- `attention_weights`: 注意力权重字典
  - `atom_to_text`: `[batch, heads, num_atoms, seq_len]`
  - `text_to_atom`: `[batch, heads, seq_len, num_atoms]`

#### 2. 配置参数

在 `ALIGNNConfig` 中添加：

```python
# Fine-grained attention settings
use_fine_grained_attention: bool = False  # 启用细粒度注意力
fine_grained_hidden_dim: int = 256        # 隐藏层维度
fine_grained_num_heads: int = 8           # 注意力头数量
fine_grained_dropout: float = 0.1         # Dropout率
fine_grained_use_projection: bool = True  # 是否投影到相同维度
```

---

## 使用方法

### 1. 训练新模型

修改配置文件，启用细粒度注意力：

```python
config = ALIGNNConfig(
    name="alignn",
    alignn_layers=4,
    gcn_layers=4,
    hidden_features=256,

    # 启用细粒度注意力
    use_fine_grained_attention=True,
    fine_grained_hidden_dim=256,
    fine_grained_num_heads=8,
    fine_grained_dropout=0.1,
)

model = ALIGNN(config)
```

然后正常训练即可。

### 2. 推理和分析

使用示例脚本：

```bash
python band/demo_fine_grained_attention.py \
    --model_path /path/to/checkpoint.pt \
    --cif_path /path/to/structure.cif \
    --text "Material description text" \
    --save_dir ./results
```

### 3. 在已有模型上启用（需重新训练）

**注意**：细粒度注意力是架构的一部分，需要重新训练模型。无法直接在已训练的模型上启用。

但是，可以：
1. 加载已训练模型的特征提取部分（ALIGNN layers）
2. 添加细粒度注意力层
3. 冻结ALIGNN层，只训练注意力层（快速适应）

---

## 可视化示例

### 热力图：Atom → Text

```
                semiconductor  band  gap  coordination  bond  ...
Bi原子            ████████     ████  ████    ██         ██
Ba原子1           ██           ██    ██      ████████   ██████
Ba原子2           ██           ██    ██      ████████   ██████
Na原子            █            █     █       ████       ███
```

- **深色区域**：该原子高度关注这个词
- **浅色区域**：关注度低

### 解读

- Bi原子高度关注"semiconductor", "band", "gap" → 模型理解Bi是带隙的关键元素
- Ba原子关注"coordination", "bond" → 模型理解Ba的结构角色
- Na原子关注度普遍较低 → 对性质影响小

---

## 可解释性分析

### 6个维度的分析

#### 1. 全局模式

```python
atom_to_text_avg = attention_weights['atom_to_text'].mean()  # 原子对文本的依赖度
text_to_atom_avg = attention_weights['text_to_atom'].mean()  # 文本对原子的依赖度
```

#### 2. 原子级模式

哪些原子更依赖文本？

```python
for i, atom in enumerate(atoms):
    atom_text_dep = attention_weights['atom_to_text'][:, :, i, :].mean()
    print(f"{atom.element}: {atom_text_dep:.3f}")
```

#### 3. 词语重要性

哪些词对预测重要？

```python
word_importance = attention_weights['atom_to_text'].mean(axis=(0, 1, 2))  # 对所有原子平均
top_words = sorted(zip(tokens, word_importance), key=lambda x: x[1], reverse=True)
```

#### 4. 原子-词语对应

特定原子关注哪些词？

```python
bi_atom_attention = attention_weights['atom_to_text'][0, :, bi_index, :]  # [heads, seq_len]
top_words_for_bi = bi_atom_attention.mean(axis=0).argsort()[-10:]  # Top 10词
```

#### 5. 多头语义分离

不同的头学到什么？

```python
for head in range(num_heads):
    head_attn = attention_weights['atom_to_text'][0, head, :, :]
    # 分析每个头的模式
```

#### 6. 反向分析

词语影响哪些原子？

```python
word_atom_influence = attention_weights['text_to_atom'][0, :, word_idx, :]  # [heads, num_atoms]
top_atoms = word_atom_influence.mean(axis=0).argsort()[-5:]  # Top 5原子
```

---

## 技术细节

### DGL批处理格式转换

DGL通过拼接图进行批处理：`x` 是 `[total_atoms, dim]` 而不是 `[batch, num_atoms, dim]`。

细粒度注意力需要后者，因此需要转换：

```python
# DGL格式 → 批处理格式
batch_num_nodes = g.batch_num_nodes().tolist()
node_features_batched = torch.zeros(batch_size, max_atoms, node_dim)
offset = 0
for i, num_nodes in enumerate(batch_num_nodes):
    node_features_batched[i, :num_nodes] = x[offset:offset+num_nodes]
    offset += num_nodes

# 应用注意力
enhanced_nodes, _, attn_weights = fine_grained_attention(
    node_features_batched,
    text_tokens,
    node_mask=node_mask,
    token_mask=token_mask,
    return_attention=True
)

# 批处理格式 → DGL格式
x_enhanced = torch.zeros_like(x)
offset = 0
for i, num_nodes in enumerate(batch_num_nodes):
    x_enhanced[offset:offset+num_nodes] = enhanced_nodes[i, :num_nodes]
    offset += num_nodes
```

### Padding和Masking

- **原子padding**：不同图有不同数量的原子，需要pad到 `max_atoms`
- **文本padding**：BERT自动处理，使用 `attention_mask`
- **Masking**：在softmax前，将padding位置设为 `-inf`

```python
if token_mask is not None:
    attn_scores = attn_scores.masked_fill(~token_mask_expanded, float('-inf'))
attn_weights = F.softmax(attn_scores, dim=-1)
```

---

## 性能考虑

### 计算复杂度

- **原子-文本注意力**：`O(num_atoms × seq_len × hidden_dim)`
- **对于典型材料**：
  - num_atoms ≈ 50
  - seq_len ≈ 100
  - hidden_dim = 256
  - → 1.28M 乘法/样本

### 优化策略

1. **降维**：`fine_grained_use_projection=True` 并降低 `hidden_dim`
2. **减少头数**：从8降到4
3. **混合精度**：使用FP16训练
4. **稀疏注意力**（未实现）：只计算top-k关联

### 内存使用

- 存储注意力权重：`[batch, heads, num_atoms, seq_len]`
- 对于 batch=32, heads=8, atoms=50, seq_len=100：
  - 32 × 8 × 50 × 100 × 4 bytes = 5.12 MB/batch

---

## 调试和验证

### 检查注意力权重

```python
output = model([g, lg, text], return_attention=True)
fg_attn = output['fine_grained_attention_weights']

print(f"Atom→Text shape: {fg_attn['atom_to_text'].shape}")
print(f"Text→Atom shape: {fg_attn['text_to_atom'].shape}")

# 检查是否正确归一化（每行和为1）
attn_sum = fg_attn['atom_to_text'][0, 0, 0, :].sum()
print(f"Attention sum: {attn_sum:.4f}")  # 应该接近1.0
```

### 验证注意力的有效性

1. **注意力-梯度一致性**：高注意力的词应该有高梯度
2. **消融实验**：遮盖高注意力的词，预测应该变差
3. **人类评估**：化学家判断注意力模式是否合理

---

## 与全局注意力的对比

| 特性 | 全局注意力 | 细粒度注意力 |
|------|-----------|-------------|
| 粒度 | 图向量 ⟷ 文本向量 | 原子 ⟷ 文本tokens |
| 注意力矩阵 | `[1, 1]` (永远1.0) | `[num_atoms, seq_len]` |
| 可解释性 | 低 | 高 |
| 计算开销 | 低 | 中等 |
| 内存使用 | 低 | 中等 |
| 训练难度 | 易 | 中等 |

---

## 示例输出

```
================================================================================
🔬 Fine-Grained Cross-Modal Attention Analysis
================================================================================

✅ Prediction: 2.3456

✅ Fine-grained attention extracted:
   - atom_to_text shape: torch.Size([1, 8, 6, 128])
   - text_to_atom shape: torch.Size([1, 8, 128, 6])

================================================================================
📊 Fine-Grained Attention Analysis
================================================================================

🔤 Top 10 Most Important Words (overall):
Rank   Word                 Importance
----------------------------------------
1      semiconductor        0.4521
2      band                 0.4123
3      gap                  0.3987
4      Bi                   0.3456
5      bond                 0.2987
6      length               0.2876
7      coordination         0.2654
8      Ba                   0.2341
9      geometry             0.2123
10     cubic                0.1987

⚛️  Top 10 Most Important Atoms (overall):
Rank   Atom                 Importance
----------------------------------------
1      Bi_4                 0.5678
2      Ba_0                 0.3456
3      Ba_1                 0.3421
4      Ba_2                 0.3398
5      Ba_3                 0.3387
6      Na_5                 0.1234

🔍 Top Words for Each Atom:
------------------------------------------------------------

Bi_4:
  - semiconductor      0.7234
  - band               0.6543
  - gap                0.6432
  - Bi                 0.5876
  - electronic         0.4321

Ba_0:
  - coordination       0.5876
  - bond               0.5432
  - length             0.5123
  - geometry           0.4987
  - Ba                 0.4321

================================================================================

✅ Analysis complete! Results saved to: ./fine_grained_results
```

---

## 故障排除

### 问题1：`fine_grained_attention_weights` 为 None

**原因**：模型没有启用细粒度注意力

**解决**：
```python
config.use_fine_grained_attention = True
model = ALIGNN(config)
```

### 问题2：形状不匹配错误

**原因**：节点特征或文本tokens的维度不对

**检查**：
```python
print(f"Node features: {node_feat.shape}")  # 应该是 [batch, num_atoms, 256]
print(f"Text tokens: {token_feat.shape}")   # 应该是 [batch, seq_len, 768]
```

### 问题3：OOM (Out of Memory)

**解决**：
- 减少batch size
- 减少注意力头数：`fine_grained_num_heads = 4`
- 降低隐藏维度：`fine_grained_hidden_dim = 128`
- 使用梯度检查点（未实现）

### 问题4：训练不收敛

**可能原因**：
- 学习率过高
- 注意力参数需要预热

**解决**：
- 降低学习率
- 使用warmup schedule
- 先冻结ALIGNN，只训练注意力层

---

## 未来改进

- [ ] 稀疏注意力：只计算top-k关联
- [ ] 层次注意力：先原子组→词语组，再细粒度
- [ ] 注意力正则化：鼓励稀疏、可解释的模式
- [ ] 交互式可视化：点击原子→高亮相关词语
- [ ] 注意力引导：用领域知识引导注意力（如Bi应该关注"semiconductor"）

---

## 引用

如果使用这个功能，请引用：

```bibtex
@misc{fine_grained_attention_alignn,
  title={Fine-Grained Cross-Modal Attention for Material Property Prediction},
  author={Your Name},
  year={2025},
  note={Extension to ALIGNN architecture}
}
```

---

## 联系方式

问题和建议请提交到GitHub Issues。
