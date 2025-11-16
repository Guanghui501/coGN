"""
CrysMMNet 可解释性分析工具

功能：
1. 跨模态注意力权重可视化
2. 原子重要性分数计算
3. 文本 token 重要性分析
4. 对比学习特征空间可视化
5. 预测结果解释
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import pandas as pd
from pathlib import Path
import json


class InterpretabilityAnalyzer:
    """CrysMMNet 可解释性分析器"""

    def __init__(self, model, device='cuda'):
        """
        Args:
            model: 训练好的 ALIGNN 模型
            device: 计算设备
        """
        self.model = model
        self.device = device
        self.model.eval()

        # 用于存储中间激活和注意力权重
        self.activations = {}
        self.attention_weights = {}

    def register_hooks(self):
        """注册钩子以捕获中间层激活和注意力权重"""

        def get_activation(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook

        # 如果模型有跨模态注意力，注册钩子
        if hasattr(self.model, 'cross_modal_attention') and self.model.cross_modal_attention is not None:
            self.model.cross_modal_attention.register_forward_hook(
                get_activation('cross_modal_attention')
            )

    def extract_attention_weights(self, graph, return_prediction=True):
        """
        提取跨模态注意力权重

        Args:
            graph: DGL graph with text
            return_prediction: 是否返回预测值

        Returns:
            Dict containing attention weights and optionally prediction
        """
        self.activations.clear()
        self.attention_weights.clear()

        with torch.no_grad():
            # 修改模型forward以返回注意力权重
            # 临时保存原始forward
            original_forward = self.model.forward

            # 定义新的forward函数来捕获注意力
            attention_dict = {}

            def forward_with_attention(g, return_features=False):
                """修改后的forward，返回注意力权重"""

                # 复制原始forward逻辑，但保存注意力权重
                # ... (需要根据实际模型结构实现)

                output = original_forward(g, return_features=True)

                # 如果有跨模态注意力，提取权重
                if hasattr(self.model, 'cross_modal_attention') and self.model.use_cross_modal_attention:
                    # 注意力权重应该在forward过程中被计算
                    # 这里我们需要修改模型代码来暴露这些权重
                    pass

                return output

            # 临时替换forward
            # self.model.forward = forward_with_attention

            # 执行forward
            output = self.model(graph, return_features=True)

            # 恢复原始forward
            # self.model.forward = original_forward

        result = {
            'activations': self.activations,
            'attention_weights': self.attention_weights
        }

        if return_prediction:
            if isinstance(output, dict):
                result['prediction'] = output['predictions'].cpu().numpy()
            else:
                result['prediction'] = output.cpu().numpy()

        return result

    def compute_atom_importance(self, graph, target_output=None, method='gradient'):
        """
        计算每个原子对预测的重要性

        Args:
            graph: DGL graph
            target_output: 目标输出索引（如果是分类任务）
            method: 'gradient' 或 'integrated_gradients'

        Returns:
            atom_importance: 每个原子的重要性分数 [num_atoms]
        """
        self.model.eval()

        if method == 'gradient':
            return self._compute_gradient_importance(graph, target_output)
        elif method == 'integrated_gradients':
            return self._compute_integrated_gradients(graph, target_output)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _compute_gradient_importance(self, graph, target_output=None):
        """使用梯度计算原子重要性"""

        # 启用梯度
        graph = graph.to(self.device)

        # 获取节点特征并启用梯度
        node_features = graph.ndata['atom_features'].clone().detach().requires_grad_(True)
        original_features = graph.ndata['atom_features']
        graph.ndata['atom_features'] = node_features

        # Forward pass
        output = self.model(graph)

        if isinstance(output, dict):
            prediction = output['predictions']
        else:
            prediction = output

        # 如果是多输出，选择目标输出
        if target_output is not None:
            prediction = prediction[target_output]
        else:
            prediction = prediction.sum()  # 对所有输出求和

        # Backward pass
        prediction.backward()

        # 计算重要性（梯度的L2范数）
        gradients = node_features.grad
        importance = torch.norm(gradients, dim=1).cpu().numpy()

        # 恢复原始特征
        graph.ndata['atom_features'] = original_features

        return importance

    def _compute_integrated_gradients(self, graph, target_output=None, steps=50):
        """使用积分梯度计算原子重要性"""

        graph = graph.to(self.device)
        original_features = graph.ndata['atom_features'].clone()

        # 基线：零特征
        baseline_features = torch.zeros_like(original_features)

        # 积分路径
        alphas = torch.linspace(0, 1, steps).to(self.device)

        integrated_grads = torch.zeros_like(original_features)

        for alpha in alphas:
            # 插值特征
            interpolated_features = baseline_features + alpha * (original_features - baseline_features)
            interpolated_features = interpolated_features.clone().detach().requires_grad_(True)

            graph.ndata['atom_features'] = interpolated_features

            # Forward
            output = self.model(graph)
            if isinstance(output, dict):
                prediction = output['predictions']
            else:
                prediction = output

            if target_output is not None:
                prediction = prediction[target_output]
            else:
                prediction = prediction.sum()

            # Backward
            prediction.backward()

            # 累积梯度
            integrated_grads += interpolated_features.grad

        # 平均并乘以特征差
        integrated_grads = integrated_grads / steps
        importance = torch.norm(integrated_grads * (original_features - baseline_features), dim=1)

        # 恢复原始特征
        graph.ndata['atom_features'] = original_features

        return importance.cpu().numpy()

    def analyze_text_importance(self, graph, text_embedding):
        """
        分析文本描述中各部分的重要性

        Args:
            graph: DGL graph
            text_embedding: 文本嵌入 [hidden_dim]

        Returns:
            text_importance: 文本特征的重要性分数
        """
        # 这需要文本的token级别信息
        # 简化版本：计算文本嵌入的梯度

        text_embedding = text_embedding.clone().detach().requires_grad_(True)

        # 这里需要修改模型以接受外部文本嵌入
        # 简化处理：返回整体重要性

        return text_embedding.grad.cpu().numpy() if text_embedding.grad is not None else None

    def visualize_cross_modal_attention(self, graph, attention_weights,
                                       atom_symbols=None, text_tokens=None,
                                       save_path=None):
        """
        可视化跨模态注意力权重

        Args:
            graph: DGL graph
            attention_weights: 注意力权重矩阵 [num_atoms, text_dim] 或 [num_atoms, num_tokens]
            atom_symbols: 原子符号列表
            text_tokens: 文本token列表
            save_path: 保存路径
        """
        plt.figure(figsize=(12, 8))

        # 如果attention_weights是3D (batch, atoms, text)，取第一个样本
        if len(attention_weights.shape) == 3:
            attention_weights = attention_weights[0]

        # 绘制热图
        sns.heatmap(
            attention_weights,
            cmap='YlOrRd',
            xticklabels=text_tokens if text_tokens else False,
            yticklabels=atom_symbols if atom_symbols else False,
            cbar_kws={'label': 'Attention Weight'}
        )

        plt.xlabel('Text Tokens / Features')
        plt.ylabel('Atoms')
        plt.title('Cross-Modal Attention: Graph ↔ Text')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Attention visualization saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def visualize_atom_importance(self, atoms_object, importance_scores,
                                  save_path=None, top_k=10):
        """
        可视化原子重要性

        Args:
            atoms_object: jarvis.core.atoms.Atoms对象
            importance_scores: 原子重要性分数 [num_atoms]
            save_path: 保存路径
            top_k: 显示top-k重要原子
        """
        # 归一化重要性分数
        importance_scores = (importance_scores - importance_scores.min()) / \
                          (importance_scores.max() - importance_scores.min() + 1e-8)

        # 创建DataFrame
        elements = [atom for atom in atoms_object.elements]
        coords = atoms_object.cart_coords

        df = pd.DataFrame({
            'Atom': elements,
            'X': coords[:, 0],
            'Y': coords[:, 1],
            'Z': coords[:, 2],
            'Importance': importance_scores
        })

        df = df.sort_values('Importance', ascending=False)

        # 打印top-k重要原子
        print(f"\nTop {top_k} Most Important Atoms:")
        print("="*60)
        print(df.head(top_k).to_string(index=False))
        print("="*60)

        # 可视化
        fig = plt.figure(figsize=(15, 5))

        # 1. 条形图：所有原子的重要性
        ax1 = fig.add_subplot(131)
        ax1.bar(range(len(importance_scores)), importance_scores)
        ax1.set_xlabel('Atom Index')
        ax1.set_ylabel('Importance Score')
        ax1.set_title('Atom Importance Distribution')
        ax1.grid(True, alpha=0.3)

        # 2. 按元素类型的平均重要性
        ax2 = fig.add_subplot(132)
        element_importance = df.groupby('Atom')['Importance'].mean().sort_values(ascending=False)
        ax2.barh(range(len(element_importance)), element_importance.values)
        ax2.set_yticks(range(len(element_importance)))
        ax2.set_yticklabels(element_importance.index)
        ax2.set_xlabel('Average Importance')
        ax2.set_title('Importance by Element Type')
        ax2.grid(True, alpha=0.3)

        # 3. 3D结构可视化（简化版：2D投影）
        ax3 = fig.add_subplot(133)
        scatter = ax3.scatter(coords[:, 0], coords[:, 1],
                            c=importance_scores, s=200,
                            cmap='YlOrRd', alpha=0.7, edgecolors='black')

        # 标注top-k原子
        top_indices = df.head(top_k).index[:top_k]
        for idx in top_indices:
            ax3.annotate(elements[idx], (coords[idx, 0], coords[idx, 1]),
                        fontsize=9, fontweight='bold')

        ax3.set_xlabel('X Coordinate (Å)')
        ax3.set_ylabel('Y Coordinate (Å)')
        ax3.set_title('Spatial Distribution (X-Y Projection)')
        plt.colorbar(scatter, ax=ax3, label='Importance')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nAtom importance visualization saved to {save_path}")
        else:
            plt.show()

        plt.close()

        return df

    def visualize_feature_space(self, graph_features, text_features, labels=None,
                               method='tsne', save_path=None):
        """
        可视化图和文本特征在嵌入空间的分布

        Args:
            graph_features: 图特征 [N, D]
            text_features: 文本特征 [N, D]
            labels: 标签（用于着色）
            method: 'tsne' 或 'pca'
            save_path: 保存路径
        """
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA

        # 合并特征
        all_features = np.vstack([graph_features, text_features])

        # 降维
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
        else:
            reducer = PCA(n_components=2)

        embedded = reducer.fit_transform(all_features)

        # 分离图和文本特征
        n = len(graph_features)
        graph_embedded = embedded[:n]
        text_embedded = embedded[n:]

        # 可视化
        plt.figure(figsize=(12, 5))

        # 左图：分离显示
        ax1 = plt.subplot(121)
        ax1.scatter(graph_embedded[:, 0], graph_embedded[:, 1],
                   c='blue', alpha=0.6, s=50, label='Graph', marker='o')
        ax1.scatter(text_embedded[:, 0], text_embedded[:, 1],
                   c='red', alpha=0.6, s=50, label='Text', marker='^')

        # 绘制配对连线
        for i in range(min(n, 100)):  # 只绘制前100个以避免过于密集
            ax1.plot([graph_embedded[i, 0], text_embedded[i, 0]],
                    [graph_embedded[i, 1], text_embedded[i, 1]],
                    'gray', alpha=0.2, linewidth=0.5)

        ax1.set_xlabel(f'{method.upper()} Component 1')
        ax1.set_ylabel(f'{method.upper()} Component 2')
        ax1.set_title('Graph-Text Feature Alignment')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 右图：根据标签着色（如果提供）
        ax2 = plt.subplot(122)
        if labels is not None:
            scatter1 = ax2.scatter(graph_embedded[:, 0], graph_embedded[:, 1],
                                  c=labels, cmap='viridis', alpha=0.6, s=50,
                                  marker='o', edgecolors='black', linewidth=0.5)
            scatter2 = ax2.scatter(text_embedded[:, 0], text_embedded[:, 1],
                                  c=labels, cmap='viridis', alpha=0.6, s=50,
                                  marker='^', edgecolors='black', linewidth=0.5)
            plt.colorbar(scatter1, ax=ax2, label='Target Value')
            ax2.set_title('Features Colored by Target Value')
        else:
            ax2.scatter(graph_embedded[:, 0], graph_embedded[:, 1],
                       c='blue', alpha=0.6, s=50, label='Graph', marker='o')
            ax2.scatter(text_embedded[:, 0], text_embedded[:, 1],
                       c='red', alpha=0.6, s=50, label='Text', marker='^')
            ax2.set_title('Feature Distribution')
            ax2.legend()

        ax2.set_xlabel(f'{method.upper()} Component 1')
        ax2.set_ylabel(f'{method.upper()} Component 2')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature space visualization saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def explain_prediction(self, graph, atoms_object, text_description,
                          true_value=None, save_dir=None):
        """
        为单个预测生成完整的解释

        Args:
            graph: DGL graph
            atoms_object: jarvis.core.atoms.Atoms对象
            text_description: 文本描述
            true_value: 真实值（可选）
            save_dir: 保存目录

        Returns:
            explanation: 解释字典
        """
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "="*80)
        print("预测结果可解释性分析")
        print("="*80)

        # 1. 获取预测
        with torch.no_grad():
            output = self.model(graph.to(self.device), return_features=True)
            if isinstance(output, dict):
                prediction = output['predictions'].cpu().item()
            else:
                prediction = output.cpu().item()

        print(f"\n预测值: {prediction:.4f}")
        if true_value is not None:
            print(f"真实值: {true_value:.4f}")
            print(f"误差: {abs(prediction - true_value):.4f}")

        # 2. 计算原子重要性
        print("\n计算原子重要性...")
        atom_importance = self.compute_atom_importance(graph, method='gradient')

        # 3. 可视化原子重要性
        atom_df = None
        if atoms_object is not None:
            atom_save_path = save_dir / 'atom_importance.png' if save_dir else None
            atom_df = self.visualize_atom_importance(
                atoms_object, atom_importance,
                save_path=atom_save_path, top_k=10
            )

        # 4. 提取特征（如果模型支持）
        explanation = {
            'prediction': prediction,
            'true_value': true_value,
            'error': abs(prediction - true_value) if true_value is not None else None,
            'atom_importance': atom_importance.tolist(),
            'text_description': text_description,
            'num_atoms': len(atom_importance),
        }

        # 5. 保存解释
        if save_dir:
            with open(save_dir / 'explanation.json', 'w') as f:
                json.dump(explanation, f, indent=2)
            print(f"\n解释已保存到: {save_dir / 'explanation.json'}")

        print("\n" + "="*80)

        return explanation


def create_interpretability_report(analyzer, test_loader, save_dir, num_samples=5):
    """
    为测试集样本创建可解释性报告

    Args:
        analyzer: InterpretabilityAnalyzer实例
        test_loader: 测试数据加载器
        save_dir: 保存目录
        num_samples: 分析的样本数量
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n生成可解释性报告...")
    print(f"分析 {num_samples} 个样本")
    print(f"保存目录: {save_dir}\n")

    all_graph_features = []
    all_text_features = []
    all_labels = []

    for i, batch in enumerate(test_loader):
        if i >= num_samples:
            break

        g, lg, text, labels = batch
        g = g.to(analyzer.device)

        # 为每个样本创建单独的解释
        sample_dir = save_dir / f'sample_{i:03d}'
        sample_dir.mkdir(exist_ok=True)

        # 这里需要从batch中提取atoms对象
        # 简化处理：只分析特征

        with torch.no_grad():
            output = analyzer.model(g, return_features=True)
            if isinstance(output, dict):
                prediction = output['predictions']
                if 'graph_features' in output:
                    all_graph_features.append(output['graph_features'].cpu().numpy())
                if 'text_features' in output:
                    all_text_features.append(output['text_features'].cpu().numpy())

            all_labels.append(labels.cpu().numpy())

    # 可视化特征空间
    if all_graph_features and all_text_features:
        graph_features = np.vstack(all_graph_features)
        text_features = np.vstack(all_text_features)
        labels = np.concatenate(all_labels)

        analyzer.visualize_feature_space(
            graph_features, text_features, labels,
            method='tsne',
            save_path=save_dir / 'feature_space_tsne.png'
        )

        analyzer.visualize_feature_space(
            graph_features, text_features, labels,
            method='pca',
            save_path=save_dir / 'feature_space_pca.png'
        )

    print(f"\n✓ 可解释性报告已生成: {save_dir}")
