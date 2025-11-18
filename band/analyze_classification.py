#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可合成性分类模型分析脚本
用于评估分类模型性能和生成可解释性分析
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
import seaborn as sns

def load_predictions(output_dir):
    """加载预测结果"""
    # 尝试加载测试集预测
    test_file = os.path.join(output_dir, "prediction_results.csv")
    val_file = os.path.join(output_dir, "prediction_results_val_set.csv")

    results = {}

    if os.path.exists(test_file):
        df = pd.read_csv(test_file)
        results['test'] = df
        print(f"加载测试集预测: {len(df)} 条记录")

    if os.path.exists(val_file):
        df = pd.read_csv(val_file)
        results['val'] = df
        print(f"加载验证集预测: {len(df)} 条记录")

    return results


def compute_classification_metrics(y_true, y_pred, y_prob=None):
    """计算分类指标"""
    metrics = {}

    # 基本指标
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # 二分类特有指标
    if len(np.unique(y_true)) == 2:
        metrics['precision_binary'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall_binary'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1_binary'] = f1_score(y_true, y_pred, zero_division=0)

        if y_prob is not None:
            metrics['auc_roc'] = roc_auc_score(y_true, y_prob)

    return metrics


def plot_confusion_matrix(y_true, y_pred, save_path, class_names=None):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)

    if class_names is None:
        class_names = ['Not Synthesizable', 'Synthesizable']

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    return cm


def plot_roc_curve(y_true, y_prob, save_path):
    """绘制ROC曲线"""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', label=f'ROC (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'r--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    return auc


def plot_precision_recall_curve(y_true, y_prob, save_path):
    """绘制Precision-Recall曲线"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, 'b-')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def analyze_prediction_distribution(y_true, y_pred, y_prob, save_path):
    """分析预测分布"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 预测概率分布
    if y_prob is not None:
        ax1 = axes[0]
        ax1.hist(y_prob[y_true == 0], bins=50, alpha=0.5, label='Class 0 (Not Synth)')
        ax1.hist(y_prob[y_true == 1], bins=50, alpha=0.5, label='Class 1 (Synth)')
        ax1.set_xlabel('Predicted Probability (Class 1)')
        ax1.set_ylabel('Count')
        ax1.set_title('Prediction Probability Distribution by True Class')
        ax1.legend()

    # 类别分布
    ax2 = axes[1]
    classes = ['Class 0\n(Not Synth)', 'Class 1\n(Synth)']
    true_counts = [np.sum(y_true == 0), np.sum(y_true == 1)]
    pred_counts = [np.sum(y_pred == 0), np.sum(y_pred == 1)]

    x = np.arange(len(classes))
    width = 0.35
    ax2.bar(x - width/2, true_counts, width, label='True', alpha=0.8)
    ax2.bar(x + width/2, pred_counts, width, label='Predicted', alpha=0.8)
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Count')
    ax2.set_title('Class Distribution')
    ax2.set_xticks(x)
    ax2.set_xticklabels(classes)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def generate_report(metrics, output_dir, dataset_name='test'):
    """生成分析报告"""
    report_path = os.path.join(output_dir, f'classification_report_{dataset_name}.txt')

    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write(f"Classification Analysis Report - {dataset_name.upper()} Set\n")
        f.write("=" * 60 + "\n\n")

        f.write("Performance Metrics:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Accuracy:  {metrics['accuracy']:.4f}\n")
        f.write(f"  Precision: {metrics['precision']:.4f}\n")
        f.write(f"  Recall:    {metrics['recall']:.4f}\n")
        f.write(f"  F1 Score:  {metrics['f1']:.4f}\n")

        if 'auc_roc' in metrics:
            f.write(f"  AUC-ROC:   {metrics['auc_roc']:.4f}\n")

        if 'precision_binary' in metrics:
            f.write("\nBinary Classification Metrics:\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Precision (Positive): {metrics['precision_binary']:.4f}\n")
            f.write(f"  Recall (Positive):    {metrics['recall_binary']:.4f}\n")
            f.write(f"  F1 (Positive):        {metrics['f1_binary']:.4f}\n")

        f.write("\n" + "=" * 60 + "\n")

    print(f"报告已保存: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description='Analyze classification model predictions')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory containing prediction results')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold for probability')
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("可合成性分类模型分析")
    print("=" * 60 + "\n")

    # 创建分析输出目录
    analysis_dir = os.path.join(args.output_dir, 'classification_analysis')
    os.makedirs(analysis_dir, exist_ok=True)

    # 加载预测结果
    results = load_predictions(args.output_dir)

    if not results:
        print("错误: 未找到预测结果文件")
        return

    # 分析每个数据集
    for dataset_name, df in results.items():
        print(f"\n分析 {dataset_name} 集...")

        # 提取数据
        y_true = df['target'].values

        # 检查是否是概率预测（0-1之间）还是已经是类别
        predictions = df['prediction'].values

        if predictions.max() <= 1.0 and predictions.min() >= 0.0:
            # 概率预测
            y_prob = predictions
            y_pred = (predictions >= args.threshold).astype(int)
        else:
            # 类别预测
            y_pred = predictions.astype(int)
            y_prob = None

        # 确保标签是整数
        y_true = y_true.astype(int)

        # 计算指标
        metrics = compute_classification_metrics(y_true, y_pred, y_prob)

        print(f"\n{dataset_name.upper()} Set Performance:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        if 'auc_roc' in metrics:
            print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")

        # 绘制混淆矩阵
        cm_path = os.path.join(analysis_dir, f'confusion_matrix_{dataset_name}.png')
        plot_confusion_matrix(y_true, y_pred, cm_path)
        print(f"  混淆矩阵: {cm_path}")

        # 绘制ROC曲线（仅当有概率预测时）
        if y_prob is not None:
            roc_path = os.path.join(analysis_dir, f'roc_curve_{dataset_name}.png')
            plot_roc_curve(y_true, y_prob, roc_path)
            print(f"  ROC曲线: {roc_path}")

            pr_path = os.path.join(analysis_dir, f'pr_curve_{dataset_name}.png')
            plot_precision_recall_curve(y_true, y_prob, pr_path)
            print(f"  PR曲线: {pr_path}")

        # 分析预测分布
        dist_path = os.path.join(analysis_dir, f'prediction_distribution_{dataset_name}.png')
        analyze_prediction_distribution(y_true, y_pred, y_prob, dist_path)
        print(f"  预测分布: {dist_path}")

        # 生成报告
        generate_report(metrics, analysis_dir, dataset_name)

        # 保存详细分类报告
        report = classification_report(y_true, y_pred,
                                       target_names=['Not Synthesizable', 'Synthesizable'],
                                       digits=4)
        report_path = os.path.join(analysis_dir, f'sklearn_report_{dataset_name}.txt')
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"  详细报告: {report_path}")

    print("\n" + "=" * 60)
    print("分析完成！")
    print(f"所有结果保存在: {analysis_dir}")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
