#!/bin/bash
# 可合成性分类训练脚本
# 用于预测材料是否可合成（二分类任务）

echo "=========================================="
echo "开始可合成性分类模型训练"
echo "任务类型: 二分类"
echo "数据规模: 根据数据集确定"
echo "=========================================="

# 检查数据集属性参数
PROPERTY=${1:-"synthesizable"}  # 默认属性名，可通过参数修改

echo "预测属性: $PROPERTY"
echo ""

python train_with_cross_modal_attention.py \
    --root_dir /public/home/ghzhang/crysmmnet-main/dataset \
    --dataset jarvis \
    --property $PROPERTY \
    \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1 \
    \
    --batch_size 32 \
    --epochs 200 \
    --learning_rate 5e-4 \
    --weight_decay 1e-5 \
    --warmup_steps 2000 \
    \
    --alignn_layers 4 \
    --gcn_layers 4 \
    --hidden_features 256 \
    \
    --use_fine_grained_attention 1 \
    --fine_grained_hidden_dim 256 \
    --fine_grained_num_heads 8 \
    --fine_grained_dropout 0.15 \
    --fine_grained_use_projection 1 \
    \
    --use_cross_modal 1 \
    --cross_modal_hidden_dim 256 \
    --cross_modal_num_heads 4 \
    --cross_modal_dropout 0.15 \
    \
    --classification 1 \
    --num_classes 2 \
    \
    --output_dir ./output_synthesizability_classification \
    --num_workers 4 \
    --random_seed 42

echo "=========================================="
echo "训练完成！"
echo "模型保存在: ./output_synthesizability_classification/"
echo "=========================================="
