#!/bin/bash
# 配置4：全融合配置（实验性）
# 同时开启细粒度注意力 + 晚期融合 + 中期融合
# 适合：实验验证中期融合的额外收益
# 注意：计算开销大，GPU内存需求高

echo "=========================================="
echo "开始细粒度注意力模型训练 - 全融合配置（实验性）"
echo "数据规模: 18000样本"
echo "注意：同时开启三种融合机制，需要更多GPU内存"
echo "=========================================="

python train_with_cross_modal_attention.py \
    --root_dir /public/home/ghzhang/crysmmnet-main/dataset \
    --dataset jarvis \
    --property mbj_bandgap \
    \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1 \
    \
    --batch_size 16 \
    --epochs 300 \
    --learning_rate 3e-4 \
    --weight_decay 1e-5 \
    --warmup_steps 4000 \
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
    --use_middle_fusion 1 \
    --middle_fusion_layers "2" \
    --middle_fusion_hidden_dim 128 \
    --middle_fusion_num_heads 2 \
    --middle_fusion_dropout 0.1 \
    \
    --output_dir ./output_full_fusion \
    --num_workers 4 \
    --random_seed 42

echo "=========================================="
echo "训练完成！"
echo "模型保存在: ./output_full_fusion/"
echo "=========================================="
