#!/bin/bash
# 消融实验脚本集合
# 用法: ./run_ablation_experiments.sh <experiment_id>
# 例如: ./run_ablation_experiments.sh A0

# 公共参数
ROOT_DIR="/public/home/ghzhang/crysmmnet-main/dataset"
DATASET="jarvis"
PROPERTY="mbj_bandgap"
TRAIN_RATIO=0.8
VAL_RATIO=0.1
TEST_RATIO=0.1
BATCH_SIZE=32
EPOCHS=300
LR=5e-4
WEIGHT_DECAY=1e-5
WARMUP_STEPS=3000
NUM_WORKERS=4
SEED=42

# 设置环境变量以消除警告
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8

experiment=$1

if [ -z "$experiment" ]; then
    echo "=========================================="
    echo "消融实验脚本"
    echo "=========================================="
    echo ""
    echo "用法: ./run_ablation_experiments.sh <experiment_id>"
    echo ""
    echo "可用实验:"
    echo ""
    echo "=== 融合机制消融 (Fusion Mechanism) ==="
    echo "  A0  - 纯GNN (无跨模态融合)"
    echo "  A1  - 仅晚期融合"
    echo "  A2  - 纯TEXT (仅文本特征)"
    echo "  A3  - 仅细粒度注意力"
    echo "  A4  - 细粒度+晚期融合 [Baseline]"
    echo "  A6  - 全融合 (细粒度+晚期+中期)"
    echo ""
    echo "=== 注意力头数消融 (Number of Heads) ==="
    echo "  C1  - 1 head"
    echo "  C2  - 2 heads"
    echo "  C3  - 4 heads"
    echo "  C4  - 8 heads [Baseline]"
    echo "  C5  - 16 heads"
    echo ""
    echo "=== 隐藏维度消融 (Hidden Dimension) ==="
    echo "  D1  - 64 dim"
    echo "  D2  - 128 dim"
    echo "  D3  - 256 dim [Baseline]"
    echo "  D4  - 384 dim"
    echo ""
    echo "=== Dropout消融 ==="
    echo "  G1  - dropout=0.0"
    echo "  G2  - dropout=0.1"
    echo "  G3  - dropout=0.15 [Baseline]"
    echo "  G4  - dropout=0.2"
    echo "  G5  - dropout=0.3"
    echo ""
    echo "=== 融合方式消融 (已在代码中切换) ==="
    echo "  E1  - 拼接融合 (需修改代码)"
    echo "  E2  - 平均融合 [当前配置]"
    echo ""
    exit 0
fi

echo "=========================================="
echo "开始消融实验: $experiment"
echo "=========================================="

case $experiment in
    # ========== 融合机制消融 ==========

    # A0: 纯GNN (无跨模态融合)
    "A0")
        echo "实验 A0: 纯GNN (无跨模态融合)"
        python train_with_cross_modal_attention.py \
            --root_dir $ROOT_DIR \
            --dataset $DATASET \
            --property $PROPERTY \
            --train_ratio $TRAIN_RATIO \
            --val_ratio $VAL_RATIO \
            --test_ratio $TEST_RATIO \
            --batch_size $BATCH_SIZE \
            --epochs $EPOCHS \
            --learning_rate $LR \
            --weight_decay $WEIGHT_DECAY \
            --warmup_steps $WARMUP_STEPS \
            --alignn_layers 4 \
            --gcn_layers 4 \
            --hidden_features 256 \
            --use_fine_grained_attention 0 \
            --use_cross_modal 0 \
            --use_middle_fusion 0 \
            --output_dir ./ablation_A0_pure_gnn/ \
            --num_workers $NUM_WORKERS \
            --random_seed $SEED
        ;;

    # A1: 仅晚期融合
    "A1")
        echo "实验 A1: 仅晚期融合"
        python train_with_cross_modal_attention.py \
            --root_dir $ROOT_DIR \
            --dataset $DATASET \
            --property $PROPERTY \
            --train_ratio $TRAIN_RATIO \
            --val_ratio $VAL_RATIO \
            --test_ratio $TEST_RATIO \
            --batch_size $BATCH_SIZE \
            --epochs $EPOCHS \
            --learning_rate $LR \
            --weight_decay $WEIGHT_DECAY \
            --warmup_steps $WARMUP_STEPS \
            --alignn_layers 4 \
            --gcn_layers 4 \
            --hidden_features 256 \
            --use_fine_grained_attention 0 \
            --use_cross_modal 1 \
            --cross_modal_hidden_dim 256 \
            --cross_modal_num_heads 4 \
            --cross_modal_dropout 0.15 \
            --use_middle_fusion 0 \
            --output_dir ./ablation_A1_late_fusion_only/ \
            --num_workers $NUM_WORKERS \
            --random_seed $SEED
        ;;

    # A2: 纯TEXT (仅文本特征)
    "A2")
        echo "实验 A2: 纯TEXT (仅文本特征)"
        python train_with_cross_modal_attention.py \
            --root_dir $ROOT_DIR \
            --dataset $DATASET \
            --property $PROPERTY \
            --train_ratio $TRAIN_RATIO \
            --val_ratio $VAL_RATIO \
            --test_ratio $TEST_RATIO \
            --batch_size $BATCH_SIZE \
            --epochs $EPOCHS \
            --learning_rate $LR \
            --weight_decay $WEIGHT_DECAY \
            --warmup_steps $WARMUP_STEPS \
            --alignn_layers 4 \
            --gcn_layers 4 \
            --hidden_features 256 \
            --use_fine_grained_attention 0 \
            --use_cross_modal 0 \
            --use_middle_fusion 0 \
            --text_only 1 \
            --output_dir ./ablation_A2_text_only/ \
            --num_workers $NUM_WORKERS \
            --random_seed $SEED
        ;;

    # A3: 仅细粒度注意力 (无晚期融合)
    "A3")
        echo "实验 A3: 仅细粒度注意力"
        python train_with_cross_modal_attention.py \
            --root_dir $ROOT_DIR \
            --dataset $DATASET \
            --property $PROPERTY \
            --train_ratio $TRAIN_RATIO \
            --val_ratio $VAL_RATIO \
            --test_ratio $TEST_RATIO \
            --batch_size $BATCH_SIZE \
            --epochs $EPOCHS \
            --learning_rate $LR \
            --weight_decay $WEIGHT_DECAY \
            --warmup_steps $WARMUP_STEPS \
            --alignn_layers 4 \
            --gcn_layers 4 \
            --hidden_features 256 \
            --use_fine_grained_attention 1 \
            --fine_grained_hidden_dim 256 \
            --fine_grained_num_heads 8 \
            --fine_grained_dropout 0.15 \
            --fine_grained_use_projection 1 \
            --use_cross_modal 0 \
            --use_middle_fusion 0 \
            --output_dir ./ablation_A3_fine_grained_only/ \
            --num_workers $NUM_WORKERS \
            --random_seed $SEED
        ;;

    # A4: 细粒度+晚期融合 [Baseline]
    "A4")
        echo "实验 A4: 细粒度+晚期融合 [Baseline]"
        python train_with_cross_modal_attention.py \
            --root_dir $ROOT_DIR \
            --dataset $DATASET \
            --property $PROPERTY \
            --train_ratio $TRAIN_RATIO \
            --val_ratio $VAL_RATIO \
            --test_ratio $TEST_RATIO \
            --batch_size $BATCH_SIZE \
            --epochs $EPOCHS \
            --learning_rate $LR \
            --weight_decay $WEIGHT_DECAY \
            --warmup_steps $WARMUP_STEPS \
            --alignn_layers 4 \
            --gcn_layers 4 \
            --hidden_features 256 \
            --use_fine_grained_attention 1 \
            --fine_grained_hidden_dim 256 \
            --fine_grained_num_heads 8 \
            --fine_grained_dropout 0.15 \
            --fine_grained_use_projection 1 \
            --use_cross_modal 1 \
            --cross_modal_hidden_dim 256 \
            --cross_modal_num_heads 4 \
            --cross_modal_dropout 0.15 \
            --use_middle_fusion 0 \
            --output_dir ./ablation_A4_baseline/ \
            --num_workers $NUM_WORKERS \
            --random_seed $SEED
        ;;

    # A6: 全融合
    "A6")
        echo "实验 A6: 全融合 (细粒度+晚期+中期)"
        python train_with_cross_modal_attention.py \
            --root_dir $ROOT_DIR \
            --dataset $DATASET \
            --property $PROPERTY \
            --train_ratio $TRAIN_RATIO \
            --val_ratio $VAL_RATIO \
            --test_ratio $TEST_RATIO \
            --batch_size $BATCH_SIZE \
            --epochs $EPOCHS \
            --learning_rate $LR \
            --weight_decay $WEIGHT_DECAY \
            --warmup_steps $WARMUP_STEPS \
            --alignn_layers 4 \
            --gcn_layers 4 \
            --hidden_features 256 \
            --use_fine_grained_attention 1 \
            --fine_grained_hidden_dim 256 \
            --fine_grained_num_heads 8 \
            --fine_grained_dropout 0.15 \
            --fine_grained_use_projection 1 \
            --use_cross_modal 1 \
            --cross_modal_hidden_dim 256 \
            --cross_modal_num_heads 4 \
            --cross_modal_dropout 0.15 \
            --use_middle_fusion 1 \
            --middle_fusion_layers "2" \
            --middle_fusion_hidden_dim 128 \
            --middle_fusion_num_heads 2 \
            --middle_fusion_dropout 0.1 \
            --output_dir ./ablation_A6_full_fusion/ \
            --num_workers $NUM_WORKERS \
            --random_seed $SEED
        ;;

    # ========== 注意力头数消融 ==========

    # C1: 1 head
    "C1")
        echo "实验 C1: 1 attention head"
        python train_with_cross_modal_attention.py \
            --root_dir $ROOT_DIR \
            --dataset $DATASET \
            --property $PROPERTY \
            --train_ratio $TRAIN_RATIO \
            --val_ratio $VAL_RATIO \
            --test_ratio $TEST_RATIO \
            --batch_size $BATCH_SIZE \
            --epochs $EPOCHS \
            --learning_rate $LR \
            --weight_decay $WEIGHT_DECAY \
            --warmup_steps $WARMUP_STEPS \
            --alignn_layers 4 \
            --gcn_layers 4 \
            --hidden_features 256 \
            --use_fine_grained_attention 1 \
            --fine_grained_hidden_dim 256 \
            --fine_grained_num_heads 1 \
            --fine_grained_dropout 0.15 \
            --fine_grained_use_projection 1 \
            --use_cross_modal 1 \
            --cross_modal_hidden_dim 256 \
            --cross_modal_num_heads 4 \
            --cross_modal_dropout 0.15 \
            --use_middle_fusion 0 \
            --output_dir ./ablation_C1_heads_1/ \
            --num_workers $NUM_WORKERS \
            --random_seed $SEED
        ;;

    # C2: 2 heads
    "C2")
        echo "实验 C2: 2 attention heads"
        python train_with_cross_modal_attention.py \
            --root_dir $ROOT_DIR \
            --dataset $DATASET \
            --property $PROPERTY \
            --train_ratio $TRAIN_RATIO \
            --val_ratio $VAL_RATIO \
            --test_ratio $TEST_RATIO \
            --batch_size $BATCH_SIZE \
            --epochs $EPOCHS \
            --learning_rate $LR \
            --weight_decay $WEIGHT_DECAY \
            --warmup_steps $WARMUP_STEPS \
            --alignn_layers 4 \
            --gcn_layers 4 \
            --hidden_features 256 \
            --use_fine_grained_attention 1 \
            --fine_grained_hidden_dim 256 \
            --fine_grained_num_heads 2 \
            --fine_grained_dropout 0.15 \
            --fine_grained_use_projection 1 \
            --use_cross_modal 1 \
            --cross_modal_hidden_dim 256 \
            --cross_modal_num_heads 4 \
            --cross_modal_dropout 0.15 \
            --use_middle_fusion 0 \
            --output_dir ./ablation_C2_heads_2/ \
            --num_workers $NUM_WORKERS \
            --random_seed $SEED
        ;;

    # C3: 4 heads
    "C3")
        echo "实验 C3: 4 attention heads"
        python train_with_cross_modal_attention.py \
            --root_dir $ROOT_DIR \
            --dataset $DATASET \
            --property $PROPERTY \
            --train_ratio $TRAIN_RATIO \
            --val_ratio $VAL_RATIO \
            --test_ratio $TEST_RATIO \
            --batch_size $BATCH_SIZE \
            --epochs $EPOCHS \
            --learning_rate $LR \
            --weight_decay $WEIGHT_DECAY \
            --warmup_steps $WARMUP_STEPS \
            --alignn_layers 4 \
            --gcn_layers 4 \
            --hidden_features 256 \
            --use_fine_grained_attention 1 \
            --fine_grained_hidden_dim 256 \
            --fine_grained_num_heads 4 \
            --fine_grained_dropout 0.15 \
            --fine_grained_use_projection 1 \
            --use_cross_modal 1 \
            --cross_modal_hidden_dim 256 \
            --cross_modal_num_heads 4 \
            --cross_modal_dropout 0.15 \
            --use_middle_fusion 0 \
            --output_dir ./ablation_C3_heads_4/ \
            --num_workers $NUM_WORKERS \
            --random_seed $SEED
        ;;

    # C4: 8 heads [Baseline]
    "C4")
        echo "实验 C4: 8 attention heads [Baseline]"
        ./run_ablation_experiments.sh A4
        ;;

    # C5: 16 heads
    "C5")
        echo "实验 C5: 16 attention heads"
        python train_with_cross_modal_attention.py \
            --root_dir $ROOT_DIR \
            --dataset $DATASET \
            --property $PROPERTY \
            --train_ratio $TRAIN_RATIO \
            --val_ratio $VAL_RATIO \
            --test_ratio $TEST_RATIO \
            --batch_size $BATCH_SIZE \
            --epochs $EPOCHS \
            --learning_rate $LR \
            --weight_decay $WEIGHT_DECAY \
            --warmup_steps $WARMUP_STEPS \
            --alignn_layers 4 \
            --gcn_layers 4 \
            --hidden_features 256 \
            --use_fine_grained_attention 1 \
            --fine_grained_hidden_dim 256 \
            --fine_grained_num_heads 16 \
            --fine_grained_dropout 0.15 \
            --fine_grained_use_projection 1 \
            --use_cross_modal 1 \
            --cross_modal_hidden_dim 256 \
            --cross_modal_num_heads 4 \
            --cross_modal_dropout 0.15 \
            --use_middle_fusion 0 \
            --output_dir ./ablation_C5_heads_16/ \
            --num_workers $NUM_WORKERS \
            --random_seed $SEED
        ;;

    # ========== 隐藏维度消融 ==========

    # D1: 64 dim
    "D1")
        echo "实验 D1: hidden_dim=64"
        python train_with_cross_modal_attention.py \
            --root_dir $ROOT_DIR \
            --dataset $DATASET \
            --property $PROPERTY \
            --train_ratio $TRAIN_RATIO \
            --val_ratio $VAL_RATIO \
            --test_ratio $TEST_RATIO \
            --batch_size $BATCH_SIZE \
            --epochs $EPOCHS \
            --learning_rate $LR \
            --weight_decay $WEIGHT_DECAY \
            --warmup_steps $WARMUP_STEPS \
            --alignn_layers 4 \
            --gcn_layers 4 \
            --hidden_features 256 \
            --use_fine_grained_attention 1 \
            --fine_grained_hidden_dim 64 \
            --fine_grained_num_heads 8 \
            --fine_grained_dropout 0.15 \
            --fine_grained_use_projection 1 \
            --use_cross_modal 1 \
            --cross_modal_hidden_dim 256 \
            --cross_modal_num_heads 4 \
            --cross_modal_dropout 0.15 \
            --use_middle_fusion 0 \
            --output_dir ./ablation_D1_dim_64/ \
            --num_workers $NUM_WORKERS \
            --random_seed $SEED
        ;;

    # D2: 128 dim
    "D2")
        echo "实验 D2: hidden_dim=128"
        python train_with_cross_modal_attention.py \
            --root_dir $ROOT_DIR \
            --dataset $DATASET \
            --property $PROPERTY \
            --train_ratio $TRAIN_RATIO \
            --val_ratio $VAL_RATIO \
            --test_ratio $TEST_RATIO \
            --batch_size $BATCH_SIZE \
            --epochs $EPOCHS \
            --learning_rate $LR \
            --weight_decay $WEIGHT_DECAY \
            --warmup_steps $WARMUP_STEPS \
            --alignn_layers 4 \
            --gcn_layers 4 \
            --hidden_features 256 \
            --use_fine_grained_attention 1 \
            --fine_grained_hidden_dim 128 \
            --fine_grained_num_heads 8 \
            --fine_grained_dropout 0.15 \
            --fine_grained_use_projection 1 \
            --use_cross_modal 1 \
            --cross_modal_hidden_dim 256 \
            --cross_modal_num_heads 4 \
            --cross_modal_dropout 0.15 \
            --use_middle_fusion 0 \
            --output_dir ./ablation_D2_dim_128/ \
            --num_workers $NUM_WORKERS \
            --random_seed $SEED
        ;;

    # D3: 256 dim [Baseline]
    "D3")
        echo "实验 D3: hidden_dim=256 [Baseline]"
        ./run_ablation_experiments.sh A4
        ;;

    # D4: 384 dim
    "D4")
        echo "实验 D4: hidden_dim=384"
        python train_with_cross_modal_attention.py \
            --root_dir $ROOT_DIR \
            --dataset $DATASET \
            --property $PROPERTY \
            --train_ratio $TRAIN_RATIO \
            --val_ratio $VAL_RATIO \
            --test_ratio $TEST_RATIO \
            --batch_size 16 \
            --epochs $EPOCHS \
            --learning_rate $LR \
            --weight_decay $WEIGHT_DECAY \
            --warmup_steps $WARMUP_STEPS \
            --alignn_layers 4 \
            --gcn_layers 4 \
            --hidden_features 256 \
            --use_fine_grained_attention 1 \
            --fine_grained_hidden_dim 384 \
            --fine_grained_num_heads 8 \
            --fine_grained_dropout 0.15 \
            --fine_grained_use_projection 1 \
            --use_cross_modal 1 \
            --cross_modal_hidden_dim 256 \
            --cross_modal_num_heads 4 \
            --cross_modal_dropout 0.15 \
            --use_middle_fusion 0 \
            --output_dir ./ablation_D4_dim_384/ \
            --num_workers $NUM_WORKERS \
            --random_seed $SEED
        ;;

    # ========== Dropout消融 ==========

    # G1: dropout=0.0
    "G1")
        echo "实验 G1: dropout=0.0"
        python train_with_cross_modal_attention.py \
            --root_dir $ROOT_DIR \
            --dataset $DATASET \
            --property $PROPERTY \
            --train_ratio $TRAIN_RATIO \
            --val_ratio $VAL_RATIO \
            --test_ratio $TEST_RATIO \
            --batch_size $BATCH_SIZE \
            --epochs $EPOCHS \
            --learning_rate $LR \
            --weight_decay $WEIGHT_DECAY \
            --warmup_steps $WARMUP_STEPS \
            --alignn_layers 4 \
            --gcn_layers 4 \
            --hidden_features 256 \
            --use_fine_grained_attention 1 \
            --fine_grained_hidden_dim 256 \
            --fine_grained_num_heads 8 \
            --fine_grained_dropout 0.0 \
            --fine_grained_use_projection 1 \
            --use_cross_modal 1 \
            --cross_modal_hidden_dim 256 \
            --cross_modal_num_heads 4 \
            --cross_modal_dropout 0.0 \
            --use_middle_fusion 0 \
            --output_dir ./ablation_G1_dropout_0/ \
            --num_workers $NUM_WORKERS \
            --random_seed $SEED
        ;;

    # G2: dropout=0.1
    "G2")
        echo "实验 G2: dropout=0.1"
        python train_with_cross_modal_attention.py \
            --root_dir $ROOT_DIR \
            --dataset $DATASET \
            --property $PROPERTY \
            --train_ratio $TRAIN_RATIO \
            --val_ratio $VAL_RATIO \
            --test_ratio $TEST_RATIO \
            --batch_size $BATCH_SIZE \
            --epochs $EPOCHS \
            --learning_rate $LR \
            --weight_decay $WEIGHT_DECAY \
            --warmup_steps $WARMUP_STEPS \
            --alignn_layers 4 \
            --gcn_layers 4 \
            --hidden_features 256 \
            --use_fine_grained_attention 1 \
            --fine_grained_hidden_dim 256 \
            --fine_grained_num_heads 8 \
            --fine_grained_dropout 0.1 \
            --fine_grained_use_projection 1 \
            --use_cross_modal 1 \
            --cross_modal_hidden_dim 256 \
            --cross_modal_num_heads 4 \
            --cross_modal_dropout 0.1 \
            --use_middle_fusion 0 \
            --output_dir ./ablation_G2_dropout_01/ \
            --num_workers $NUM_WORKERS \
            --random_seed $SEED
        ;;

    # G3: dropout=0.15 [Baseline]
    "G3")
        echo "实验 G3: dropout=0.15 [Baseline]"
        ./run_ablation_experiments.sh A4
        ;;

    # G4: dropout=0.2
    "G4")
        echo "实验 G4: dropout=0.2"
        python train_with_cross_modal_attention.py \
            --root_dir $ROOT_DIR \
            --dataset $DATASET \
            --property $PROPERTY \
            --train_ratio $TRAIN_RATIO \
            --val_ratio $VAL_RATIO \
            --test_ratio $TEST_RATIO \
            --batch_size $BATCH_SIZE \
            --epochs $EPOCHS \
            --learning_rate $LR \
            --weight_decay $WEIGHT_DECAY \
            --warmup_steps $WARMUP_STEPS \
            --alignn_layers 4 \
            --gcn_layers 4 \
            --hidden_features 256 \
            --use_fine_grained_attention 1 \
            --fine_grained_hidden_dim 256 \
            --fine_grained_num_heads 8 \
            --fine_grained_dropout 0.2 \
            --fine_grained_use_projection 1 \
            --use_cross_modal 1 \
            --cross_modal_hidden_dim 256 \
            --cross_modal_num_heads 4 \
            --cross_modal_dropout 0.2 \
            --use_middle_fusion 0 \
            --output_dir ./ablation_G4_dropout_02/ \
            --num_workers $NUM_WORKERS \
            --random_seed $SEED
        ;;

    # G5: dropout=0.3
    "G5")
        echo "实验 G5: dropout=0.3"
        python train_with_cross_modal_attention.py \
            --root_dir $ROOT_DIR \
            --dataset $DATASET \
            --property $PROPERTY \
            --train_ratio $TRAIN_RATIO \
            --val_ratio $VAL_RATIO \
            --test_ratio $TEST_RATIO \
            --batch_size $BATCH_SIZE \
            --epochs $EPOCHS \
            --learning_rate $LR \
            --weight_decay $WEIGHT_DECAY \
            --warmup_steps $WARMUP_STEPS \
            --alignn_layers 4 \
            --gcn_layers 4 \
            --hidden_features 256 \
            --use_fine_grained_attention 1 \
            --fine_grained_hidden_dim 256 \
            --fine_grained_num_heads 8 \
            --fine_grained_dropout 0.3 \
            --fine_grained_use_projection 1 \
            --use_cross_modal 1 \
            --cross_modal_hidden_dim 256 \
            --cross_modal_num_heads 4 \
            --cross_modal_dropout 0.3 \
            --use_middle_fusion 0 \
            --output_dir ./ablation_G5_dropout_03/ \
            --num_workers $NUM_WORKERS \
            --random_seed $SEED
        ;;

    *)
        echo "未知实验ID: $experiment"
        echo "运行 ./run_ablation_experiments.sh 查看可用实验列表"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "实验 $experiment 完成!"
echo "=========================================="
