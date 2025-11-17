#!/bin/bash
# 训练监控脚本
# 实时显示GPU使用情况和训练日志

echo "=========================================="
echo "训练监控工具"
echo "=========================================="
echo ""
echo "选择监控类型："
echo "1. GPU使用情况"
echo "2. 训练日志 (baseline)"
echo "3. 训练日志 (high_perf)"
echo "4. 训练日志 (fast)"
echo "5. 所有checkpoint列表"
echo ""
read -p "请输入选项 (1-5): " choice

case $choice in
    1)
        echo "实时监控GPU使用情况 (按Ctrl+C退出)..."
        watch -n 1 nvidia-smi
        ;;
    2)
        echo "监控baseline训练日志 (按Ctrl+C退出)..."
        if [ -f "output_full_training_baseline/train.log" ]; then
            tail -f output_full_training_baseline/train.log
        else
            echo "错误: 未找到训练日志文件"
            echo "请确认训练已开始"
        fi
        ;;
    3)
        echo "监控high_perf训练日志 (按Ctrl+C退出)..."
        if [ -f "output_full_training_high_perf/train.log" ]; then
            tail -f output_full_training_high_perf/train.log
        else
            echo "错误: 未找到训练日志文件"
        fi
        ;;
    4)
        echo "监控fast训练日志 (按Ctrl+C退出)..."
        if [ -f "output_full_training_fast/train.log" ]; then
            tail -f output_full_training_fast/train.log
        else
            echo "错误: 未找到训练日志文件"
        fi
        ;;
    5)
        echo "所有checkpoint列表："
        echo ""
        for dir in output_full_training_*/; do
            if [ -d "$dir" ]; then
                echo "目录: $dir"
                ls -lh "$dir"/*.pt 2>/dev/null | awk '{print "  "$9" ("$5")"}'
                echo ""
            fi
        done
        ;;
    *)
        echo "无效选项"
        exit 1
        ;;
esac
