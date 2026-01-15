#!/bin/bash
# Effective Rank SVD 压缩脚本 (新流程)
# 第一级分配: 层间余弦相似度 (Cosine Similarity)
# 第二级分配: 模块有效秩 (Effective Rank)

export CUDA_VISIBLE_DEVICES=3

MODEL="meta-llama/Llama-3.2-3B"
# MODEL="Enoch/llama-7b-hf"
MODEL_NAME=$(echo "$MODEL" | tr '/-' '_')
RATIOS=(0.3 0.4) # 0.3 0.4 0.5 0.6 0.7 0.8
SAVE_PATH="./output/cgsvr"
DATASET="wikitext2"
TEMP=0.4
LAYER_FLUCT_RATIO=-1
MODULE_FLUCT_RATIO=-1
mkdir -p $SAVE_PATH
mkdir -p ./logs_auto

echo "=========================================="
echo "SVD 压缩 (新流程 V23)，自动寻找最优 layer_fluctuation_ratio 和 module_fluctuation_ratio"
echo "模型: $MODEL"
# echo "目标压缩比: $RATIO (保留 $(python3 -c "print(1-$RATIO)")))"
echo "Strategy: Layer Cosine Similarity + Module Effective Rank"
echo "Compensation Limit: ${LIMIT}"
echo "Layer Fluctuation Ratio: ${LAYER_FLUCT_RATIO}"
echo "Module Fluctuation Ratio: ${MODULE_FLUCT_RATIO}"
echo "=========================================="

for RATIO in ${RATIOS[@]}
do
    # 运行 CGSVD_V23.py (新流程代码)
    python -u CGSVD_V23.py \
        --model $MODEL \
        --ratio $RATIO \
        --dataset $DATASET \
        --whitening_nsamples 256 \
        --model_seq_len 2048 \
        --save_path $SAVE_PATH \
        --min_rank_ratio 0.05 \
        --max_rank_ratio 0.95 \
        --layer_fluctuation_ratio $LAYER_FLUCT_RATIO \
        --module_fluctuation_ratio $MODULE_FLUCT_RATIO \
        --importance_metric angular \
        --use_ppl \
        --DEV cuda \
        --step 0 \
        2>&1 | tee ./logs_auto/${MODEL_NAME}_run_cgsvr_v23_angular_ratio_${RATIO}.log
            #     --enable_compensation \
done
echo "压缩完成!"
echo "模型保存于: $SAVE_PATH"
echo "日志保存于: ./logs_auto/${MODEL_NAME}_run_cgsvr_v23_angular_ratio_${RATIO}.log"
