#!/bin/bash

set -x

model="meta-llama/Llama-2-7b-hf"
model_name=$(echo "$model" | tr '/-' '_')
FINE_TUNE_PATH=".lora"
# sparsity_ratios=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8) # 
sparsity_ratios=(0.9) #  0.4 0.5 0.6
whitening_nsamples=256
seed=3


export CUDA_VISIBLE_DEVICES=0

# run data whitening with 20% compression ratio
# python SVDLLM.py --model Enoch/llama-7b-hf --step 1 --ratio 0.2 --whitening_nsamples 256 --dataset wikitext2 --seed 3 --model_seq_len 2048 --save_path .
## you can also run the following command for low-resource gpu (ex. llama 7b will only need 15G gpu memory to compress) or to compress large-scale llm (ex. llama 65b)
# python SVDLLM.py --model jeffwan/llama-7b-hf --step 1 --ratio 0.2 --whitening_nsamples 256 --dataset wikitext2 --model_seq_len 2048 --save_path ./ --run_low_resource


finetune(){
    python utils/LoRA.py \
    --prune_model  "profiles/${model_name}_whitening_only_${1}.pt" \
    --data_path yahma/alpaca-cleaned \
    --output_dir $FINE_TUNE_PATH/${model_name}_whitening_only_${1} \
    --lora_target_modules q_u_proj,k_u_proj,v_u_proj,o_u_proj,gate_u_proj,down_u_proj,up_u_proj,q_v_proj,k_v_proj,v_v_proj,o_v_proj,gate_v_proj,down_v_proj,up_v_proj \
    --lora_r 8 \
    --num_epochs 2 \
    --learning_rate 1e-4 \
    --batch_size 64 >logs_lora/${model_name}_whitening_only_ratio_${1}_lora_all.log
}

evaluate(){
    python SVDLLM.py \
    --step 4 \
    --lora  ${FINE_TUNE_PATH}/${model_name}_whitening_only_${1} \
    --model_path "profiles/${model_name}_whitening_only_${1}.pt" >logs_lora/${model_name}_whitening_only_ratio_${1}eval_step4_lora_all.log
}

for sparsity_ratio in "${sparsity_ratios[@]}"
do
    echo "Create  $model_name profile with ratio $sparsity_ratio"
    ratio=$(python3 -c "print(f'{1 - $sparsity_ratio:.1f}')")
    echo "ratio:$ratio"

    finetune "${ratio}"
    evaluate  "${ratio}"

done

set +x