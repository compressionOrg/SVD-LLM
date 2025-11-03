#!/bin/bash

set -x

model="Enoch/llama-7b-hf"
model_name=$(echo "$model" | tr '/-' '_')

sparsity_ratios=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9) # 
# sparsity_ratios=(0.2)
whitening_nsamples=256
seed=3


export CUDA_VISIBLE_DEVICES=2,3

# run data whitening with 20% compression ratio
# python SVDLLM.py --model Enoch/llama-7b-hf --step 1 --ratio 0.2 --whitening_nsamples 256 --dataset wikitext2 --seed 3 --model_seq_len 2048 --save_path .
## you can also run the following command for low-resource gpu (ex. llama 7b will only need 15G gpu memory to compress) or to compress large-scale llm (ex. llama 65b)
# python SVDLLM.py --model jeffwan/llama-7b-hf --step 1 --ratio 0.2 --whitening_nsamples 256 --dataset wikitext2 --model_seq_len 2048 --save_path ./ --run_low_resource
create_whitening(){
    python SVDLLM.py \
    --model ${model} \
    --step 1 \
    --ratio $1 \
    --whitening_nsamples ${whitening_nsamples} \
    --dataset wikitext2 \
    --seed ${seed} \
    --model_seq_len 2048 \
    --save_path "profiles" >logs/${model_name}_whitening_only_ratio_${1}.log
}
    # --profiling_mat_path "profiles/${model_name}_profiling_wikitext2_${whitening_nsamples}_${seed}.pt" \

evaluate_whitening(){
    python SVDLLM.py \
    --step $1 \
    --model_path "profiles/${model_name}_whitening_only_${2}.pt" >logs/${model_name}_whitening_only_ratio_${2}eval_step${1}.log
}

for sparsity_ratio in "${sparsity_ratios[@]}"
do
    echo "Create  $model_name profile with ratio $sparsity_ratio"
    ratio=$(python3 -c "print(f'{1 - $sparsity_ratio:.1f}')")
    echo "ratio:$ratio"
    # create whitening 
    # create_whitening ${sparsity_ratio}
    # evaluate
    evaluate_whitening 4 "${ratio}"
    evaluate_whitening 5 "${ratio}"
done

set +x