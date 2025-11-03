#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
set -x
# "meta-llama/Llama-2-13b-hf"  "meta-llama/Meta-Llama-3-8B" "Enoch/llama-13b-hf"  "facebook/opt-6.7b"
models=("Enoch/llama-7b-hf") # "meta-llama/Llama-2-7b-hf"


# sparsity_ratios=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
whitening_nsamples=256
seed=3
pt=./${model_name}_whitening_only_${seed}.pt

# ratio=$(echo "1 - $sparsity_ratio" | bc)



# run data whitening with 20% compression ratio
# python SVDLLM.py --model Enoch/llama-7b-hf --step 1 --ratio 0.2 --whitening_nsamples 256 --dataset wikitext2 --seed 3 --model_seq_len 2048 --save_path .
## you can also run the following command for low-resource gpu (ex. llama 7b will only need 15G gpu memory to compress) or to compress large-scale llm (ex. llama 65b)
# python SVDLLM.py --model jeffwan/llama-7b-hf --step 1 --ratio 0.2 --whitening_nsamples 256 --dataset wikitext2 --model_seq_len 2048 --save_path ./ --run_low_resource

run_whitening() {
    python SVDLLM.py \
        --model ${1} \
        --step 1 \
        --ratio 0.2 \
        --whitening_nsamples ${whitening_nsamples} \
        --dataset wikitext2 \
        --seed ${seed} \
        --model_seq_len 2048 \
        --save_path "profiles"
}

# run data whitening with 20% compression ratio
for model in "${models[@]}"; do
    echo "Create  $model profile"
    run_whitening $model
done

# evaluate the perplexity of llama_7b_whitening_0.2.pt


set +x
