#!/bin/bash

model_id="llava-hf/llava-v1.6-vicuna-13b-hf"
model_short_name="1.6-vicuna-13b"

benchmarks=("mme" "mmmu_val" "docvqa_val" "chartqa" "ocrbench" "vqav2_val")
# benchmarks=("vqav2_val")
alpha_vision_token_budgets=(0.2 0.4)
beta_sub_images_budgets=(0.5)
spatial_budgets=(0)
clip_attn_layers=(22)

# Refactored code with the specified parameters

for task in "${benchmarks[@]}"; do
    for alpha_vision_token_budget in "${alpha_vision_token_budgets[@]}"; do
        for spatial_budget in "${spatial_budgets[@]}"; do
            for clip_attn_layer in "${clip_attn_layers[@]}"; do
                for beta_sub_images_budget in "${beta_sub_images_budgets[@]}"; do
                    output_path="./logs/${model_short_name}/alpha_${alpha_vision_token_budget}_spatial_${spatial_budget}_clip_${clip_attn_layer}_beta_${beta_sub_images_budget}/"
                    echo "Running benchmark for task: $task with alpha_vision_token_budget: $alpha_vision_token_budget, spatial_budget: $spatial_budget, clip_attn_layer: $clip_attn_layer, beta_sub_images_budget: $beta_sub_images_budget"
                    accelerate launch --num_processes=1 -m lmms_eval --model llava_hf  \
                        --model_args pretrained=$model_id,device_map=cuda,alpha_vision_token_budget=$alpha_vision_token_budget,spatial_budget=$spatial_budget,clip_attn_layer=$clip_attn_layer,beta_sub_images_budget=$beta_sub_images_budget \
                        --tasks "$task" \
                        --batch_size 1 --log_samples --output_path "$output_path"
                done
            done
        done
    done
done

echo "All benchmarks are completed!"
