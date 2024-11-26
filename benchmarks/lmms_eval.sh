#!/bin/bash

# todo

python3 -m accelerate.commands.launch \
    --num_processes=1 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="liuhaotian/llava-v1.5-7b" \
    --tasks mme \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.5_POPE \
    --output_path ./lmms_log/