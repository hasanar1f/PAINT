#!/bin/bash

# single round evaluation
python modPAI/pope_eval.py --model llava-1.5 --batch_size 1  --data-path modPAI/coco/val2014 --pope-type random --use-attn --alpha 0.5 --use-cfg --gamma 1.1 --start-layer 2 --end-layer 32

# multi round evaluation
python modPAI/pope_chat_eval.py --model llava-1.5 --data-path modPAI/coco/val2014 --pope-type random --use-attn --alpha 0.5 --use-cfg --gamma 1.1 --start-layer 2 --end-layer 32

# Calculate POPE
python modPAI/pope_ans.py --ans_file results/pope/


# CHAIR 

conda activate modpai

python modPAI/chair_eval.py --model llava-1.5 --data-path modPAI/coco/val2014 --use-attn --alpha 0.9 --beta 0.3 --use-cfg --gamma 1.1 --spatial 0 --start-layer 2 --end-layer 32

python modPAI/chair.py --cap_file results/chair/llava-1.5/chair_eval_layers_2-32_tokens_512_bs_1_attn_0.5_cfg_1.1.jsonl

# remove the last ans or you can rename the file with hyper parameters
rm results/chair/llava-1.5/chair_eval_layers_2-32_tokens_512_bs_1_attn_0.5_cfg_1.1.jsonl 