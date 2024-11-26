#!/bin/bash

# <!-- single round evaluation -->
python modPAI/pope_eval.py --model llava-1.5 --batch_size 1  --data-path modPAI/coco/val2014 --pope-type random --use-attn --alpha 0.5 --use-cfg --gamma 1.1 --start-layer 2 --end-layer 32

# <!-- multi round evaluation -->
python modPAI/pope_chat_eval.py --model llava-1.5 --data-path modPAI/coco/val2014 --pope-type random --use-attn --alpha 0.5 --use-cfg --gamma 1.1 --start-layer 2 --end-layer 32

# Calculate POPE using the answer file
python modPAI/pope_ans.py --ans_file results/pope/



# <!-- chair evaluation -->
python modPAI/chair_eval.py --model llava-1.5 --data-path modPAI/coco/val2014 --use-attn --alpha 0.5 --use-cfg --gamma 1.1 --start-layer 2 --end-layer 32

# <!-- chair evaluation -->
python modPAI/chair.py --cap_file results/chair/llava-1.5/chair_eval_layers_2-32_tokens_512_bs_1_attn_0.5_cfg_1.1.jsonl