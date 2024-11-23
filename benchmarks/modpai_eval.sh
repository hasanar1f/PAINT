#!/bin/bash

# <!-- single round evaluation -->
python modPAI/pope_eval.py --model MODEL_NAME --data-path /path/to/COCO --pope-type random --use-attn --alpha 0.2 --use-cfg --gamma 1.1 --start-layer 2 --end-layer 32

# <!-- multi round evaluation -->
python pope_chat_eval.py --model MODEL_NAME --data-path /path/to/COCO --pope-type random --use-attn --alpha 0.2 --use-cfg --gamma 1.1 --start-layer 2 --end-layer 32

# Calculate POPE using the answer file
python pope_ans.py --ans_file /path/to/answer.json



# <!-- chair evaluation -->
python chair_eval.py --model MODEL_NAME --data-path /path/to/COCO --use-attn --alpha 0.2 --use-cfg --gamma 1.1 --start-layer 2 --end-layer 32

# <!-- chair evaluation -->
python chair.py --cap_file /path/to/jsonl