from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, LlavaProcessor, LlavaForConditionalGeneration
import torch
from PIL import Image
import requests
import time
import numpy as np
from helper import *

# ------------------- set params -------------------

attn_layer = 0 # which LLM layer's attention map to visualize

model_id = "llava-hf/llava-v1.6-mistral-7b-hf"

quantization = False # set to True to load the model in 8-bit mode

processor = LlavaNextProcessor.from_pretrained(model_id)
model = LlavaNextForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
    load_in_8bit=quantization,
    attn_implementation="eager",
)
if quantization is False: # hot fix for: .to` is not supported for `4-bit` or `8-bit` bitsandbytes models. 
    # Please use the model as it is, since the model has already been set to the correct devices and casted to the correct `dtype`.
    model = model.to("cuda:0")

# -------------- set KVTO params --------------

model.config.fast_vlm_config = {
    "spatial_budget": 0,
    "alpha_vision_token_budget": 1,
    "beta_sub_images_budget": 0.5,
    "clip_attn_layer": 22,
}

# -------------- generate random input image --------------
image = get_random_image(h=200, w=400)
prompt = "[INST] <image>\nExplain the image in details. Tell about everything you see in the image.\n[/INST]"
inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")

with torch.inference_mode():
    output = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=500,
        use_cache=True,
        return_dict_in_generate=True,
        )
output_text = processor.decode(output['sequences'][0], skip_special_tokens=False)
inputs = processor(output_text, image, return_tensors='pt').to(0, torch.float16)
with torch.inference_mode():
    output = model(**inputs, output_attentions=True, return_dict=True)

llm_attn = output.attentions[attn_layer][0]
vit_to_llm_mapping = model.vit_to_llm_mapping[0]
# plot_attn_map(llm_attn, stride=10)

def calculate_kvt_distro(attn,vit_to_llm_mapping,n_percent=0.95):
    text_token_start_idx = vit_to_llm_mapping[-1]+1
    attn = attn[:, text_token_start_idx:, vit_to_llm_mapping] # Shape: (n_heads, n_text_tokens, n_visual_tokens)
    attn = torch.mean(attn, dim=0) # Shape: (n_text_tokens, n_visual_tokens)
    attn = torch.mean(attn, dim=0) # Shape: (n_visual_tokens)
    attn_sorted, _ = torch.sort(attn, descending=True)
    attn_sorted_cumsum = torch.cumsum(attn_sorted, dim=0)
    attn_sum = attn_sorted_cumsum[-1]
    n_tokens = torch.sum(attn_sorted_cumsum <= attn_sum * n_percent)
    print(f"Number of tokens that cover {n_percent*100}% of the attention: {n_tokens} which is {n_tokens/len(attn)*100}% of the total tokens.")

calculate_kvt_distro(llm_attn, vit_to_llm_mapping, n_percent=0.99)