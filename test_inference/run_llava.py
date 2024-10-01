import requests
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
import time
import numpy as np
from helper import *

# Initialize model and processor
model_id = "llava-hf/llava-1.5-7b-hf"
# model_id = "llava-hf/llava-1.5-13b-hf"

quantization = True

model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
    load_in_4bit=quantization,
)

if quantization is False: # hot fix for: .to` is not supported for `4-bit` or `8-bit` bitsandbytes models. 
    # Please use the model as it is, since the model has already been set to the correct devices and casted to the correct `dtype`.
    model = model.to("cuda:0")

# adjust token budget 

model.config.fast_vlm_config = {
    "spatial_budget": 0,
    "alpha_vision_token_budget": 0.5,
}

processor = AutoProcessor.from_pretrained(model_id)

# Number of runs for averaging
num_runs = 1
generation_times = []
token_counts = []

for _ in range(num_runs):
    raw_image = get_random_image(h=400, w=400)
    prompt = "USER: <image>\nWhat are these?\nASSISTANT:"
    inputs = processor(prompt, raw_image, return_tensors='pt').to("cuda:0", torch.float16)

    # Measure generation stage time
    generation_start_time = time.time()
    with torch.inference_mode():
        output = model.generate(**inputs, do_sample=False, output_attentions=False, return_dict=True)
    generation_end_time = time.time()
    generation_times.append(generation_end_time - generation_start_time)

    # Count the number of generated tokens
    num_tokens_generated = len(output[0])
    token_counts.append(num_tokens_generated)

# Calculate averages
average_generation_time = np.mean(generation_times)
average_token_count = np.mean(token_counts)

# Calculate token generation throughput (tokens per second)
token_throughput = average_token_count / average_generation_time

# Print results
print(f"Average generation time: {average_generation_time:.4f} seconds")
print(f"Average number of tokens generated: {average_token_count:.2f}")
print(f"Token generation throughput: {token_throughput:.2f} tokens/second")

# Decode and print one of the outputs for verification
# print(processor.decode(output[0], skip_special_tokens=True))
# print(f"Time taken for last run: {generation_times[-1]:.4f} seconds")
