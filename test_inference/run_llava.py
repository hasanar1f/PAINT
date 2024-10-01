import requests
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
import time
import numpy as np

# Initialize model and processor
model_id = "llava-hf/llava-1.5-7b-hf"
# model_id = "llava-hf/llava-1.5-13b-hf"

quantization = False

model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
    load_in_4bit=quantization,
)
processor = AutoProcessor.from_pretrained(model_id)


if quantization is False: # hot fix for: .to` is not supported for `4-bit` or `8-bit` bitsandbytes models. 
    # Please use the model as it is, since the model has already been set to the correct devices and casted to the correct `dtype`.
    model = model.to("cuda:0")

raw_image = Image.open("images/kitchen.jpg")
prompt = "USER: <image>\nExplain the image in detail.\nASSISTANT:"
inputs = processor(prompt, raw_image, return_tensors='pt').to("cuda:0", torch.float16)

# Measure generation stage time
generation_start_time = time.time()
with torch.inference_mode():
    output = model.generate(**inputs, do_sample=False, output_attentions=False, return_dict=True)

print(processor.decode(output[0], skip_special_tokens=True))