import torch
from PIL import Image
from model_loader import ModelLoader
from attention import llama_modify
from constants import INSTRUCTION_TEMPLATE, SYSTEM_MESSAGE


# Helper function to preprocess and load an image
def load_image(image_path, image_processor):
    image = Image.open(image_path).convert("RGB")
    processed_image = image_processor(image)
    processed_image = torch.tensor(processed_image)
    return processed_image

# Function to generate a caption for an image
def generate_caption(model, image, prompt, start_layer, end_layer, use_attn, alpha):
    query = [prompt]
    template = INSTRUCTION_TEMPLATE[model.model_name]
    template = SYSTEM_MESSAGE + template
    questions, kwargs = model.prepare_inputs_for_model(template, query, image)
    
    # Modify the model for multimodal input if necessary
    llama_modify(
        model.llm_model,
        start_layer=start_layer,
        end_layer=end_layer,
        use_attn=use_attn,
        alpha=alpha,
        use_cfg=False,
        img_start_idx=model.img_start_idx,
        img_end_idx=model.img_end_idx,
    )
    
    # Perform inference
    with torch.inference_mode():
        outputs = model.llm_model.generate(
            do_sample=False,
            max_new_tokens=512,
            use_cache=True,
            num_beams=1,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            **kwargs,
        )
    
    return model.decode(outputs)[0]

if __name__ == "__main__":
    # Configuration
    model_name = "llava-1.5"  
    image_path = "images/kitchen.jpg" 
    prompt = "Please help me describe the image in detail."
    start_layer = 2
    end_layer = 32
    use_attn = True
    alpha = 0.5
    
    # Load the model
    model_loader = ModelLoader(model_name)
    
    # Load the image
    image = load_image(image_path, model_loader.image_processor)
    
    # Generate a caption
    caption = generate_caption(model_loader, image, prompt, start_layer, end_layer, use_attn, alpha)
    
   