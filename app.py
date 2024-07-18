import streamlit as st
from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

class CFG:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed = 42
    generator = torch.Generator(device).manual_seed(seed)
    image_gen_steps = 35
    image_gen_model_id = 'runwayml/stable-diffusion-v1-5'
    image_gen_size = (400, 400)
    image_gen_guidance_scale = 9
    prompt_gen_model_id = 'gpt2'
    prompt_dataset_size = 6
    prompt_max_length = 12

@st.cache_resource
def load_model():
    return StableDiffusionPipeline.from_pretrained(
        CFG.image_gen_model_id, torch_dtype=torch.float16,
        revision='fp16', use_auth_token='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'
    ).to(CFG.device)

def generate_image(prompt, model):
    try:
        result = model(
            prompt, num_inference_steps=CFG.image_gen_steps,
            generator=CFG.generator,
            guidance_scale=CFG.image_gen_guidance_scale
        )
        image = result.images[0]
        image = image.resize(CFG.image_gen_size, Image.LANCZOS)
        return image
    except Exception as e:
        st.error(f"Error generating image: {e}")
        return None

st.title("Image Generation with Stable Diffusion")

prompt = st.text_input("Enter a prompt for the image generation:")

if st.button("Generate Image"):
    image_gen_model = load_model()
    image = generate_image(prompt, image_gen_model)
    if image:
        st.image(image, caption="Generated Image", use_column_width=True)
    else:
        st.error("Failed to generate image.")
