import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

st.set_page_config(page_title="Image Generation", page_icon="🖼️", layout="centered")

st.header("Image Generation 🖼️")

# Define the device to use (GPU if available, otherwise CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the pre-trained model from Hugging Face
@st.cache_resource
def load_model():
    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)
    return pipe

pipe = load_model()

# Input field for text prompt
text_prompt = st.text_input("Enter your text prompt here")

# Generate button
generate_button = st.button("Generate Image")

# Generate and display image
if generate_button and text_prompt:
    with st.spinner("Generating image..."):
        with torch.autocast(device):
            image = pipe(text_prompt).images[0]  # Access the image directly from the 'images' attribute
        image.save("generated_image.png")
        st.image(image, caption="Generated Image", use_column_width=True)
