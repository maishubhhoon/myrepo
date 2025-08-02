# app.py

import streamlit as st
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, GPT2Tokenizer, GPT2LMHeadModel
from PIL import Image

# -----------------------------
# 🧠 Load Models
# -----------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_models():
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    gpt2_model.eval()
    return blip_processor, blip_model, tokenizer, gpt2_model

blip_processor, blip_model, tokenizer, gpt2_model = load_models()

# -----------------------------
# 🔍 Image Captioning
# -----------------------------

def image_to_caption(image):
    image = image.convert("RGB")
    inputs = blip_processor(image, return_tensors="pt").to(device)
    with torch.no_grad():
        caption_ids = blip_model.generate(**inputs)
        caption = blip_processor.decode(caption_ids[0], skip_special_tokens=True)
    return caption

# -----------------------------
# ✍️ Text to Story
# -----------------------------

def generate_story(prompt, max_length=200):
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    outputs = gpt2_model.generate(
        inputs,
        max_length=max_length,
        do_sample=True,
        temperature=0.9,
        top_k=50,
        top_p=0.95
    )
    story = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return story

# -----------------------------
# 🎨 Streamlit UI
# -----------------------------

st.set_page_config(page_title="AI Procedural Story Generator", layout="centered")
st.title("🎮 Procedural Narrative Generator with Multi-Modal AI")

st.markdown("Generate unique fantasy stories using image or text prompts powered by BLIP + GPT-2.")

input_mode = st.radio("Choose your input mode:", ["Upload Image", "Enter Text"])

if input_mode == "Upload Image":
    uploaded_image = st.file_uploader("Upload an image to analyze", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        img = Image.open(uploaded_image)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        st.info("Analyzing image and generating caption...")
        caption = image_to_caption(img)
        st.success(f"🖼️ Caption Detected: '{caption}'")

        prompt = f"In a distant world, {caption} begins an epic journey. "
        if st.button("📜 Generate Story"):
            story = generate_story(prompt)
            st.subheader("📝 Generated Story")
            st.write(story)

elif input_mode == "Enter Text":
    user_text = st.text_input("Describe your character or setting:")
    if user_text:
        prompt = f"In a distant world, {user_text} begins an epic journey. "
        if st.button("📜 Generate Story"):
            story = generate_story(prompt)
            st.subheader("📝 Generated Story")
            st.write(story)
