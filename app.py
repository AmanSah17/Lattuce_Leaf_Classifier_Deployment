# app.py

import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from utils.model_utils import load_model, preprocess_image, predict_image

# === Device Configuration ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load class names ===
with open("classes.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# === Define image transforms ===
custom_image_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# === Load Model ===
model_path = "models/TinyVGG_DNN_model_0_weights.pth"
model = load_model(model_path, device, num_classes=len(class_names))

# === Streamlit Frontend ===
st.title("🧠 Lettuce Leaf Disease Classifier")
st.markdown("""
Welcome to the Lettuce Leaf Disease Classifier! Upload a clear image of a lettuce leaf, and the model will predict the disease class with confidence.
""")

# Sidebar with instructions and credits
st.sidebar.title("ℹ️ About & Instructions")
st.sidebar.markdown("""
**How to use:**
1. Click 'Browse files' to upload a lettuce leaf image (JPG, JPEG, PNG).
2. Wait for the prediction to appear below the image.
3. The predicted class and confidence will be shown.

**Example images:**
- (Add your example images here if available)

**Credits:**  
Developed by [Aman Sah](https://github.com/AmanSah17)
""")

uploaded_file = st.file_uploader("Upload a Lettuce Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Predicting..."):
        image_tensor = preprocess_image(image, custom_image_transform)
        prediction, confidence = predict_image(model, image_tensor, class_names, device)

    st.success(f"🟢 Predicted Class: `{prediction}`")
    st.info(f"Confidence: {confidence * 100:.2f}%")
