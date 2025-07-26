import streamlit as st
import torch
import numpy as np
import os
import tempfile
from PIL import Image
from torchvision import transforms
from utils.model_utils import load_model, preprocess_image, predict_image, get_class_display_name

# === Page Configuration ===
st.set_page_config(
    page_title="DNN Model - Lettuce Leaf Classifier",
    layout="centered",
    initial_sidebar_state="expanded"
)

# === Custom CSS for DNN page ===
st.markdown("""
<style>
    .dnn-header {
        text-align: center;
        color: #2E86AB;
        font-size: 2.5rem;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, #2E86AB 0%, #A23B72 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .model-info {
        background: linear-gradient(135deg, #2E86AB 0%, #A23B72 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
    }
    .prediction-result {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #2E86AB;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .confidence-bar {
        background-color: #e9ecef;
        border-radius: 10px;
        padding: 0.5rem;
        margin: 0.5rem 0;
    }
    .confidence-fill {
        background: linear-gradient(90deg, #2E86AB 0%, #A23B72 100%);
        height: 20px;
        border-radius: 8px;
        transition: width 0.3s ease;
    }
    .upload-area {
        border: 2px dashed #2E86AB;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #f8f9fa;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# === Navigation Buttons ===
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🏠 Home", key="home_dnn", use_container_width=True):
        st.switch_page("app")
with col2:
    if st.button("🔬 Classical ML", key="classical_dnn", use_container_width=True):
        st.switch_page("Classical_ML")
with col3:
    if st.button("⚖️ Compare Models", key="compare_dnn", use_container_width=True):
        st.switch_page("performace_comparison")

# === Main Header ===
st.markdown('<h1 class="dnn-header">🧠 Deep Neural Network (TinyVGG)</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666; font-size: 1.2rem;">Advanced CNN-based Lettuce Leaf Disease Classification</p>', unsafe_allow_html=True)

# === Model Information Section ===
with st.container():
    st.markdown('<div class="model-info">', unsafe_allow_html=True)
    st.markdown("""
    ## 🧠 TinyVGG Architecture
    
    **Model Type**: Custom Convolutional Neural Network  
    **Input Size**: 64x64 RGB images  
    **Architecture**: 
    - 2 Convolutional Blocks with ReLU activation
    - MaxPooling layers for feature reduction
    - Fully connected classifier layer
    - Softmax output for 8 disease classes
    
    **Training**: Custom dataset with PyTorch  
    **Optimizer**: Adam  
    **Loss Function**: CrossEntropyLoss
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# === Device Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load Class Names ===
@st.cache_data
def load_class_names():
    with open("classes.txt", "r") as f:
        return [line.strip() for line in f.readlines()]

class_names = load_class_names()

# === Load DNN Model ===
@st.cache_resource
def load_dnn_model():
    model = load_model("models/TinyVGG_DNN_model_0_weights.pth", device, num_classes=len(class_names))
    return model

dnn_model = load_dnn_model()

# === Define Image Transforms ===
custom_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# === Prediction Function ===
def predict_dnn_image(image):
    """Predict disease using DNN model"""
    image_tensor = preprocess_image(image, custom_transform)
    prediction, confidence = predict_image(dnn_model, image_tensor, class_names, device)
    return prediction, confidence

# === Upload Section ===
with st.container():
    st.markdown("## 📤 Upload Lettuce Leaf Image")
    st.markdown('<div class="upload-area">', unsafe_allow_html=True)
    st.markdown("**Drag and drop or click to upload a lettuce leaf image**")
    st.markdown("Supported formats: JPG, JPEG, PNG")
    st.markdown('</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose an image", 
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of a lettuce leaf for disease classification"
    )

# === Results Display ===
if uploaded_file is not None:
    with st.container():
        st.markdown("## 🔍 Analysis Results")
        
        # Load and display image
        image = Image.open(uploaded_file).convert("RGB")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            # Get prediction
            with st.spinner("🧠 Analyzing with DNN..."):
                prediction, confidence = predict_dnn_image(image)
            
            # Display results
            st.markdown('<div class="prediction-result">', unsafe_allow_html=True)
            st.markdown(f"### 🎯 Predicted Class")
            st.success(f"**{get_class_display_name(prediction)}**")
            
            st.markdown(f"### 📊 Confidence Score")
            st.info(f"**{confidence * 100:.2f}%**")
            
            # Confidence bar
            st.markdown("### 📈 Confidence Visualization")
            st.markdown('<div class="confidence-bar">', unsafe_allow_html=True)
            confidence_width = min(confidence * 100, 100)
            st.markdown(f'<div class="confidence-fill" style="width: {confidence_width}%;"></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

# === Class Information ===
with st.container():
    st.markdown("## 📚 Disease Categories")
    
    # Create a grid of disease categories
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🔬 Disease Types:**
        - **BACT**: Bacterial Diseases
        - **DML**: Downey Mildew on Lettuce
        - **PML**: Powdery Mildew on Lettuce
        - **SBL**: Septorial Blight on Lettuce
        """)
    
    with col2:
        st.markdown("""
        **🌱 Other Categories:**
        - **HLTY**: Healthy Leaves
        - **SPW**: Shepherd Purse Weed
        - **VIRL**: Viral Diseases
        - **WLBL**: Wilt and Leaf Blight
        """)

# === Model Performance Info ===
with st.container():
    st.markdown("## ⚡ Model Performance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Model Type", "CNN (TinyVGG)")
    
    with col2:
        st.metric("Input Size", "64x64")
    
    with col3:
        st.metric("Output Classes", "8")

# === Footer ===
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem 0;">
    <p>🧠 Deep Neural Network Model | Lettuce Leaf Disease Classifier</p>
    <p>Developed by <a href="https://github.com/amansah17" target="_blank">Aman Sah</a></p>
</div>
""", unsafe_allow_html=True) 