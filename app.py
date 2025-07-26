# app.py

import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from utils.model_utils import load_model, preprocess_image, predict_image, get_class_display_name

# === Page Configuration ===
st.set_page_config(
    page_title="Lettuce Leaf Disease Classifier",
    layout="centered",
    initial_sidebar_state="expanded"
)

# === Custom CSS for main app ===
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 3rem;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, #1f77b4 0%, #ff7f0e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.3rem;
        margin-bottom: 2rem;
    }
    .nav-buttons {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin: 2rem 0;
        flex-wrap: wrap;
    }
    .nav-button {
        background: linear-gradient(135deg, #1f77b4 0%, #ff7f0e 100%);
        color: white;
        padding: 1rem 2rem;
        border: none;
        border-radius: 10px;
        font-size: 1.1rem;
        font-weight: bold;
        cursor: pointer;
        text-decoration: none;
        display: inline-block;
        transition: transform 0.2s ease;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .nav-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .welcome-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        text-align: center;
    }
    .feature-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .prediction-result {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# === Main Header ===
st.markdown('<h1 class="main-header">🥬 Lettuce Leaf Disease Classifier</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced Machine Learning Solutions for Agricultural Disease Detection</p>', unsafe_allow_html=True)

# === Navigation Buttons ===
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🔬 Classical ML", key="classical_main", use_container_width=True):
        st.switch_page("Classical_ML")
with col2:
    if st.button("🧠 DNN Model", key="dnn_main", use_container_width=True):
        st.switch_page("DNN_Model")
with col3:
    if st.button("⚖️ Compare Models", key="compare_main", use_container_width=True):
        st.switch_page("performace_comparison")



# === Welcome Section ===
with st.container():
    st.markdown('<div class="welcome-card">', unsafe_allow_html=True)
    st.markdown("""
    ## 🎯 Welcome to Lettuce Leaf Disease Classification
    
    This application provides three powerful approaches to detect and classify diseases in lettuce leaves:
    
    - **🔬 Classical Machine Learning**: Traditional feature-based approach using brightness, texture, and histogram analysis
    - **🧠 Deep Neural Network**: Advanced CNN architecture (TinyVGG) for high-accuracy classification
    - **⚖️ Model Comparison**: Compare both approaches side-by-side for comprehensive analysis
    
    **Perfect for**: Farmers, Agronomists, Researchers, and Agricultural Technologists
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# === Quick Demo Section ===
with st.container():
    st.markdown("## 🚀 Quick Demo")
    st.markdown("Try our basic classification demo below, or use the navigation buttons above for specialized analysis.")

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

# === Quick Demo Upload ===
uploaded_file = st.file_uploader("📤 Quick Demo: Upload a Lettuce Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("🧠 Analyzing..."):
        image_tensor = preprocess_image(image, custom_image_transform)
        prediction, confidence = predict_image(model, image_tensor, class_names, device)

    st.markdown('<div class="prediction-result">', unsafe_allow_html=True)
    st.success(f"🟢 Predicted Class: `{get_class_display_name(prediction)}`")
    st.info(f"📊 Confidence: {confidence * 100:.2f}%")
    st.markdown('</div>', unsafe_allow_html=True)

# === Disease Categories Info ===
with st.container():
    st.markdown("## 📚 Disease Categories")
    
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

# === Developer Information ===
with st.container():
    st.markdown("## 👨‍💻 Developer Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Aman Sah**  
        🐙 GitHub: [github.com/amansah17](https://github.com/amansah17)  
        📧 Email: amansah1717@gmail.com  
        🔗 Project: [Lettuce Leaf Classification](https://github.com/AmanSah17/Lettuce_leaf_classification)
        """)
    
    with col2:
        st.markdown("""
        **Technologies Used:**
        - 🐍 Python & Streamlit
        - 🧠 PyTorch & TorchVision
        - 🔬 Scikit-learn & Joblib
        - 🖼️ OpenCV & PIL
        - 📊 NumPy & Pandas
        """)

# === Footer ===
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem 0;">
    <p>© 2025 Lettuce Leaf Disease Classifier | Developed by Aman Sah</p>
    
</div>
""", unsafe_allow_html=True)
