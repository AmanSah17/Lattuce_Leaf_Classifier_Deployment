import streamlit as st
import numpy as np
import cv2
import joblib
import torch
import mahotas
import os
import tempfile
from PIL import Image
from torchvision import transforms
from utils.model_utils import load_model, preprocess_image, predict_image, get_class_display_name

# === Page Configuration ===
st.set_page_config(
    page_title="Lettuce Leaf Classifier - Comparison",
    layout="centered",
    initial_sidebar_state="expanded"
)

# === Custom CSS for better styling ===
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .contact-form {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid #dee2e6;
    }
    .youtube-container {
        display: flex;
        justify-content: center;
        margin: 2rem 0;
    }
    .project-info {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# === JavaScript for interactivity ===
st.markdown("""
<script>
    // Auto-refresh predictions
    function refreshPredictions() {
        if (window.location.search.includes('uploaded_file')) {
            setTimeout(() => {
                window.location.reload();
            }, 1000);
        }
    }
    
    // Smooth scrolling
    function scrollToSection(sectionId) {
        document.getElementById(sectionId).scrollIntoView({
            behavior: 'smooth'
        });
    }
    
    // Initialize on page load
    document.addEventListener('DOMContentLoaded', function() {
        refreshPredictions();
    });
</script>
""", unsafe_allow_html=True)

# === Navigation Buttons ===
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🏠 Home", key="home_compare", use_container_width=True):
        st.switch_page("app")
with col2:
    if st.button("🔬 Classical ML", key="classical_compare", use_container_width=True):
        st.switch_page("Classical_ML")
with col3:
    if st.button("🧠 DNN Model", key="dnn_compare", use_container_width=True):
        st.switch_page("DNN_Model")

# === Main Header ===
st.markdown('<h1 class="main-header">🥬 Lettuce Leaf Disease Classification</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Compare Deep Neural Network (TinyVGG) vs Classical ML (Voting Classifier) Predictions</p>', unsafe_allow_html=True)

# === Project Information Section ===
with st.container():
    st.markdown('<div class="project-info">', unsafe_allow_html=True)
    st.markdown("""
    ## 📋 About This Project
    
    This application demonstrates a comprehensive comparison between two machine learning approaches for lettuce leaf disease classification:
    
    - **🧠 Deep Neural Network (TinyVGG)**: A custom CNN architecture trained on image data
    - **🔬 Classical ML (Voting Classifier)**: Traditional machine learning using feature extraction
    
    ### 🎯 Classification Categories:
    - **BACT**: Bacterial Diseases
    - **DML**: Downey Mildew on Lettuce
    - **HLTY**: Healthy Leaves
    - **PML**: Powdery Mildew on Lettuce
    - **SBL**: Septorial Blight on Lettuce
    - **SPW**: Shepherd Purse Weed
    - **VIRL**: Viral Diseases
    - **WLBL**: Wilt and Leaf Blight on Lettuce
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# === YouTube Video Section ===
with st.container():
    st.markdown("## 📺 Project Explanation Video")
    st.markdown('<div class="youtube-container">', unsafe_allow_html=True)
    
    # YouTube embed with dummy link
    youtube_embed = """
    <iframe width="560" height="315" 
            src="https://www.youtube.com/embed/dQw4w9WgXcQ" 
            title="Lettuce Leaf Classification Project" 
            frameborder="0" 
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
            allowfullscreen>
    </iframe>
    """
    st.markdown(youtube_embed, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# === Device Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load Class Names ===
@st.cache_data
def load_class_names():
    with open("classes.txt", "r") as f:
        return [line.strip() for line in f.readlines()]

class_names = load_class_names()

# === Load Models and Scalers ===
@st.cache_resource
def load_models():
    dnn_model = load_model("models/TinyVGG_DNN_model_0_weights.pth", device, num_classes=len(class_names))
    voting_model = joblib.load("models/Lctf_voting_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return dnn_model, voting_model, scaler

dnn_model, voting_model, scaler = load_models()

# === Define Image Transforms for DNN ===
custom_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# === Feature Extraction for Classical ML ===
def compute_brightness(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

def extract_texture(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return np.zeros(13)
    gray = cv2.cvtColor(cv2.resize(img, (240, 240)), cv2.COLOR_BGR2GRAY)
    return mahotas.features.haralick(gray).mean(axis=0)

def extract_histogram(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(cv2.resize(img, (240, 240)), cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
    return hist.flatten() / hist.sum()

# === Prediction Function ===
def predict_single_image(image, image_name):
    """Predict disease for a single image using both models"""
    # Save image temporarily for OpenCV processing
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        image.save(tmp_file.name)
        temp_path = tmp_file.name
    
    try:
        # DNN Prediction
        image_tensor = preprocess_image(image, custom_transform)
        prediction_dnn, confidence_dnn = predict_image(dnn_model, image_tensor, class_names, device)
        
        # Classical ML Prediction
        brightness = compute_brightness(temp_path)
        texture = extract_texture(temp_path)
        histogram = extract_histogram(temp_path)
        
        feature_vector = np.hstack([brightness, texture, histogram])
        feature_vector = scaler.transform([feature_vector])
        
        prediction_ml = voting_model.predict(feature_vector)[0]
        probas_ml = voting_model.predict_proba(feature_vector)[0]
        confidence_ml = np.max(probas_ml)
        
        return {
            'image_name': image_name,
            'dnn_prediction': prediction_dnn,
            'dnn_confidence': confidence_dnn,
            'ml_prediction': prediction_ml,
            'ml_confidence': confidence_ml
        }
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)

# === Multiple Image Upload Section ===
with st.container():
    st.markdown("## 📤 Upload Multiple Images")
    st.markdown("Upload one or more lettuce leaf images to compare predictions from both models.")
    
    uploaded_files = st.file_uploader(
        "Choose images", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True,
        help="Select multiple images to analyze"
    )

# === Results Display Section ===
if uploaded_files:
    with st.container():
        st.markdown("## 🔍 Analysis Results")
        
        # Process all uploaded images
        results = []
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file).convert("RGB")
            result = predict_single_image(image, uploaded_file.name)
            results.append((image, result))
        
        # Display results in a grid layout
        for i, (image, result) in enumerate(results):
            st.markdown(f"### 📷 Image {i+1}: {result['image_name']}")
            
            # Create columns for image and predictions
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.image(image, caption=f"Uploaded Image: {result['image_name']}", use_column_width=True)
            
            with col2:
                st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                st.markdown("**🧠 TinyVGG (DNN)**")
                st.success(f"**Class:** {get_class_display_name(result['dnn_prediction'])}")
                st.info(f"**Confidence:** {result['dnn_confidence']*100:.2f}%")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                st.markdown("**🔬 Voting Classifier**")
                st.success(f"**Class:** {get_class_display_name(result['ml_prediction'])}")
                st.info(f"**Confidence:** {result['ml_confidence']*100:.2f}%")
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("---")

# === Contact Form Section ===
with st.container():
    st.markdown("## 📧 Contact Information")
    st.markdown('<div class="contact-form">', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 👨‍💻 Developer Information")
        st.markdown("""
        **Aman Sah**  
        🐙 GitHub: [github.com/amansah17](https://github.com/amansah17)  
        📧 Email: amansah1717@gmail.com  
        🔗 Project: [Lettuce Leaf Classification](https://github.com/AmanSah17/Lettuce_leaf_classification)
        """)
    
    with col2:
        st.markdown("### 💬 Get in Touch")
        
        # Contact form using formsubmit
        contact_form = """
        <form action="https://formsubmit.co/amansah1717@gmail.com" method="POST">
            <div style="margin-bottom: 1rem;">
                <label for="name" style="display: block; margin-bottom: 0.5rem; font-weight: bold;">Name:</label>
                <input type="text" id="name" name="name" required style="width: 100%; padding: 0.5rem; border: 1px solid #ccc; border-radius: 4px;">
            </div>
            
            <div style="margin-bottom: 1rem;">
                <label for="email" style="display: block; margin-bottom: 0.5rem; font-weight: bold;">Email:</label>
                <input type="email" id="email" name="email" required style="width: 100%; padding: 0.5rem; border: 1px solid #ccc; border-radius: 4px;">
            </div>
            
            <div style="margin-bottom: 1rem;">
                <label for="subject" style="display: block; margin-bottom: 0.5rem; font-weight: bold;">Subject:</label>
                <input type="text" id="subject" name="subject" required style="width: 100%; padding: 0.5rem; border: 1px solid #ccc; border-radius: 4px;">
            </div>
            
            <div style="margin-bottom: 1rem;">
                <label for="message" style="display: block; margin-bottom: 0.5rem; font-weight: bold;">Message:</label>
                <textarea id="message" name="message" rows="4" required style="width: 100%; padding: 0.5rem; border: 1px solid #ccc; border-radius: 4px;"></textarea>
            </div>
            
            <button type="submit" style="background-color: #1f77b4; color: white; padding: 0.75rem 1.5rem; border: none; border-radius: 4px; cursor: pointer; font-size: 1rem;">
                Send Message
            </button>
        </form>
        """
        st.markdown(contact_form, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# === Footer ===
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem 0;">
    <p>© 2024 Lettuce Leaf Disease Classifier | Developed by Aman Sah</p>
    <p>Built with ❤️ using Streamlit, PyTorch, and Scikit-learn</p>
</div>
""", unsafe_allow_html=True)

