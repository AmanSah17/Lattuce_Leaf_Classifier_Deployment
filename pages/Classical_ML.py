import streamlit as st
import numpy as np
import cv2
import joblib
import os
import tempfile
from PIL import Image
from utils.model_utils import compute_brightness, extract_texture, extract_histogram, get_class_display_name

# === Page Configuration ===
st.set_page_config(
    page_title="Classical ML - Lettuce Leaf Classifier",
    layout="centered",
    initial_sidebar_state="expanded"
)

# === Custom CSS for Classical ML page ===
st.markdown("""
<style>
    .ml-header {
        text-align: center;
        color: #FF6B35;
        font-size: 2.5rem;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, #FF6B35 0%, #F7931E 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .model-info {
        background: linear-gradient(135deg, #FF6B35 0%, #F7931E 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
    }
    .prediction-result {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #FF6B35;
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
        background: linear-gradient(90deg, #FF6B35 0%, #F7931E 100%);
        height: 20px;
        border-radius: 8px;
        transition: width 0.3s ease;
    }
    .upload-area {
        border: 2px dashed #FF6B35;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #f8f9fa;
        margin: 1rem 0;
    }
    .feature-info {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# === Navigation Buttons ===
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🏠 Home", key="home_ml", use_container_width=True):
        st.switch_page("app")
with col2:
    if st.button("🧠 DNN Model", key="dnn_ml", use_container_width=True):
        st.switch_page("DNN_Model")
with col3:
    if st.button("⚖️ Compare Models", key="compare_ml", use_container_width=True):
        st.switch_page("performace_comparison")

# === Main Header ===
st.markdown('<h1 class="ml-header">🔬 Classical Machine Learning</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666; font-size: 1.2rem;">Feature-based Voting Classifier for Lettuce Leaf Disease Classification</p>', unsafe_allow_html=True)

# === Model Information Section ===
with st.container():
    st.markdown('<div class="model-info">', unsafe_allow_html=True)
    st.markdown("""
    ## 🔬 Voting Classifier Architecture
    
    **Model Type**: Ensemble Voting Classifier  
    **Base Models**: Multiple traditional ML algorithms  
    **Feature Extraction**: 
    - Brightness analysis
    - Haralick texture features (13 features)
    - Histogram analysis (64 bins)
    
    **Total Features**: 78 engineered features  
    **Preprocessing**: StandardScaler normalization  
    **Voting Strategy**: Soft voting with probability averaging
    
    **Advantages**: Interpretable, fast inference, feature importance analysis
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# === Load Class Names ===
@st.cache_data
def load_class_names():
    with open("classes.txt", "r") as f:
        return [line.strip() for line in f.readlines()]

class_names = load_class_names()

# === Load Models and Scalers ===
@st.cache_resource
def load_ml_models():
    voting_model = joblib.load("models/Lctf_voting_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return voting_model, scaler

voting_model, scaler = load_ml_models()

# === Prediction Function ===
def predict_ml_image(image_path):
    """Predict disease using Classical ML model"""
    try:
        # Extract features
        brightness = compute_brightness(image_path)
        texture = extract_texture(image_path)
        histogram = extract_histogram(image_path)
        
        # Combine features
        feature_vector = np.hstack([brightness, texture, histogram])
        feature_vector = scaler.transform([feature_vector])
        
        # Get prediction
        prediction_id = voting_model.predict(feature_vector)[0]
        probas = voting_model.predict_proba(feature_vector)[0]
        confidence = np.max(probas)
        
        # Map numeric ID to class name
        if isinstance(prediction_id, (int, np.integer)) and prediction_id < len(class_names):
            prediction = class_names[prediction_id]
        else:
            prediction = str(prediction_id)  # Fallback to string representation
        
        return prediction, confidence, probas
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None, None, None

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
        
        # Save image temporarily for OpenCV processing
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            image.save(tmp_file.name)
            temp_path = tmp_file.name
        
        try:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(image, caption="Uploaded Image", use_column_width=True)
            
            with col2:
                # Get prediction
                with st.spinner("🔬 Analyzing with Classical ML..."):
                    prediction, confidence, probas = predict_ml_image(temp_path)
                
                if prediction is not None:
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
                    
                    # Feature information
                    st.markdown('<div class="feature-info">', unsafe_allow_html=True)
                    st.markdown("### 🔍 Feature Analysis")
                    st.markdown(f"**Brightness Level**: {compute_brightness(temp_path):.2f}")
                    st.markdown(f"**Texture Features**: 13 Haralick features extracted")
                    st.markdown(f"**Histogram Bins**: 64 grayscale intensity bins")
                    st.markdown('</div>', unsafe_allow_html=True)
        
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)

# === Feature Engineering Details ===
with st.container():
    st.markdown("## 🔧 Feature Engineering Process")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📊 Brightness Analysis
        - Converts image to grayscale
        - Calculates mean pixel intensity
        - Indicates overall leaf health
        
        ### 🎨 Texture Features
        - Haralick texture analysis
        - 13 statistical texture measures
        - Captures surface patterns
        """)
    
    with col2:
        st.markdown("""
        ### 📈 Histogram Analysis
        - 64-bin grayscale histogram
        - Normalized frequency distribution
        - Captures intensity patterns
        
        ### 🔄 Preprocessing
        - Feature scaling with StandardScaler
        - Ensures equal feature importance
        - Improves model performance
        """)

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
        st.metric("Model Type", "Voting Classifier")
    
    with col2:
        st.metric("Features", "78")
    
    with col3:
        st.metric("Inference Speed", "Fast")

# === Advantages Section ===
with st.container():
    st.markdown("## ✅ Classical ML Advantages")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🎯 Interpretability**
        - Clear feature importance
        - Understandable decision process
        - Explainable predictions
        
        **⚡ Speed**
        - Fast feature extraction
        - Quick inference time
        - Lightweight model
        """)
    
    with col2:
        st.markdown("""
        **🔧 Flexibility**
        - Easy to modify features
        - Simple to retrain
        - Low computational cost
        
        **📊 Robustness**
        - Works with small datasets
        - Less prone to overfitting
        - Stable performance
        """)

# === Footer ===
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem 0;">
    <p>🔬 Classical Machine Learning Model | Lettuce Leaf Disease Classifier</p>
    <p>Developed by <a href="https://github.com/amansah17" target="_blank">Aman Sah</a></p>
</div>
""", unsafe_allow_html=True) 