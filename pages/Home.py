import streamlit as st
import numpy as np
import cv2
import joblib
import torch
import mahotas
from PIL import Image
from torchvision import transforms
from utils.model_utils import load_model, preprocess_image, predict_image


# === Streamlit UI ===
st.set_page_config(page_title="Lettuce Leaf Classifier", layout="wide")
st.title("🥬 Lettuce Leaf Health Classification App")

st.markdown("Upload a lettuce leaf image and compare predictions from a Deep Neural Network (TinyVGG) and a Classical Voting Classifier.")


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


uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image.save("temp.jpg")  # For OpenCV processing
    st.image(image, caption="Uploaded Image", use_column_width=True)

    col1, col2 = st.columns(2)

    # === DNN Prediction ===
    with col1:
        st.subheader("🧠 TinyVGG (DNN) Prediction")
        image_tensor = preprocess_image(image, custom_transform)
        prediction_dnn, confidence_dnn = predict_image(dnn_model, image_tensor, class_names, device)

        st.success(f"Predicted Class: `{prediction_dnn}`")
        st.info(f"Confidence: `{confidence_dnn * 100:.2f}%`")
        st.session_state["dnn_pred"] = prediction_dnn
        st.session_state["dnn_conf"] = confidence_dnn

    # === Classical ML Prediction ===
    with col2:
        st.subheader("🔬 Voting Classifier (ML) Prediction")
        brightness = compute_brightness("temp.jpg")
        texture = extract_texture("temp.jpg")
        histogram = extract_histogram("temp.jpg")

        feature_vector = np.hstack([brightness, texture, histogram])
        feature_vector = scaler.transform([feature_vector])

        prediction_ml = voting_model.predict(feature_vector)[0]
        probas_ml = voting_model.predict_proba(feature_vector)[0]
        confidence_ml = np.max(probas_ml)

        st.success(f"Predicted Class: `{prediction_ml}`")
        st.info(f"Confidence: `{confidence_ml * 100:.2f}%`")
        st.session_state["ml_pred"] = prediction_ml
        st.session_state["ml_conf"] = confidence_ml


    # === Display Raw Predictions ===
    with st.expander("🧾 View Session Summary"):
        st.write("🔁 Predictions for last image:")
        st.write({
            "DNN": {
                "class": st.session_state.get("dnn_pred"),
                "confidence": round(st.session_state.get("dnn_conf", 0) * 100, 2)
            },
            "Voting Classifier": {
                "class": st.session_state.get("ml_pred"),
                "confidence": round(st.session_state.get("ml_conf", 0) * 100, 2)
            }
        })

