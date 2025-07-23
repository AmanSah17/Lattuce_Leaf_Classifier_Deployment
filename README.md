# 🧠 Lettuce Leaf Disease Classifier | Deep Learning Deployment

A full-stack deep learning web application for **automated detection of leaf diseases in lettuce crops** using a custom-trained PyTorch `TinyVGG` model. This project integrates **image classification**, **Streamlit UI**, and **cloud deployment via Render** to enable real-time inference on lettuce leaf conditions.

### 🚀 Live Web App
🔗 [Try the Classifier](https://lattuce-leaf-classifier-deployment.onrender.com/)

---

## 📚 Project Overview

This application classifies high-resolution images of lettuce leaves into one of the following classes:

| Class Code | Class Name                          |
|------------|-------------------------------------|
| BACT       | Bacterial                           |
| DML        | Downey Mildew on Lettuce            |
| HLTY       | Healthy                             |
| PML        | Powdery Mildew on Lettuce           |
| SBL        | Septorial Blight on Lettuce         |
| SPW        | Shephered Purse Weed                |
| VIRL       | Viral                               |
| WLBL       | Wilt and Leaf Blight on Lettuce     |

The goal is to assist farmers and agronomists in **early detection and classification of diseases** using machine learning tools.

---

## 🧪 Model Architecture

The deployed model is based on a lightweight, custom-built **TinyVGG** CNN architecture in PyTorch:

- Input: RGB images resized to 64x64
- Architecture: 2 Convolutional Blocks + 1 Dense Layer
- Output: Softmax over 8 classes
- Optimizer: Adam
- Loss: CrossEntropyLoss

Training was conducted on a **custom-labeled image dataset** containing multiple disease categories and healthy samples.

---

## 🖥️ Technologies Used

- **Frontend**: Streamlit
- **Backend**: PyTorch, TorchVision
- **Deployment**: Render (free tier)
- **Image Processing**: PIL, torchvision.transforms

---

## 📦 Project Structure

