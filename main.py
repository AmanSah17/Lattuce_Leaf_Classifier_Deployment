import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from torchvision import transforms
from utils.model_utils import load_model, preprocess_image, predict_image

app = FastAPI(title="Lettuce Leaf Disease Classifier API")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load class names
with open("classes.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Define image transforms
custom_image_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# Load model
model_path = "models/TinyVGG_DNN_model_0_weights.pth"
model = load_model(model_path, device, num_classes=len(class_names))

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Only JPG, JPEG, PNG are supported.")
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_tensor = preprocess_image(image, custom_image_transform)
        pred_class, confidence = predict_image(model, image_tensor, class_names, device)
        response = {
            "predicted_class": pred_class,
            "confidence": confidence,
            "probability": confidence  # For clarity, probability == confidence here
        }
        return JSONResponse(content=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}") 