# utils/model_utils.py

import torch
from torchvision import transforms
from PIL import Image

# TinyVGG class (copied here or imported from another file)
import torch.nn as nn

class TinyVGG(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(input_shape, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_units * 16 * 16, output_shape)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x

def load_model(weights_path, device, num_classes):
    model = TinyVGG(input_shape=3, hidden_units=10, output_shape=num_classes).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model

def preprocess_image(image, transform):
    return transform(image).unsqueeze(0)

def predict_image(model, image_tensor, class_names, device):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        pred_logits = model(image_tensor)
        pred_probs = torch.softmax(pred_logits, dim=1)
        pred_confidence, pred_label = torch.max(pred_probs, dim=1)
        pred_label = pred_label.item()
        pred_confidence = pred_confidence.item()
        return class_names[pred_label], pred_confidence
