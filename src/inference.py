# src/inference.py
# 可以实现单张图片的推理
import torch
from PIL import Image
from transformers import AutoProcessor

def predict_image(image_path, model, processor, device):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=[image], return_tensors="pt").to(device)
    model.eval()
    with torch.no_grad():
        logits = model(inputs["pixel_values"])
        probs = torch.softmax(logits, dim=1)
        predicted_label = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted_label].item()
    return predicted_label, confidence