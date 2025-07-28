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
        probabilities = torch.sigmoid(logits)
        predicted_label = (probabilities > 0.5).float() 
        confidence = probabilities.max().item()  # 获取最大概率作为置信度
    return predicted_label, confidence