# src/model.py
import torch
from torch import nn
from transformers import AutoModel

class BinaryClassifier(nn.Module):
    def __init__(self, base_model, embedding_dim=1152):  # SigLIP 2 嵌入维度
        super(BinaryClassifier, self).__init__()
        self.base_model = base_model
        self.classifier = nn.Linear(embedding_dim, 2)  # 二分类：0 或 1

    def forward(self, pixel_values):
        with torch.no_grad():
            outputs = self.base_model(pixel_values=pixel_values)
            image_embeddings = outputs.image_embeds
        logits = self.classifier(image_embeddings)
        return logits

def load_model(checkpoint, device):
    model = AutoModel.from_pretrained(checkpoint, device_map="auto").eval()
    classifier_model = BinaryClassifier(model).to(device)
    return classifier_model