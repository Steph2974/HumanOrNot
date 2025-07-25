# src/model.py
import torch
from torch import nn
from transformers import AutoModel

class BinaryClassifier(nn.Module):
    def __init__(self, base_model, embedding_dim=768, hidden_dim=512, dropout_rate=0.3):
        super(BinaryClassifier, self).__init__()
        self.base_model = base_model
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),  # 768 -> 512
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),  # 512 -> 256
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1)  # 256 -> 1（二分类）
        )

    def forward(self, pixel_values, lables=None):
        image_embeddings = self.base_model.get_image_features(pixel_values=pixel_values)
        logits = self.classifier(image_embeddings)
        return logits

    def get_image_embeddings(self, pixel_values):
        return self.base_model.get_image_features(pixel_values=pixel_values)

def load_model(checkpoint, device, cache_dir):
    try:
        # 加载预训练模型
        model = AutoModel.from_pretrained(
            checkpoint,
            cache_dir=cache_dir,
            local_files_only=False  # 允许在线下载
        )
        # 冻结视觉塔参数
        for param in model.parameters():
            param.requires_grad = False
        model.eval()  # 设置为评估模式
        # 初始化分类器
        classifier_model = BinaryClassifier(model, embedding_dim=768).to(device)
        # 确保只有 classifier 参数需要梯度
        for param in classifier_model.classifier.parameters():
            param.requires_grad = True
        return classifier_model
    except Exception as e:
        print(f"加载模型失败：{e}")
        raise