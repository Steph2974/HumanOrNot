# src/train.py
import torch
from torch.utils.data import DataLoader
from torch import nn
from transformers import AdamW
from tqdm import tqdm

def train_model(model, train_dataset, eval_dataset, config):
    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=config.BATCH_SIZE)

    # 定义优化器和损失函数
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    model.train()
    for epoch in range(config.NUM_EPOCHS):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            pixel_values = batch["pixel_values"].to(config.DEVICE)
            labels = batch["labels"].to(config.DEVICE)

            outputs = model(pixel_values)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")

    # 评估
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in eval_loader:
            pixel_values = batch["pixel_values"].to(config.DEVICE)
            labels = batch["labels"].to(config.DEVICE)
            outputs = model(pixel_values)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total * 100
    print(f"Validation Accuracy: {accuracy:.2f}%")

    # 保存模型
    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
    print(f"Model saved to {config.MODEL_SAVE_PATH}")