# src/train.py
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm
import logging
import os

def train_model(model, train_dataset, eval_dataset, config):
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    try:
        # 创建 DataLoader
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        eval_loader = DataLoader(eval_dataset, batch_size=config.BATCH_SIZE, num_workers=4, pin_memory=True)

        # 定义优化器和损失函数
        optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=0.01)
        criterion = nn.BCEWithLogitsLoss()  # 二分类损失函数

        # 训练循环
        model.train()
        for epoch in range(config.NUM_EPOCHS):
            total_loss = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}"):
                try:
                    pixel_values = batch["pixel_values"].to(config.DEVICE, non_blocking=True)
                    labels = batch["labels"].to(config.DEVICE, non_blocking=True).float().unsqueeze(1)  # 形状 (batch_size, 1)

                    outputs = model(pixel_values)
                    loss = criterion(outputs, labels)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                except Exception as e:
                    logger.error(f"训练批次处理失败：{e}")
                    continue

            avg_loss = total_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1}/{config.NUM_EPOCHS}, Average Loss: {avg_loss:.4f}")

            # 每轮进行一次验证
            model.eval()
            correct = 0
            total = 0
            eval_loss = 0
            with torch.no_grad():
                for batch in tqdm(eval_loader, desc="Evaluating"):
                    try:
                        pixel_values = batch["pixel_values"].to(config.DEVICE, non_blocking=True)
                        labels = batch["labels"].to(config.DEVICE, non_blocking=True).float().unsqueeze(1)

                        outputs = model(pixel_values)
                        loss = criterion(outputs, labels)
                        eval_loss += loss.item()

                        probabilities = torch.sigmoid(outputs)  # 转换为概率
                        predicted = (probabilities > 0.5).float()  # 阈值 0.5

                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                    except Exception as e:
                        logger.error(f"验证批次处理失败：{e}")
                        continue

            if total > 0:
                accuracy = correct / total * 100
                avg_eval_loss = eval_loss / len(eval_loader)
                logger.info(f"Validation - Epoch {epoch+1}/{config.NUM_EPOCHS}, Loss: {avg_eval_loss:.4f}, Accuracy: {accuracy:.2f}%")
            else:
                logger.warning("验证集为空，无法计算准确率。")

        # 保存模型
        try:
            os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            logger.info(f"模型已保存至 {config.MODEL_SAVE_PATH}")
        except Exception as e:
            logger.error(f"保存模型失败：{e}")

        return accuracy, avg_eval_loss

    except Exception as e:
        logger.error(f"训练过程失败：{e}")
        raise