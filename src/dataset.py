# src/dataset.py
import pandas as pd
from PIL import Image
import os
import torch
from transformers import AutoProcessor
from datasets import Dataset

def load_dataset(image_folder, csv_file, processor):
    # 读取 CSV
    # df = pd.read_csv(csv_file).head(5)
    df = pd.read_csv(csv_file)

    # 加载图片
    def load_image(row):
        image_path = os.path.join(image_folder, row['Image'])
        image = Image.open(image_path).convert("RGB")
        return {"image": image, "label": row['Label']}

    # 转换为 Hugging Face Dataset
    dataset = Dataset.from_pandas(df).map(load_image)

    # 拆分数据集
    train_test_split = dataset.train_test_split(test_size=0.2)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']

    # 预处理函数
    def preprocess(batch):
        images = [image for image in batch["image"]]
        inputs = processor(images=images, return_tensors="pt", padding=True)
        batch["pixel_values"] = inputs["pixel_values"]
        batch["labels"] = torch.tensor(batch["label"], dtype=torch.long)
        return batch

    # 应用预处理
    train_dataset = train_dataset.map(preprocess, batched=True, remove_columns=["image", "Image", "Label"])
    eval_dataset = eval_dataset.map(preprocess, batched=True, remove_columns=["image", "Image", "Label"])

    # 设置格式
    train_dataset.set_format("torch", columns=["pixel_values", "labels"])
    eval_dataset.set_format("torch", columns=["pixel_values", "labels"])

    return train_dataset, eval_dataset
