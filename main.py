# main.py
import argparse
from src.dataset import load_dataset
from src.model import load_model
from src.train import train_model
from src.inference import predict_image
from config import *
from transformers import AutoProcessor
import os

def main():
    parser = argparse.ArgumentParser(description="Face Classification with SigLIP 2")
    parser.add_argument("--mode", choices=["train", "infer"], default="train", help="Mode: train or infer")
    parser.add_argument("--image_path", type=str, help="Path to image for inference")
    args = parser.parse_args()

    # 加载处理器
    processor = AutoProcessor.from_pretrained(MODEL_CHECKPOINT)

    if args.mode == "train":
        # 加载数据集
        train_dataset, eval_dataset = load_dataset(IMAGE_FOLDER, CSV_FILE, processor)
        # 加载模型
        model = load_model(MODEL_CHECKPOINT, DEVICE)
        # 训练
        train_model(model, train_dataset, eval_dataset, config=globals())
    elif args.mode == "infer":
        if not args.image_path:
            raise ValueError("Please provide --image_path for inference")
        # 加载模型
        model = load_model(MODEL_CHECKPOINT, DEVICE)
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
        # 推理
        label, confidence = predict_image(args.image_path, model, processor, DEVICE)
        print(f"Predicted Label: {label}, Confidence: {confidence:.4f}")

if __name__ == "__main__":
    main()