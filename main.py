# main.py
import argparse
from src.dataset import load_dataset
from src.model import load_model
from src.train import train_model
from src.inference import predict_image
import config  # 直接导入 config 模块
from transformers import AutoProcessor
import os

def main():
    parser = argparse.ArgumentParser(description="Face Classification with SigLIP 2")
    parser.add_argument("--mode", choices=["train", "infer"], default="train", help="Mode: train or infer")
    parser.add_argument("--image_path", type=str, help="Path to image for inference")
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(config.MODEL_CHECKPOINT)

    if args.mode == "train":
        # 加载模型 ❌，尝试加载本地模型，从so400m改成base
        model = load_model(config.MODEL_CHECKPOINT, config.DEVICE, config.CACHE_DIR)
        # 加载数据集 ✅
        train_dataset, eval_dataset = load_dataset(config.IMAGE_FOLDER, config.CSV_FILE, processor)
        # 训练 ✅
        train_model(model, train_dataset, eval_dataset, config=config)
    elif args.mode == "infer":
        if not args.image_path:
            raise ValueError("Please provide --image_path for inference")
        # 加载模型
        model = load_model(config.MODEL_CHECKPOINT, config.DEVICE)
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
        # 推理
        label, confidence = predict_image(args.image_path, model, processor, config.DEVICE)
        print(f"Predicted Label: {label}, Confidence: {confidence:.4f}")

if __name__ == "__main__":
    main()