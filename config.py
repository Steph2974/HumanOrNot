# config.py
import os

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_FOLDER = os.path.join(BASE_DIR, "data/images")
CSV_FILE = os.path.join(BASE_DIR, "data/image_list.csv") 
MODEL_CHECKPOINT = "google/siglip2-so400m-patch16-384"
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "binary_classifier.pth")

# 训练超参数
BATCH_SIZE = 16
NUM_EPOCHS = 5
LEARNING_RATE = 5e-5
DEVICE = "cpu"

# 数据拆分
TEST_SIZE = 0.2