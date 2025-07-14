import torch
import os

# 获取项目根目录的绝对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- 数据相关配置 ---
CSV_FILE = os.path.join(BASE_DIR, 'four_corner_data.csv')
IMAGE_DIR = os.path.join(BASE_DIR, 'char_images')
FONT_PATH = os.path.join(BASE_DIR, 'SimSun.ttc')

# --- 图像生成配置 ---
IMAGE_SIZE = (64, 64)
FONT_SIZE = 64

# --- 模型训练配置 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'four_corner_model.pth')

# --- 预测配置 ---
# 用于测试的示例汉字
TEST_CHARACTER = "汉"