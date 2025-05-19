import cv2
import numpy as np
import os
from config import config

def preprocess_images():
    # 创建输出目录
    os.makedirs(os.path.join(config.PROCESSED_DIR, "real"), exist_ok=True)
    os.makedirs(os.path.join(config.PROCESSED_DIR, "fake"), exist_ok=True)

    # 处理真实图像
    real_dir = os.path.join(config.RAW_DATA_DIR, "real_images")
    for img_file in os.listdir(real_dir):
        process_single_image(img_file, real_dir, "real")

    # 处理伪造图像
    fake_dir = os.path.join(config.RAW_DATA_DIR, "fake_images")
    for img_file in os.listdir(fake_dir):
        process_single_image(img_file, fake_dir, "fake")

def process_single_image(filename, input_dir, label):
    # 读取并调整尺寸
    img_path = os.path.join(input_dir, filename)
    img = cv2.imread(img_path)
    img = cv2.resize(img, config.IMG_SIZE, interpolation=cv2.INTER_AREA)
    
    # 转换颜色空间并标准化
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    normalized = (img / 255.0) * np.pi
    
    # 保存预处理结果
    output_path = os.path.join(config.PROCESSED_DIR, label, f"{os.path.splitext(filename)[0]}.npy")
    np.save(output_path, normalized)

if __name__ == "__main__":
    preprocess_images()