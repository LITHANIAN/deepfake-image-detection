import os
import torch

class Config:
    # 路径配置
    RAW_DATA_DIR = "deepfake-image-detection"
    PROCESSED_DIR = "processed"
    MODEL_SAVE_PATH = "models/hybrid_model.pth"
    LOG_DIR = "runs"
    MODEL_SAVE_PATH_CLASSIC = "models/classic_model.pth"
    # 模型参数
    PCA_COMPONENTS = 8
    N_QUBITS = 8
    QUANTUM_LAYERS = 1
    
    # 训练参数
    BATCH_SIZE = 64
    EPOCHS = 30
    BASE_LR = 0.001
    MAX_LR = 0.003
    
    # 设备配置
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

config = Config()