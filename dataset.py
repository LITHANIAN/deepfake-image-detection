import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from quantum_encoder import QuantumEncoder
import joblib
from config import config

class DeepfakeDataset(Dataset):
    def __init__(self):
        # 加载编码器
        self.encoder = QuantumEncoder()
        self.encoder.pca = joblib.load("models/pca_model.pkl")
        
        # 加载数据路径
        self.real_files = self._load_npy_files(os.path.join(config.PROCESSED_DIR, "real"))
        self.fake_files = self._load_npy_files(os.path.join(config.PROCESSED_DIR, "fake"))
        self.all_files = self.real_files + self.fake_files
        self.labels = [0]*len(self.real_files) + [1]*len(self.fake_files)
        
        # 计算类别权重
        class_counts = np.bincount(self.labels)
        self.weights = 1. / torch.tensor(class_counts, dtype=torch.float32)
    
    def _load_npy_files(self, dir_path):
        return [os.path.join(dir_path, f) 
                for f in os.listdir(dir_path) if f.endswith('.npy')]
    
    def __len__(self):
        return len(self.all_files)
    
    def __getitem__(self, idx):
        img_array = np.load(self.all_files[idx])
        encoding = self.encoder.encode(img_array)
        return torch.tensor(encoding, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

def create_loaders():
    dataset = DeepfakeDataset()
    
    # 创建带类别平衡的采样器
    sample_weights = dataset.weights[dataset.labels]
    sampler = WeightedRandomSampler(
        sample_weights, 
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # 划分训练测试集
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_set.dataset,  # 保持原始采样
        batch_size=config.BATCH_SIZE,
        sampler=sampler,
        num_workers=2
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )
    return train_loader, test_loader