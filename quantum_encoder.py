import numpy as np
from sklearn.decomposition import PCA
import joblib
from config import config
import os

class QuantumEncoder:
    def __init__(self, train_data=None):
        self.pca = PCA(n_components=config.PCA_COMPONENTS)
        if train_data is not None:
            self.train_pca(train_data)
            
    def train_pca(self, train_data):
        """使用训练数据拟合PCA模型"""
        self.pca.fit(train_data)
        joblib.dump(self.pca, "models/pca_model.pkl")
        
    def encode(self, img_array):
        """编码单个图像"""
        flattened = img_array.reshape(1, -1)
        return self.pca.transform(flattened)[0]
    
    def load_full_dataset(self):
        """加载所有预处理后的真实与伪造图像数据"""
        real_dir = os.path.join(config.PROCESSED_DIR, "real")
        fake_dir = os.path.join(config.PROCESSED_DIR, "fake")
        
        # 加载真实图像数据
        real_data = []
        for f in os.listdir(real_dir):
            if f.endswith('.npy'):
                data = np.load(os.path.join(real_dir, f))
                real_data.append(data.flatten())  # 展平为1D向量
                
        # 加载伪造图像数据
        fake_data = []
        for f in os.listdir(fake_dir):
            if f.endswith('.npy'):
                data = np.load(os.path.join(fake_dir, f))
                fake_data.append(data.flatten())
        
        # 合并并转换为numpy数组
        full_data = np.vstack(real_data + fake_data)
        print(f"成功加载 {full_data.shape[0]} 个样本，每个样本 {full_data.shape[1]} 维特征")
        return full_data

def build_encoder():
    # 收集真实图像训练PCA
    real_dir = os.path.join(config.PROCESSED_DIR, "real")
    train_data = np.array([np.load(os.path.join(real_dir, f)) for f in os.listdir(real_dir)[:100]])
    train_data = train_data.reshape(len(train_data), -1)
    
    encoder = QuantumEncoder(train_data)
    return encoder

if __name__ == "__main__":
    # 创建输出目录
    os.makedirs("models", exist_ok=True)
    
    # 初始化编码器
    encoder = QuantumEncoder()
    
    # 加载完整数据集
    train_data = encoder.load_full_dataset()
    
    # 训练并保存PCA模型
    encoder.train_pca(train_data)
    print("PCA模型已使用完整数据集训练完成！")