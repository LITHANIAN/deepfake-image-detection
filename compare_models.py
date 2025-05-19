# compare_models.py
import torch
from dataset import create_loaders
from hybrid_model import HybridModel
from classic_model import ClassicModel
from config import config
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, test_loader):
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            outputs = model(features)
            probs = torch.softmax(outputs, dim=1)
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc

def plot_confusion_matrix(model, test_loader, model_name):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            outputs = model(features)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f"results/{model_name}_confusion.png")
    plt.close()

def compare_models():
    # 加载数据
    _, test_loader = create_loaders()
    
    # 初始化模型
    hybrid_model = HybridModel().to(config.DEVICE)
    hybrid_model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
    
    classic_model = ClassicModel().to(config.DEVICE)
    classic_model.load_state_dict(torch.load(config.MODEL_SAVE_PATH_CLASSIC))
    
    # 评估两个模型
    hybrid_fpr, hybrid_tpr, hybrid_auc = evaluate_model(hybrid_model, test_loader)
    classic_fpr, classic_tpr, classic_auc = evaluate_model(classic_model, test_loader)
    
    # 绘制ROC对比
    plt.figure(figsize=(8,6))
    plt.plot(hybrid_fpr, hybrid_tpr, 
            label=f'Hybrid (AUC = {hybrid_auc:.2f})')
    plt.plot(classic_fpr, classic_tpr,
            label=f'Classic (AUC = {classic_auc:.2f})')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend()
    plt.savefig("results/roc_comparison.png")
    
    # 绘制混淆矩阵对比
    plot_confusion_matrix(hybrid_model, test_loader, "Hybrid")
    plot_confusion_matrix(classic_model, test_loader, "Classic")
    
    # 打印性能对比
    print("\n=== 模型性能对比 ===")
    print(f"{'指标':<15} | {'混合模型':<10} | {'经典模型':<10}")
    print("-"*40)
    print(f"{'AUC值':<15} | {hybrid_auc:<10.2f} | {classic_auc:<10.2f}")

if __name__ == "__main__":
    compare_models()