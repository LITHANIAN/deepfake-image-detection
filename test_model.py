import torch
from dataset import create_loaders
from hybrid_model import HybridModel
from config import config
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import os

def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            outputs = model(features)
            probs = torch.softmax(outputs, dim=1)
            
            all_probs.extend(probs[:, 1].cpu().numpy())  # 获取正类概率
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算评估指标
    cm = confusion_matrix(all_labels, all_preds)
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    # 创建结果目录
    os.makedirs("results", exist_ok=True)
    
    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.1)
    ax = sns.heatmap(
        cm, 
        annot=False, 
        fmt="d",
        cmap="Blues",
        cbar=False,
        annot_kws={"size": 16},
        linewidths=0.5
    )
    
    # 添加百分比标签
    total = cm.sum(axis=1)[:, np.newaxis]
    percent = cm / total * 100
    annotations = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            perc = percent[i, j]
            annotations.append(f"{val}\n({perc:.1f}%)")
    annotations = np.array(annotations).reshape(cm.shape)
    
    # 手动添加文本标注
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text_color = "white" if cm[i, j] > cm.max()/2 else "black"
            ax.text(
                x=j + 0.5, 
                y=i + 0.5,
                s=annotations[i, j],
                ha="center",
                va="center",
                color=text_color,
                fontsize=12
            )
    
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xticklabels(["Authentic", "Deepfake"])
    ax.set_yticklabels(["Authentic", "Deepfake"], rotation=90)
    plt.title("Confusion Matrix with Percentage Distribution", pad=20, fontsize=14)
    plt.savefig("results/confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 绘制ROC曲线
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic', fontsize=14)
    plt.legend(loc="lower right")
    plt.savefig("results/roc_curve.png", dpi=300)
    plt.close()
    
    # 打印指标
    TN, FP, FN, TP = cm.ravel()
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    tpr = TP / (TP + FN)
    fpr = FP / (FP + TN)
    fnr = FN / (TP + FN)  # 真判假率/漏检率
    tnr = TN / (TN + FP)  # 假判假率/特异度

    print("=== 模型性能报告 ===")
    print(f"准确率: {accuracy:.2%}")
    print(f"AUC值: {roc_auc:.2f}")
    print(f"真判真率(TPR): {tpr:.2%}")
    print(f"假判真率(FPR): {fpr:.2%}")
    print(f"真判假率(FNR): {fnr:.2%}")
    print(f"假判假率(TNR): {tnr:.2%}")

if __name__ == "__main__":
    # 加载模型
    model = HybridModel()
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
    
    # 获取测试集
    _, test_loader = create_loaders()
    
    # 执行评估
    evaluate_model(model, test_loader)