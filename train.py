import torch
from torch.utils.tensorboard import SummaryWriter
from dataset import create_loaders
from hybrid_model import HybridModel
from config import config
import torch.nn as nn
import matplotlib.pyplot as plt
import os

def train():
    # 创建结果目录
    os.makedirs("results/training_curves", exist_ok=True)
    
    # 初始化模型
    model = HybridModel().to(config.DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.BASE_LR,
        weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=config.BASE_LR,
        max_lr=config.MAX_LR,
        step_size_up=500,
        cycle_momentum=False
    )
    criterion = nn.CrossEntropyLoss()
    train_loader, test_loader = create_loaders()
    writer = SummaryWriter(config.LOG_DIR)
    
    # 初始化记录列表
    train_losses = []
    val_accuracies = []
    learning_rates = []
    
    # 训练循环
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0.0
        
        # 训练阶段
        for batch_idx, (features, labels) in enumerate(train_loader):
            features = features.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            
            # 记录梯度
            if batch_idx % 10 == 0:
                for name, param in model.named_parameters():
                    writer.add_histogram(f"grads/{name}", param.grad, epoch*len(train_loader)+batch_idx)
        
        # 验证阶段
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for features, labels in test_loader:
                features = features.to(config.DEVICE)
                labels = labels.to(config.DEVICE)
                outputs = model(features)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            acc = 100 * correct / total
        
        # 记录指标
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        val_accuracies.append(acc)
        current_lr = scheduler.get_last_lr()[0]
        learning_rates.append(current_lr)
        
        # TensorBoard记录
        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('Accuracy/val', acc, epoch)
        writer.add_scalar('Learning Rate', current_lr, epoch)
        
        # 打印进度
        print(f"Epoch {epoch+1}/{config.EPOCHS} | "
              f"Loss: {avg_loss:.4f} | "
              f"Acc: {acc:.2f}% | "
              f"LR: {current_lr:.6f}")
        
        # 每5个epoch保存训练曲线
        if (epoch+1) % 5 == 0 or epoch == 0:
            plt.figure(figsize=(15, 5))
            
            # 损失曲线
            plt.subplot(1, 3, 1)
            plt.plot(train_losses, 'b-o')
            plt.title(f"Training Loss (Epoch {epoch+1})")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.grid(True)
            
            # 准确率曲线
            plt.subplot(1, 3, 2)
            plt.plot(val_accuracies, 'r-s')
            plt.title(f"Validation Accuracy (Epoch {epoch+1})")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy (%)")
            plt.ylim(0, 100)
            plt.grid(True)
            
            # 学习率曲线
            plt.subplot(1, 3, 3)
            plt.semilogy(learning_rates, 'g-d')
            plt.title("Learning Rate Schedule")
            plt.xlabel("Epoch")
            plt.ylabel("Learning Rate (log scale)")
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(f"results/training_curves/epoch_{epoch+1}.png")
            plt.close()
    
    # 保存最终训练曲线
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'b-o', label='Train Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, 'r-s', label='Val Acc')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.title("Validation Accuracy Curve")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("results/final_training_curves.png")
    plt.close()
    
    # 保存模型
    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
    writer.close()

if __name__ == "__main__":
    train()