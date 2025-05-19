# train_classic.py
import torch
from torch.utils.tensorboard import SummaryWriter
from dataset import create_loaders
from classic_model import ClassicModel
from config import config
import torch.nn as nn
import matplotlib.pyplot as plt
import os

def train_classic():
    # 初始化配置
    os.makedirs("results/classic_training", exist_ok=True)
    
    # 初始化模型
    model = ClassicModel().to(config.DEVICE)
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
    writer = SummaryWriter(os.path.join(config.LOG_DIR, "classic"))
    
    # 训练记录
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
        print(f"[Classic] Epoch {epoch+1}/{config.EPOCHS} | "
              f"Loss: {avg_loss:.4f} | "
              f"Acc: {acc:.2f}% | "
              f"LR: {current_lr:.6f}")
        
        # 保存训练曲线
        if (epoch+1) % 5 == 0 or epoch == 0:
            plt.figure(figsize=(15, 5))
            plt.subplot(1,3,1)
            plt.plot(train_losses, 'b-o')
            plt.title(f"Classic Training Loss (Epoch {epoch+1})")
            plt.subplot(1,3,2)
            plt.plot(val_accuracies, 'r-s')
            plt.title(f"Validation Accuracy (Epoch {epoch+1})")
            plt.subplot(1,3,3)
            plt.semilogy(learning_rates, 'g-d')
            plt.title("Learning Rate Schedule")
            plt.savefig(f"results/classic_training/epoch_{epoch+1}.png")
            plt.close()
    
    # 保存最终模型
    torch.save(model.state_dict(), config.MODEL_SAVE_PATH_CLASSIC)
    writer.close()

if __name__ == "__main__":
    train_classic()