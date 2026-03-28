import os
import sys
import warnings

# 禁用警告
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TORCHCODEC_DISABLE'] = '1'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import time

from config import Config
from dataset import SpeechDataset
from model import LightweightCNN

def train():
    config = Config()
    
    # 检查GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"{'='*60}\n")
    
    # 加载数据集
    print("加载训练集...")
    train_dataset = SpeechDataset(
        config.data_root, config, mode='train', 
        val_split=config.val_split
    )
    
    print("\n加载验证集...")
    val_dataset = SpeechDataset(
        config.data_root, config, mode='val',
        val_split=config.val_split
    )
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True if config.num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True if config.num_workers > 0 else False
    )
    
    print(f"\n训练集大小: {len(train_dataset):,} 条")
    print(f"验证集大小: {len(val_dataset):,} 条")
    print(f"每epoch批次数: {len(train_loader):,}")
    
    # 创建模型
    model = LightweightCNN(num_classes=2).to(device)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数量: {total_params:,}")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate, 
        weight_decay=1e-5
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        steps_per_epoch=len(train_loader),
        epochs=config.epochs,
        pct_start=0.3
    )
    
    # 混合精度训练
    if config.use_amp and device.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()
        print("启用混合精度训练 (AMP)")
    else:
        scaler = None
    
    # 梯度累积
    accumulation_steps = getattr(config, 'gradient_accumulation', 1)
    if accumulation_steps > 1:
        print(f"启用梯度累积: {accumulation_steps} 步")
    
    # 训练记录
    best_val_acc = 0.0
    train_losses = []
    val_accs = []
    start_time = time.time()
    
    print(f"\n开始训练...")
    print(f"预计每个epoch时间: {len(train_loader) * config.batch_size / 1000:.1f} 秒")
    print(f"{'='*60}\n")
    
    for epoch in range(config.epochs):
        epoch_start = time.time()
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.epochs} [训练]')
        optimizer.zero_grad()
        
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(device), labels.to(device)
            
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                loss = loss / accumulation_steps
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss = loss / accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
            
            train_loss += loss.item() * accumulation_steps
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            # 更新进度条
            current_acc = 100. * train_correct / train_total
            pbar.set_postfix({
                'loss': f'{train_loss/(batch_idx+1):.4f}',
                'acc': f'{current_acc:.2f}%'
            })
        
        train_acc = 100. * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{config.epochs} [验证]'):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        val_accs.append(val_acc)
        
        # 计算时间
        epoch_time = time.time() - epoch_start
        
        # 打印结果
        print(f"\nEpoch {epoch+1}/{config.epochs} 完成 (耗时: {epoch_time:.1f}秒)")
        print(f"  训练损失: {avg_train_loss:.4f} | 训练准确率: {train_acc:.2f}%")
        print(f"  验证准确率: {val_acc:.2f}% | 学习率: {scheduler.get_last_lr()[0]:.6f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
            }, 'best_model.pth')
            print(f"  ✓ 保存最佳模型! (验证准确率: {val_acc:.2f}%)")
        
        # 定期保存检查点
        if config.save_checkpoint and (epoch + 1) % config.checkpoint_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, f'checkpoint_epoch_{epoch+1}.pth')
            print(f"  ✓ 保存检查点: checkpoint_epoch_{epoch+1}.pth")
        
        print("-" * 60)
    
    # 训练完成
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"训练完成！")
    print(f"总耗时: {total_time/60:.1f} 分钟")
    print(f"最佳验证准确率: {best_val_acc:.2f}%")
    print(f"最终训练准确率: {train_acc:.2f}%")
    print(f"最终验证准确率: {val_acc:.2f}%")
    print(f"{'='*60}")

def test_model(model_path, audio_path):
    """测试单个音频文件"""
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    model = LightweightCNN(num_classes=2).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 创建临时dataset用于处理音频
    temp_dataset = SpeechDataset(config.data_root, config, mode='val', val_split=0.1)
    
    # 处理音频
    waveform = temp_dataset.load_audio(audio_path)
    mel = temp_dataset.extract_mel_spectrogram(waveform)
    mel = mel.unsqueeze(0).to(device)
    
    # 预测
    with torch.no_grad():
        outputs = model(mel)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(outputs, dim=1)
    
    result = "AI合成" if pred.item() == 1 else "真人"
    confidence = probs[0][pred.item()].item()
    
    print(f"\n音频: {audio_path}")
    print(f"预测结果: {result} (置信度: {confidence:.2%})")
    
    return result, confidence

if __name__ == "__main__":
    train()