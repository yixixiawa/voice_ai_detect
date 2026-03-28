import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
import time
import random

from config import Config
from mmap_dataset import MMAPSpeechDataset
from model import LightweightCNN

def create_batch_datasets(dataset, num_batches=10):
    """将数据集分成多个批次"""
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    
    batch_size = len(indices) // num_batches
    batches = []
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size if i < num_batches - 1 else len(indices)
        batch_indices = indices[start_idx:end_idx]
        batches.append(Subset(dataset, batch_indices))
    
    return batches

def train():
    config = Config()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 加载完整数据集
    print("加载数据集...")
    full_dataset = MMAPSpeechDataset(
        config.data_root, config, mode='train', 
        val_split=config.val_split
    )
    
    # 将训练集分成10个批次
    num_batches = 10
    train_batches = create_batch_datasets(full_dataset, num_batches)
    
    print(f"训练集分为 {num_batches} 个批次，每批约 {len(train_batches[0]):,} 条")
    
    # 加载验证集
    val_dataset = MMAPSpeechDataset(
        config.data_root, config, mode='val',
        val_split=config.val_split
    )
    
    # 创建模型
    model = LightweightCNN(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # 训练记录
    best_val_acc = 0.0
    start_time = time.time()
    
    print(f"\n开始分批训练...")
    print(f"{'='*60}\n")
    
    for epoch in range(config.epochs):
        epoch_start = time.time()
        
        # 训练每个批次
        for batch_idx, train_subset in enumerate(train_batches):
            train_loader = DataLoader(
                train_subset,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=config.num_workers,
                pin_memory=True
            )
            
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.epochs} [Batch {batch_idx+1}/{num_batches}]')
            
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{train_loss/len(pbar):.4f}',
                    'acc': f'{100.*train_correct/train_total:.2f}%'
                })
        
        # 验证
        model.eval()
        val_correct = 0
        val_total = 0
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True
        )
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc='验证'):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        # 计算时间
        epoch_time = time.time() - epoch_start
        
        print(f"\nEpoch {epoch+1}/{config.epochs} 完成 (耗时: {epoch_time:.1f}秒)")
        print(f"  验证准确率: {val_acc:.2f}%")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
            }, 'best_model.pth')
            print(f"  ✓ 保存最佳模型! (验证准确率: {val_acc:.2f}%)")
        
        print("-" * 60)
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"训练完成！总耗时: {total_time/60:.1f} 分钟")
    print(f"最佳验证准确率: {best_val_acc:.2f}%")
    print(f"{'='*60}")

if __name__ == "__main__":
    train()