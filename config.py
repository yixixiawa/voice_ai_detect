import os

class Config:
    # 数据路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(current_dir, "data")
    
    # 音频参数（平衡速度和内存）
    sample_rate = 16000
    duration = 2  # 2秒足够
    n_mels = 64   # 64个梅尔频带
    
    # 训练参数
    batch_size = 38   # 在当前基础上再上调约 5%
    val_batch_size = 76
    num_workers = 3   # 适度提高数据加载并行度
    epochs = 4
    learning_rate = 1.15e-3
    
    # 数据划分
    val_split = 0.1
    
    # 数据增强（关闭以提升速度）
    use_augmentation = False
    
    # 混合精度训练
    use_amp = True
    
    # 梯度累积
    gradient_accumulation = 4
    
    # 模型保存
    save_checkpoint = True
    checkpoint_interval = 2

    # 续训设置
    # 优先从 model 目录中的已导出权重继续训练
    resume = True
    resume_model_path = os.path.join(current_dir, "model", "best_model_merged.pth")