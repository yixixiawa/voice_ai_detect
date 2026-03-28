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
    batch_size = 256  # 充分利用GPU
    num_workers = 8   # 多线程加载
    epochs = 20
    learning_rate = 1e-3
    
    # 数据划分
    val_split = 0.1
    
    # 数据增强（关闭以提升速度）
    use_augmentation = False
    
    # 混合精度训练
    use_amp = True
    
    # 梯度累积
    gradient_accumulation = 2
    
    # 模型保存
    save_checkpoint = True
    checkpoint_interval = 2