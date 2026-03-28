import torch
import torchaudio
import os
import random
import numpy as np
import soundfile as sf
from tqdm import tqdm

class CachedSpeechDataset(torch.utils.data.Dataset):
    """缓存梅尔频谱到内存，大幅提升速度"""
    def __init__(self, root_dir, config, mode='train', val_split=0.1):
        self.config = config
        self.mode = mode
        self.cache = []  # 缓存梅尔频谱和标签
        self.samples = []
        
        print(f"\n加载数据集 (模式: {mode})...")
        
        # 加载文件列表
        real_dir = os.path.join(root_dir, 'real')
        fake_dir = os.path.join(root_dir, 'fake')
        
        if os.path.exists(real_dir):
            self._load_audio_files(real_dir, 0)
        if os.path.exists(fake_dir):
            self._load_audio_files(fake_dir, 1)
        
        print(f"  找到 {len(self.samples):,} 个音频文件")
        {}
        # 划分数据集
        random.shuffle(self.samples)
        val_size = int(len(self.samples) * val_split)
        
        if mode == 'train':
            self.samples = self.samples[val_size:]
        else:
            self.samples = self.samples[:val_size]
        
        print(f"  {mode}集: {len(self.samples):,} 条")
        
        # 缓存梅尔频谱到内存（首次加载会慢，后续训练飞快）
        print(f"\n缓存梅尔频谱到内存（首次加载需要时间）...")
        self._cache_all()
        print(f"  缓存完成！共 {len(self.cache):,} 条")
    
    def _load_audio_files(self, directory, label):
        """递归加载音频文件"""
        supported_extensions = ('.wav', '.flac', '.ogg')
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(supported_extensions):
                    filepath = os.path.join(root, file)
                    self.samples.append((filepath, label))
    
    def _cache_all(self):
        """预计算并缓存所有梅尔频谱"""
        for path, label in tqdm(self.samples, desc="缓存音频"):
            try:
                mel = self._extract_mel(path)
                self.cache.append((mel, label))
            except:
                # 出错时缓存随机数据
                self.cache.append((
                    torch.randn(1, self.config.n_mels, 94),
                    torch.tensor(random.randint(0, 1), dtype=torch.long)
                ))
    
    def _extract_mel(self, path):
        """提取梅尔频谱"""
        target_len = self.config.duration * self.config.sample_rate
        
        # 加载音频
        audio, sr = sf.read(path)
        waveform = torch.from_numpy(audio).float()
        
        if len(waveform.shape) > 1:
            waveform = waveform.mean(dim=1)
        waveform = waveform.unsqueeze(0)
        
        # 重采样
        if sr != self.config.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sr, self.config.sample_rate
            )
        
        # 固定长度
        if waveform.shape[1] > target_len:
            start = random.randint(0, waveform.shape[1] - target_len)
            waveform = waveform[:, start:start + target_len]
        elif waveform.shape[1] < target_len:
            padding = target_len - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        
        # 提取梅尔频谱
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.config.sample_rate,
            n_mels=self.config.n_mels,
            n_fft=1024,
            hop_length=512,
        )
        mel = mel_spectrogram(waveform)
        mel = torchaudio.transforms.AmplitudeToDB()(mel)
        
        if mel.std() > 0:
            mel = (mel - mel.mean()) / (mel.std() + 1e-8)
        
        return mel
    
    def __len__(self):
        return len(self.cache)
    
    def __getitem__(self, idx):
        mel, label = self.cache[idx]
        return mel, label