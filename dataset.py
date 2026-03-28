import torch
import torchaudio
import os
import random
import numpy as np
import soundfile as sf

class SpeechDataset(torch.utils.data.Dataset):
    """按需加载音频，避免内存爆炸"""
    def __init__(self, root_dir, config, mode='train', val_split=0.1):
        self.config = config
        self.mode = mode
        self.samples = []
        
        print(f"\n加载数据集 (模式: {mode})...")
        
        # 递归加载真人素材
        real_dir = os.path.join(root_dir, 'real')
        fake_dir = os.path.join(root_dir, 'fake')
        
        if os.path.exists(real_dir):
            self._load_audio_files(real_dir, 0)
        if os.path.exists(fake_dir):
            self._load_audio_files(fake_dir, 1)
        
        print(f"  找到 {len(self.samples):,} 个音频文件")
        real_count = len([s for s in self.samples if s[1] == 0])
        fake_count = len([s for s in self.samples if s[1] == 1])
        print(f"    真人: {real_count:,}")
        print(f"    AI: {fake_count:,}")
        
        # 划分训练集和验证集
        random.shuffle(self.samples)
        val_size = int(len(self.samples) * val_split)
        
        if mode == 'train':
            self.samples = self.samples[val_size:]
        else:
            self.samples = self.samples[:val_size]
        
        print(f"  {mode}集: {len(self.samples):,} 条")
    
    def _load_audio_files(self, directory, label):
        """递归遍历目录，加载所有音频文件"""
        supported_extensions = ('.wav', '.flac', '.ogg')
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(supported_extensions):
                    filepath = os.path.join(root, file)
                    self.samples.append((filepath, label))
    
    def load_audio(self, path):
        """加载并预处理音频"""
        target_len = self.config.duration * self.config.sample_rate
        
        try:
            # 使用soundfile加载
            audio, sr = sf.read(path)
            waveform = torch.from_numpy(audio).float()
            
            # 转单声道
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
            
            return waveform
            
        except Exception as e:
            return torch.zeros(1, target_len)
    
    def extract_mel_spectrogram(self, waveform):
        """提取梅尔频谱图"""
        try:
            mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.config.sample_rate,
                n_mels=self.config.n_mels,
                n_fft=1024,
                hop_length=512,
                power=2.0,
            )
            mel = mel_spectrogram(waveform)
            mel = torchaudio.transforms.AmplitudeToDB()(mel)
            
            # 归一化
            if mel.std() > 0:
                mel = (mel - mel.mean()) / (mel.std() + 1e-8)
            
            return mel
        except:
            return torch.zeros(1, self.config.n_mels, 94)
    
    def add_augmentation(self, waveform):
        """数据增强"""
        if not self.config.use_augmentation:
            return waveform
        
        # 添加噪声
        if random.random() < 0.3:
            noise = torch.randn_like(waveform) * 0.005
            waveform = waveform + noise
        
        # 音量变化
        if random.random() < 0.3:
            gain = random.uniform(0.7, 1.3)
            waveform = waveform * gain
        
        return waveform
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        
        try:
            waveform = self.load_audio(path)
            waveform = self.add_augmentation(waveform)
            mel = self.extract_mel_spectrogram(waveform)
            return mel, torch.tensor(label, dtype=torch.long)
        except:
            # 出错时返回随机数据
            return torch.randn(1, self.config.n_mels, 94), torch.tensor(random.randint(0, 1), dtype=torch.long)