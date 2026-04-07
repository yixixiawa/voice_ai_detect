import torch
import torchaudio
import os
import sys
from pathlib import Path
import soundfile as sf

# 允许在 test 目录直接运行脚本时导入项目根目录模块
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import LightweightCNN

CONFIDENCE_THRESHOLD = 0.8
SUPPORTED_AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac', '.wma'}

def load_model(model_path, device='cuda'):
    """加载pth模型文件"""
    # 创建模型实例
    model = LightweightCNN(num_classes=2)
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    # 处理不同的保存格式
    if 'model_state_dict' in checkpoint:
        # 如果保存的是完整checkpoint
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        val_acc = checkpoint.get('val_acc', 'unknown')
        print(f"加载checkpoint: epoch {epoch}, 验证准确率 {val_acc}%")
    else:
        # 如果直接保存的是state_dict
        model.load_state_dict(checkpoint)
        print("加载模型权重")
    
    model.to(device)
    model.eval()
    return model

def _load_audio(audio_path):
    """优先使用 torchaudio 读取，失败后回退到 soundfile。"""
    try:
        waveform, sr = torchaudio.load(audio_path)
        return waveform.float(), sr
    except Exception:
        audio, sr = sf.read(audio_path)
        waveform = torch.from_numpy(audio).float()
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.ndim == 2:
            waveform = waveform.transpose(0, 1)
        return waveform, sr

def preprocess_audio(audio_path, sample_rate=16000, duration=2):
    """预处理音频文件"""
    target_len = duration * sample_rate
    
    try:
        # 加载音频
        waveform, sr = _load_audio(audio_path)
        
        # 转单声道
        if waveform.ndim > 1 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        elif waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        
        # 重采样
        if sr != sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sr, sample_rate
            )
        
        # 固定长度
        if waveform.shape[1] > target_len:
            waveform = waveform[:, :target_len]
        elif waveform.shape[1] < target_len:
            padding = target_len - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        
        # 提取梅尔频谱
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=64,
            n_fft=1024,
            hop_length=512,
        )
        mel = mel_spectrogram(waveform)
        mel = torchaudio.transforms.AmplitudeToDB()(mel)
        
        # 归一化
        if mel.std() > 0:
            mel = (mel - mel.mean()) / (mel.std() + 1e-8)
        
        return mel.unsqueeze(0)  # 添加batch维度
        
    except Exception as e:
        print(f"音频处理失败: {e}")
        return None

def collect_audio_files(test_folder):
    """递归收集支持的音频文件。"""
    folder = Path(test_folder)
    return [
        str(p) for p in folder.rglob('*')
        if p.is_file() and p.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    ]

def predict(model, audio_path, device='cuda'):
    """预测单个音频文件"""
    # 预处理音频
    mel = preprocess_audio(audio_path)
    if mel is None:
        return None, None
    
    mel = mel.to(device)
    
    # 推理
    with torch.no_grad():
        outputs = model(mel)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(outputs, dim=1)
    
    pred_idx = int(pred.item())
    confidence = probs[0][pred_idx].item()
    result = "AI合成" if pred_idx == 1 else "真人"

    # 低置信度时按 AI 合成处理，避免将不确定样本判为真人
    if confidence < CONFIDENCE_THRESHOLD:
        result = "AI合成"
    
    return result, confidence

def predict_batch(model, audio_paths, device='cuda'):
    """批量预测多个音频文件"""
    results = []
    interrupted = False
    
    for idx, audio_path in enumerate(audio_paths, start=1):
        try:
            result, confidence = predict(model, audio_path, device)
            if result:
                results.append({
                    'file': audio_path,
                    'result': result,
                    'confidence': confidence
                })
                print(f"{audio_path}: {result} (置信度: {confidence:.2%})")
        except KeyboardInterrupt:
            interrupted = True
            print(f"\n检测被用户中断，已处理 {idx - 1}/{len(audio_paths)} 个文件。")
            break
    
    return results, interrupted


def print_summary(results, total_files, interrupted=False):
    """输出汇总和基于当前结果的结论。"""
    processed = len(results)
    real_cnt = sum(1 for item in results if item['result'] == '真人')
    fake_cnt = sum(1 for item in results if item['result'] == 'AI合成')

    if interrupted:
        print("\n[中断汇总] 基于已处理样本给出当前结论：")
    else:
        print("\n[完整汇总]")

    print(f"已处理: {processed}/{total_files}")
    print(f"真人: {real_cnt}，AI合成: {fake_cnt}")

    if processed == 0:
        print("结论: 暂无有效结果（未成功处理任何文件）。")
        return

    fake_ratio = fake_cnt / processed
    real_ratio = real_cnt / processed

    if fake_cnt > real_cnt:
        print(f"结论: 当前样本以 AI合成 为主（占比 {fake_ratio:.2%}）。")
    elif real_cnt > fake_cnt:
        print(f"结论: 当前样本以 真人 为主（占比 {real_ratio:.2%}）。")
    else:
        print("结论: 当前样本中 真人 与 AI合成 数量持平。")

if __name__ == "__main__":
    # 配置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = str(PROJECT_ROOT / 'model' / 'best_model_merged.pth')  # 你的pth文件路径
    
    print(f"加载模型: {model_path}")
    model = load_model(model_path, device)
    
    # 默认测试整个文件夹
    test_folder = str(PROJECT_ROOT / "data" / "test_data")
    audio_files = collect_audio_files(test_folder)

    if not audio_files:
        exts = ', '.join(sorted(SUPPORTED_AUDIO_EXTENSIONS))
        print(f"未找到可测试的音频文件: {test_folder}")
        print(f"支持的格式: {exts}")
    else:
        print(f"\n开始批量预测，共 {len(audio_files)} 个文件...")
        results, interrupted = predict_batch(model, audio_files, device)
        print_summary(results, len(audio_files), interrupted)