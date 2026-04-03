import subprocess
import os
import sys

def mp4_to_wav_ffmpeg(input_file, output_file=None, bitrate='128k', sample_rate=44100):
    """
    使用ffmpeg将MP4转换为WAV
    
    参数:
        input_file: 输入的MP4文件路径
        output_file: 输出的WAV文件路径（可选）
        bitrate: 音频比特率（默认128k）
        sample_rate: 采样率（默认44100 Hz）
    """
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 输入文件 '{input_file}' 不存在")
        return None
    
    # 如果没有指定输出文件，使用输入文件名但扩展名改为.wav
    if output_file is None:
        output_file = os.path.splitext(input_file)[0] + '.wav'
    
    # 检查输出文件是否已存在
    if os.path.exists(output_file):
        response = input(f"输出文件 '{output_file}' 已存在，是否覆盖？(y/n): ")
        if response.lower() != 'y':
            print("操作已取消")
            return None
    
    try:
        # 构建ffmpeg命令
        cmd = [
            'ffmpeg',
            '-i', input_file,           # 输入文件
            '-vn',                      # 不处理视频
            '-acodec', 'pcm_s16le',     # 音频编码器（WAV格式）
            '-ar', str(sample_rate),    # 采样率
            '-ac', '2',                 # 声道数（立体声）
            '-ab', bitrate,             # 比特率
            '-y',                       # 自动覆盖输出文件
            output_file
        ]
        
        print(f"正在转换: {input_file} -> {output_file}")
        print(f"参数: 采样率={sample_rate}Hz, 比特率={bitrate}, 声道=立体声")
        
        # 执行命令
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"转换成功: {output_file}")
            # 显示文件大小
            file_size = os.path.getsize(output_file) / (1024 * 1024)
            print(f"文件大小: {file_size:.2f} MB")
            return output_file
        else:
            print(f"转换失败: {result.stderr}")
            return None
            
    except FileNotFoundError:
        print("错误: 未找到ffmpeg，请先安装ffmpeg")
        print("安装方法:")
        print("  Windows: 从 https://ffmpeg.org/download.html 下载")
        print("  Mac: brew install ffmpeg")
        print("  Ubuntu/Debian: sudo apt-get install ffmpeg")
        return None
    except Exception as e:
        print(f"发生错误: {e}")
        return None

def batch_convert_ffmpeg(input_folder, output_folder=None, **kwargs):
    """
    批量转换文件夹中的所有MP4文件
    
    参数:
        input_folder: 输入文件夹路径
        output_folder: 输出文件夹路径（可选）
        **kwargs: 传递给mp4_to_wav_ffmpeg的其他参数
    """
    if output_folder is None:
        output_folder = input_folder
    
    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 获取所有MP4文件
    mp4_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.mp4')]
    
    if not mp4_files:
        print("没有找到MP4文件")
        return
    
    print(f"找到 {len(mp4_files)} 个MP4文件")
    success_count = 0
    
    for mp4_file in mp4_files:
        input_path = os.path.join(input_folder, mp4_file)
        output_filename = os.path.splitext(mp4_file)[0] + '.wav'
        output_path = os.path.join(output_folder, output_filename)
        
        result = mp4_to_wav_ffmpeg(input_path, output_path, **kwargs)
        if result:
            success_count += 1
    
    print(f"\n批量转换完成: {success_count}/{len(mp4_files)} 个文件成功")

# 使用示例
if __name__ == "__main__":
    # 简单的命令行接口
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        if len(sys.argv) > 2:
            output_file = sys.argv[2]
            mp4_to_wav_ffmpeg(input_file, output_file)
        else:
            mp4_to_wav_ffmpeg(input_file)
    else:
        # 单个文件转换示例
        # mp4_to_wav_ffmpeg("input.mp4", "output.wav")
        # mp4_to_wav_ffmpeg("input.mp4", "output.wav", bitrate='192k', sample_rate=48000)
        
        # 批量转换示例
        # batch_convert_ffmpeg("videos_folder", "audio_folder", bitrate='128k', sample_rate=44100)
        print("请指定输入文件: python mp4_to_wav.py input.mp4 [output.wav]")