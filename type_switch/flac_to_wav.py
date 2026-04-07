import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def convert_flac_to_wav(input_file, output_file, overwrite=False):
    """使用 ffmpeg 转换单个 FLAC 文件到 WAV"""
    if not overwrite and output_file.exists():
        return False
    
    cmd = [
        'ffmpeg',
        '-i', str(input_file),
        '-acodec', 'pcm_s16le',  # 16-bit PCM WAV
        '-ar', '16000',           # 16kHz 采样率（语音识别常用）
        '-ac', '1',               # 单声道
        '-y',                     # 覆盖输出文件
        str(output_file)
    ]
    
    try:
        # 静默模式，不输出信息
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"转换失败 {input_file.name}: {e.stderr.decode()}")
        return False

def process_speaker_folder(flac_folder, output_root, overwrite=False):
    """处理单个说话人的文件夹"""
    flac_path = Path(flac_folder)
    speaker_id = flac_path.name  # 例如 id11446
    
    # 在输出目录中创建对应的文件夹
    output_folder = Path(output_root) / speaker_id
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # 查找所有 flac 文件
    flac_files = list(flac_path.glob("*.flac"))
    
    converted = 0
    for flac_file in flac_files:
        wav_file = output_folder / f"{flac_file.stem}.wav"
        if convert_flac_to_wav(flac_file, wav_file, overwrite):
            converted += 1
    
    return speaker_id, len(flac_files), converted

def batch_convert_cnceleb2(input_root, output_root, max_workers=4, overwrite=False):
    """
    批量转换 CN-Celeb2 数据集
    
    Args:
        input_root: 输入根目录 (例如: data/CN-Celeb2_flac/data)
        output_root: 输出根目录 (例如: data/CN-Celeb2_wav/data)
        max_workers: 并行线程数
        overwrite: 是否覆盖已存在的文件
    """
    input_path = Path(input_root)
    output_path = Path(output_root)
    
    if not input_path.exists():
        print(f"错误：输入目录不存在 {input_root}")
        return
    
    # 获取所有说话人文件夹
    speaker_folders = [f for f in input_path.iterdir() if f.is_dir()]
    
    if not speaker_folders:
        print(f"在 {input_root} 中没有找到子文件夹")
        return
    
    print(f"找到 {len(speaker_folders)} 个说话人文件夹")
    print(f"输出目录: {output_root}")
    print(f"使用 {max_workers} 个线程并行处理\n")
    
    start_time = time.time()
    total_files = 0
    total_converted = 0
    
    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_speaker_folder, folder, output_root, overwrite): folder 
            for folder in speaker_folders
        }
        
        for i, future in enumerate(as_completed(futures), 1):
            speaker_id, total, converted = future.result()
            total_files += total
            total_converted += converted
            print(f"[{i}/{len(speaker_folders)}] {speaker_id}: {converted}/{total} 转换成功")
    
    elapsed_time = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"转换完成！")
    print(f"总文件数: {total_files}")
    print(f"成功转换: {total_converted}")
    print(f"失败数量: {total_files - total_converted}")
    print(f"用时: {elapsed_time:.2f} 秒")
    print(f"输出目录: {output_root}")

def convert_single_speaker(speaker_folder, output_root, overwrite=False):
    """只转换单个说话人的文件夹"""
    input_path = Path(speaker_folder)
    if not input_path.exists():
        print(f"错误：文件夹不存在 {speaker_folder}")
        return
    
    speaker_id = input_path.name
    output_folder = Path(output_root) / speaker_id
    output_folder.mkdir(parents=True, exist_ok=True)
    
    flac_files = list(input_path.glob("*.flac"))
    print(f"找到 {len(flac_files)} 个 FLAC 文件")
    
    for i, flac_file in enumerate(flac_files, 1):
        wav_file = output_folder / f"{flac_file.stem}.wav"
        print(f"[{i}/{len(flac_files)}] 转换: {flac_file.name} -> {wav_file.name}")
        
        if convert_flac_to_wav(flac_file, wav_file, overwrite):
            print(f"  ✓ 成功")
        else:
            print(f"  ✗ 失败")

# ============= 使用示例 =============

if __name__ == "__main__":
    # 设置路径（基于项目根目录，避免硬编码绝对路径）
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    INPUT_ROOT = PROJECT_ROOT / "data" / "CN-Celeb2_flac" / "data"
    OUTPUT_ROOT = PROJECT_ROOT / "data" / "dt"
    
    # 方式1：批量转换所有说话人
    batch_convert_cnceleb2(
        input_root=INPUT_ROOT,
        output_root=OUTPUT_ROOT,
        max_workers=4,      # 并行线程数，可根据 CPU 核心数调整
        overwrite=False     # 是否覆盖已存在的文件
    )
    
    # 方式2：只转换单个说话人（测试用）
    # convert_single_speaker(
    #     speaker_folder=PROJECT_ROOT / "data" / "CN-Celeb2_flac" / "data" / "id11446",
    #     output_root=PROJECT_ROOT / "data" / "CN-Celeb2_wav" / "data",
    #     overwrite=False
    # )