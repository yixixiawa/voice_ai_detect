import os
import torch
import torchaudio

print("=" * 60)
print("数据诊断工具")
print("=" * 60)

# 1. 检查当前目录
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f"\n1. 当前工作目录: {current_dir}")

# 2. 检查data文件夹
data_path = os.path.join(current_dir, "data")
print(f"\n2. data文件夹路径: {data_path}")
print(f"   data文件夹是否存在: {os.path.exists(data_path)}")

if not os.path.exists(data_path):
    print("   ❌ data文件夹不存在！")
    print("   请确保在项目根目录创建data文件夹")
    exit()

# 3. 检查real和fake文件夹
real_path = os.path.join(data_path, "real")
fake_path = os.path.join(data_path, "fake")

print(f"\n3. real文件夹: {real_path}")
print(f"   是否存在: {os.path.exists(real_path)}")
print(f"\n4. fake文件夹: {fake_path}")
print(f"   是否存在: {os.path.exists(fake_path)}")

# 4. 递归查找所有音频文件
print("\n" + "=" * 60)
print("递归查找音频文件:")
print("=" * 60)

def find_audio_files(directory, label_name):
    """递归查找所有音频文件"""
    audio_files = []
    supported_extensions = ('.wav', '.mp3', '.flac', '.m4a', '.ogg', '.wma')
    
    if not os.path.exists(directory):
        print(f"   ❌ {label_name} 文件夹不存在: {directory}")
        return audio_files
    
    print(f"\n   📁 搜索 {label_name} 文件夹: {directory}")
    file_count = 0
    folder_count = 0
    
    for root, dirs, files in os.walk(directory):
        folder_count += 1
        for file in files:
            if file.lower().endswith(supported_extensions):
                filepath = os.path.join(root, file)
                audio_files.append(filepath)
                file_count += 1
                
                # 只显示前10个文件示例
                if file_count <= 10:
                    print(f"      ✅ 找到: {os.path.relpath(filepath, directory)}")
        
        # 只显示前5个文件夹
        if folder_count <= 5:
            print(f"      📂 扫描子文件夹: {os.path.relpath(root, directory)}")
    
    print(f"\n   📊 {label_name} 文件夹统计:")
    print(f"      总文件数: {file_count}")
    print(f"      扫描子文件夹数: {folder_count}")
    
    return audio_files

# 搜索真人音频
real_audio_files = find_audio_files(real_path, "真人(Real)")

# 搜索AI音频
fake_audio_files = find_audio_files(fake_path, "AI合成(Fake)")

# 5. 检查音频文件是否可以正常加载
print("\n" + "=" * 60)
print("测试音频文件加载:")
print("=" * 60)

def test_audio_loading(audio_files, max_test=5):
    """测试音频文件是否可以正常加载"""
    success_count = 0
    fail_count = 0
    
    for i, filepath in enumerate(audio_files[:max_test]):
        try:
            print(f"\n   测试 {i+1}: {filepath}")
            waveform, sample_rate = torchaudio.load(filepath)
            print(f"      ✅ 成功加载!")
            print(f"         采样率: {sample_rate} Hz")
            print(f"         形状: {waveform.shape}")
            print(f"         时长: {waveform.shape[1] / sample_rate:.2f} 秒")
            success_count += 1
        except Exception as e:
            print(f"      ❌ 加载失败: {e}")
            fail_count += 1
    
    return success_count, fail_count

if real_audio_files:
    print("\n测试真人音频:")
    success, fail = test_audio_loading(real_audio_files)
    print(f"\n   真人音频测试结果: 成功 {success}, 失败 {fail}")
else:
    print("\n❌ 没有找到真人音频文件！")

if fake_audio_files:
    print("\n测试AI音频:")
    success, fail = test_audio_loading(fake_audio_files)
    print(f"\n   AI音频测试结果: 成功 {success}, 失败 {fail}")
else:
    print("\n❌ 没有找到AI音频文件！")

# 6. 总结
print("\n" + "=" * 60)
print("诊断总结:")
print("=" * 60)

if not real_audio_files and not fake_audio_files:
    print("❌ 严重问题: 没有找到任何音频文件！")
    print("\n可能的原因:")
    print("  1. 数据文件夹路径不正确")
    print("  2. 音频文件格式不支持（支持的格式: .wav, .mp3, .flac, .m4a, .ogg）")
    print("  3. 文件夹权限问题")
    print("\n建议解决方案:")
    print("  1. 确认当前在项目根目录再运行脚本")
    print("  2. 检查data文件夹确实包含real和fake子文件夹")
    print("  3. 确认音频文件确实在这些子文件夹中")
elif not real_audio_files:
    print("⚠️  警告: 找到AI音频，但没有真人音频")
    print("   请确保将真人音频放入 data/real/ 及其子文件夹")
elif not fake_audio_files:
    print("⚠️  警告: 找到真人音频，但没有AI音频")
    print("   请确保将AI音频放入 data/fake/ 及其子文件夹")
else:
    print(f"✅ 数据准备就绪!")
    print(f"   真人音频: {len(real_audio_files)} 个")
    print(f"   AI音频: {len(fake_audio_files)} 个")
    print(f"   总计: {len(real_audio_files) + len(fake_audio_files)} 个音频文件")
    print("\n现在可以运行训练了: python train.py")

print("\n" + "=" * 60)