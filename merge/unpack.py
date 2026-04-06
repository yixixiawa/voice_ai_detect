import os
import shutil
from pathlib import Path

def extract_files_to_real():
    # 源目录和目标目录
    source_dir = r"D:\workspace\python\data\test\cv-corpus-25.0-2026-03-09\en\clips"
    target_dir = r"D:\workspace\python\data\test"
    
    # 确保目标目录存在
    os.makedirs(target_dir, exist_ok=True)
    
    # 遍历源目录
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            # 源文件完整路径
            source_file = os.path.join(root, file)
            # 目标文件路径
            target_file = os.path.join(target_dir, file)
            
            # 处理文件名冲突：如果目标文件已存在，添加序号
            counter = 1
            original_target = target_file
            while os.path.exists(target_file):
                name, ext = os.path.splitext(original_target)
                target_file = f"{name}_{counter}{ext}"
                counter += 1
            
            # 移动文件
            shutil.move(source_file, target_file)
            print(f"移动: {source_file} -> {target_file}")
    
    print("所有文件移动完成！")

if __name__ == "__main__":
    extract_files_to_real()