import shutil
from pathlib import Path
import math

def group_files_by_count(target_dir, files_per_folder=500):
    """
    将real目录中的文件按数量分组到子文件夹
    
    Args:
        target_dir: 需要分组的目录
        files_per_folder: 每个子文件夹存放的文件数量
    """
    target_dir = Path(target_dir)
    
    # 获取所有文件
    all_files = [f for f in target_dir.iterdir() if f.is_file()]
    total_files = len(all_files)
    
    if total_files == 0:
        print("没有找到文件！")
        return
    
    # 计算需要的文件夹数量
    num_folders = math.ceil(total_files / files_per_folder)
    
    print(f"总文件数: {total_files}")
    print(f"每个文件夹存放: {files_per_folder} 个文件")
    print(f"需要创建: {num_folders} 个文件夹")
    
    # 创建子文件夹并移动文件
    for i in range(num_folders):
        # 创建子文件夹，命名为 group_001, group_002, ...
        group_folder = target_dir / f"group_{i+1:03d}"
        group_folder.mkdir(exist_ok=True)
        
        # 计算当前文件夹的文件范围
        start_idx = i * files_per_folder
        end_idx = min((i + 1) * files_per_folder, total_files)
        
        # 移动文件
        for j in range(start_idx, end_idx):
            file_path = all_files[j]
            target_path = group_folder / file_path.name
            
            # 处理重名（理论上不会有重名）
            if target_path.exists():
                counter = 1
                stem = file_path.stem
                suffix = file_path.suffix
                while target_path.exists():
                    target_path = group_folder / f"{stem}_{counter}{suffix}"
                    counter += 1
            
            shutil.move(str(file_path), str(target_path))
        
        print(f"✓ 已创建 {group_folder.name}，移动了 {end_idx - start_idx} 个文件")
    
    print(f"\n完成！文件已分组到 {num_folders} 个文件夹中")

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    real_dir = project_root / "data" / "test"
    group_files_by_count(real_dir, files_per_folder=500)