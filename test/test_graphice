import torch
import sys

print("=" * 60)
print("CUDA环境诊断")
print("=" * 60)

print(f"\n1. Python版本: {sys.version}")

print(f"\n2. PyTorch版本: {torch.__version__}")

print(f"\n3. CUDA是否可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"   CUDA版本: {torch.version.cuda}")
    print(f"   GPU数量: {torch.cuda.device_count()}")
    print(f"   GPU名称: {torch.cuda.get_device_name(0)}")
    print(f"   显存大小: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("\n❌ CUDA不可用！可能的原因：")
    print("   1. 安装了CPU版本的PyTorch")
    print("   2. CUDA驱动未安装或版本不匹配")
    print("   3. PyTorch版本与CUDA版本不兼容")
    
    print("\n检查CUDA驱动:")
    import subprocess
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        print(result.stdout)
    except:
        print("   无法运行nvidia-smi，可能NVIDIA驱动未安装")

print("\n" + "=" * 60)