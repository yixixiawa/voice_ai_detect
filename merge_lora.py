import os
import json
import torch

from model import LightweightCNN

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from accelerate import init_empty_weights, load_checkpoint_and_dispatch
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

# ================= 配置区域 =================
# 1. 基座模型目录（你说模型在 model 目录）
base_model_path = "./model"

# 2. 权重路径
#    - 你的训练文件通常是 best_model.pth / checkpoint_epoch_xx.pth
#    - 如果传入 LoRA 目录，也支持 HF 合并
weight_path = "./best_model.pth"

# 3. 输出目录（默认直接写回 model 目录）
save_path = "./model"

# 4. 分类数（与你训练时一致）
num_classes = 2
# ===========================================


def export_lightweightcnn_checkpoint(checkpoint_path: str, output_dir: str, classes: int) -> None:
    """导出本项目的 LightweightCNN 权重到 model 目录。"""
    print("🚀 检测到本地 PyTorch checkpoint，走 LightweightCNN 导出流程...")
    os.makedirs(output_dir, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model = LightweightCNN(num_classes=classes)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    model.eval()

    out_pth = os.path.join(output_dir, "best_model_merged.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_classes": classes,
        },
        out_pth,
    )

    # 保存一份元信息，方便后续推理脚本读取
    meta = {
        "model_name": "LightweightCNN",
        "num_classes": classes,
        "source_checkpoint": checkpoint_path,
        "missing_keys": len(missing_keys),
        "unexpected_keys": len(unexpected_keys),
    }
    with open(os.path.join(output_dir, "model_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"ℹ️ 缺失参数数量: {len(missing_keys)}")
    print(f"ℹ️ 多余参数数量: {len(unexpected_keys)}")
    print(f"✅ 导出完成：{out_pth}")


def merge_hf_lora(base_path: str, lora_dir: str, output_dir: str) -> None:
    """可选：保留 HuggingFace LoRA 合并能力。"""
    if not HF_AVAILABLE:
        raise RuntimeError("transformers/peft/accelerate 不可用，无法执行 HF LoRA 合并。")

    print("🚀 开始 HF LoRA 合并流程 (CPU+GPU 混合模式)...")
    print("⏳ 正在构建模型架构...")
    with init_empty_weights():
        base_model = AutoModelForCausalLM.from_pretrained(
            base_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

    print("⏳ 正在加载权重 (自动分配 CPU/GPU)...")
    base_model = load_checkpoint_and_dispatch(
        base_model,
        base_path,
        device_map="auto",
        dtype=torch.float16,
        trust_remote_code=True,
    )

    print("🔧 正在加载 LoRA 补丁...")
    model = PeftModel.from_pretrained(base_model, lora_dir)

    print("🔥 正在执行合并 (Merge and Unload)...")
    merged_model = model.merge_and_unload()

    print("💾 正在保存模型到磁盘...")
    merged_model.save_pretrained(output_dir, max_shard_size="2GB")
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ HF LoRA 合并完成！模型已保存至: {output_dir}")


if __name__ == "__main__":
    # 你的当前工程是 best_model.pth + model.py，默认走这里
    if os.path.isfile(weight_path):
        export_lightweightcnn_checkpoint(weight_path, save_path, num_classes)
    # 仅当传入 LoRA 目录时，才尝试 HF 合并
    elif os.path.isdir(weight_path):
        merge_hf_lora(base_model_path, weight_path, save_path)
    else:
        raise FileNotFoundError(f"找不到权重路径: {weight_path}")