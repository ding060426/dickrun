#!/usr/bin/env python3
"""
X-ASR 模型下载脚本
从 Hugging Face 下载 sherpa-onnx 流式模型文件

模型来源: GilgameshWind/X-ASR-zh-en (Apache 2.0)
模型大小: ~586MB (encoder 565MB, decoder 11MB, joiner 10MB)

手动下载:
  浏览器打开 https://huggingface.co/GilgameshWind/X-ASR-zh-en
  下载 deployment/models/chunk-160ms-model/ 中的文件到 backend/xasr/models/:
    - encoder-160ms.onnx  (~565MB)
    - decoder-160ms.onnx  (~11MB)
    - joiner-160ms.onnx   (~10MB)
    - tokens.txt          (~63KB)
"""

import os
import sys
import shutil
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "models"
REPO_ID = "GilgameshWind/X-ASR-zh-en"
REMOTE_PATH = "deployment/models/chunk-160ms-model"

FILES = [
    "encoder-160ms.onnx",
    "decoder-160ms.onnx",
    "joiner-160ms.onnx",
    "tokens.txt",
]

MIN_SIZES = {
    "encoder-160ms.onnx": 100 * 1024 * 1024,
    "decoder-160ms.onnx": 1 * 1024 * 1024,
    "joiner-160ms.onnx": 1 * 1024 * 1024,
    "tokens.txt": 40 * 1024,
}


def download_file(repo_id: str, remote_path: str, dest: Path, desc: str):
    """下载文件并显示进度"""
    from huggingface_hub import hf_hub_download

    print(f"  下载 {desc}...", end=" ", flush=True)

    try:
        full_path = f"{remote_path}/{desc}"
        local = hf_hub_download(
            repo_id=repo_id,
            filename=full_path,
            local_dir=str(dest.parent),
            local_dir_use_symlinks=False,
        )
        # hf_hub_download 会把文件放到子目录，移到正确位置
        if local != str(dest):
            shutil.move(local, str(dest))
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"✅ ({size_mb:.1f}MB)")
        return True
    except Exception as e:
        print(f"❌ 失败: {e}")
        return False


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  X-ASR 模型下载工具")
    print(f"  模型来源: {REPO_ID}")
    print(f"  目标目录: {MODEL_DIR}")
    print("=" * 60)
    print()

    # 检查 huggingface_hub
    try:
        import huggingface_hub
    except ImportError:
        print("请先安装 huggingface_hub:")
        print("  pip install huggingface_hub")
        return

    success = True
    for filename in FILES:
        dest = MODEL_DIR / filename

        if dest.exists() and dest.stat().st_size >= MIN_SIZES.get(filename, 1):
            size_mb = dest.stat().st_size / (1024 * 1024)
            print(f"  ⏭  跳过 {filename} (已存在, {size_mb:.1f}MB)")
            continue
        if dest.exists():
            size_kb = dest.stat().st_size / 1024
            print(f"  ⚠️  {filename} 文件过小 ({size_kb:.1f}KB)，重新下载")
            dest.unlink()

        if not download_file(REPO_ID, REMOTE_PATH, dest, filename):
            success = False

    # 清理可能的临时文件夹
    for tmp in [MODEL_DIR / "deployment", MODEL_DIR / ".cache"]:
        if tmp.exists():
            import shutil
            shutil.rmtree(tmp)

    print()
    if success:
        print("✅ 所有模型下载完成！")
        print(f"   模型目录: {MODEL_DIR}")
    else:
        print("⚠️  部分文件下载失败。")
        print("   请手动下载:")
        print(f"   https://huggingface.co/{REPO_ID}")
        print(f"   路径: {REMOTE_PATH}/")
        print(f"   放入: {MODEL_DIR}")


if __name__ == "__main__":
    main()
