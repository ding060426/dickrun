"""
谛听 DiTing - 标点恢复模型模块
============================================================================
基于 sherpa-onnx OfflinePunctuation (CT-Transformer) 的标点恢复。

模型: sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12
大小: ~266MB (解压后 model.onnx ~281MB)
来源: https://github.com/k2-fsa/sherpa-onnx/releases/tag/punctuation-models

用法:
    from modules.punctuation_model import PunctuationRestorer
    restorer = PunctuationRestorer(model_dir="backend/xasr/models/punct")
    text = restorer.add_punctuation("今天我们来讨论一下产品的转化率问题")
    # -> "今天我们来讨论一下产品的转化率问题。"
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("punct_model")

# Check for sherpa_onnx
try:
    import sherpa_onnx
    HAS_SHERPA_PUNCT = hasattr(sherpa_onnx, 'OfflinePunctuation')
except ImportError:
    HAS_SHERPA_PUNCT = False


# ======================================================================
# Model download URLs
# ======================================================================

PUNCT_MODEL_INFO = {
    "name": "sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12",
    "url_github": "https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2",
    "url_modelscope": "https://modelscope.cn/models/k2-fsa/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12/resolve/master/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2",
    "files": [
        "model.onnx",       # ~281MB
        "tokens.json",      # token→id mapping
    ],
}

# Default model directory
DEFAULT_PUNCT_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "xasr", "models", "punct"
)


# ======================================================================
# PunctuationRestorer
# ======================================================================

class PunctuationRestorer:
    """
    sherpa-onnx OfflinePunctuation wrapper.

    This model adds punctuation to raw ASR text using a CT-Transformer
    trained specifically for Chinese + English punctuation restoration.
    """

    def __init__(self, model_dir: str = None):
        """
        Args:
            model_dir: Path to the punctuation model directory.
                       Should contain model.onnx (tokens.json is auto-detected).
        """
        self.model_dir = model_dir or DEFAULT_PUNCT_MODEL_DIR
        self._available = False
        self._punct = None

        if not HAS_SHERPA_PUNCT:
            logger.warning("sherpa_onnx.OfflinePunctuation not available")
            return

        model_path = os.path.join(self.model_dir, "model.onnx")

        if not os.path.exists(model_path):
            logger.info(f"Punctuation model not found at {model_path}")
            logger.info(f"Run: python -m modules.punctuation_model --download")
            return

        try:
            config = sherpa_onnx.OfflinePunctuationConfig(
                model=sherpa_onnx.OfflinePunctuationModelConfig(
                    ct_transformer=model_path,
                    num_threads=1,
                    provider="cpu",
                ),
            )
            self._punct = sherpa_onnx.OfflinePunctuation(config)
            self._available = True
            logger.info(f"Punctuation model loaded: {model_path}")
        except Exception as e:
            logger.error(f"Failed to load punctuation model: {e}")
            self._punct = None
            self._available = False

    @property
    def is_available(self) -> bool:
        return self._available and self._punct is not None

    def add_punctuation(self, text: str) -> Optional[str]:
        """
        Add punctuation to raw ASR text.

        Args:
            text: Raw text without punctuation

        Returns:
            Text with punctuation, or None if unavailable
        """
        if not self.is_available or not text or not text.strip():
            return None

        try:
            result = self._punct.add_punctuation(text.strip())
            return result if result else None
        except Exception as e:
            logger.warning(f"Punctuation inference failed: {e}")
            return None


# Singleton
_punct_instance: Optional[PunctuationRestorer] = None


def get_punctuation_restorer(model_dir: str = None) -> PunctuationRestorer:
    """Get or create the singleton PunctuationRestorer."""
    global _punct_instance
    if _punct_instance is None or (model_dir and model_dir != _punct_instance.model_dir):
        _punct_instance = PunctuationRestorer(model_dir=model_dir)
    return _punct_instance


# ======================================================================
# CLI: download model
# ======================================================================

def download_punctuation_model(target_dir: str = None) -> bool:
    """
    Download the punctuation model from GitHub / ModelScope.

    Returns True if successful.
    """
    target_dir = target_dir or DEFAULT_PUNCT_MODEL_DIR
    os.makedirs(target_dir, exist_ok=True)

    # Check if already downloaded
    model_onnx = os.path.join(target_dir, "model.onnx")
    if os.path.exists(model_onnx):
        size_mb = os.path.getsize(model_onnx) / (1024 * 1024)
        print(f"Punctuation model already exists: {model_onnx} ({size_mb:.0f}MB)")
        return True

    import urllib.request
    import shutil
    import tempfile
    import tarfile

    print(f"Downloading punctuation model ({PUNCT_MODEL_INFO['name']})...")
    print(f"This model is ~170MB (compressed ~80MB)")

    # Try ModelScope first (faster in China), then GitHub
    urls = [
        ("ModelScope", PUNCT_MODEL_INFO["url_modelscope"]),
        ("GitHub", PUNCT_MODEL_INFO["url_github"]),
    ]

    for source, url in urls:
        try:
            print(f"  Trying {source}...")
            tmp_file = os.path.join(tempfile.gettempdir(),
                                    f"punct_model_{os.urandom(4).hex()}.tar.bz2")
            urllib.request.urlretrieve(url, tmp_file)
            print(f"  Downloaded to {tmp_file}")

            # Extract
            print(f"  Extracting to {target_dir}...")
            with tarfile.open(tmp_file, "r:bz2") as tar:
                # Strip the top-level directory
                for member in tar.getmembers():
                    # Remove the top-level directory component
                    parts = member.name.split('/', 1)
                    if len(parts) > 1:
                        member.name = parts[1]
                        if member.name:  # skip empty
                            tar.extract(member, target_dir)

            os.unlink(tmp_file)
            print(f"  Done! Model extracted to {target_dir}")

            # Verify
            if os.path.exists(model_onnx):
                size_mb = os.path.getsize(model_onnx) / (1024 * 1024)
                print(f"  Verified: model.onnx ({size_mb:.0f}MB)")
                return True

        except Exception as e:
            print(f"  {source} failed: {e}")
            continue

    print("\nERROR: Could not download from any source.")
    print("Please download manually:")
    print(f"  1. Visit: {PUNCT_MODEL_INFO['url_github']}")
    print(f"  2. Extract to: {target_dir}")
    return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DiTing Punctuation Model Manager")
    parser.add_argument("--download", action="store_true",
                        help="Download punctuation model")
    parser.add_argument("--dir", default=None,
                        help=f"Model directory (default: {DEFAULT_PUNCT_MODEL_DIR})")
    parser.add_argument("--test", action="store_true",
                        help="Test with sample text")
    args = parser.parse_args()

    if args.download:
        success = download_punctuation_model(args.dir)
        sys.exit(0 if success else 1)

    if args.test:
        restorer = PunctuationRestorer(model_dir=args.dir)
        if restorer.is_available:
            tests = [
                "今天我们来讨论一下产品的转化率问题",
                "我觉得这个方案不太行但是我们可以调整一下",
                "首先确认一下目标人群然后讨论市场策略最后分配预算",
                "这个数据准确吗我们需要核实一下",
            ]
            print("=== Punctuation Model Test ===\n")
            for t in tests:
                result = restorer.add_punctuation(t)
                print(f"  RAW : {t}")
                print(f"  PROC: {result}")
                print()
        else:
            print("Punctuation model not available.")
            print(f"Expected at: {restorer.model_dir}")
            print("Run with --download to download the model.")
        sys.exit(0)

    # Default: show status
    restorer = PunctuationRestorer(model_dir=args.dir)
    model_onnx = os.path.join(restorer.model_dir, "model.onnx")
    if os.path.exists(model_onnx):
        size_mb = os.path.getsize(model_onnx) / (1024 * 1024)
        print(f"Punctuation model: FOUND ({size_mb:.0f}MB)")
    else:
        print(f"Punctuation model: NOT FOUND")
        print(f"Expected at: {model_onnx}")
        print(f"Run with --download to download.")
