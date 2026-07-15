import tempfile
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
import sys

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from xasr.hotwords import prepare_hotword_assets


class HotwordAssetTests(unittest.TestCase):
    def test_prepares_bpe_vocab_and_scored_hotword_variants(self):
        with tempfile.TemporaryDirectory() as root:
            model_dir = Path(root) / "models"
            runtime_dir = Path(root) / "runtime"
            model_dir.mkdir()
            (model_dir / "tokens.txt").write_text(
                "<blk> 0\n▁Open 1\n▁你 2\n▁好 3\n",
                encoding="utf-8",
            )

            assets = prepare_hotword_assets(
                model_dir,
                ["openai", "你好"],
                score=5.0,
                runtime_dir=runtime_dir,
            )

            self.assertEqual(assets.decoding_method, "modified_beam_search")
            hotword_text = assets.hotwords_file.read_text(encoding="utf-8")
            self.assertIn("openai :2.5", hotword_text)
            self.assertIn("Openai :2.5", hotword_text)
            self.assertIn("你 好 :5", hotword_text)
            vocab_text = assets.bpe_vocab.read_text(encoding="utf-8")
            self.assertIn("▁\t-1", vocab_text)
            self.assertIn("你\t-999999", vocab_text)


if __name__ == "__main__":
    unittest.main()
