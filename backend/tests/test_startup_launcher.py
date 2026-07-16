from pathlib import Path
import tempfile
import unittest

import start


class StartupLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_prefers_explicit_override(self):
        configured = self.root / "custom-python.exe"
        configured.touch()

        resolved = start.resolve_backend_python(
            root_dir=self.root,
            environ={"DITING_BACKEND_PYTHON": str(configured)},
            current_python="fallback-python",
        )

        self.assertEqual(resolved, str(configured))

    def test_uses_qwen_environment_when_present(self):
        qwen_python = self.root / ".venv-qwen3" / "Scripts" / "python.exe"
        qwen_python.parent.mkdir(parents=True)
        qwen_python.touch()

        resolved = start.resolve_backend_python(
            root_dir=self.root,
            environ={},
            current_python="fallback-python",
        )

        self.assertEqual(resolved, str(qwen_python))

    def test_falls_back_to_current_interpreter(self):
        resolved = start.resolve_backend_python(
            root_dir=self.root,
            environ={},
            current_python="fallback-python",
        )

        self.assertEqual(resolved, "fallback-python")

    def test_rejects_missing_override(self):
        with self.assertRaisesRegex(RuntimeError, "DITING_BACKEND_PYTHON"):
            start.resolve_backend_python(
                root_dir=self.root,
                environ={"DITING_BACKEND_PYTHON": str(self.root / "missing.exe")},
                current_python="fallback-python",
            )


if __name__ == "__main__":
    unittest.main()
