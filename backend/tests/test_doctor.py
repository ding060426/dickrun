import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DOCTOR = ROOT / "tools" / "doctor.py"


class DoctorTests(unittest.TestCase):
    def test_json_report_contains_core_sections(self):
        completed = subprocess.run(
            [sys.executable, str(DOCTOR), "--json", "--backend-url", "http://127.0.0.1:9"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=10,
            check=True,
        )

        report = json.loads(completed.stdout)

        self.assertIn("python", report)
        self.assertIn("dependencies", report)
        self.assertIn("models", report)
        self.assertIn("ports", report)
        self.assertIn("apis", report)
        self.assertIn("xasr_profiles", report["models"])


if __name__ == "__main__":
    unittest.main()
