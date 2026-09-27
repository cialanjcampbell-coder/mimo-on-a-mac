import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

DATA = Path(__file__).parent / "data"
TEXT = ("5 items, 626 units, value £143.74\n"
        "  garage: 2 items, 3 units, £120.99\n"
        "  office: 1 items, 3 units, £6.75\n"
        "  shed: 2 items, 620 units, £16.00\n")
SAMPLE_JSON = {"items": 5, "total_qty": 626, "total_value": "143.74", "locations": [
    {"name": "garage", "items": 2, "qty": 3, "value": "120.99"},
    {"name": "office", "items": 1, "qty": 3, "value": "6.75"},
    {"name": "shed", "items": 2, "qty": 620, "value": "16.00"}]}
OTHER_JSON = {"items": 3, "total_qty": 7, "total_value": "25.39", "locations": [
    {"name": "bin-a", "items": 2, "qty": 6, "value": "24.40"},
    {"name": "bin-b", "items": 1, "qty": 1, "value": "0.99"}]}


def run(*args):
    return subprocess.run([sys.executable, "-m", "inventory", *args], capture_output=True, text=True,
                          timeout=30, env=dict(os.environ, PYTHONIOENCODING="utf-8"))


class TestJsonFormat(unittest.TestCase):
    def test_default_text_unchanged(self):
        r = run(str(DATA / "sample.csv"))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout, TEXT)

    def test_explicit_text(self):
        self.assertEqual(run("--format", "text", str(DATA / "sample.csv")).stdout, TEXT)

    def test_json(self):
        r = run("--format", "json", str(DATA / "sample.csv"))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(json.loads(r.stdout), SAMPLE_JSON)

    def test_json_flag_after_path(self):
        r = run(str(DATA / "other.csv"), "--format", "json")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(json.loads(r.stdout), OTHER_JSON)

    def test_json_empty(self):
        r = run("--format", "json", str(DATA / "empty.csv"))
        self.assertEqual(json.loads(r.stdout),
                         {"items": 0, "total_qty": 0, "total_value": "0.00", "locations": []})

    def test_bad_format_rejected(self):
        self.assertNotEqual(run("--format", "xml", str(DATA / "sample.csv")).returncode, 0)

    def test_readme_documents_flag(self):
        readme = Path("README.md").read_text()
        self.assertIn("--format", readme)
        self.assertIn("json", readme.lower())
