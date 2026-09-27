import ast
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

FILES = {"IMG_20260314_101500.jpg": "a", "scan 2025-11-02.pdf": "b", "notes-2026.01.07.txt": "c",
         "budget_2024_12_31.csv": "d", "x_2026_02_30.txt": "e", "20261301.log": "f",
         "report-v2.txt": "g", "123456789.bin": "h", "old 1989-05-05.txt": "i",
         "photo-20260314.jpg": "j", "a_20260301.jpg": "k"}
EXPECTED_DEST = {
    "2026/03/IMG_20260314_101500.jpg": "a", "2025/11/scan 2025-11-02.pdf": "b",
    "2026/01/notes-2026.01.07.txt": "c", "2024/12/budget_2024_12_31.csv": "d",
    "unsorted/x_2026_02_30.txt": "e", "unsorted/20261301.log": "f", "unsorted/report-v2.txt": "g",
    "unsorted/123456789.bin": "h", "unsorted/old 1989-05-05.txt": "i",
    "2026/03/photo-20260314.jpg": "j",
    "2026/03/a_20260301.jpg": "old", "2026/03/a_20260301-1.jpg": "old1", "2026/03/a_20260301-2.jpg": "k"}
LEFT_IN_SRC = {".hidden": "z", "sub/inner_20260101.txt": "y"}


def snapshot(root):
    return {str(p.relative_to(root)): p.read_text() for p in sorted(Path(root).rglob("*")) if p.is_file()}


class TestOrganise(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name)
        self.src, self.dest = t / "in", t / "out"
        for name, content in {**FILES, **LEFT_IN_SRC}.items():
            (self.src / name).parent.mkdir(parents=True, exist_ok=True)
            (self.src / name).write_text(content)
        (self.dest / "2026" / "03").mkdir(parents=True)
        (self.dest / "2026" / "03" / "a_20260301.jpg").write_text("old")
        (self.dest / "2026" / "03" / "a_20260301-1.jpg").write_text("old1")

    def tearDown(self):
        self.tmp.cleanup()

    def run_tool(self, *extra):
        return subprocess.run([sys.executable, "organise.py", *extra, str(self.src), str(self.dest)],
                              capture_output=True, text=True, timeout=60)

    def test_dry_run_changes_nothing(self):
        before = (snapshot(self.src), snapshot(self.dest))
        r = self.run_tool("--dry-run")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual((snapshot(self.src), snapshot(self.dest)), before)
        for name in FILES:
            self.assertIn(name, r.stdout)

    def test_real_run(self):
        r = self.run_tool()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(snapshot(self.dest), EXPECTED_DEST)
        self.assertEqual(snapshot(self.src), LEFT_IN_SRC)
        self.assertEqual(sum(1 for line in r.stdout.splitlines() if "->" in line), len(FILES))

    def test_second_run_is_noop(self):
        self.assertEqual(self.run_tool().returncode, 0)
        before = (snapshot(self.src), snapshot(self.dest))
        r = self.run_tool()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual((snapshot(self.src), snapshot(self.dest)), before)

    def test_standard_library_only(self):
        tree = ast.parse(Path("organise.py").read_text())
        mods = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        mods |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
        self.assertEqual(mods - set(sys.stdlib_module_names), set())
