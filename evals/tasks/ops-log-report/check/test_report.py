import ast
import gzip
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

DATA = Path(__file__).parent / "data"
SMALL_A = [
    '1.1.1.1 - - [10/Oct/2026:13:55:36 +0000] "GET /a?x=1 HTTP/1.1" 200 10',
    '1.1.1.1 - - [10/Oct/2026:13:55:37 +0000] "GET /b HTTP/1.1" 404 -',
    'garbage line',
    '',
    '1.1.1.1 - - [10/Oct/2026:13:55:38 +0000] "POST /a HTTP/1.1" 302 0',
]
SMALL_B = [
    '2.2.2.2 - - [09/Oct/2026:10:00:00 +0000] "GET /c HTTP/1.1" 503 7',
    '2.2.2.2 - - [09/Oct/2026:10:00:01 +0000] "GET /b HTTP/1.1" 200 7',
    '2.2.2.2 - - [09/Oct/2026:10:00:02 +0000] "GET /x HTTP/1.1" 999 7',
]
SMALL_EXPECTED = """top paths:
  2 /a
  2 /b
  1 /c
status classes:
  2xx 2
  3xx 1
  4xx 1
  5xx 1
malformed: 2"""


def run(logdir):
    return subprocess.run([sys.executable, "report.py", str(logdir)], capture_output=True, text=True, timeout=60)


def norm(s):
    return "\n".join(line.rstrip() for line in s.strip().splitlines())


class TestReport(unittest.TestCase):
    def test_small_handmade(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            (d / "access.log").write_text("\n".join(SMALL_A) + "\n")
            (d / "access.log.1.gz").write_bytes(gzip.compress(("\n".join(SMALL_B) + "\n").encode()))
            (d / "notes.txt").write_text(SMALL_B[1] + "\n")  # not an access.log* file: ignored
            r = run(d)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(norm(r.stdout), SMALL_EXPECTED)

    def test_shipped_logs(self):
        r = run(DATA / "set1")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(norm(r.stdout), norm((DATA / "expected1.txt").read_text()))

    def test_hidden_logs(self):
        r = run(DATA / "set2")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(norm(r.stdout), norm((DATA / "expected2.txt").read_text()))

    def test_standard_library_only(self):
        tree = ast.parse(Path("report.py").read_text())
        mods = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        mods |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
        self.assertEqual(mods - set(sys.stdlib_module_names), set())
