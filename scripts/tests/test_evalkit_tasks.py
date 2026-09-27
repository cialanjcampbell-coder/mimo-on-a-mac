import json
import os
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evalkit import tasks  # noqa: E402


def make_demo_task(root):
    d = root / "demo"
    (d / "repo" / "tests").mkdir(parents=True)
    (d / "check").mkdir()
    (d / "task.json").write_text(json.dumps({
        "title": "Fix add", "category": "bug fix", "prompt": "add() subtracts; fix it.",
        "tools": "read,edit,write", "protected": ["tests/test_visible.py"]}))
    (d / "repo" / "mod.py").write_text("def add(a, b):\n    return a - b\n")
    (d / "repo" / "tests" / "__init__.py").write_text("")
    (d / "repo" / "tests" / "test_visible.py").write_text("# visible\n")
    (d / "check" / "__init__.py").write_text("")
    (d / "check" / "test_hidden.py").write_text(
        "import unittest\nfrom mod import add\n\n"
        "class T(unittest.TestCase):\n"
        "    def test_add(self):\n        self.assertEqual(add(2, 3), 5)\n"
        "    def test_zero(self):\n        self.assertEqual(add(0, 0), 0)\n")
    return d


class TestTasks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name)
        make_demo_task(t / "tasks")
        self.old = {k: os.environ.get(k) for k in ("EVAL_TASKS_DIR", "EVAL_SCRATCH")}
        os.environ["EVAL_TASKS_DIR"] = str(t / "tasks")
        os.environ["EVAL_SCRATCH"] = str(t / "scratch")

    def tearDown(self):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self.tmp.cleanup()

    def test_list_and_load(self):
        self.assertEqual([t.name for t in tasks.list_tasks()], ["demo"])
        self.assertEqual(tasks.load_task("demo").protected, ["tests/test_visible.py"])
        with self.assertRaises(KeyError):
            tasks.load_task("nope")

    def test_setup_makes_git_baseline_without_hidden_files(self):
        d = tasks.setup("demo", "t1")
        meta = tasks.read_meta(d)
        self.assertEqual((meta["task"], meta["tag"]), ("demo", "t1"))
        self.assertEqual(len(meta["baseline"]), 40)
        self.assertFalse((d / "check").exists())
        self.assertFalse((d / "task.json").exists())
        self.assertIn(".eval-*", (d / ".git" / "info" / "exclude").read_text())

    def test_setup_same_second_gets_unique_dirs(self):
        now = datetime(2026, 9, 27, 20, 0, 0)
        self.assertNotEqual(tasks.setup("demo", "x", now=now), tasks.setup("demo", "x", now=now))

    def test_check_fails_untouched_then_passes_fixed(self):
        d = tasks.setup("demo")
        r = tasks.check(d)
        self.assertFalse(r.passed)
        self.assertEqual((r.tests_run, r.tests_failed), (2, 1))
        self.assertTrue(r.summary().startswith("FAIL (1/2 checks)"))
        (d / "mod.py").write_text("def add(a, b):\n    return a + b\n")
        r = tasks.check(d)
        self.assertTrue(r.passed, r.output)
        self.assertEqual(r.summary(), "PASS (2/2 checks)")
        self.assertIn("mod.py", r.diff_stat)
        self.assertTrue((d / ".eval-check.txt").exists())

    def test_protected_file_edit_fails(self):
        d = tasks.setup("demo")
        (d / "mod.py").write_text("def add(a, b):\n    return a + b\n")
        (d / "tests" / "test_visible.py").write_text("# changed\n")
        r = tasks.check(d)
        self.assertFalse(r.passed)
        self.assertEqual(r.protected_changed, ["tests/test_visible.py"])

    def test_deleted_protected_file_fails(self):
        d = tasks.setup("demo")
        (d / "mod.py").write_text("def add(a, b):\n    return a + b\n")
        (d / "tests" / "test_visible.py").unlink()
        self.assertEqual(tasks.check(d).protected_changed, ["tests/test_visible.py"])

    def test_import_time_hang_times_out(self):
        d = tasks.setup("demo")
        (d / "mod.py").write_text("while True:\n    pass\n")
        r = tasks.check(d, timeout=3)
        self.assertTrue(r.timed_out)
        self.assertFalse(r.passed)

    def test_syntax_error_fails_cleanly(self):
        d = tasks.setup("demo")
        (d / "mod.py").write_text("def add(:\n")
        r = tasks.check(d)
        self.assertFalse(r.passed)
        self.assertIn("SyntaxError", r.output)

    def test_diff_includes_new_files_without_touching_index(self):
        d = tasks.setup("demo")
        (d / "new.py").write_text("x = 1\n")
        self.assertIn("new.py", tasks.diff(d, tasks.read_meta(d)["baseline"]))
        self.assertIn("+x = 1", tasks.diff(d, tasks.read_meta(d)["baseline"], stat=False))
        self.assertEqual(tasks.git(d, "status", "--porcelain").strip(), "?? new.py")

    def test_stdlib_shadowing_rejected(self):
        d = tasks.setup("demo")
        (d / "unittest").mkdir()
        (d / "unittest" / "__init__.py").write_text("")
        (d / "unittest" / "__main__.py").write_text("print('Ran 6 tests')\nprint('OK')\n")
        r = tasks.check(d)
        self.assertFalse(r.passed)
        self.assertIn("shadows a standard-library module: unittest", r.output)

    def test_eval_check_dir_in_workdir_does_not_crash(self):
        d = tasks.setup("demo")
        (d / "_eval_check").mkdir()
        (d / "_eval_check" / "test_fake.py").write_text("import unittest\nclass T(unittest.TestCase):\n    def test_ok(self): pass\n")
        r = tasks.check(d)
        self.assertFalse(r.passed)
        self.assertEqual(r.tests_run, 2)

    def test_timeout_kills_spawned_processes(self):
        d = tasks.setup("demo")
        pidfile = Path(self.tmp.name) / "child.pid"
        (d / "mod.py").write_text(
            "import subprocess, sys\n"
            "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
            f"open({str(pidfile)!r}, 'w').write(str(p.pid))\n"
            "while True:\n    pass\n")
        r = tasks.check(d, timeout=3)
        self.assertTrue(r.timed_out)
        pid = int(pidfile.read_text())
        import time
        time.sleep(0.5)
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)

    def test_check_uses_given_meta_not_workdir_file(self):
        d = tasks.setup("demo")
        meta = tasks.read_meta(d)
        (d / "mod.py").write_text("def add(a, b):\n    return a + b\n")
        (d / "tests" / "test_visible.py").write_text("# tampered\n")
        tasks.git(d, "add", "-A")
        tasks.git(d, "commit", "-q", "-m", "tamper")
        tasks.write_meta(d, dict(meta, baseline=tasks.git(d, "rev-parse", "HEAD").strip()))
        r = tasks.check(d, meta=meta)
        self.assertEqual(r.protected_changed, ["tests/test_visible.py"])
        self.assertFalse(r.passed)


if __name__ == "__main__":
    unittest.main()
