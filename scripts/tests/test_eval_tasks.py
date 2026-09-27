"""Every real task: hidden checks FAIL on the untouched repo and PASS with the reference solution."""
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evalkit import tasks  # noqa: E402


def apply_solution(task, workdir):
    shutil.copytree(task.dir / "solution", workdir, dirs_exist_ok=True)


class TestTaskPack(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = os.environ.get("EVAL_SCRATCH")
        os.environ["EVAL_SCRATCH"] = self.tmp.name
        os.environ.pop("EVAL_TASKS_DIR", None)

    def tearDown(self):
        if self.old is None:
            os.environ.pop("EVAL_SCRATCH", None)
        else:
            os.environ["EVAL_SCRATCH"] = self.old
        self.tmp.cleanup()

    def test_tasks_exist(self):
        self.assertGreater(len(tasks.list_tasks()), 0)

    def test_expected_task_set(self):
        self.assertEqual({t.name for t in tasks.list_tasks()},
                         {"bugfix-report", "feature-json-output", "refactor-dedupe",
                          "fix-failing-suite", "ops-log-report", "ops-organise-files"})

    def test_protected_files_are_announced_in_prompt(self):
        for task in tasks.list_tasks():
            if task.protected:
                with self.subTest(task=task.name):
                    self.assertRegex(task.prompt, r"(?i)(do not modify|without changes).*tests/|tests/.*(without changes|do not modify)")

    def test_fail_untouched_pass_with_solution(self):
        for task in tasks.list_tasks():
            with self.subTest(task=task.name):
                d = tasks.setup(task.name, "selftest")
                self.assertFalse((d / "solution").exists())
                self.assertFalse((d / "check").exists())
                before = tasks.check(d)
                self.assertFalse(before.passed, f"{task.name} passes untouched:\n{before.output}")
                self.assertGreater(before.tests_run, 0, before.output)
                apply_solution(task, d)
                after = tasks.check(d)
                self.assertTrue(after.passed, f"{task.name} solution fails:\n{after.output}")


if __name__ == "__main__":
    unittest.main()
