import os, sys, tempfile, time, unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from agent import tools  # noqa: E402


class ToolsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = os.getcwd()
        os.chdir(self.tmp.name)

    def tearDown(self):
        os.chdir(self.old)
        self.tmp.cleanup()

    def run_tool(self, name, **args):
        return tools.run(name, args)

    def test_schemas(self):
        s = tools.schemas(["bash", "read"])
        self.assertEqual([t["function"]["name"] for t in s], ["bash", "read"])
        self.assertTrue(all(t["type"] == "function" and "parameters" in t["function"] for t in tools.schemas(tools.ALL)))

    def test_read_offset_limit(self):
        Path("f.txt").write_text("".join(f"line{i}\n" for i in range(1, 11)))
        text, err = self.run_tool("read", path="f.txt", offset=3, limit=2)
        self.assertFalse(err)
        self.assertEqual(text.splitlines()[:2], ["3\tline3", "4\tline4"])
        self.assertIn("use offset=5", text)
        text, _ = self.run_tool("read", path="f.txt", offset=9)
        self.assertEqual(text, "9\tline9\n10\tline10")
        self.assertTrue(self.run_tool("read", path="f.txt", offset=50)[1])
        self.assertTrue(self.run_tool("read", path="missing.txt")[1])

    def test_read_truncation(self):
        Path("big.txt").write_text("x\n" * 3000)
        text, err = self.run_tool("read", path="big.txt")
        self.assertFalse(err)
        self.assertIn("showing lines 1-2000 of 3000; use offset=2001", text)
        Path("wide.txt").write_text(("y" * 1000 + "\n") * 200)
        text, _ = self.run_tool("read", path="wide.txt")
        self.assertLess(len(text), tools.MAX_BYTES + 200)
        self.assertIn("[truncated", text)

    def test_edit(self):
        Path("a.py").write_text("x = 1\ny = 1\n")
        text, err = self.run_tool("edit", path="a.py", old_text="z = 1", new_text="z = 2")
        self.assertTrue(err)
        self.assertIn("0 matches", text)
        text, err = self.run_tool("edit", path="a.py", old_text=" = 1", new_text=" = 2")
        self.assertTrue(err)
        self.assertIn("2 matches", text)
        self.assertEqual(Path("a.py").read_text(), "x = 1\ny = 1\n")
        self.assertFalse(self.run_tool("edit", path="a.py", old_text="y = 1", new_text="y = 3")[1])
        self.assertEqual(Path("a.py").read_text(), "x = 1\ny = 3\n")

    def test_write_creates_dirs(self):
        text, err = self.run_tool("write", path="deep/er/f.txt", content="hi")
        self.assertFalse(err)
        self.assertEqual(Path("deep/er/f.txt").read_text(), "hi")

    def test_grep_find_ls(self):
        Path("src/pkg").mkdir(parents=True)
        Path("src/pkg/m.py").write_text("def foo():\n    pass\n")
        Path("src/n.txt").write_text("foo bar\n")
        Path("node_modules").mkdir()
        Path("node_modules/x.py").write_text("foo\n")
        for rg in ("/usr/bin/rg", None):
            if rg and not tools.shutil.which("rg"):
                continue
            with mock.patch.object(tools.shutil, "which", (lambda _: None) if rg is None else tools.shutil.which):
                text, err = self.run_tool("grep", pattern="def fo+", path="src")
                self.assertFalse(err)
                self.assertEqual(text, "src/pkg/m.py:1:def foo():")
                text, _ = self.run_tool("grep", pattern="foo", glob="*.txt")
                self.assertEqual(text, "src/n.txt:1:foo bar")
                self.assertEqual(self.run_tool("grep", pattern="nomatch")[0], "No matches found")
        self.assertEqual(self.run_tool("find", pattern="*.py")[0], "src/pkg/m.py")
        self.assertEqual(self.run_tool("find", pattern="src/**/*.py")[0], "src/pkg/m.py")
        self.assertEqual(self.run_tool("ls", path="src")[0], "n.txt\npkg/")
        self.assertTrue(self.run_tool("ls", path="nope")[1])

    def test_bash(self):
        self.assertEqual(self.run_tool("bash", command="echo out; echo err >&2"), ("out\nerr", False))
        text, err = self.run_tool("bash", command="echo bad; exit 3")
        self.assertTrue(err)
        self.assertIn("[exit code 3]", text)
        self.assertEqual(self.run_tool("bash", command="pwd")[0], os.path.realpath(os.getcwd()))

    def test_bash_timeout_kills_group(self):
        t0 = time.time()
        text, err = self.run_tool("bash", command="echo started; (sleep 30; echo leaked > leak.txt) & sleep 30",
                                  timeout=1)
        self.assertLess(time.time() - t0, 10)
        self.assertTrue(err)
        self.assertIn("started", text)
        self.assertIn("timed out after 1s", text)

    def test_bash_output_keeps_tail(self):
        text, _ = self.run_tool("bash", command="seq 1 5000")
        self.assertTrue(text.startswith("[output truncated"))
        self.assertTrue(text.endswith("5000"))

    def test_bash_strips_agent_repo_from_pythonpath(self):
        with mock.patch.dict(os.environ, {"PYTHONPATH": tools.REPO + os.pathsep + "/elsewhere"}):
            self.assertEqual(self.run_tool("bash", command='echo "$PYTHONPATH"')[0], "/elsewhere")

    def test_errors(self):
        text, err = tools.run("bash", {"command": "echo hi"}, enabled=("read",))
        self.assertTrue(err)
        self.assertIn("not enabled", text)
        self.assertTrue(tools.run("nope", {})[1])
        self.assertIn("Missing required", tools.run("read", {})[0])
        self.assertTrue(tools.run("read", {"path": "x", "bogus": 1})[1])
        self.assertTrue(tools.run("read", ["x"])[1])


if __name__ == "__main__":
    unittest.main()
