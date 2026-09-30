import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from evalkit import runner, tasks  # noqa: E402
from test_transcript import SAMPLE_EVENTS  # noqa: E402

STUB_AGENT = '''#!/usr/bin/env python3
import json, os, shutil, sys, time
mode = os.environ.get("STUB_AGENT_MODE", "noop")
with open(os.environ["STUB_AGENT_ARGS"], "w") as f:
    json.dump(sys.argv[1:], f)
if mode == "wreck":
    shutil.rmtree(".git")
if mode == "tamper":
    import subprocess
    shutil.copytree(os.environ["STUB_AGENT_SOLUTION"], os.getcwd(), dirs_exist_ok=True)
    open(os.environ["STUB_AGENT_TAMPER"], "a").write("# tampered\\n")
    g = ["git", "-c", "user.name=x", "-c", "user.email=x@x"]
    subprocess.run(g + ["add", "-A"], check=True)
    subprocess.run(g + ["commit", "-qm", "t"], check=True)
    m = json.load(open(".eval-task.json"))
    m["baseline"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    json.dump(m, open(".eval-task.json", "w"))
if mode == "hang":
    time.sleep(120)
if mode == "solve":
    shutil.copytree(os.environ["STUB_AGENT_SOLUTION"], os.getcwd(), dirs_exist_ok=True)
if os.environ.get("STUB_AGENT_LOG"):
    with open(os.environ["STUB_AGENT_LOG"], "a") as f:
        f.write("20:00:00 [req] extend reuse=10 new=5 prefill 0.5s (10.0 tok/s) | decode 100 tok 12.5 tok/s"
                " | misses/tok 20.0 | stop | ctx 115\\n")
for e in json.loads(os.environ["STUB_AGENT_EVENTS"]):
    print(json.dumps(e), flush=True)
sys.exit(3 if mode == "crash" else 0)
'''


class TestRunner(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name)
        stub = t / "mimo-agent"
        stub.write_text(STUB_AGENT)
        stub.chmod(0o755)
        self.log = t / "mimo.log"
        self.log.write_text("")
        self.args_file = t / "args.json"
        self.saved_env = dict(os.environ)
        os.environ.update(EVAL_AGENT_BIN=str(stub), EVAL_SCRATCH=str(t / "scratch"),
                          EVAL_RESULTS=str(t / "runs.jsonl"), STUB_AGENT_ARGS=str(self.args_file),
                          STUB_AGENT_EVENTS=json.dumps(SAMPLE_EVENTS), STUB_AGENT_LOG=str(self.log))
        os.environ.pop("EVAL_TASKS_DIR", None)
        self.logs = {"mimo": self.log}

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.saved_env)
        self.tmp.cleanup()

    def test_build_command(self):
        task = tasks.load_task("bugfix-report")
        cmd = runner.build_command("/bin/mimo-agent", task)
        self.assertEqual(cmd, ["/bin/mimo-agent", "-p", task.prompt, "--json", "--no-start",
                               "--tools", "read,edit,write,grep,find,ls", "--thinking", "on"])
        cmd = runner.build_command("/bin/mimo-agent", task, thinking="off", allow_bash=True)
        self.assertEqual(cmd[cmd.index("--tools") + 1], "read,edit,write,grep,find,ls,bash")
        self.assertEqual(cmd[cmd.index("--thinking") + 1], "off")

    def test_agent_bin_default(self):
        os.environ.pop("EVAL_AGENT_BIN")
        self.assertEqual(runner.agent_bin(), str(tasks.REPO / "scripts" / "mimo-agent"))

    def test_solved_run_passes_and_records(self):
        os.environ.update(STUB_AGENT_MODE="solve",
                          STUB_AGENT_SOLUTION=str(tasks.load_task("bugfix-report").dir / "solution"))
        recs = runner.run(["bugfix-report"], repeats=1, check_server=False, logs=self.logs,
                          echo=lambda *_: None)
        r = recs[0]
        self.assertTrue(r["pass"])
        self.assertEqual((r["task"], r["model"], r["repeat"], r["timed_out"], r["agent_exit"]),
                         ("bugfix-report", "mimo", 1, False, 0))
        self.assertEqual((r["generated_tokens"], r["decode_tps"], r["requests"]), (100, 12.5, 1))
        self.assertEqual((r["turns"], r["tool_calls"], r["tool_errors"]), (3, 2, 1))
        args = json.loads(self.args_file.read_text())
        self.assertEqual(args[:2], ["-p", tasks.load_task("bugfix-report").prompt])
        self.assertIn("--no-start", args)
        wd = Path(r["workdir"])
        self.assertEqual(tasks.read_meta(wd)["model"], "mimo")
        self.assertTrue((wd / ".eval-transcript.jsonl").exists())
        self.assertIn("expenses/dates.py", r["diff_files"])
        self.assertIn("1 file changed", r["diff_stat"])
        lines = Path(os.environ["EVAL_RESULTS"]).read_text().splitlines()
        self.assertEqual(json.loads(lines[0])["task"], "bugfix-report")

    def test_noop_run_fails(self):
        os.environ["STUB_AGENT_MODE"] = "noop"
        r = runner.run_once("bugfix-report", 1, logs=self.logs)
        self.assertFalse(r["pass"])

    def test_crash_recorded(self):
        os.environ["STUB_AGENT_MODE"] = "crash"
        r = runner.run_once("bugfix-report", 1, logs=self.logs)
        self.assertEqual(r["agent_exit"], 3)
        self.assertFalse(r["pass"])

    def test_hang_times_out_and_still_checks(self):
        os.environ["STUB_AGENT_MODE"] = "hang"
        r = runner.run_once("bugfix-report", 1, timeout=2, logs=self.logs)
        self.assertTrue(r["timed_out"])
        self.assertFalse(r["pass"])
        self.assertGreater(r["checks_total"], 0)
        self.assertLess(r["wall_s"], 20)

    def test_wrecked_workdir_recorded_not_raised(self):
        os.environ["STUB_AGENT_MODE"] = "wreck"
        r = runner.run_once("bugfix-report", 1, logs=self.logs)
        self.assertFalse(r["pass"])
        self.assertIn("check_error", r)

    def test_baseline_tamper_ignored(self):
        os.environ.update(STUB_AGENT_MODE="tamper", STUB_AGENT_TAMPER="tests/test_importers.py",
                          STUB_AGENT_SOLUTION=str(tasks.load_task("refactor-dedupe").dir / "solution"))
        r = runner.run_once("refactor-dedupe", 1, logs=self.logs)
        self.assertEqual(r["protected_changed"], ["tests/test_importers.py"])
        self.assertFalse(r["pass"])

    def test_refuses_when_server_down(self):
        orig = runner.server_healthy
        runner.server_healthy = lambda url, timeout=3: False
        try:
            with self.assertRaises(RuntimeError) as cm:
                runner.run(["bugfix-report"], repeats=1, logs=self.logs)
            self.assertIn("mimo-server-ctl start", str(cm.exception))
        finally:
            runner.server_healthy = orig

    def test_summary_and_save(self):
        recs = [{"task": "t", "model": "mimo", "pass": p, "wall_s": w, "decode_tps": 12.0, "turns": 3}
                for p, w in ((True, 10.0), (False, 30.0), (True, 20.0))]
        md = runner.summarize_runs(recs)
        self.assertIn("| t | mimo | 3 | 2/3 (67%) | 20.0 | 12.0 | 3 |", md)
        jl, mdp = runner.save_results(recs, Path(self.tmp.name) / "res")
        self.assertTrue(jl.name.endswith("-evalrun-mimo.jsonl"))
        self.assertTrue(mdp.exists())
        jl2, _ = runner.save_results(recs, Path(self.tmp.name) / "res")
        self.assertNotEqual(jl, jl2)


if __name__ == "__main__":
    unittest.main()
