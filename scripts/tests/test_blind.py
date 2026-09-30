import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from evalkit import blind, tasks  # noqa: E402
from test_transcript import SAMPLE_EVENTS  # noqa: E402

LEAKY = [{"type": "message_end", "message": {"role": "assistant", "provider": "mimo", "model": "mimo-v2.6-flash",
          "content": [{"type": "text", "text": "As MiMo (Xiaomi) I edited it. Qwen or Flash-Next would too. mlx-serve!"}],
          "usage": {"output": 5}, "stopReason": "stop"}}]


def fake_run(task, model, repeat, events):
    d = tasks.setup(task, tag=f"{model}-r{repeat}")
    m = tasks.read_meta(d)
    m.update(model=model, repeat=repeat)
    tasks.write_meta(d, m)
    (d / ".eval-transcript.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
    (d / "notes_mimo.txt").write_text(f"workdir is {d}\n")  # the diff will contain the path and a model name
    tasks.check(d)
    return d


class TestBlind(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.saved = dict(os.environ)
        os.environ["EVAL_SCRATCH"] = str(Path(self.tmp.name) / "scratch")
        os.environ.pop("EVAL_TASKS_DIR", None)
        self.runs = [fake_run("bugfix-report", "mimo", 1, SAMPLE_EVENTS + LEAKY),
                     fake_run("bugfix-report", "mimo", 2, SAMPLE_EVENTS),
                     fake_run("bugfix-report", "flashnext", 1, SAMPLE_EVENTS)]
        self.out = Path(self.tmp.name) / "review"

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.saved)
        self.tmp.cleanup()

    def test_anonymise(self):
        s = blind.anonymise("MiMo, mimo-v2.6-flash, XIAOMI, Qwen3.8, Flash-Next, flash next, mlx-serve at /x/y", ["/x/y"])
        self.assertNotRegex(s.lower(), "mimo|xiaomi|qwen|flash.?next|mlx.?serve|/x/y")
        self.assertIn("<workdir>", s)

    def test_make_blind_layout_and_no_leaks(self):
        key = blind.make_blind(self.runs, self.out, seed=1)
        self.assertEqual(sorted(key["labels"]["bugfix-report"].values()), ["flashnext", "mimo"])
        self.assertTrue((self.out / "bugfix-report" / "PROMPT.md").exists())
        run_dirs = sorted(p.name for p in (self.out / "bugfix-report").iterdir() if p.is_dir())
        self.assertEqual(len(run_dirs), 3)
        for name in run_dirs:
            self.assertRegex(name, r"^[AB]-r[12]$")
        self.assertFalse(any(p.name.endswith("key.json") for p in self.out.rglob("*")))
        self.assertTrue(blind.key_path(self.out).exists())
        for p in self.out.rglob("*"):
            if p.is_file():
                text = p.read_text().lower()
                self.assertIsNone(re.search(r"mimo|xiaomi|qwen|flash.?next|mlx.?serve|scratch/", text),
                                  f"leak in {p}")

    def test_refuses_non_empty_out(self):
        self.out.mkdir()
        (self.out / "x").write_text("")
        with self.assertRaises(FileExistsError):
            blind.make_blind(self.runs, self.out)

    def test_unblind(self):
        key = blind.make_blind(self.runs, self.out, seed=1)
        scores = {"runs": {}}
        for rid, info in key["runs"].items():
            v = 3 if info["model"] == "mimo" else 1
            scores["runs"][rid] = {k: v for k in blind.RUBRIC} | {"notes": f"n {rid}"}
        (self.out / "scores.json").write_text(json.dumps(scores))
        md = blind.unblind(self.out)
        self.assertIn("| mimo | 2 |", md)
        self.assertIn("| flashnext | 1 |", md)
        self.assertTrue((self.out / "unblinded.md").exists())

    def test_unblind_unknown_run(self):
        blind.make_blind(self.runs, self.out, seed=1)
        (self.out / "scores.json").write_text(json.dumps({"runs": {"bugfix-report/Z-r9": {}}}))
        with self.assertRaises(KeyError):
            blind.unblind(self.out)


if __name__ == "__main__":
    unittest.main()
