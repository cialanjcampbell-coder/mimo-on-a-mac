import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "trial-record"
STUB_MACMON = '''#!/usr/bin/env python3
import json, sys, time
i = 0
while True:
    print(json.dumps({"sys_power": 20.0 + i, "all_power": 10.0, "gpu_power": 8.0,
                      "gpu_active_ratio": 0.9, "gpu_freq_mhz": 1500,
                      "temp": {"gpu_temp_avg": 50.0, "cpu_temp_avg": 45.0},
                      "fans": [{"rpm": 1000, "max_rpm": 5000}],
                      "memory": {"ram_usage": 9e10, "swap_usage": 0}}), flush=True)
    i += 1
    time.sleep(0.1)
'''
DEAD_MACMON = "#!/bin/sh\nexit 0\n"
MIMO = ("19:45:45 [req] extend reuse=2394 new=652 prefill 6.8s (95.2 tok/s) | decode 134 tok 11.36 tok/s"
        " | misses/tok 32.0 | tool_calls | ctx 3180\n")


class TestTrialRecord(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name)
        self.mimo_log = t / "mimo.log"
        self.mimo_log.write_text("before\n")
        self.results = t / "results"
        self.results.mkdir()
        self.env = dict(os.environ, TRIAL_STATE_DIR=str(t / "state"), TRIAL_MIMO_LOG=str(self.mimo_log),
                        TRIAL_MLXSERVE_LOG=str(t / "absent.log"), TRIAL_RESULTS_DIR=str(self.results))
        self.stub(STUB_MACMON)

    def stub(self, body):
        p = Path(self.tmp.name) / "macmon"
        p.write_text(body)
        p.chmod(0o755)
        self.env["TRIAL_MACMON_BIN"] = str(p)

    def tearDown(self):
        self.cli("stop")  # never leave a stub running
        self.tmp.cleanup()

    def cli(self, *args):
        return subprocess.run([sys.executable, str(SCRIPT), *args], env=self.env,
                              capture_output=True, text=True, timeout=30)

    def test_start_stop_save(self):
        self.assertEqual(self.cli("start", "demo run").returncode, 0)
        time.sleep(0.5)
        with open(self.mimo_log, "a") as f:
            f.write(MIMO)
        st = self.cli("status")
        self.assertIn("demo-run", st.stdout)
        r = self.cli("stop", "--save")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("**Requests:** 1", r.stdout)
        self.assertIn("System power", r.stdout)
        saved = list(self.results.glob("*-trial-demo-run.md"))
        self.assertEqual(len(saved), 1)

    def test_one_trial_at_a_time(self):
        self.assertEqual(self.cli("start").returncode, 0)
        self.assertNotEqual(self.cli("start").returncode, 0)

    def test_stop_without_start(self):
        r = self.cli("stop")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("no active trial", r.stderr)

    def test_macmon_died(self):
        self.stub(DEAD_MACMON)
        self.assertEqual(self.cli("start", "dead").returncode, 0)
        time.sleep(0.3)
        r = self.cli("stop")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("No macmon samples", r.stdout)
        self.assertIn("macmon exited before", r.stdout)


if __name__ == "__main__":
    unittest.main()
