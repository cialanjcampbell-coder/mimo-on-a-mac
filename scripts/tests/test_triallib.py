import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import triallib as tl  # noqa: E402

MIMO = ("19:45:45 [req] extend reuse=2394 new=652 prefill 6.8s (95.2 tok/s) | decode 134 tok 11.36 tok/s"
        " | misses/tok 32.0 | tool_calls | ctx 3180")
MLX_STREAM = ("  <- 16789+226 tokens streamed [prefill: 1791.6 tok/s (11792 cached / 16789 total),"
              " decode: 71.4 tok/s] [stop]")
MLX_PLAIN = "  <- 52+262 tokens (3042ms) [prefill: 174.5 tok/s, decode: 97.3 tok/s] [stop]"


class TestParseLine(unittest.TestCase):
    def test_mimo(self):
        r = tl.parse_line(MIMO)
        self.assertEqual(r["server"], "mimo")
        self.assertEqual((r["mode"], r["reused"], r["new"], r["gen"]), ("extend", 2394, 652, 134))
        self.assertEqual((r["prefill_s"], r["decode_tps"], r["misses"]), (6.8, 11.36, 32.0))
        self.assertEqual((r["finish"], r["ctx"]), ("tool_calls", 3180))

    def test_mlxserve_cached(self):
        r = tl.parse_line(MLX_STREAM)
        self.assertEqual(r["server"], "mlxserve")
        self.assertEqual((r["reused"], r["new"], r["gen"]), (11792, 4997, 226))
        self.assertAlmostEqual(r["prefill_s"], 4997 / 1791.6, places=2)
        self.assertIsNone(r["misses"])
        self.assertEqual(r["finish"], "stop")

    def test_mlxserve_uncached_plain(self):
        r = tl.parse_line(MLX_PLAIN)
        self.assertEqual((r["reused"], r["new"], r["gen"], r["decode_tps"]), (0, 52, 262, 97.3))

    def test_other_lines(self):
        self.assertIsNone(tl.parse_line("  prompt=16789 tokens, max_gen=8192, ctx=262144"))
        self.assertIsNone(tl.parse_line("12:00:00 loading model"))


class TestParseLog(unittest.TestCase):
    def test_counts_skipped_request_lines(self):
        text = "\n".join([MIMO, "12:00:01 [req] error: RuntimeError('x')", "noise", MLX_PLAIN])
        reqs, skipped = tl.parse_log_text(text)
        self.assertEqual(len(reqs), 2)
        self.assertEqual(skipped, 1)

    def test_disconnect_not_counted(self):
        self.assertEqual(tl.parse_log_text("12:00 [req] client disconnected"), ([], 0))


class TestReadSince(unittest.TestCase):
    def test_appended_only(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "s.log"
            p.write_text("old line\n")
            off = tl.log_size(p)
            with open(p, "a") as f:
                f.write(MIMO + "\n")
            self.assertEqual(tl.read_since(p, off), MIMO + "\n")

    def test_truncated_log_read_from_start(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "s.log"
            p.write_text("x" * 1000)
            off = tl.log_size(p)
            p.write_text(MIMO + "\n")  # rotated / truncated
            self.assertEqual(tl.read_since(p, off), MIMO + "\n")

    def test_missing_file(self):
        self.assertEqual(tl.log_size("/nonexistent/x.log"), 0)
        self.assertEqual(tl.read_since("/nonexistent/x.log", 0), "")


class TestSummarizeRequests(unittest.TestCase):
    def test_token_weighted_decode(self):
        reqs = [dict(tl.parse_line(MIMO)), dict(tl.parse_line(MLX_PLAIN))]
        s = tl.summarize_requests(reqs)
        self.assertEqual(s["requests"], 2)
        self.assertEqual(s["generated_tokens"], 396)
        self.assertAlmostEqual(s["decode_tps"], 396 / (134 / 11.36 + 262 / 97.3), places=1)
        self.assertEqual(s["servers"], ["mimo", "mlxserve"])
        self.assertEqual(s["prefill_s_max"], 6.8)

    def test_empty(self):
        s = tl.summarize_requests([])
        self.assertEqual((s["requests"], s["generated_tokens"], s["decode_tps"]), (0, 0, None))


def sample(i, busy=True, freq=1500):
    return {"sys_power": 20.0 + i, "all_power": 10.0, "gpu_power": 8.0,
            "gpu_active_ratio": 0.9 if busy else 0.1, "gpu_freq_mhz": freq,
            "temp": {"gpu_temp_avg": 50.0 + i, "cpu_temp_avg": 45.0},
            "fans": [{"rpm": 1000 * i, "max_rpm": 5000}],
            "memory": {"ram_usage": 90e9 + i * 1e9, "swap_usage": 1e8 * i}}


class TestSamples(unittest.TestCase):
    def test_summary(self):
        s = tl.summarize_samples([sample(0), sample(1), sample(2, freq=1200), sample(3, busy=False)], 1.0)
        self.assertEqual(s["samples"], 4)
        self.assertEqual(s["sys_power_w"], {"mean": 21.5, "peak": 23.0})
        self.assertEqual(s["energy_j"], 86.0)
        self.assertEqual(s["gpu_temp_c"]["peak"], 53.0)
        self.assertEqual(s["fan_peak_pct"], 60.0)
        self.assertEqual(s["gpu"]["busy_samples"], 3)
        self.assertEqual(s["gpu"]["peak_freq_mhz"], 1500)
        self.assertAlmostEqual(s["gpu"]["throttled_share"], 1 / 3, places=2)
        self.assertEqual(s["ram_peak_gb"], 93.0)
        self.assertEqual(s["swap_delta_gb"], 0.3)

    def test_glitch_samples_dropped(self):
        glitch = sample(1)
        glitch["sys_power"] = glitch["all_power"] = 1905.4
        s = tl.summarize_samples([sample(0), glitch, sample(2)], 1.0)
        self.assertEqual(s["glitches"], 1)
        self.assertEqual(s["samples"], 2)
        self.assertEqual(s["sys_power_w"]["peak"], 22.0)
        self.assertEqual(s["energy_j"], 63.0)  # 20 + 22 over kept samples, scaled to 3 s of wall time

    def test_no_samples(self):
        self.assertEqual(tl.summarize_samples([]), {"samples": 0})

    def test_load_skips_bad_lines(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "m.jsonl"
            p.write_text('{"sys_power": 1}\nnot json\n{"sys_power": 2}\n{"sys_po')
            self.assertEqual(len(tl.load_samples(p)), 2)


class TestReport(unittest.TestCase):
    META = {"label": "demo", "start": "2026-09-27T20:00:00", "interval_s": 1.0}

    def test_report_contents(self):
        reqs = [tl.parse_line(MIMO)]
        s = tl.build_summary(self.META, "2026-09-27T20:01:40", reqs, 0, [sample(0), sample(1)], False)
        self.assertEqual(s["duration_s"], 100.0)
        self.assertAlmostEqual(s["j_per_token"], 41.0 / 134, places=3)
        md = tl.render_report(s)
        for text in ("# Trial: demo", "**Requests:** 1", "extend", "11.36", "System power",
                     "per generated token", "GPU temp"):
            self.assertIn(text, md)

    def test_report_without_samples_or_requests(self):
        s = tl.build_summary(self.META, "2026-09-27T20:00:05", [], 2, [], True)
        md = tl.render_report(s)
        self.assertIn("No macmon samples", md)
        self.assertIn("macmon exited before", md)
        self.assertIn("**Requests:** 0 (2 other", md)


if __name__ == "__main__":
    unittest.main()
