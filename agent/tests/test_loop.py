import json, os, signal, subprocess, sys, tempfile, time, unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
from agent.loop import Agent  # noqa: E402
from agent.prompt import SYSTEM_PROMPT  # noqa: E402
from fake_server import FakeServer, reply  # noqa: E402

# Deliberately odd formatting: must come back byte-for-byte, not re-serialized.
RAW_ARGS = '{"path":  "out/hello.txt",\n "content": "hi\\u00e9"}'


class WorkDir(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = os.getcwd()
        os.chdir(self.tmp.name)
        self.servers = []

    def tearDown(self):
        os.chdir(self.old)
        self.tmp.cleanup()
        for s in self.servers:
            s.close()

    def server(self, *script):
        s = FakeServer(script)
        self.servers.append(s)
        return s


class LoopTest(WorkDir):
    def agent(self, srv, **kw):
        self.events = []
        return Agent(srv.base_url, "m", emit=self.events.append, system="SYS", **kw)

    def test_tool_loop_round_trips_ids_and_raw_args(self):
        srv = self.server(reply("Writing.", reasoning="need a file", tool_calls=[("call_abc", "write", RAW_ARGS)]),
                          reply("Done."))
        msg = self.agent(srv).run("make hello")
        self.assertEqual(msg["stopReason"], "stop")
        self.assertEqual(Path("out/hello.txt").read_text(), "hié")
        self.assertEqual(len(srv.requests), 2)
        r1, r2 = srv.requests
        self.assertTrue(r1["stream"] and r1["stream_options"]["include_usage"])
        self.assertTrue(r1["enable_thinking"] and r1["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual([t["function"]["name"] for t in r1["tools"]], ["read", "edit", "write", "grep", "find", "ls", "bash"])
        self.assertEqual(r1["messages"], [{"role": "system", "content": "SYS"}, {"role": "user", "content": "make hello"}])
        asst, tool = r2["messages"][2], r2["messages"][3]
        self.assertEqual(asst, {"role": "assistant", "content": "Writing.", "reasoning_content": "need a file",
                                "tool_calls": [{"id": "call_abc", "type": "function",
                                                "function": {"name": "write", "arguments": RAW_ARGS}}]})
        self.assertEqual(tool["role"], "tool")
        self.assertEqual(tool["tool_call_id"], "call_abc")
        self.assertIn("Wrote", tool["content"])
        self.assertEqual(r2["messages"][:2], r1["messages"])
        self.assertEqual([e["role"] for e in self.events], ["user", "assistant", "toolResult", "assistant"])
        self.assertEqual(self.events[1]["stopReason"], "toolUse")
        self.assertEqual(self.events[1]["content"][2],
                         {"type": "toolCall", "id": "call_abc", "name": "write",
                          "arguments": {"path": "out/hello.txt", "content": "hié"}})
        self.assertEqual(self.events[1]["usage"], {"input": 100, "output": 7})

    def test_thinking_off_disabled_tool_bad_json(self):
        srv = self.server(reply(tool_calls=[("c1", "bash", '{"command": "echo x"}'), ("c2", "read", '{"path": ')]),
                          reply("ok"))
        a = self.agent(srv, tool_names=("read",), thinking=False)
        self.assertEqual(a.run("go")["stopReason"], "stop")
        r1 = srv.requests[0]
        self.assertFalse(r1["enable_thinking"] or r1["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual([t["function"]["name"] for t in r1["tools"]], ["read"])
        results = [e for e in self.events if e["role"] == "toolResult"]
        self.assertTrue(all(r["isError"] for r in results))
        self.assertIn("not enabled", results[0]["content"][0]["text"])
        self.assertIn("Invalid JSON", results[1]["content"][0]["text"])
        self.assertEqual(self.events[1]["content"][1]["arguments"], {"_raw": '{"path": '})

    def test_max_turns_and_length(self):
        srv = self.server(*[reply(tool_calls=[(f"c{i}", "ls", "{}")]) for i in range(3)])
        msg = self.agent(srv, max_turns=2).run("loop")
        self.assertEqual(msg["stopReason"], "error")
        self.assertIn("max-turns", msg["errorMessage"])
        srv = self.server(reply("trunc", finish="length"))
        self.assertEqual(self.agent(srv).run("x")["stopReason"], "length")

    def test_server_error_content(self):
        srv = self.server(reply("partial\n[server error: boom]"))
        msg = self.agent(srv).run("x")
        self.assertEqual((msg["stopReason"], msg["errorMessage"]), ("error", "[server error: boom]"))

    def test_system_prompt_short(self):
        self.assertLess(len(SYSTEM_PROMPT.split()), 300)


class CliTest(WorkDir):
    def cli(self, *args, launcher=None):
        env = dict(os.environ, PYTHONPATH=str(REPO))
        cmd = [launcher] if launcher else [sys.executable, "-m", "agent"]
        return subprocess.run(cmd + list(args), capture_output=True, text=True, env=env, timeout=60)

    def test_headless_json_events(self):
        srv = self.server(reply("Let me look.", reasoning="hmm", tool_calls=[("call_1", "ls", '{"path": "."}')]),
                          reply("There is one file."))
        Path("x.txt").write_text("x")
        p = self.cli("-p", "what is here?", "--json", "--no-start", "--base-url", srv.base_url)
        self.assertEqual(p.returncode, 0, p.stderr)
        events = [json.loads(ln) for ln in p.stdout.splitlines()]
        self.assertTrue(all(e["type"] == "message_end" for e in events))
        msgs = [e["message"] for e in events]
        self.assertEqual(msgs[0], {"role": "user", "content": [{"type": "text", "text": "what is here?"}]})
        self.assertEqual(msgs[1], {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "hmm"}, {"type": "text", "text": "Let me look."},
            {"type": "toolCall", "id": "call_1", "name": "ls", "arguments": {"path": "."}}],
            "usage": {"input": 100, "output": 7}, "stopReason": "toolUse"})
        self.assertEqual(msgs[2], {"role": "toolResult", "toolCallId": "call_1", "toolName": "ls",
                                   "content": [{"type": "text", "text": "x.txt"}], "isError": False})
        self.assertEqual(msgs[3]["stopReason"], "stop")
        self.assertEqual(msgs[3]["content"], [{"type": "text", "text": "There is one file."}])
        # the eval harness's parser reads these events
        sys.path.insert(0, str(REPO / "scripts"))
        from evalkit import transcript
        Path("t.jsonl").write_text(p.stdout)
        st = transcript.stats(transcript.load_events("t.jsonl"))
        self.assertEqual((st["turns"], st["tool_calls"], st["stop_reason"]), (2, 1, "stop"))

    def test_headless_text_and_tools_flag(self):
        srv = self.server(reply(tool_calls=[("c", "read", '{"path": "a"}')]), reply("Final answer."))
        Path("a").write_text("A")
        p = self.cli("-p", "hi", "--no-start", "--base-url", srv.base_url, "--tools", "read,ls", "--thinking", "off")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(p.stdout, "Final answer.\n")
        self.assertIn("> read a", p.stderr)
        self.assertEqual([t["function"]["name"] for t in srv.requests[0]["tools"]], ["read", "ls"])
        self.assertFalse(srv.requests[0]["enable_thinking"])

    def test_unreachable_server_fails(self):
        p = self.cli("-p", "hi", "--json", "--no-start", "--base-url", "http://127.0.0.1:9/v1")
        self.assertNotEqual(p.returncode, 0)
        last = json.loads(p.stdout.splitlines()[-1])["message"]
        self.assertEqual(last["stopReason"], "error")
        self.assertIn("cannot reach", last["errorMessage"])

    def test_interactive_ctrl_c_aborts_turn(self):
        srv = FakeServer([reply("word " * 60), reply("second answer")], delay=0.05)
        self.servers.append(srv)
        p = subprocess.Popen([sys.executable, "-m", "agent", "--no-start", "--base-url", srv.base_url],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             env=dict(os.environ, PYTHONPATH=str(REPO)))
        p.stdin.write("first\n")
        p.stdin.flush()
        time.sleep(1)
        p.send_signal(signal.SIGINT)
        time.sleep(0.3)
        out = p.communicate("/reset\nagain\n", timeout=30)[0]
        self.assertEqual(p.returncode, 0, out)
        self.assertIn("[aborted]", out)
        self.assertIn("second answer", out)
        self.assertEqual(srv.requests[1]["messages"][1:], [{"role": "user", "content": "again"}])

    def test_interactive_commands_and_turn_stats(self):
        srv = self.server(reply("an answer"))
        p = subprocess.run([sys.executable, "-m", "agent", "--no-start", "--base-url", srv.base_url],
                           input="/help\n/rest\nhi\n/exit\n", capture_output=True, text=True, timeout=30,
                           env=dict(os.environ, PYTHONPATH=str(REPO)))
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("/reset   clear the conversation", p.stdout)
        self.assertIn("unknown command /rest", p.stdout)
        self.assertIn("an answer", p.stdout)
        self.assertIn("· 7 tokens out]", p.stdout)
        self.assertEqual(len(srv.requests), 1)  # only "hi" reached the model

    def test_server_start_failure_exits_fast(self):
        # default base URL with nothing on it: mimo-server-ctl start must fail on the bad model dir, not hang
        from agent.__main__ import DEFAULT_BASE, healthy
        if healthy(DEFAULT_BASE):
            self.skipTest("a server is running on the default port")
        t0 = time.time()
        p = subprocess.run([sys.executable, "-m", "agent", "-p", "hi"], capture_output=True, text=True, timeout=60,
                           env=dict(os.environ, PYTHONPATH=str(REPO), MIMO_MODEL_DIR="/nonexistent/model"))
        self.assertEqual(p.returncode, 1, p.stderr)
        self.assertLess(time.time() - t0, 20)
        self.assertIn("has no config.json", p.stderr)
        self.assertIn("failed to start", p.stderr)

    def test_launcher_via_symlink(self):
        srv = self.server(reply("hello from launcher"))
        link = Path(self.tmp.name) / "bin" / "mimo-agent"
        link.parent.mkdir()
        link.symlink_to(REPO / "scripts" / "mimo-agent")
        p = self.cli("-p", "hi", "--no-start", "--base-url", srv.base_url, launcher=str(link))
        self.assertEqual((p.returncode, p.stdout), (0, "hello from launcher\n"), p.stderr)
        self.assertIn(os.path.realpath(self.tmp.name), srv.requests[0]["messages"][0]["content"])


if __name__ == "__main__":
    unittest.main()
