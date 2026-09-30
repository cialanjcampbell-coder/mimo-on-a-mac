import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evalkit import transcript  # noqa: E402

USAGE = {"input": 900, "output": 42, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 942,
         "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0}}
SAMPLE_EVENTS = [
    {"type": "session", "version": 3, "id": "u", "timestamp": "2026-09-27T20:00:00Z", "cwd": "/w"},
    {"type": "agent_start"},
    {"type": "message_end", "message": {"role": "user", "content": "Fix add()", "timestamp": 1}},
    {"type": "message_update", "usage": USAGE, "assistantMessageEvent": {"type": "text_delta", "contentIndex": 0, "delta": "x"}},
    {"type": "message_end", "message": {"role": "assistant", "provider": "mimo", "model": "mimo-v2.6-flash",
     "content": [{"type": "thinking", "thinking": "Let me read mod.py first."},
                 {"type": "toolCall", "id": "c1", "name": "read", "arguments": {"path": "mod.py"}}],
     "usage": USAGE, "stopReason": "toolUse", "timestamp": 2}},
    {"type": "message_end", "message": {"role": "toolResult", "toolCallId": "c1", "toolName": "read",
     "content": [{"type": "text", "text": "def add(a, b):\n    return a - b\n"}], "isError": False, "timestamp": 3}},
    {"type": "message_end", "message": {"role": "assistant", "provider": "mimo", "model": "mimo-v2.6-flash",
     "content": [{"type": "toolCall", "id": "c2", "name": "edit", "arguments": {"path": "nope.py"}}],
     "usage": USAGE, "stopReason": "toolUse", "timestamp": 4}},
    {"type": "message_end", "message": {"role": "toolResult", "toolCallId": "c2", "toolName": "edit",
     "content": [{"type": "text", "text": "File not found"}], "isError": True, "timestamp": 5}},
    {"type": "message_end", "message": {"role": "assistant", "provider": "mimo", "model": "mimo-v2.6-flash",
     "content": [{"type": "text", "text": "Fixed: add() now adds."}], "usage": USAGE, "stopReason": "stop",
     "timestamp": 6}},
    {"type": "agent_end", "messages": [], "willRetry": False},
]


def write_events(path, events):
    Path(path).write_text("\n".join(json.dumps(e) for e in events) + "\n{broken\n")


class TestTranscript(unittest.TestCase):
    def test_load_skips_bad_lines(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "e.jsonl"
            write_events(p, SAMPLE_EVENTS)
            self.assertEqual(len(transcript.load_events(p)), len(SAMPLE_EVENTS))
        self.assertEqual(transcript.load_events("/nonexistent.jsonl"), [])

    def test_stats(self):
        s = transcript.stats(SAMPLE_EVENTS)
        self.assertEqual(s, {"turns": 3, "tool_calls": 2, "tool_errors": 1, "output_tokens": 126,
                             "stop_reason": "stop", "error": None})

    def test_render(self):
        md = transcript.render_markdown(SAMPLE_EVENTS)
        for text in ("## User", "Fix add()", "## Assistant (turn 1)", "Let me read mod.py first.",
                     "**tool call:** `read`", '"path": "mod.py"', "### Tool result: `read`",
                     "return a - b", "### Tool result: `edit` (error)", "Fixed: add() now adds."):
            self.assertIn(text, md)
        self.assertNotIn("mimo-v2.6-flash", md)  # provider/model metadata is not rendered

    def test_render_clips_long_tool_output(self):
        ev = [{"type": "message_end", "message": {"role": "toolResult", "toolName": "read", "isError": False,
               "content": [{"type": "text", "text": "x" * 5000}]}}]
        md = transcript.render_markdown(ev, max_tool_output=100)
        self.assertIn("[4900 more chars]", md)

    def test_empty(self):
        self.assertEqual(transcript.stats([])["turns"], 0)
        self.assertEqual(transcript.render_markdown([]), "")


if __name__ == "__main__":
    unittest.main()
