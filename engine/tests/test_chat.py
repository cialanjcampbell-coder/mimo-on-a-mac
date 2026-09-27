"""engine.mimo_stream.chat vs MiMo's own chat_template.jinja, plus parsing / streaming split.

  python3 -m unittest discover -s engine/tests -v
"""
import json, os, sys, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from engine.mimo_stream.chat import render, split_generation, coerce_args, StreamSplitter

MODEL = Path(os.environ["MIMO_MODEL_DIR"]) if os.environ.get("MIMO_MODEL_DIR") else None
TOOLS = [{"type": "function", "function": {"name": "read", "description": "Read a file",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "offset": {"type": "integer"}},
                         "required": ["path"]}}},
         {"type": "function", "function": {"name": "bash", "description": "Run a command",
          "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}}]
CONV = [
    {"role": "system", "content": "You are a coding agent. Unicode: é ✓"},
    {"role": "user", "content": [{"type": "text", "text": "Read setup.py"}]},
    {"role": "assistant", "reasoning_content": "I should read it.\n", "content": "",
     "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read", "arguments": {"path": "setup.py", "offset": 10}}}]},
    {"role": "tool", "tool_call_id": "c1", "content": "from setuptools import setup\n"},
    {"role": "assistant", "reasoning_content": "", "content": "It uses setuptools."},
    {"role": "user", "content": "thanks"},
]


class TestChat(unittest.TestCase):
    def test_render_matches_jinja(self):
        if MODEL is None or not MODEL.exists():
            raise unittest.SkipTest("set MIMO_MODEL_DIR to the converted model to run this test")
        try:
            from mlx_lm.tokenizer_utils import load as load_tokenizer
        except ImportError:
            raise unittest.SkipTest("mlx_lm not available")
        tok = load_tokenizer(MODEL)
        for thinking in (True, False):
            ref = tok.apply_chat_template(CONV, tools=TOOLS, add_generation_prompt=True, tokenize=False, enable_thinking=thinking)
            got = render(CONV, TOOLS, add_generation_prompt=True, enable_thinking=thinking)
            diff = next((i for i, (a, b) in enumerate(zip(got, ref)) if a != b), None)
            self.assertEqual(got, ref, f"thinking={thinking}, first diff at index {diff}")

    def test_json_string_arguments_render_native(self):
        conv = [dict(CONV[2])]
        conv[0]["tool_calls"] = [{"id": "c1", "type": "function", "function": {"name": "read", "arguments": json.dumps({"path": "setup.py", "offset": 10})}}]
        self.assertIn("<parameter=path>setup.py</parameter><parameter=offset>10</parameter>", render(conv, add_generation_prompt=False))

    def test_split_and_coerce(self):
        raw = "<think>Let me look.</think>Sure.<tool_call><function=read><parameter=path>a b.py</parameter><parameter=offset>5</parameter></function></tool_call>"
        r, c, calls = split_generation(raw)
        self.assertEqual(r, "Let me look.")
        self.assertEqual(c, "Sure.")
        self.assertEqual(calls, [("read", {"path": "a b.py", "offset": "5"})])
        self.assertEqual(coerce_args("read", calls[0][1], TOOLS), {"path": "a b.py", "offset": 5})
        # re-rendering the parsed turn reproduces the raw text (prefix-cache property)
        msg = {"role": "assistant", "reasoning_content": r, "content": c,
               "tool_calls": [{"function": {"name": "read", "arguments": json.dumps(coerce_args("read", calls[0][1], TOOLS))}}]}
        self.assertEqual(render([msg], add_generation_prompt=False), "<|im_start|>assistant\n" + raw + "<|im_end|>")

    def test_stream_splitter_tags_split_across_chunks(self):
        raw = "<think>abc</think>Hello <tool_call><function=x></function></tool_call>"
        for step in (1, 2, 3, 7):
            s, got = StreamSplitter(True), []
            for i in range(0, len(raw), step):
                got += s.feed(raw[i:i + step])
            got += s.flush()
            reasoning = "".join(t for k, t in got if k == "reasoning")
            content = "".join(t for k, t in got if k == "content")
            self.assertEqual(reasoning, "abc", f"step={step}, got={got}")
            self.assertEqual(content, "Hello ", f"step={step}, got={got}")


if __name__ == "__main__":
    unittest.main()
