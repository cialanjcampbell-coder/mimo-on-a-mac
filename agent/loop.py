"""The tool loop: send the conversation, run the requested tools, repeat until the model stops.

Finished messages are reported via emit(message) as JSON-event shapes (user / assistant / toolResult).
History keeps each assistant turn exactly as the server sent it (content, reasoning, tool-call ids and raw
argument strings) so the server's prefix cache can re-render it from the raw generated text.
"""
import json, re

from agent import client, tools
from agent.prompt import system_prompt

STOP = {"stop": "stop", "tool_calls": "toolUse", "length": "length"}
SERVER_ERROR = re.compile(r"\n\[server error: .*\]$", re.S)


class Agent:
    def __init__(self, base_url, model, tool_names=tools.ALL, thinking=True, max_turns=100,
                 emit=lambda m: None, on_delta=None, on_tool_start=None, system=None):
        self.base_url, self.model, self.tool_names = base_url, model, tuple(tool_names)
        self.thinking, self.max_turns = thinking, max_turns
        self.emit, self.on_delta, self.on_tool_start = emit, on_delta, on_tool_start
        self.system = system if system is not None else system_prompt()
        self.reset()

    def reset(self):
        self.messages = [{"role": "system", "content": self.system}]

    def request(self):
        p = {"model": self.model, "messages": self.messages, "enable_thinking": self.thinking,
             "chat_template_kwargs": {"enable_thinking": self.thinking}}
        if self.tool_names:
            p["tools"] = tools.schemas(self.tool_names)
        return p

    def run(self, prompt):
        """One user turn through to the final answer. Returns the last assistant event message."""
        self.messages.append({"role": "user", "content": prompt})
        self.emit({"role": "user", "content": [{"type": "text", "text": prompt}]})
        for _ in range(self.max_turns):
            msg = self.step()
            if msg["stopReason"] != "toolUse":
                return msg
            if not self.run_tools(msg):
                return dict(msg, stopReason="aborted")
        msg = {"role": "assistant", "content": [], "usage": {"input": 0, "output": 0}, "stopReason": "error",
               "errorMessage": f"stopped after {self.max_turns} turns (--max-turns)"}
        self.emit(msg)
        return msg

    def step(self):
        """One completion; appends it to history and emits it."""
        try:
            r, stop, err = client.chat(self.base_url, self.request(), self.on_delta), None, None
        except client.Aborted as a:
            r, stop, err = a.partial, "aborted", None
        except Exception as e:
            r, stop, err = {"reasoning": "", "content": "", "tool_calls": [], "usage": None}, "error", str(e)
        if stop is None:
            stop = STOP.get(r["finish"], "error")
            if r["finish"] is None:
                err = "stream ended without a finish reason"
            elif SERVER_ERROR.search(r["content"]):
                stop, err = "error", SERVER_ERROR.search(r["content"]).group(0).strip()
            elif stop == "error":
                err = f"finish_reason {r['finish']!r}"
            elif stop == "toolUse" and not r["tool_calls"]:
                stop = "stop"
        if stop != "error" and (r["content"] or r["tool_calls"]):
            h = {"role": "assistant", "content": r["content"]}
            if r["reasoning"]:
                h["reasoning_content"] = r["reasoning"]
            if r["tool_calls"]:
                h["tool_calls"] = r["tool_calls"]
            self.messages.append(h)
        content = ([{"type": "thinking", "thinking": r["reasoning"]}] if r["reasoning"] else []) + \
                  ([{"type": "text", "text": r["content"]}] if r["content"] else [])
        for tc in r["tool_calls"]:
            content.append({"type": "toolCall", "id": tc["id"], "name": tc["function"]["name"],
                            "arguments": parse_args(tc["function"]["arguments"])})
        u = r.get("usage") or {}
        msg = {"role": "assistant", "content": content,
               "usage": {"input": u.get("prompt_tokens", 0),
                         "output": u.get("completion_tokens", (len(r["reasoning"]) + len(r["content"])) // 4)},
               "stopReason": stop}
        if err:
            msg["errorMessage"] = err
        self.emit(msg)
        return msg

    def run_tools(self, msg):
        """Execute the assistant's tool calls in order. False if interrupted (remaining calls get an error)."""
        calls = [c for c in msg["content"] if c["type"] == "toolCall"]
        raw = {tc["id"]: tc["function"]["arguments"] for tc in self.messages[-1]["tool_calls"]}
        interrupted = False
        for c in calls:
            if interrupted:
                text, is_err = "Aborted by user", True
            else:
                try:
                    args = json.loads(raw[c["id"]] or "{}")
                    if self.on_tool_start:
                        self.on_tool_start(c["name"], args)
                    text, is_err = tools.run(c["name"], args, self.tool_names)
                except json.JSONDecodeError as e:
                    text, is_err = f"Invalid JSON arguments for {c['name']}: {e}", True
                except KeyboardInterrupt:
                    interrupted, text, is_err = True, "Aborted by user", True
            self.messages.append({"role": "tool", "tool_call_id": c["id"], "content": text})
            self.emit({"role": "toolResult", "toolCallId": c["id"], "toolName": c["name"],
                       "content": [{"type": "text", "text": text}], "isError": is_err})
        return not interrupted


def parse_args(s):
    try:
        v = json.loads(s or "{}")
        return v if isinstance(v, dict) else {"_raw": s}
    except json.JSONDecodeError:
        return {"_raw": s}
