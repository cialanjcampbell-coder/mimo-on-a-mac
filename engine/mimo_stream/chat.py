"""MiMo-V2.6 chat formatting: prompt rendering mirrors chat_template.jinja, output parsing, tool-call coercion.

Prefix-cache friendliness: past assistant turns that this server generated are rendered from the exact raw
text the model produced (looked up by tool-call id or by content), because OpenAI-compatible clients send tool
arguments back as JSON strings, which the template would render differently from MiMo's native
<parameter=...> form, breaking the cached prefix at every tool call.
"""
import json, re

TOOL_CALL_RE = re.compile(r"<tool_call>\s*<function=([^>\s]+)>(.*?)</function>\s*</tool_call>", re.S)
PARAM_RE = re.compile(r"<parameter=([^>\s]+)>(.*?)</parameter>", re.S)


def content_text(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    out = []
    for part in content:
        if isinstance(part, dict):
            if part.get("type") == "text" or "text" in part:
                out.append(part.get("text", ""))
            elif part.get("type") in ("image", "image_url") or "image_url" in part:
                out.append("<|vision_start|><|image_pad|><|vision_end|>")
        else:
            out.append(str(part))
    return "".join(out)


def _render_value(v):
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def _tool_json(t):
    # jinja tojson == json.dumps with default separators and ensure_ascii=False
    return json.dumps(t, ensure_ascii=False)


def render_tool_calls(tool_calls):
    out = []
    for tc in tool_calls:
        fn = tc.get("function") or tc.get("custom") or tc
        out.append("<tool_call><function=" + fn["name"] + ">")
        if isinstance(fn.get("input"), str):
            out.append(fn["input"])
        else:
            args = fn.get("arguments")
            if isinstance(args, str):
                try:  # JSON-string arguments (OpenAI clients) -> native parameter form
                    args = json.loads(args)
                except json.JSONDecodeError:
                    out.append(args)
                    args = None
            if isinstance(args, dict):
                for k, v in args.items():
                    out.append(f"<parameter={k}>{_render_value(v)}</parameter>")
        out.append("</function></tool_call>")
    return "".join(out)


def render(messages, tools=None, add_generation_prompt=True, enable_thinking=True, raw_lookup=None):
    """Render a conversation exactly like MiMo's chat_template.jinja, with raw
    substitution of assistant turns this server generated. raw_lookup(message) -> raw text or None."""
    out = []
    if tools:
        out.append("<|im_start|>system\nYou are provided with the following tools:\n\n<tools>"
                   + "".join("\n" + _tool_json(t) for t in tools) + "\n</tools><|im_end|>")
    for m in messages:
        role = m.get("role")
        if role == "assistant":
            raw = raw_lookup(m) if raw_lookup else None
            if raw is not None:
                out.append("<|im_start|>assistant\n" + raw + "<|im_end|>")
                continue
            reasoning = m.get("reasoning_content") if isinstance(m.get("reasoning_content"), str) else ""
            s = "<|im_start|>assistant\n<think>" + reasoning + "</think>" + content_text(m.get("content"))
            if m.get("tool_calls"):
                s += render_tool_calls(m["tool_calls"])
            out.append(s + "<|im_end|>")
        else:
            if role == "developer":
                role = "system"
            body = content_text(m.get("content"))
            s = "<|im_start|>" + role + "\n" + body
            if m.get("tools"):
                if body:
                    s += "\n\n"
                s += "You are provided with the following tools:\n\n<tools>" + "".join(
                    "\n" + _tool_json(t) for t in m["tools"]) + "\n</tools>"
            out.append(s + "<|im_end|>")
    if add_generation_prompt:
        out.append("<|im_start|>assistant\n" + ("" if enable_thinking else "<think></think>"))
    return "".join(out)


def split_generation(text):
    """Split raw generated text (after '<|im_start|>assistant\\n') into (reasoning, content, tool_calls)."""
    reasoning, rest = "", text
    if rest.startswith("<think>"):
        end = rest.find("</think>")
        if end < 0:
            return rest[len("<think>"):], "", []
        reasoning, rest = rest[len("<think>"):end], rest[end + len("</think>"):]
    idx = rest.find("<tool_call>")
    content = rest if idx < 0 else rest[:idx]
    calls = []
    for name, body in TOOL_CALL_RE.findall(rest if idx >= 0 else ""):
        params = PARAM_RE.findall(body)
        if params:
            args = {k: v for k, v in params}
        else:
            body = body.strip()
            try:
                args = json.loads(body) if body else {}
            except json.JSONDecodeError:
                args = {"input": body}
        calls.append((name, args))
    return reasoning, content, calls


def coerce_args(name, args, tools):
    """Parameter values arrive as strings; convert per the tool's JSON schema (numbers, booleans, arrays...)."""
    schema = {}
    for t in tools or []:
        fn = t.get("function", t)
        if fn.get("name") == name:
            schema = (fn.get("parameters") or {}).get("properties", {})
    out = {}
    for k, v in args.items():
        typ = (schema.get(k) or {}).get("type")
        if isinstance(v, str) and typ in ("integer", "number", "boolean", "array", "object", "null"):
            try:
                v = json.loads(v)
            except json.JSONDecodeError:
                pass
        out[k] = v
    return out


class StreamSplitter:
    """Incrementally route generated text into reasoning/content deltas; holds back text that might be the
    start of a tag, and stops emitting content once a tool call starts."""

    def __init__(self, thinking):
        self.mode = "start" if thinking else "content"
        self.buf = ""

    def feed(self, text):
        """Returns list of (kind, text) with kind in {'reasoning', 'content'}."""
        self.buf += text
        out = []
        while True:
            if self.mode == "start":
                if self.buf.startswith("<think>"):
                    self.buf = self.buf[len("<think>"):]
                    self.mode = "reasoning"
                    continue
                if len(self.buf) < len("<think>") and "<think>".startswith(self.buf):
                    return out
                self.mode = "content"  # model skipped thinking
                continue
            tag = "</think>" if self.mode == "reasoning" else "<tool_call>"
            if self.mode == "tool":
                self.buf = ""
                return out
            i = self.buf.find(tag)
            if i >= 0:
                if i:
                    out.append((self.mode, self.buf[:i]))
                self.buf = self.buf[i + len(tag):]
                self.mode = "content" if self.mode == "reasoning" else "tool"
                continue
            keep = 0  # hold back a possible partial tag at the end
            for k in range(min(len(tag) - 1, len(self.buf)), 0, -1):
                if tag.startswith(self.buf[-k:]):
                    keep = k
                    break
            emit = self.buf[: len(self.buf) - keep]
            if emit:
                out.append((self.mode, emit))
            self.buf = self.buf[len(self.buf) - keep:]
            return out

    def flush(self):
        out = [(self.mode, self.buf)] if self.buf and self.mode in ("reasoning", "content") else []
        self.buf = ""
        return out
