"""Render the agent's (`scripts/mimo-agent --json`) event stream as Markdown; count turns, tool calls and tokens."""
import json
from pathlib import Path


def load_events(path):
    p = Path(path)
    if not p.exists():
        return []
    events = []
    for raw in p.read_bytes().split(b"\n"):
        raw = raw.strip()
        if not raw:
            continue
        try:
            e = json.loads(raw)
        except ValueError:
            continue
        if isinstance(e, dict):
            events.append(e)
    return events


def final_messages(events):
    return [e["message"] for e in events if e.get("type") == "message_end" and isinstance(e.get("message"), dict)]


def stats(events):
    msgs = final_messages(events)
    asst = [m for m in msgs if m.get("role") == "assistant"]
    return {
        "turns": len(asst),
        "tool_calls": sum(1 for m in asst for c in m.get("content") or [] if c.get("type") == "toolCall"),
        "tool_errors": sum(1 for m in msgs if m.get("role") == "toolResult" and m.get("isError")),
        "output_tokens": sum((m.get("usage") or {}).get("output", 0) for m in asst),
        "stop_reason": asst[-1].get("stopReason") if asst else None,
        "error": next((m["errorMessage"] for m in reversed(asst) if m.get("errorMessage")), None),
    }


def _clip(text, n):
    return text if len(text) <= n else text[:n] + f"\n… [{len(text) - n} more chars]"


def _text(content):
    if isinstance(content, str):
        return content
    return "\n".join(c.get("text", "") for c in content or [] if c.get("type") == "text")


def render_markdown(events, max_tool_output=2000, max_thinking=1500):
    out, turn = [], 0
    for m in final_messages(events):
        role = m.get("role")
        if role == "user":
            out += ["## User", "", _text(m.get("content")), ""]
        elif role == "assistant":
            turn += 1
            out += [f"## Assistant (turn {turn})", ""]
            for c in m.get("content") or []:
                kind = c.get("type")
                if kind == "thinking" and c.get("thinking"):
                    out += ["<details><summary>thinking</summary>", "", _clip(c["thinking"], max_thinking),
                            "", "</details>", ""]
                elif kind == "text" and c.get("text"):
                    out += [c["text"], ""]
                elif kind == "toolCall":
                    out += [f"**tool call:** `{c.get('name')}`", "", "````json",
                            json.dumps(c.get("arguments"), indent=1, ensure_ascii=False), "````", ""]
            if m.get("errorMessage"):
                out += [f"**error:** {m['errorMessage']}", ""]
        elif role == "toolResult":
            flag = " (error)" if m.get("isError") else ""
            out += [f"### Tool result: `{m.get('toolName')}`{flag}", "", "````",
                    _clip(_text(m.get("content")), max_tool_output), "````", ""]
    return "\n".join(out)
