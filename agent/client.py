"""Streaming OpenAI chat-completions client (stdlib urllib + SSE)."""
import json, urllib.error, urllib.request


class Aborted(Exception):
    """Ctrl-C during a stream; .partial holds what arrived so far."""

    def __init__(self, partial):
        super().__init__("aborted")
        self.partial = partial


def chat(base_url, payload, on_delta=None, timeout=1800):
    """POST a streamed chat completion. on_delta(kind, text) with kind 'reasoning' or 'content'.
    Returns {reasoning, content, tool_calls, finish, usage}. Tool-call ids and argument strings are kept verbatim."""
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions",
                                 data=json.dumps(dict(payload, stream=True, stream_options={"include_usage": True})).encode(),
                                 headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
    r = {"reasoning": "", "content": "", "tool_calls": [], "finish": None, "usage": None}
    calls = {}
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for line in resp:
                line = line.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                ev = json.loads(data)
                if ev.get("error"):
                    raise RuntimeError(f"server error: {ev['error']}")
                if ev.get("usage"):
                    r["usage"] = ev["usage"]
                for ch in ev.get("choices") or []:
                    d = ch.get("delta") or {}
                    for kind, key in (("reasoning", "reasoning_content"), ("content", "content")):
                        if d.get(key):
                            r[kind] += d[key]
                            if on_delta:
                                on_delta(kind, d[key])
                    for tc in d.get("tool_calls") or []:
                        c = calls.setdefault(tc.get("index", len(calls)), {"id": "", "type": "function",
                                                                        "function": {"name": "", "arguments": ""}})
                        c["id"] = tc.get("id") or c["id"]
                        fn = tc.get("function") or {}
                        c["function"]["name"] += fn.get("name") or ""
                        c["function"]["arguments"] += fn.get("arguments") or ""
                    if ch.get("finish_reason"):
                        r["finish"] = ch["finish_reason"]
    except KeyboardInterrupt:
        r["tool_calls"] = []
        raise Aborted(r) from None
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode(errors='replace')[:500]}") from None
    except urllib.error.URLError as e:
        raise RuntimeError(f"cannot reach {base_url}: {e.reason}") from None
    except (OSError, ValueError) as e:  # connection reset / timeout / bad JSON mid-stream
        raise RuntimeError(f"stream failed: {e!r}") from None
    r["tool_calls"] = [calls[i] for i in sorted(calls)]
    return r
