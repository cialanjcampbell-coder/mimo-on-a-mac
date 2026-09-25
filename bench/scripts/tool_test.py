import json, sys, time, urllib.request
think = sys.argv[1] == "on"; stream = len(sys.argv) > 2 and sys.argv[2] == "stream"
tools = [{"type": "function", "function": {"name": "read_file", "description": "Read a file from the local filesystem and return its contents.",
  "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Absolute path of the file"},
  "offset": {"type": "integer", "description": "Line to start at"}, "limit": {"type": "integer"}}, "required": ["path"]}}},
  {"type": "function", "function": {"name": "bash", "description": "Run a shell command", "parameters": {"type": "object",
  "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]
msgs = [{"role": "system", "content": "You are a coding agent. Use the provided tools to inspect files before answering."},
        {"role": "user", "content": "What does /Users/me/project/src/main.py do? Read it first (first 50 lines)."}]
body = {"model": "m", "messages": msgs, "tools": tools, "tool_choice": "auto", "max_tokens": 2048, "stream": stream,
        "enable_thinking": think, "chat_template_kwargs": {"enable_thinking": think}}
def post(b):
    req = urllib.request.Request("http://127.0.0.1:8081/v1/chat/completions", data=json.dumps(b).encode(), headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=600)
t0 = time.time()
if not stream:
    r = json.load(post(body)); m = r["choices"][0]["message"]
    print("finish_reason:", r["choices"][0]["finish_reason"], "time %.1fs" % (time.time()-t0), "usage", r.get("usage"))
    print("reasoning chars:", len(m.get("reasoning_content") or ""), "content:", repr((m.get("content") or "")[:300]))
    print("tool_calls:", json.dumps(m.get("tool_calls"), indent=1))
    # round 2: feed tool result back
    tc = m.get("tool_calls")
    if tc:
        msgs2 = msgs + [m, {"role": "tool", "tool_call_id": tc[0]["id"], "content": "import sys\n\ndef main():\n    print(sum(int(a) for a in sys.argv[1:]))\n\nif __name__ == '__main__':\n    main()\n"}]
        body["messages"] = msgs2; r2 = json.load(post(body)); m2 = r2["choices"][0]["message"]
        print("ROUND2 finish:", r2["choices"][0]["finish_reason"], "content:", repr((m2.get("content") or "")[:400]), "tool_calls:", m2.get("tool_calls"))
else:
    calls = {}; fin = None; content = ""; nreason = 0
    for line in post(body):
        line = line.decode().strip()
        if not line.startswith("data:") or line[5:].strip() == "[DONE]": continue
        j = json.loads(line[5:])
        for c in j.get("choices", []):
            d = c.get("delta", {}); fin = c.get("finish_reason") or fin
            content += d.get("content") or ""; nreason += len(d.get("reasoning_content") or "")
            for t in d.get("tool_calls") or []:
                e = calls.setdefault(t["index"], {"id": None, "name": "", "args": ""})
                e["id"] = t.get("id") or e["id"]; f = t.get("function", {})
                e["name"] += f.get("name") or ""; e["args"] += f.get("arguments") or ""
    print("STREAM finish:", fin, "reasoning chars", nreason, "content", repr(content[:200]), "calls", calls,
          "args-json-valid:", all(json.loads(v["args"]) is not None for v in calls.values()) if calls else None)
