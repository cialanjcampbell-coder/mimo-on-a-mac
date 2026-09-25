#!/usr/bin/env python3
"""Streaming bench client for OpenAI-compatible server.
Usage: mlx_bench.py LABEL PROMPTFILE|-  MAX_TOKENS [--think] [--model=M] [--url=BASE] [--extra=JSON]
--url defaults to mlx-serve (http://127.0.0.1:8081); e.g. --url=http://127.0.0.1:8083 for ds4-server."""
import json, sys, time, urllib.request
BASE = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--url=")), "http://127.0.0.1:8081")
URL = BASE.rstrip("/") + "/v1/chat/completions"
label, pf, mt = sys.argv[1], sys.argv[2], int(sys.argv[3])
think = "--think" in sys.argv
model = next((a.split("=",1)[1] for a in sys.argv if a.startswith("--model=")), "default")
prompt = sys.stdin.read() if pf == "-" else open(pf).read()
body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": mt,
        "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
        "enable_thinking": think, "chat_template_kwargs": {"enable_thinking": think}}
for a in sys.argv:
    if a.startswith("--extra="): body.update(json.loads(a[8:]))
req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
t0 = time.time(); tfirst = None; text = ""; reasoning = ""; usage = None; extra = {}; nchunks = 0
with urllib.request.urlopen(req, timeout=3600) as r:
    for line in r:
        line = line.decode().strip()
        if not line.startswith("data:"): continue
        d = line[5:].strip()
        if d == "[DONE]": break
        j = json.loads(d)
        if j.get("usage"): usage = j["usage"]
        for k in ("timings", "x_timings", "stats", "perf"):
            if k in j: extra[k] = j[k]
        for c in j.get("choices", []):
            dl = c.get("delta", {})
            piece = (dl.get("content") or "") + (dl.get("reasoning_content") or dl.get("reasoning") or "")
            if piece:
                nchunks += 1
                if tfirst is None: tfirst = time.time()
            text += dl.get("content") or ""; reasoning += dl.get("reasoning_content") or dl.get("reasoning") or ""
t1 = time.time()
pt = (usage or {}).get("prompt_tokens"); ct = (usage or {}).get("completion_tokens") or nchunks
ttft = (tfirst or t1) - t0; dec = (t1 - tfirst) if tfirst else 0
res = {"label": label, "prompt_tokens": pt, "cached": ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens"),
       "completion_tokens": ct, "ttft_s": round(ttft, 2), "prefill_tok_s(client)": round(pt / ttft, 1) if pt else None,
       "decode_tok_s(client)": round((ct - 1) / dec, 1) if dec > 0 and ct else None, "total_s": round(t1 - t0, 2),
       "answer": text.strip()[-300:], "reasoning_len": len(reasoning), "usage": usage, "extra": extra}
print(json.dumps(res, indent=1))
