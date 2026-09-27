#!/usr/bin/env python3
"""OpenAI-compatible server for MiMo-V2.6-Flash with SSD-streamed experts (engine/mimo_stream).

  python -m engine.server [--port 8082] [--slots 125] [--model DIR]

Endpoints: GET /health, GET /v1/models, GET /stats, POST /v1/chat/completions (stream and non-stream).
- One request at a time (a lock); a second request waits.
- Reasoning streams as `reasoning_content`, text as `content`; tool calls are parsed from MiMo's native
  <tool_call><function=..><parameter=..> format at the end and returned as OpenAI tool_calls
  (arguments coerced to the tools' JSON-schema types).
- Thinking: on by default; `enable_thinking` (top level, as some OpenAI-compatible clients send it) or
  chat_template_kwargs.enable_thinking turn it off.
- Prefix cache (one conversation): the KV cache is reused when the rendered prompt extends the text already
  in the cache. Past assistant turns generated here are re-rendered from their exact raw text. The 128-token
  sliding-window caches are snapshotted at the end of each prompt, so a request can roll back into the
  previous generation instead of re-prefilling everything.
"""
import argparse, json, os, sys, threading, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import mlx.core as mx
from mlx_lm.models.cache import RotatingKVCache
from mlx_lm.tokenizer_utils import load as load_tokenizer

from engine.mimo_stream.model import Engine
from engine.mimo_stream import chat

MODEL_ID = "mimo-v2.6-flash"
DEFAULT_MODEL_DIR = os.environ.get("MIMO_MODEL_DIR", "")
PROFILE = Path(__file__).resolve().parent / "mimo_stream" / "data" / "expert_profile.npz"
EOS = {151643, 151645, 151672}
PREFILL_CHUNK = 4096


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, file=sys.stderr, flush=True)


def load_profile():
    if PROFILE.exists():
        z = np.load(PROFILE)
        return {int(k[1:]): z[k] for k in z.files}
    return None


def snapshot(caches):
    snap = []
    for c in caches:
        if isinstance(c, RotatingKVCache):
            k = None if c.keys is None else c.keys + 0      # +0 detaches from later in-place updates
            v = None if c.values is None else c.values + 0
            snap.append(("rot", k, v, c.offset, c._idx))
        else:
            snap.append(("kv", c.offset))
    mx.eval([s[1] for s in snap if s[0] == "rot" and s[1] is not None] + [s[2] for s in snap if s[0] == "rot" and s[2] is not None])
    return snap


def restore(caches, snap):
    for c, s in zip(caches, snap):
        if s[0] == "rot":
            _, k, v, off, idx = s
            c.keys = None if k is None else k + 0
            c.values = None if v is None else v + 0
            c.offset, c._idx = off, idx
        else:
            c.offset = s[1]


def sample(logits, temperature, top_p, top_k):
    if temperature is None or temperature <= 0:
        return int(mx.argmax(logits).item())
    lg = logits / temperature
    if top_k and 0 < top_k < lg.shape[-1]:
        kth = mx.sort(lg)[-top_k]
        lg = mx.where(lg < kth, -mx.inf, lg)
    if top_p is not None and 0 < top_p < 1:
        order = mx.argsort(-lg)
        probs = mx.softmax(lg[order])
        cum = mx.cumsum(probs)
        keep = (cum - probs) < top_p
        masked = mx.where(keep, lg[order], -mx.inf)
        return int(order[mx.random.categorical(masked)].item())
    return int(mx.random.categorical(lg).item())


class Conversation:
    """Tokens / text currently in the KV cache, plus a snapshot taken at the end of the last prompt."""

    def __init__(self, engine):
        self.engine = engine
        self.reset()

    def reset(self):
        self.cache = self.engine.make_cache()
        self.tokens, self.text = [], ""
        self.snap = None          # (n_tokens, text_len, cache snapshot) at the end of the last prompt
        self.gen_tokens, self.gen_ends = [], []   # tokens generated after the snapshot, cumulative char ends


class Server:
    def __init__(self, args):
        if not args.model or not Path(args.model).exists():
            print(f"Model directory '{args.model}' not found. Please specify --model /path/to/converted-model or set MIMO_MODEL_DIR.", file=sys.stderr)
            sys.exit(1)
        t0 = time.time()
        self.engine = Engine(args.model, args.slots, profile=load_profile(), log=log, dense_bits=args.dense_bits,
                             prefetch_k=args.prefetch_k)
        self.tok = load_tokenizer(Path(args.model))
        self.conv = Conversation(self.engine)
        self.raw_by_call, self.raw_by_content = {}, {}
        self.lock = threading.Lock()
        self.last = {}
        self.max_context = args.max_context
        log(f"[server] ready in {time.time() - t0:.1f}s")

    # ---- prompt construction with prefix reuse
    def raw_lookup(self, m):
        for tc in m.get("tool_calls") or []:
            raw = self.raw_by_call.get(tc.get("id"))
            if raw is not None:
                return raw
        if not m.get("tool_calls"):
            return self.raw_by_content.get(chat.content_text(m.get("content")))
        return None

    def prepare(self, text):
        """Bring the KV cache to a prefix of `text`; returns the new tokens still to prefill and stats."""
        c, enc = self.conv, lambda s: self.tok.encode(s, add_special_tokens=False)
        n = len(os.path.commonprefix([text, c.text]))
        if c.tokens and n == len(c.text):
            return enc(text[n:]), {"reused": len(c.tokens), "mode": "extend"}
        if c.snap is not None and n >= c.snap[1]:
            n_tok, tlen, snap = c.snap
            restore(c.cache, snap)
            j = 0
            while j < len(c.gen_ends) and tlen + c.gen_ends[j] <= n:
                j += 1
            keep_gen = c.gen_tokens[:j]
            cut = tlen + (c.gen_ends[j - 1] if j else 0)
            c.tokens = c.tokens[:n_tok]
            c.text = c.text[:tlen]
            c.gen_tokens, c.gen_ends = [], []
            return keep_gen + enc(text[cut:]), {"reused": n_tok, "mode": "rollback", "regen": len(keep_gen)}
        c.reset()
        return enc(text), {"reused": 0, "mode": "fresh"}

    def prefill(self, new_tokens, text):
        c = self.conv
        logits = None
        for i in range(0, len(new_tokens), PREFILL_CHUNK):
            logits = self.engine.forward(new_tokens[i:i + PREFILL_CHUNK], c.cache)
        c.tokens += new_tokens
        c.text = text
        c.snap = (len(c.tokens), len(c.text), snapshot(c.cache))
        c.gen_tokens, c.gen_ends = [], []
        return logits

    # ---- one completion
    def complete(self, req, emit):
        messages, tools = req.get("messages", []), req.get("tools")
        thinking = req.get("enable_thinking")
        if thinking is None:
            thinking = (req.get("chat_template_kwargs") or {}).get("enable_thinking", True)
        max_tokens = int(req.get("max_tokens") or req.get("max_completion_tokens") or 8192)
        temperature = req.get("temperature", 1.0)
        top_p = req.get("top_p", 0.95)
        top_k = req.get("top_k", 0)
        text = chat.render(messages, tools, add_generation_prompt=True, enable_thinking=bool(thinking),
                           raw_lookup=self.raw_lookup)
        t0 = time.time()
        new_tokens, pstats = self.prepare(text)
        if len(self.conv.tokens) + len(new_tokens) + max_tokens > self.max_context:
            max_tokens = max(1, self.max_context - len(self.conv.tokens) - len(new_tokens))
        io0, m0 = self.engine.io.bytes, sum(s.misses for s in self.engine.slots.values())
        if not new_tokens:   # identical prompt (e.g. a retry): no logits to continue from, so start over
            self.conv.reset()
            new_tokens, pstats = self.tok.encode(text, add_special_tokens=False), {"reused": 0, "mode": "fresh"}
        logits = self.prefill(new_tokens, text)
        t_prefill = time.time() - t0
        prefix = "" if thinking else "<think></think>"
        splitter = chat.StreamSplitter(bool(thinking))
        out, gen_text, t1, finish = [], "", time.time(), "length"
        c = self.conv
        for _ in range(max_tokens):
            tok = sample(logits, temperature, top_p, top_k)
            if tok in EOS:
                finish = "stop"
                break
            out.append(tok)
            new_text = self.tok.decode(out)
            delta, gen_text = new_text[len(gen_text):], new_text
            for kind, piece in splitter.feed(delta):
                if not emit(kind, piece):
                    finish = "abort"
                    break
            if finish == "abort":
                break
            logits = self.engine.decode_step(tok, c.cache)
            c.tokens.append(tok)
            c.gen_tokens.append(tok)
            c.gen_ends.append(len(gen_text))
        if finish != "abort":
            for kind, piece in splitter.flush():
                emit(kind, piece)
        c.text += gen_text[: c.gen_ends[-1]] if c.gen_ends else ""   # only tokens that are in the KV cache
        t_decode = time.time() - t1
        reasoning, content, calls = chat.split_generation(prefix + gen_text if prefix else gen_text)
        tool_calls = []
        raw = prefix + gen_text
        for name, args in calls:
            cid = "call_" + uuid.uuid4().hex[:12]
            tool_calls.append({"id": cid, "type": "function",
                               "function": {"name": name, "arguments": json.dumps(chat.coerce_args(name, args, tools), ensure_ascii=False)}})
            self.raw_by_call[cid] = raw
        if not tool_calls:
            self.raw_by_content[content] = raw
        if tool_calls and finish == "stop":
            finish = "tool_calls"
        misses = sum(s.misses for s in self.engine.slots.values()) - m0
        self.last = {"prompt_tokens": c.snap[0], "new_prompt_tokens": len(new_tokens),
                     "prefix": pstats, "prefill_s": round(t_prefill, 2),
                     "prefill_tok_s": round(len(new_tokens) / max(t_prefill, 1e-9), 1),
                     "completion_tokens": len(out), "decode_s": round(t_decode, 2),
                     "decode_tok_s": round(len(out) / max(t_decode, 1e-9), 2),
                     "expert_misses_per_token": round(misses / max(len(out), 1), 1),
                     "ssd_GB": round((self.engine.io.bytes - io0) / 1e9, 2), "finish": finish,
                     "context_tokens": len(c.tokens)}
        log(f"[req] {pstats['mode']} reuse={pstats['reused']} new={len(new_tokens)} prefill {t_prefill:.1f}s "
            f"({self.last['prefill_tok_s']} tok/s) | decode {len(out)} tok {self.last['decode_tok_s']} tok/s | "
            f"misses/tok {self.last['expert_misses_per_token']} | {finish} | ctx {len(c.tokens)}")
        usage = {"prompt_tokens": c.snap[0] if c.snap else 0, "completion_tokens": len(out),
                 "total_tokens": (c.snap[0] if c.snap else 0) + len(out)}
        return {"reasoning": reasoning, "content": content, "tool_calls": tool_calls, "finish": finish, "usage": usage}


def make_handler(srv):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _json(self, code, obj):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path.rstrip("/") in ("/health", ""):
                return self._json(200, {"status": "ok", "model": MODEL_ID})
            if self.path.startswith("/v1/models"):
                return self._json(200, {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "local"}]})
            if self.path.startswith("/stats"):
                hr, _ = srv.engine.hit_rate()
                return self._json(200, {"last_request": srv.last, "expert_hit_rate_total": round(hr, 4),
                                        "slots_per_layer": srv.engine.S, "context_tokens": len(srv.conv.tokens)})
            self._json(404, {"error": "not found"})

        def do_POST(self):
            if not self.path.startswith("/v1/chat/completions"):
                return self._json(404, {"error": "not found"})
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            stream = bool(req.get("stream"))
            rid, created = "chatcmpl-" + uuid.uuid4().hex[:16], int(time.time())
            include_usage = bool((req.get("stream_options") or {}).get("include_usage"))

            def chunk(delta, finish=None, usage=None):
                d = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": MODEL_ID,
                     "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                if usage is not None:
                    d["usage"] = usage
                return ("data: " + json.dumps(d, ensure_ascii=False) + "\n\n").encode()

            with srv.lock:
                if stream:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self.end_headers()

                    def emit(kind, text):
                        if not text:
                            return True
                        try:
                            self.wfile.write(chunk({"reasoning_content": text} if kind == "reasoning" else {"content": text}))
                            self.wfile.flush()
                            return True
                        except (BrokenPipeError, ConnectionResetError):
                            return False
                    try:
                        self.wfile.write(chunk({"role": "assistant", "content": ""}))
                        r = srv.complete(req, emit)
                        if r["finish"] == "abort":
                            return
                        if r["tool_calls"]:
                            self.wfile.write(chunk({"tool_calls": [dict(tc, index=i) for i, tc in enumerate(r["tool_calls"])]}))
                        self.wfile.write(chunk({}, finish=r["finish"], usage=r["usage"] if include_usage else None))
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        log("[req] client disconnected")
                    except Exception as e:  # report and keep serving
                        log(f"[req] error: {e!r}")
                        try:
                            self.wfile.write(chunk({"content": f"\n[server error: {e}]"}, finish="stop"))
                            self.wfile.write(b"data: [DONE]\n\n")
                        except Exception:
                            pass
                        srv.conv.reset()
                    self.close_connection = True
                else:
                    try:
                        r = srv.complete(req, lambda k, t: True)
                    except Exception as e:
                        srv.conv.reset()
                        return self._json(500, {"error": repr(e)})
                    msg = {"role": "assistant", "content": r["content"] or None}
                    if r["reasoning"]:
                        msg["reasoning_content"] = r["reasoning"]
                    if r["tool_calls"]:
                        msg["tool_calls"] = r["tool_calls"]
                    self._json(200, {"id": rid, "object": "chat.completion", "created": created, "model": MODEL_ID,
                                     "choices": [{"index": 0, "message": msg, "finish_reason": r["finish"]}],
                                     "usage": r["usage"]})

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8082)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--model", default=DEFAULT_MODEL_DIR)
    ap.add_argument("--slots", type=int, default=125, help="resident experts per MoE layer (of 256)")
    ap.add_argument("--dense-bits", type=int, default=8)
    ap.add_argument("--prefetch-k", type=int, default=4)
    ap.add_argument("--max-context", type=int, default=131072)
    a = ap.parse_args()
    if not a.model or not Path(a.model).exists():
        print(f"Model directory '{a.model}' not found. Please specify --model /path/to/converted-model or set MIMO_MODEL_DIR.", file=sys.stderr)
        sys.exit(1)
    srv = Server(a)
    httpd = ThreadingHTTPServer((a.host, a.port), make_handler(srv))
    log(f"[server] listening on http://{a.host}:{a.port}/v1 (model id {MODEL_ID})")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
