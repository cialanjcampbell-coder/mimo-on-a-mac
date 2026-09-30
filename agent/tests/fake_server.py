"""A fake OpenAI chat-completions server that plays back scripted streamed replies and records requests."""
import json, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def reply(content="", reasoning="", tool_calls=(), finish=None, usage=None):
    """SSE chunks shaped like engine/server.py's: reasoning/content deltas, one tool_calls delta, finish."""
    chunks = [{"role": "assistant", "content": ""}]
    chunks += [{"reasoning_content": reasoning[i:i + 5]} for i in range(0, len(reasoning), 5)]
    chunks += [{"content": content[i:i + 5]} for i in range(0, len(content), 5)]
    calls = [{"index": i, "id": cid, "type": "function", "function": {"name": name, "arguments": args}}
             for i, (cid, name, args) in enumerate(tool_calls)]
    if calls:
        chunks.append({"tool_calls": calls})
    events = [{"choices": [{"index": 0, "delta": d, "finish_reason": None}]} for d in chunks]
    last = {"choices": [{"index": 0, "delta": {}, "finish_reason": finish or ("tool_calls" if calls else "stop")}]}
    last["usage"] = usage or {"prompt_tokens": 100, "completion_tokens": 7, "total_tokens": 107}
    return events + [last]


class FakeServer:
    def __init__(self, script, delay=0):
        self.script, self.requests, self.delay, self.disconnects = list(script), [], delay, 0   # delay: seconds between chunks
        srv = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                b = b'{"status":"ok"}'
                self.send_response(200 if self.path == "/health" else 404)
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def do_POST(self):
                srv.requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                events = srv.script.pop(0) if srv.script else reply("(script exhausted)")
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                try:
                    for e in events:
                        self.wfile.write(("data: " + json.dumps(e) + "\n\n").encode())
                        self.wfile.flush()
                        time.sleep(srv.delay)
                    self.wfile.write(b"data: [DONE]\n\n")
                except (BrokenPipeError, ConnectionResetError):
                    srv.disconnects += 1
                self.close_connection = True

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.base_url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()
