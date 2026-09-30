"""mimo-agent: a small coding agent for the local MiMo server (engine/server.py).

  python -m agent                      interactive (/reset clears the conversation, /exit quits)
  python -m agent -p "fix the tests"   headless: run to completion, print the final answer
  python -m agent -p ... --json        headless, {"type":"message_end","message":...} events on stdout
"""
import argparse, json, os, re, signal, subprocess, sys, time, urllib.request
from pathlib import Path

from agent import tools
from agent.loop import Agent

DEFAULT_BASE = "http://127.0.0.1:8082/v1"
REPO = Path(__file__).resolve().parent.parent
TTY = sys.stdout.isatty()


def style(code, s):
    return f"\x1b[{code}m{s}\x1b[0m" if TTY else s


def healthy(base):
    try:
        with urllib.request.urlopen(re.sub(r"/v1$", "", base.rstrip("/")) + "/health", timeout=3) as r:
            return r.status == 200
    except (OSError, ValueError):
        return False


def ensure_server(base):
    """Start the local MiMo server (scripts/mimo-server-ctl start) if it's down and base is the default URL."""
    if base.rstrip("/") != DEFAULT_BASE or healthy(base):
        return
    print("[mimo-agent] server not running; starting it with scripts/mimo-server-ctl start", file=sys.stderr)
    try:
        subprocess.run([str(REPO / "scripts" / "mimo-server-ctl"), "start"], stdin=subprocess.DEVNULL,
                       stdout=sys.stderr, check=False)
    except OSError as e:
        print(f"[mimo-agent] could not run mimo-server-ctl: {e}", file=sys.stderr)
        return
    t0 = last = time.time()
    while time.time() - t0 < 300:
        if healthy(base):
            print(f"[mimo-agent] server up after {time.time() - t0:.0f}s", file=sys.stderr)
            return
        if time.time() - last >= 10:
            print(f"[mimo-agent] waiting for server ({time.time() - t0:.0f}s)", file=sys.stderr)
            last = time.time()
        time.sleep(1)
    print("[mimo-agent] server did not become healthy within 300s", file=sys.stderr)


def call_line(name, args):
    main = next((args[k] for k in ("command", "path", "pattern") if isinstance(args.get(k), str)), "")
    extra = f" ({args['pattern']} in {args['path']})" if name in ("grep", "find") and "path" in args else ""
    s = f"{name} {main.splitlines()[0] if main else ''}{extra}".rstrip()
    return s if len(s) <= 120 else s[:117] + "..."


def exit_code(msg):
    return {"stop": 0, "length": 0, "aborted": 130}.get(msg["stopReason"], 1)


def headless(a):
    def emit(m):
        if a.json:
            sys.stdout.write(json.dumps({"type": "message_end", "message": m}, ensure_ascii=False) + "\n")
            sys.stdout.flush()
        elif m["role"] == "toolResult" and m["isError"]:
            print(f"  ! {m['content'][0]['text'].splitlines()[0][:200] if m['content'][0]['text'] else 'error'}",
                  file=sys.stderr)

    def on_tool(name, args):
        if not a.json:
            print(f"> {call_line(name, args)}", file=sys.stderr, flush=True)

    agent = Agent(a.base_url, a.model, a.tools, a.thinking == "on", a.max_turns, emit=emit, on_tool_start=on_tool)
    msg = agent.run(a.prompt)
    if not a.json:
        text = "".join(c["text"] for c in msg["content"] if c["type"] == "text")
        if text:
            print(text.strip())
    if msg.get("errorMessage"):
        print(f"[mimo-agent] error: {msg['errorMessage']}", file=sys.stderr)
    elif msg["stopReason"] == "length":
        print("[mimo-agent] warning: the reply hit the max_tokens limit", file=sys.stderr)
    return exit_code(msg)


def interactive(a):
    try:
        import readline  # noqa: F401  (line editing + history for input())
    except ImportError:
        pass
    state = {"kind": None}

    def on_delta(kind, text):
        if state["kind"] and state["kind"] != kind:
            sys.stdout.write("\n\n" if kind == "content" else "\n")
        state["kind"] = kind
        sys.stdout.write(style("2", text) if kind == "reasoning" else text)
        sys.stdout.flush()

    def on_tool(name, args):
        if state["kind"]:
            print()
        state["kind"] = None
        print(style("36", f"● {call_line(name, args)}"), flush=True)

    def emit(m):
        if m["role"] == "toolResult":
            text = m["content"][0]["text"]
            lines = text.splitlines() or [""]
            summary = lines[0][:100] + (f"  (+{len(lines) - 1} lines)" if len(lines) > 1 else "")
            print(style("31" if m["isError"] else "2", f"  {summary}"), flush=True)
        elif m["role"] == "assistant" and m["stopReason"] in ("error", "aborted", "length"):
            print(style("31", f"\n[{m['stopReason']}{': ' + m['errorMessage'] if m.get('errorMessage') else ''}]"))

    agent = Agent(a.base_url, a.model, a.tools, a.thinking == "on", a.max_turns, emit=emit, on_delta=on_delta,
                  on_tool_start=on_tool)
    print(style("2", f"mimo-agent · {a.model} · {os.getcwd()} · /reset, /exit, Ctrl-C aborts a turn"))
    while True:
        try:
            line = input(style("1", "> ") if TTY else "> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line in ("/exit", "/quit"):
            return 0
        if line == "/reset":
            agent.reset()
            print(style("2", "[conversation cleared]"))
            continue
        state["kind"] = None
        try:
            agent.run(line)
        except KeyboardInterrupt:
            print(style("31", "\n[aborted]"))
        print("\n" if state["kind"] else "", end="")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mimo-agent", description="Small coding agent for the local MiMo server.")
    ap.add_argument("-p", "--prompt", help="run this task headless, then exit")
    ap.add_argument("--json", action="store_true", help="headless: JSON message_end events on stdout")
    ap.add_argument("--tools", default=",".join(tools.ALL), help="comma-separated subset of " + ",".join(tools.ALL))
    ap.add_argument("--thinking", choices=("on", "off"), default="on")
    ap.add_argument("--base-url", default=DEFAULT_BASE)
    ap.add_argument("--model", default="mimo-v2.6-flash")
    ap.add_argument("--max-turns", type=int, default=100)
    ap.add_argument("--no-start", action="store_true", help="don't auto-start the local server")
    a = ap.parse_args(argv)
    a.tools = [t for t in (s.strip() for s in a.tools.split(",")) if t]
    bad = [t for t in a.tools if t not in tools.ALL]
    if bad:
        ap.error(f"unknown tool(s): {', '.join(bad)} (choose from {', '.join(tools.ALL)})")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))  # so a running bash tool's process group is killed
    if not a.no_start:
        ensure_server(a.base_url)
    return headless(a) if a.prompt is not None else interactive(a)


if __name__ == "__main__":
    sys.exit(main())
