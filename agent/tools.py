"""The agent's tools: read, edit, write, grep, find, ls, bash. run(name, args) -> (text, is_error)."""
import fnmatch, os, re, shutil, signal, subprocess
from pathlib import Path

ALL = ("read", "edit", "write", "grep", "find", "ls", "bash")
MAX_LINES, MAX_BYTES = 2000, 50_000
MAX_RESULTS = 500
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv"}
REPO = str(Path(__file__).resolve().parent.parent)


def _fn(name, desc, props, required):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props, "required": required}}}


S, I = {"type": "string"}, {"type": "integer"}
SCHEMAS = {
    "read": _fn("read", "Read a text file; output lines are prefixed with their line number and a tab. Long files "
                "are truncated; page with offset/limit.",
                {"path": S, "offset": dict(I, description="first line (1-based)"), "limit": dict(I, description="max lines")},
                ["path"]),
    "edit": _fn("edit", "Replace old_text with new_text in a file. old_text must match exactly once (whitespace "
                "included, no line-number prefixes).", {"path": S, "old_text": S, "new_text": S},
                ["path", "old_text", "new_text"]),
    "write": _fn("write", "Create or overwrite a file (parent directories are created).",
                 {"path": S, "content": S}, ["path", "content"]),
    "grep": _fn("grep", "Search file contents with a regex; returns path:line:text matches.",
                {"pattern": S, "path": dict(S, description="file or directory (default .)"),
                 "glob": dict(S, description="only files matching this glob, e.g. *.py")}, ["pattern"]),
    "find": _fn("find", "Find files by glob pattern (e.g. *.py, src/**/test_*.py).",
                {"pattern": S, "path": dict(S, description="directory (default .)")}, ["pattern"]),
    "ls": _fn("ls", "List a directory (directories end with /).", {"path": dict(S, description="default .")}, []),
    "bash": _fn("bash", "Run a shell command in the working directory; returns combined stdout/stderr.",
                {"command": S, "timeout": dict(I, description="seconds (default 120)")}, ["command"]),
}


def schemas(names):
    return [SCHEMAS[n] for n in names]


def _path(p):
    return Path(os.path.expanduser(p or "."))


def _cap(text, keep_tail=False):
    """Truncate to MAX_LINES / MAX_BYTES, keeping the head (or the tail), with a note."""
    lines = text.splitlines()
    if len(lines) <= MAX_LINES and len(text.encode()) <= MAX_BYTES:
        return text
    lines = lines[-MAX_LINES:] if keep_tail else lines[:MAX_LINES]
    out = "\n".join(lines)
    while len(out.encode()) > MAX_BYTES:
        out = out[len(out) // 10:] if keep_tail else out[:len(out) * 9 // 10]
    return ("[output truncated; showing the end]\n" + out) if keep_tail else (out + "\n[output truncated]")


def read(path, offset=1, limit=None):
    lines = _path(path).read_text(errors="replace").splitlines()
    if not lines:
        return "(empty file)"
    start = max(int(offset or 1), 1) - 1
    if start >= len(lines):
        raise ValueError(f"offset {start + 1} is past the end of the file ({len(lines)} lines)")
    want = len(lines) if limit is None else min(len(lines), start + max(int(limit), 1))
    end = min(want, start + MAX_LINES)
    out, size = [], 0
    for i in range(start, end):
        s = f"{i + 1}\t{lines[i]}"
        if size + len(s) > MAX_BYTES and out:
            end = i
            break
        out.append(s)
        size += len(s) + 1
    if end < len(lines):
        what = "truncated: showing" if end < want else "showing"
        out.append(f"[{what} lines {start + 1}-{end} of {len(lines)}; use offset={end + 1} to continue]")
    return "\n".join(out)


def edit(path, old_text, new_text):
    p = _path(path)
    text = p.read_text()
    n = text.count(old_text) if old_text else 0
    if n != 1:
        raise ValueError(f"old_text must match exactly once in {path}, found {n} matches"
                         + ("; add surrounding lines to make it unique" if n else "; re-read the file and copy it exactly"))
    p.write_text(text.replace(old_text, new_text, 1))
    return f"Edited {path}"


def write(path, content):
    p = _path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"Wrote {len(content.encode())} bytes to {path}"


def _walk(root):
    """Files under root (relative paths as strings), skipping SKIP_DIRS."""
    root = _path(root)
    if root.is_file():
        yield str(root)
        return
    for d, dirs, files in os.walk(root):
        dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS)
        for f in sorted(files):
            yield os.path.normpath(os.path.join(d, f))


def _limit(lines, what):
    if not lines:
        return f"No {what} found"
    extra = f"\n[{len(lines) - MAX_RESULTS} more {what} not shown]" if len(lines) > MAX_RESULTS else ""
    return "\n".join(ln[:500] for ln in lines[:MAX_RESULTS]) + extra


def grep(pattern, path=None, glob=None):
    if shutil.which("rg"):
        cmd = ["rg", "-n", "--no-heading", "--color", "never", "--hidden", "-g", "!.git"] + (["-g", glob] if glob else [])
        r = subprocess.run(cmd + ["-e", pattern] + (["--", path] if path else []), capture_output=True, text=True,
                           errors="replace", stdin=subprocess.DEVNULL)
        if r.returncode > 1:
            raise ValueError(r.stderr.strip() or f"rg exited {r.returncode}")
        return _limit(r.stdout.splitlines(), "matches")
    rx, out = re.compile(pattern), []
    for f in _walk(path):
        if glob and not fnmatch.fnmatch(os.path.basename(f), glob) and not fnmatch.fnmatch(f, glob):
            continue
        try:
            with open(f, errors="replace") as fh:
                out += [f"{f}:{i}:{ln.rstrip()}" for i, ln in enumerate(fh, 1) if rx.search(ln)]
        except OSError:
            continue
        if len(out) > MAX_RESULTS:
            break
    return _limit(out, "matches")


def find(pattern, path="."):
    root = _path(path)
    if not root.is_dir():
        raise ValueError(f"not a directory: {path}")
    out = []
    for f in _walk(root):
        rel = os.path.relpath(f, root)
        if fnmatch.fnmatch(rel, pattern) or ("/" not in pattern and fnmatch.fnmatch(os.path.basename(f), pattern)) \
                or (pattern.startswith("**/") and fnmatch.fnmatch(rel, pattern[3:])):
            out.append(f)
    return _limit(out, "files")


def ls(path="."):
    ents = sorted(_path(path).iterdir(), key=lambda e: e.name)
    return _limit([e.name + ("/" if e.is_dir() else "") for e in ents], "entries") if ents else "(empty directory)"


def _env():
    """Our environment minus the agent's repo on PYTHONPATH (so it can't shadow the user's modules)."""
    env = dict(os.environ)
    pp = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p and os.path.abspath(p) != REPO]
    env.pop("PYTHONPATH", None)
    if pp:
        env["PYTHONPATH"] = os.pathsep.join(pp)
    return env


def bash(command, timeout=120):
    timeout = float(timeout or 120)
    p = subprocess.Popen(["/bin/sh", "-c", command], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, start_new_session=True, env=_env())
    try:
        out, timed_out = p.communicate(timeout=timeout)[0], False
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        out, timed_out = p.communicate()[0], True
    except BaseException:  # Ctrl-C / SIGTERM: don't leave the command running
        os.killpg(p.pid, signal.SIGKILL)
        p.wait()
        raise
    text = _cap(out.decode(errors="replace"), keep_tail=True).rstrip("\n")
    if timed_out:
        return (text + f"\n[timed out after {timeout:g}s; process killed]").lstrip("\n"), True
    if p.returncode:
        return (text + f"\n[exit code {p.returncode}]").lstrip("\n"), True
    return text or "(no output)", False


FUNCS = {"read": read, "edit": edit, "write": write, "grep": grep, "find": find, "ls": ls, "bash": bash}


def run(name, args, enabled=ALL):
    """Execute a tool call; never raises (except KeyboardInterrupt / SystemExit)."""
    if name not in FUNCS:
        return f"Unknown tool: {name}", True
    if name not in enabled:
        return f"Tool {name!r} is not enabled; available tools: {', '.join(enabled)}", True
    if not isinstance(args, dict):
        return "Tool arguments must be a JSON object", True
    missing = [k for k in SCHEMAS[name]["function"]["parameters"]["required"] if k not in args]
    if missing:
        return f"Missing required argument(s) for {name}: {', '.join(missing)}", True
    try:
        r = FUNCS[name](**args)
        return r if isinstance(r, tuple) else (r, False)
    except Exception as e:
        return f"{type(e).__name__}: {e}", True
