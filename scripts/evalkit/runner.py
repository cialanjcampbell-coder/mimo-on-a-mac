"""Headless agent runs of eval tasks: setup -> mimo-agent -p --json -> check -> record. Local only."""
import json
import os
import signal
import statistics
import subprocess
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import triallib
from evalkit import tasks, transcript

MODEL = "mimo"  # the record's model label; the only model served
HEALTH = "http://127.0.0.1:8082/health"
LOG = "mimo"  # key into triallib.SERVER_LOGS
START = "scripts/mimo-server-ctl start"
BASE_TOOLS = "read,edit,write,grep,find,ls"  # read-only discovery, no shell


def results_path():
    return Path(os.environ.get("EVAL_RESULTS", Path.home() / ".local/state/evals/runs.jsonl"))


def agent_bin():
    return os.environ.get("EVAL_AGENT_BIN", str(tasks.REPO / "scripts" / "mimo-agent"))


def server_healthy(url, timeout=3):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except OSError:
        return False


def build_command(agent, task, thinking="on", allow_bash=False):
    tools = BASE_TOOLS + (",bash" if allow_bash else "")
    return [agent, "-p", task.prompt, "--json", "--no-start", "--tools", tools, "--thinking", thinking]


def _wait(proc, timeout):
    """Wait for proc; on timeout kill its whole process group. Returns (exit code, timed_out)."""
    try:
        return proc.wait(timeout=timeout), False
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            return proc.wait(timeout=10), True
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            return proc.wait(), True


def run_once(name, repeat, thinking="on", allow_bash=False, timeout=900, logs=None):
    logs = logs or triallib.SERVER_LOGS
    log_path = Path(logs[LOG])
    task = tasks.load_task(name)
    workdir = tasks.setup(name, tag=f"{MODEL}-r{repeat}")
    meta = tasks.read_meta(workdir)
    meta.update(model=MODEL, repeat=repeat, thinking=thinking, allow_bash=allow_bash)
    tasks.write_meta(workdir, meta)
    offset = triallib.log_size(log_path)
    cmd = build_command(agent_bin(), task, thinking, allow_bash)
    t0 = time.monotonic()
    with open(workdir / ".eval-transcript.jsonl", "wb") as out, open(workdir / ".eval-stderr.txt", "wb") as err:
        proc = subprocess.Popen(cmd, cwd=workdir, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                start_new_session=True)
        rc, timed_out = _wait(proc, timeout)
    wall = time.monotonic() - t0
    try:
        result, check_error = tasks.check(workdir, meta=meta), None
    except Exception as e:  # a work dir the agent wrecked (e.g. deleted .git) is a FAIL, not a crash
        result, check_error = tasks.CheckResult(False, 0, 0, output=repr(e)), repr(e)
    reqs, _ = triallib.parse_log_text(triallib.read_since(log_path, offset))
    rt = triallib.summarize_requests(reqs)
    ts = transcript.stats(transcript.load_events(workdir / ".eval-transcript.jsonl"))
    stat_lines = result.diff_stat.strip().splitlines()
    return {"date": datetime.now().isoformat(timespec="seconds"), "task": name, "model": MODEL,
            "repeat": repeat, "thinking": thinking, "tools": cmd[cmd.index("--tools") + 1],
            "pass": result.passed, "checks_passed": result.tests_run - result.tests_failed,
            "checks_total": result.tests_run, "protected_changed": result.protected_changed,
            "wall_s": round(wall, 1), "timed_out": timed_out, "agent_exit": rc,
            "diff_stat": stat_lines[-1].strip() if stat_lines else "",
            "diff_files": [ln.split("|")[0].strip() for ln in stat_lines[:-1]],
            "generated_tokens": rt["generated_tokens"], "decode_tps": rt["decode_tps"],
            "requests": rt["requests"], "turns": ts["turns"], "tool_calls": ts["tool_calls"],
            "tool_errors": ts["tool_errors"], "stop_reason": ts["stop_reason"], "workdir": str(workdir),
            **({"check_error": check_error} if check_error else {})}


def run(names, repeats=3, thinking="on", allow_bash=False, timeout=900, check_server=True,
        logs=None, echo=print):
    if check_server and not server_healthy(HEALTH):
        raise RuntimeError(f"the MiMo server is not up; start it with: {START}")
    rp = results_path()
    rp.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for name in names:
        for i in range(1, repeats + 1):
            rec = run_once(name, i, thinking, allow_bash, timeout, logs)
            with open(rp, "a") as f:
                f.write(json.dumps(rec) + "\n")
            records.append(rec)
            echo(f"{name} r{i}: {'PASS' if rec['pass'] else 'FAIL'} ({rec['checks_passed']}/{rec['checks_total']})"
                 f" {rec['wall_s']}s, {rec['turns']} turns, {rec['decode_tps']} tok/s"
                 f"{' TIMEOUT' if rec['timed_out'] else ''}"
                 f"{' PROTECTED CHANGED: ' + ','.join(rec['protected_changed']) if rec['protected_changed'] else ''}"
                 f"  {rec['workdir']}")
    return records


def _median(values):
    v = [x for x in values if isinstance(x, (int, float))]
    return round(statistics.median(v), 1) if v else None


def summarize_runs(records):
    groups = {}
    for r in records:
        groups.setdefault((r["task"], r["model"]), []).append(r)
    lines = ["| task | model | runs | pass rate | median wall s | median decode tok/s | median turns |",
             "|---|---|---|---|---|---|---|"]
    for (task, model), rs in sorted(groups.items()):
        n, ok = len(rs), sum(1 for r in rs if r["pass"])
        lines.append(f"| {task} | {model} | {n} | {ok}/{n} ({100 * ok / n:.0f}%) | "
                     f"{_median(r['wall_s'] for r in rs)} | {_median(r['decode_tps'] for r in rs)} | "
                     f"{_median(r['turns'] for r in rs)} |")
    return "\n".join(lines) + "\n"


def save_results(records, results_dir):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    base = f"{datetime.now():%Y-%m-%d}-evalrun-{MODEL}"
    stem, i = base, 2
    while (results_dir / f"{stem}.jsonl").exists():
        stem, i = f"{base}-{i}", i + 1
    jl, md = results_dir / f"{stem}.jsonl", results_dir / f"{stem}.md"
    jl.write_text("".join(json.dumps(r) + "\n" for r in records))
    md.write_text(f"# Eval run: {MODEL} ({datetime.now():%Y-%m-%d %H:%M})\n\n{summarize_runs(records)}\n"
                  f"Records: `{jl.name}`. Made with `scripts/eval-task run`.\n")
    return jl, md
