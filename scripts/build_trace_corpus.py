#!/usr/bin/env python3
"""Build an agentic routing-trace corpus from local Claude Code coding transcripts.

  build_trace_corpus.py <model_dir> <out_dir> --train P1,P2,... --test P3,...
      [--max-tokens 6144] [--windows 3] [--budget 260000] [--skip-session ID]

Each transcript (~/.claude/projects/<coding project>/*.jsonl) becomes up to --windows sessions: consecutive
message windows rendered with MiMo's chat template, prefixed by the bundled agent's system prompt and tools
(agent.prompt.SYSTEM_PROMPT, agent.tools.schemas(ALL)), so the text looks like what scripts/mimo-agent sends.
Per token we record role: 1 for tokens the model would generate (assistant text, reasoning, tool calls),
0 for context (system, user, tool results). Output: <out_dir>/{train,test}/<project>__<session>__wN.npz with `ids`, `role`.
Everything stays on this machine. Only the projects named on the command line are used.
"""
import argparse, glob, importlib, json, os, random, re, sys
from pathlib import Path
import numpy as np


def load_tokenizer(path):
    """Load mlx-lm lazily so static analysis does not require the optional dependency."""
    try:
        tokenizer_utils = importlib.import_module("mlx_lm.tokenizer_utils")
    except ModuleNotFoundError as exc:
        if exc.name == "mlx_lm":
            raise RuntimeError("mlx-lm is required to run this script") from exc
        raise
    return tokenizer_utils.load(path)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for agent.*
from agent.prompt import SYSTEM_PROMPT
from agent.tools import ALL, schemas

HOME = os.path.expanduser("~")
PROJ_ROOT = f"{HOME}/.claude/projects"
# Project directory names under PROJ_ROOT are passed on the command line (--train / --test); the test
# split is held out by project. Use coding projects only.
TOOL_RESULT_CHARS = 6000
REMINDER = re.compile(r"<system-reminder>.*?</system-reminder>", re.S)


def text_of(content):
    if isinstance(content, str):
        return content
    out = []
    for b in content or []:
        if isinstance(b, dict) and b.get("type") == "text":
            out.append(b["text"])
    return "\n".join(out)


def transcript_messages(path):
    msgs = []
    for line in open(path):
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        m = o.get("message")
        if o.get("type") not in ("user", "assistant") or not isinstance(m, dict) or o.get("isSidechain"):
            continue
        c = m.get("content")
        if o["type"] == "user":
            if isinstance(c, list):
                for b in c:
                    if b.get("type") == "tool_result":
                        r = b.get("content")
                        r = r if isinstance(r, str) else text_of(r)
                        msgs.append({"role": "tool", "content": r[:TOOL_RESULT_CHARS]})
            t = REMINDER.sub("", text_of(c)).strip()
            if t and not t.startswith("[Request interrupted"):
                msgs.append({"role": "user", "content": t})
        else:
            blocks = c if isinstance(c, list) else [{"type": "text", "text": c or ""}]
            text = "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()
            think = "\n".join(b.get("thinking", "") for b in blocks if b.get("type") == "thinking").strip()
            calls = [{"type": "function", "function": {"name": b["name"], "arguments": b.get("input", {})}}
                     for b in blocks if b.get("type") == "tool_use"]
            if not (text or think or calls):
                continue
            a = {"role": "assistant", "content": text}
            if think:
                a["reasoning_content"] = think
            if calls:
                a["tool_calls"] = calls
            # merge consecutive assistant fragments (CC logs one block per line)
            if msgs and msgs[-1]["role"] == "assistant" and not msgs[-1].get("tool_calls"):
                prev = msgs.pop()
                a["content"] = (prev["content"] + "\n" + a["content"]).strip()
                if prev.get("reasoning_content"):
                    a["reasoning_content"] = (prev["reasoning_content"] + "\n" + a.get("reasoning_content", "")).strip()
            msgs.append(a)
    return msgs


def render_with_roles(tok, system_msgs, tools, msgs):
    """Render incrementally; tokens added by an assistant message get role 1."""
    ids, role = [], []
    prev = ""
    conv = list(system_msgs)
    for i, m in enumerate(msgs):
        conv.append(m)
        try:
            s = tok.apply_chat_template(conv, tools=tools, tokenize=False)
        except Exception:
            conv.pop()
            continue
        if not s.startswith(prev):  # template rewrote history (e.g. dropped old reasoning); restart diff
            prev = ""
            ids, role = [], []
            chunk = s
        else:
            chunk = s[len(prev):]
        t = tok.encode(chunk, add_special_tokens=False)
        ids += t
        role += [1 if m["role"] == "assistant" else 0] * len(t)
        prev = s
    return ids, role


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--max-tokens", type=int, default=6144)
    ap.add_argument("--windows", type=int, default=3)
    ap.add_argument("--budget", type=int, default=260000, help="total tokens across all sessions")
    ap.add_argument("--train", required=True, help="comma-separated project dir names under ~/.claude/projects")
    ap.add_argument("--test", required=True, help="comma-separated project dir names, held out")
    ap.add_argument("--skip-session", action="append", default=[], help="session id to exclude (repeatable)")
    a = ap.parse_args()
    tok = load_tokenizer(Path(a.model))
    sys_msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    tools = schemas(ALL)
    random.seed(0)
    stats = {"train": [0, 0, 0], "test": [0, 0, 0]}
    jobs = []
    for split, projs in (("train", a.train.split(",")), ("test", a.test.split(","))):
        for p in projs:
            for f in sorted(glob.glob(f"{PROJ_ROOT}/{p}/*.jsonl")):
                if any(sid in f for sid in a.skip_session):
                    continue
                jobs.append((split, p, f))
    random.shuffle(jobs)
    per_split_budget = {"train": int(a.budget * 0.6), "test": int(a.budget * 0.4)}
    for split, p, f in jobs:
        if stats[split][1] >= per_split_budget[split]:
            continue
        msgs = transcript_messages(f)
        if sum(m["role"] == "assistant" for m in msgs) < 3:
            continue
        # split the message list into windows that fit max_tokens
        start, w = 0, 0
        while start < len(msgs) and w < a.windows:
            lo, hi, best = start + 1, len(msgs), None
            while lo <= hi:  # largest window that fits
                mid = (lo + hi) // 2
                ids, role = render_with_roles(tok, sys_msgs, tools, msgs[start:mid])
                if len(ids) <= a.max_tokens:
                    best, lo = (mid, ids, role), mid + 1
                else:
                    hi = mid - 1
            if best is None:  # a single huge message; truncate
                ids, role = render_with_roles(tok, sys_msgs, tools, msgs[start:start + 1])
                best = (start + 1, ids[: a.max_tokens], role[: a.max_tokens])
            end, ids, role = best
            if sum(role) >= 64:
                d = os.path.join(a.out, split)
                os.makedirs(d, exist_ok=True)
                name = f"{p.split('-Documents-')[-1].strip('-')}__{Path(f).stem[:8]}__w{w}.npz"
                np.savez_compressed(os.path.join(d, name), ids=np.array(ids, np.int32), role=np.array(role, np.uint8))
                stats[split][0] += 1
                stats[split][1] += len(ids)
                stats[split][2] += sum(role)
                w += 1
            start = end
    for s, (n, t, g) in stats.items():
        print(f"{s}: {n} sessions, {t} tokens, {g} model-generated ({g / max(t, 1) * 100:.0f}%)")


if __name__ == "__main__":
    main()
