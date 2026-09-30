"""Anonymise run transcripts for blind review, and join the scores back afterwards."""
import json
import random
import re
import statistics
from pathlib import Path

from evalkit import tasks, transcript

RUBRIC = ["correctness", "tool_use", "grounding", "scope", "recovery", "efficiency", "explanation"]
MODEL_PATTERNS = [r"mimo", r"xiaomi", r"qwen", r"flash[\s_-]?next", r"mlx[\s_-]?serve", r"mlxserve",
                  r"reasoning_content"]
LABELS = "ABCDEFGH"


def anonymise(text, literals=()):
    for lit in sorted((x for x in literals if x), key=len, reverse=True):
        text = text.replace(lit, "<workdir>")
    for pat in MODEL_PATTERNS:
        text = re.sub(pat, "[model]", text, flags=re.IGNORECASE)
    return text


def key_path(out):
    return Path(f"{Path(out)}.key.json")


def make_blind(run_dirs, out, seed=None):
    rng = random.Random(seed)
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty")
    runs = {}
    for d in map(Path, run_dirs):
        meta = tasks.read_meta(d)
        runs.setdefault(meta["task"], {}).setdefault(meta.get("model", meta["tag"]), []).append((d, meta))
    key = {"labels": {}, "runs": {}}
    for task_name, by_model in sorted(runs.items()):
        models = sorted(by_model)
        rng.shuffle(models)
        labels = dict(zip(LABELS, models))
        key["labels"][task_name] = labels
        tdir = out / task_name
        tdir.mkdir(parents=True)
        (tdir / "PROMPT.md").write_text(tasks.load_task(task_name).prompt + "\n")
        for label, model in labels.items():
            ordered = sorted(by_model[model], key=lambda x: (x[1].get("repeat", 0), str(x[0])))
            for i, (d, meta) in enumerate(ordered, 1):
                rid = f"{label}-r{i}"
                rdir = tdir / rid
                rdir.mkdir()
                lits = [str(d.resolve()), str(d), d.name]
                events = transcript.load_events(d / ".eval-transcript.jsonl")
                body = transcript.render_markdown(events) if events else "(no transcript recorded)"
                (rdir / "transcript.md").write_text(anonymise(body, lits))
                (rdir / "diff.patch").write_text(anonymise(tasks.diff(d, meta["baseline"], stat=False), lits))
                chk = d / ".eval-check.txt"
                (rdir / "check.txt").write_text(anonymise(chk.read_text() if chk.exists() else "(not checked)", lits))
                key["runs"][f"{task_name}/{rid}"] = {"model": model, "workdir": str(d)}
    key_path(out).write_text(json.dumps(key, indent=1))
    return key


def _mean(values):
    v = [x for x in values if isinstance(x, (int, float))]
    return round(statistics.fmean(v), 2) if v else ""


def unblind(out):
    out = Path(out)
    key = json.loads(key_path(out).read_text())
    scores = json.loads((out / "scores.json").read_text())["runs"]
    unknown = sorted(set(scores) - set(key["runs"]))
    if unknown:
        raise KeyError(f"scores for unknown runs: {unknown}")
    header = "| run | model | " + " | ".join(RUBRIC) + " | notes |"
    lines = [header, "|" + "---|" * (len(RUBRIC) + 3)]
    by_model = {}
    for rid in sorted(scores):
        s, model = scores[rid], key["runs"][rid]["model"]
        by_model.setdefault(model, []).append(s)
        lines.append(f"| {rid} | {model} | " + " | ".join(str(s.get(k, "")) for k in RUBRIC)
                     + f" | {s.get('notes', '')} |")
    lines += ["", "| model | runs | " + " | ".join(RUBRIC) + " | mean |", "|" + "---|" * (len(RUBRIC) + 3)]
    for model, ss in sorted(by_model.items()):
        means = [_mean(x.get(k) for x in ss) for k in RUBRIC]
        overall = _mean(m for m in means if m != "")
        lines.append(f"| {model} | {len(ss)} | " + " | ".join(str(m) for m in means) + f" | {overall} |")
    text = "\n".join(lines) + "\n"
    (out / "unblinded.md").write_text(text)
    return text
