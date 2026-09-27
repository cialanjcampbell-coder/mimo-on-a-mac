"""Task pack: load tasks, set up scratch work dirs, run the hidden checks."""
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXCLUDES = [".eval-*", "__pycache__/", "*.pyc"]
GIT_ID = ["-c", "user.name=eval-task", "-c", "user.email=eval-task@localhost", "-c", "commit.gpgsign=false"]


def tasks_dir():
    return Path(os.environ.get("EVAL_TASKS_DIR", REPO / "evals" / "tasks"))


def scratch_dir():
    return Path(os.environ.get("EVAL_SCRATCH", Path.home() / "scratch" / "evals"))


@dataclass
class Task:
    name: str
    title: str
    category: str
    prompt: str
    tools: str
    protected: list
    dir: Path


def load_task(name):
    d = tasks_dir() / name
    f = d / "task.json"
    if not f.exists():
        raise KeyError(f"unknown task: {name}")
    j = json.loads(f.read_text())
    return Task(name, j["title"], j["category"], j["prompt"], j.get("tools", "read,edit,write"),
                list(j.get("protected", [])), d)


def list_tasks():
    root = tasks_dir()
    if not root.exists():
        return []
    return [load_task(p.name) for p in sorted(root.iterdir()) if (p / "task.json").exists()]


def git(d, *args, env=None, check=True):
    return subprocess.run(["git", "-C", str(d), *GIT_ID, *args], capture_output=True, text=True,
                          env=env, check=check).stdout


def read_meta(workdir):
    f = Path(workdir) / ".eval-task.json"
    if not f.exists():
        raise FileNotFoundError(f"{workdir} is not an eval-task work dir (no .eval-task.json)")
    return json.loads(f.read_text())


def write_meta(workdir, meta):
    (Path(workdir) / ".eval-task.json").write_text(json.dumps(meta, indent=1))


def setup(name, tag="manual", now=None):
    task = load_task(name)
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    base = scratch_dir() / f"{name}-{tag}-{stamp}"
    dest, i = base, 2
    while dest.exists():
        dest, i = base.with_name(f"{base.name}-{i}"), i + 1
    shutil.copytree(task.dir / "repo", dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    git(dest, "init", "-q")
    (dest / ".git" / "info").mkdir(parents=True, exist_ok=True)
    (dest / ".git" / "info" / "exclude").write_text("\n".join(EXCLUDES) + "\n")
    git(dest, "add", "-A")
    git(dest, "commit", "-q", "-m", "baseline")
    write_meta(dest, {"task": name, "tag": tag, "baseline": git(dest, "rev-parse", "HEAD").strip(),
                      "created": stamp})
    return dest


def diff(workdir, baseline, stat=True):
    """Work tree (including new files) vs baseline, using a throwaway index."""
    with tempfile.TemporaryDirectory() as t:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(t) / "index"))
        git(workdir, "read-tree", baseline, env=env)
        git(workdir, "add", "-A", env=env)
        return git(workdir, "diff", "--cached", baseline, *(["--stat"] if stat else []), env=env)


@dataclass
class CheckResult:
    passed: bool
    tests_run: int
    tests_failed: int
    protected_changed: list = field(default_factory=list)
    timed_out: bool = False
    output: str = ""
    diff_stat: str = ""

    def summary(self):
        s = f"{'PASS' if self.passed else 'FAIL'} ({self.tests_run - self.tests_failed}/{self.tests_run} checks)"
        if self.protected_changed:
            s += f"; protected files changed: {', '.join(self.protected_changed)}"
        if self.timed_out:
            s += "; timed out"
        return s


RAN_RE = re.compile(r"^Ran (\d+) tests?", re.M)
FAILED_RE = re.compile(r"^FAILED \((.*)\)", re.M)


def _count_failed(output):
    m = FAILED_RE.search(output)
    return sum(int(x) for x in re.findall(r"(?:failures|errors)=(\d+)", m.group(1))) if m else 0


def _protected_changed(workdir, baseline, files):
    changed = []
    for f in files:
        rc = subprocess.run(["git", "-C", str(workdir), "diff", "--quiet", baseline, "--", f],
                            capture_output=True).returncode
        if rc != 0 or not (Path(workdir) / f).exists():
            changed.append(f)
    return changed


def _stdlib_shadows(workdir):
    """Top-level names in the work dir that would shadow standard-library modules in the check."""
    names = set()
    for p in Path(workdir).iterdir():
        stem = p.stem if p.suffix == ".py" else (p.name if p.is_dir() else None)
        if stem and stem in sys.stdlib_module_names:
            names.add(stem)
    return sorted(names)


def _run_check(w, timeout):
    """Run the hidden tests in w; on timeout kill the whole process group. Returns (output, rc, timed_out)."""
    proc = subprocess.Popen([sys.executable, "-P", "-m", "unittest", "discover", "-s", "_eval_check", "-t", ".", "-v"],
                            cwd=w, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            start_new_session=True,
                            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8"))
    try:
        out, _ = proc.communicate(timeout=timeout)
        return out, proc.returncode, False
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        out, _ = proc.communicate()
        return f"check timed out after {timeout}s\n{out}", 1, True
    finally:
        try:
            os.killpg(proc.pid, signal.SIGKILL)  # anything the tests left running
        except ProcessLookupError:
            pass


def check(workdir, timeout=120, meta=None):
    """Run the hidden checks. Pass `meta` (from setup) to avoid trusting the work dir's own .eval-task.json."""
    workdir = Path(workdir)
    meta = meta or read_meta(workdir)
    task = load_task(meta["task"])
    shadows = _stdlib_shadows(workdir)
    if shadows:
        output, rc, timed_out = f"work dir shadows a standard-library module: {', '.join(shadows)}\n", 1, False
    else:
        with tempfile.TemporaryDirectory() as t:
            w = Path(t) / "w"
            shutil.copytree(workdir, w, ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc", ".eval-*",
                                                                      "_eval_check"))
            shutil.copytree(task.dir / "check", w / "_eval_check")
            output, rc, timed_out = _run_check(w, timeout)
    m = RAN_RE.search(output)
    run = int(m.group(1)) if m else 0
    failed = run if timed_out else _count_failed(output)
    protected = _protected_changed(workdir, meta["baseline"], task.protected)
    result = CheckResult(passed=(rc == 0 and run > 0 and not protected and not timed_out and not shadows),
                         tests_run=run, tests_failed=failed, protected_changed=protected,
                         timed_out=timed_out, output=output, diff_stat=diff(workdir, meta["baseline"]))
    (workdir / ".eval-check.txt").write_text(f"{output}\n{result.diff_stat}\n{result.summary()}\n")
    return result
