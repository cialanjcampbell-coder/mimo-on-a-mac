"""Summarise Common Log Format access logs: top paths, status classes, malformed lines."""
import gzip
import re
import sys
from collections import Counter
from pathlib import Path

LINE = re.compile(r'^\S+ \S+ \S+ \[[^\]]+\] "(\S+) (\S+) (\S+)" (\d{3}) (\d+|-)$')


def lines(logdir):
    for p in sorted(Path(logdir).iterdir()):
        if not p.is_file() or not p.name.startswith("access.log"):
            continue
        opener = gzip.open if p.name.endswith(".gz") else open
        with opener(p, "rt", encoding="utf-8", errors="replace") as f:
            yield from f


def main(logdir):
    paths, classes, bad = Counter(), Counter(), 0
    for line in lines(logdir):
        line = line.rstrip("\r\n")
        if not line.strip():
            continue
        m = LINE.match(line)
        status = int(m.group(4)) if m else 0
        if not m or not 200 <= status <= 599:
            bad += 1
            continue
        paths[m.group(2).split("?", 1)[0]] += 1
        classes[status // 100] += 1
    print("top paths:")
    for path, n in sorted(paths.items(), key=lambda kv: (-kv[1], kv[0]))[:5]:
        print(f"  {n} {path}")
    print("status classes:")
    for c in (2, 3, 4, 5):
        print(f"  {c}xx {classes[c]}")
    print(f"malformed: {bad}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python3 report.py LOGDIR")
    main(sys.argv[1])
