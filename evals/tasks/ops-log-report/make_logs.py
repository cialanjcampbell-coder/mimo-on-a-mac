"""Generate the ops-log-report fixtures deterministically. Run from anywhere: python3 make_logs.py"""
import gzip
import random
from datetime import datetime, timedelta
from pathlib import Path

PATHS = ["/", "/index.html", "/about", "/api/items", "/api/items/42", "/login",
         "/static/app.js", "/static/site.css", "/search", "/health"]
WEIGHTS = [30, 12, 5, 20, 8, 6, 15, 14, 9, 4]
METHODS = ["GET"] * 8 + ["POST", "HEAD"]
STATUSES = [200] * 70 + [304] * 8 + [301] * 3 + [404] * 10 + [403] * 2 + [500] * 4 + [503] * 3
BAD = ['this is not a log line',
       '10.0.0.9 - - "GET /x HTTP/1.1" 200 1',
       '10.0.0.9 - - [01/Sep/2026:00:00:00 +0000] "GARBAGE" 200 1',
       '10.0.0.9 - - [01/Sep/2026:00:00:00 +0000] "GET /x HTTP/1.1" 20 1',
       '10.0.0.9 - - [01/Sep/2026:00:00:00 +0000] "GET /x HTTP/1.1" 700 1']
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def stamp(t):
    return f"{t.day:02d}/{MONTHS[t.month - 1]}/{t.year}:{t:%H:%M:%S} +0000"


def line(rng, t):
    ip = f"10.0.{rng.randrange(4)}.{rng.randrange(1, 255)}"
    path = rng.choices(PATHS, WEIGHTS)[0]
    if rng.random() < 0.15:
        path += f"?q={rng.randrange(100)}"
    size = rng.choice([str(rng.randrange(50, 50000)), "-"])
    return f'{ip} - - [{stamp(t)}] "{rng.choice(METHODS)} {path} HTTP/1.1" {rng.choice(STATUSES)} {size}'


def make(outdir, seed, names, per_file):
    rng = random.Random(seed)
    outdir.mkdir(parents=True, exist_ok=True)
    t = datetime(2026, 9, 1)
    for name in names:
        rows = []
        for _ in range(per_file):
            t += timedelta(seconds=rng.randrange(1, 90))
            rows.append(rng.choice(BAD) if rng.random() < 0.02 else line(rng, t))
        data = ("\n".join(rows) + "\n").encode()
        (outdir / name).write_bytes(gzip.compress(data, mtime=0) if name.endswith(".gz") else data)


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    set1 = ["access.log", "access.log.1", "access.log.2.gz", "access.log.3.gz"]
    make(here / "repo" / "logs", 1, set1, 400)
    make(here / "check" / "data" / "set1", 1, set1, 400)
    make(here / "check" / "data" / "set2", 2, ["access.log", "access.log.1.gz", "access.log.2.gz"], 250)
