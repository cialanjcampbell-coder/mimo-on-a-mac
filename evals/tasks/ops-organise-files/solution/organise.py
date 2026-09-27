"""Sort files into DEST/YYYY/MM/ (or DEST/unsorted/) by the date in their name."""
import argparse
import re
import shutil
import sys
from datetime import date
from pathlib import Path

DATE = re.compile(r"(?<!\d)(\d{4})([-._]?)(\d{2})\2(\d{2})(?!\d)")


def find_date(name):
    for m in DATE.finditer(name):
        y, mo, d = int(m.group(1)), int(m.group(3)), int(m.group(4))
        if 1990 <= y <= 2099:
            try:
                return date(y, mo, d)
            except ValueError:
                pass
    return None


def free_target(path):
    if not path.exists():
        return path
    i = 1
    while True:
        cand = path.with_name(f"{path.stem}-{i}{path.suffix}")
        if not cand.exists():
            return cand
        i += 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true", help="print the plan, change nothing")
    p.add_argument("src")
    p.add_argument("dest")
    a = p.parse_args(argv)
    src, dest = Path(a.src), Path(a.dest)
    for f in sorted(src.iterdir(), key=lambda q: q.name):
        if f.name.startswith(".") or not f.is_file():
            continue
        d = find_date(f.name)
        folder = dest / f"{d.year:04d}" / f"{d.month:02d}" if d else dest / "unsorted"
        target = free_target(folder / f.name)
        print(f"move {f} -> {target}")
        if not a.dry_run:
            folder.mkdir(parents=True, exist_ok=True)
            shutil.move(str(f), str(target))
    return 0


if __name__ == "__main__":
    sys.exit(main())
