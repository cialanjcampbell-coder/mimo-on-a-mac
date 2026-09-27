"""Import people from CSV text."""
import csv
import io

from .records import parse_record

HEADER = ["name", "email", "age"]


def import_csv(text):
    """Parse CSV text into records; skips blank lines and an optional header row."""
    out = []
    for lineno, fields in enumerate(csv.reader(io.StringIO(text)), 1):
        if not fields or all(not f.strip() for f in fields):
            continue
        if lineno == 1 and [f.strip().lower() for f in fields] == HEADER:
            continue
        out.append(parse_record(fields, lineno))
    return out
