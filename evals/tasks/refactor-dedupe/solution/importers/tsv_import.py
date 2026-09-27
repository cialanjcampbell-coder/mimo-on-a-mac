"""Import people from tab-separated text."""
from .records import parse_record

COLUMNS = ["name", "email", "age"]


def import_tsv(text):
    """Parse tab-separated text into records; skips blank lines and an optional header row."""
    records = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split("\t")
        if line_number == 1 and [p.strip().lower() for p in parts] == COLUMNS:
            continue
        records.append(parse_record(parts, line_number))
    return records
