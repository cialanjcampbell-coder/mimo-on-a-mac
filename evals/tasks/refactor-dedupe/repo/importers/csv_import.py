"""Import people from CSV text."""
import csv
import io

HEADER = ["name", "email", "age"]


def _parse_record(fields, lineno):
    if len(fields) != 3:
        raise ValueError(f"line {lineno}: expected 3 fields, got {len(fields)}")
    name, email, raw_age = (f.strip() for f in fields)
    if not name:
        raise ValueError(f"line {lineno}: missing name")
    if "@" not in email or email.startswith("@") or email.endswith("@"):
        raise ValueError(f"line {lineno}: invalid email {email!r}")
    try:
        age = int(raw_age)
    except ValueError:
        raise ValueError(f"line {lineno}: invalid age {raw_age!r}") from None
    if not 0 <= age <= 150:
        raise ValueError(f"line {lineno}: invalid age {raw_age!r}")
    return {"name": name, "email": email.lower(), "age": age}


def import_csv(text):
    """Parse CSV text into records; skips blank lines and an optional header row."""
    out = []
    for lineno, fields in enumerate(csv.reader(io.StringIO(text)), 1):
        if not fields or all(not f.strip() for f in fields):
            continue
        if lineno == 1 and [f.strip().lower() for f in fields] == HEADER:
            continue
        out.append(_parse_record(fields, lineno))
    return out
