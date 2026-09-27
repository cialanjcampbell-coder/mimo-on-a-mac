"""Import people from tab-separated text."""

COLUMNS = ["name", "email", "age"]


def _record_from_fields(parts, line_number):
    if len(parts) != 3:
        raise ValueError(f"line {line_number}: expected 3 fields, got {len(parts)}")
    name = parts[0].strip()
    email = parts[1].strip()
    age_text = parts[2].strip()
    if not name:
        raise ValueError(f"line {line_number}: missing name")
    if "@" not in email or email.startswith("@") or email.endswith("@"):
        raise ValueError(f"line {line_number}: invalid email {email!r}")
    try:
        age = int(age_text)
    except ValueError:
        raise ValueError(f"line {line_number}: invalid age {age_text!r}") from None
    if age < 0 or age > 150:
        raise ValueError(f"line {line_number}: invalid age {age_text!r}")
    return {"name": name, "email": email.lower(), "age": age}


def import_tsv(text):
    """Parse tab-separated text into records; skips blank lines and an optional header row."""
    records = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split("\t")
        if line_number == 1 and [p.strip().lower() for p in parts] == COLUMNS:
            continue
        records.append(_record_from_fields(parts, line_number))
    return records
