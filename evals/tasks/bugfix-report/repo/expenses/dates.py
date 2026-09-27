"""Date helpers."""
from datetime import date, timedelta


def month_bounds(year, month):
    """Return (first_day, last_day) of the month, both inclusive."""
    first = date(year, month, 1)
    nxt = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    return first, nxt - timedelta(days=1)


def days_between(start, end):
    """All dates from start to end inclusive."""
    out = []
    d = start
    while d < end:
        out.append(d)
        d += timedelta(days=1)
    return out
