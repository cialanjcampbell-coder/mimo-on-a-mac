"""Ledger of expense entries."""
from dataclasses import dataclass
from datetime import date

from .dates import days_between


@dataclass(frozen=True)
class Entry:
    day: date
    amount_pence: int
    category: str


class Ledger:
    def __init__(self):
        self._entries = []

    def add(self, day, amount_pence, category):
        if amount_pence <= 0:
            raise ValueError("amount must be positive")
        self._entries.append(Entry(day, amount_pence, category))

    def entries_between(self, start, end):
        """Entries dated from start to end, inclusive, in insertion order."""
        wanted = set(days_between(start, end))
        return [e for e in self._entries if e.day in wanted]
