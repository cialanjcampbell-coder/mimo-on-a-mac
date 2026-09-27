"""Inventory items loaded from CSV."""
import csv
from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class Item:
    name: str
    qty: int
    unit_price_pence: int
    location: str


def parse_price(text):
    """'12.50' -> 1250 pence."""
    return int((Decimal(text.strip()) * 100).to_integral_value())


def load_items(path):
    with open(path, newline="") as f:
        return [Item(r["name"].strip(), int(r["qty"]), parse_price(r["unit_price"]), r["location"].strip())
                for r in csv.DictReader(f)]
