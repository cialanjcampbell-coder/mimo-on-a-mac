"""Command line: python3 -m inventory ITEMS.csv"""
import argparse

from .formatting import format_text, summarise
from .model import load_items


def main(argv=None):
    p = argparse.ArgumentParser(prog="inventory", description="Summarise an inventory CSV.")
    p.add_argument("csv", help="CSV with columns name,qty,unit_price,location")
    args = p.parse_args(argv)
    print(format_text(summarise(load_items(args.csv))))
    return 0
