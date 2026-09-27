"""Command line: python3 -m inventory ITEMS.csv [--format text|json]"""
import argparse

from .formatting import format_json, format_text, summarise
from .model import load_items


def main(argv=None):
    p = argparse.ArgumentParser(prog="inventory", description="Summarise an inventory CSV.")
    p.add_argument("csv", help="CSV with columns name,qty,unit_price,location")
    p.add_argument("--format", choices=["text", "json"], default="text", help="output format (default: text)")
    args = p.parse_args(argv)
    summary = summarise(load_items(args.csv))
    print(format_json(summary) if args.format == "json" else format_text(summary))
    return 0
