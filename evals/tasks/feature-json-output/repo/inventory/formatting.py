"""Summaries of an inventory and their text rendering."""


def summarise(items):
    """Totals overall and per location (locations sorted by name)."""
    locs = {}
    for it in items:
        s = locs.setdefault(it.location, {"items": 0, "qty": 0, "value_pence": 0})
        s["items"] += 1
        s["qty"] += it.qty
        s["value_pence"] += it.qty * it.unit_price_pence
    return {
        "items": len(items),
        "total_qty": sum(it.qty for it in items),
        "total_value_pence": sum(it.qty * it.unit_price_pence for it in items),
        "by_location": dict(sorted(locs.items())),
    }


def pounds(pence):
    return f"{pence // 100}.{pence % 100:02d}"


def format_text(summary):
    lines = [f"{summary['items']} items, {summary['total_qty']} units, value £{pounds(summary['total_value_pence'])}"]
    for name, s in summary["by_location"].items():
        lines.append(f"  {name}: {s['items']} items, {s['qty']} units, £{pounds(s['value_pence'])}")
    return "\n".join(lines)
