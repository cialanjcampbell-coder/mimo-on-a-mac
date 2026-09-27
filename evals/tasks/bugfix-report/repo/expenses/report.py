"""Monthly reports."""
from .dates import month_bounds


def monthly_totals(ledger, year, month):
    """Total pence per category for the month, plus a 'TOTAL' key."""
    first, last = month_bounds(year, month)
    totals = {}
    for e in ledger.entries_between(first, last):
        totals[e.category] = totals.get(e.category, 0) + e.amount_pence
    totals["TOTAL"] = sum(totals.values())
    return totals


def format_report(ledger, year, month):
    totals = monthly_totals(ledger, year, month)
    lines = [f"Expenses {year}-{month:02d}"]
    for cat in sorted(k for k in totals if k != "TOTAL"):
        lines.append(f"  {cat:<12} £{totals[cat] / 100:>9.2f}")
    lines.append(f"  {'TOTAL':<12} £{totals['TOTAL'] / 100:>9.2f}")
    return "\n".join(lines)
