# inventory
Summarise an inventory CSV (columns `name,qty,unit_price,location`; prices in pounds).

    python3 -m inventory data/sample.csv

Output: overall totals, then one line per location.

Options:
- `--format text` (default): the human-readable summary above.
- `--format json`: one JSON object: `{"items", "total_qty", "total_value", "locations": [{"name", "items", "qty", "value"}]}`; money values are strings in pounds.
