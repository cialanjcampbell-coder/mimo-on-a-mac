import unittest
from datetime import date

from expenses.ledger import Ledger
from expenses.report import format_report, monthly_totals


class TestReport(unittest.TestCase):
    def test_mid_month(self):
        led = Ledger()
        led.add(date(2026, 1, 10), 1250, "food")
        led.add(date(2026, 1, 12), 500, "travel")
        self.assertEqual(monthly_totals(led, 2026, 1), {"food": 1250, "travel": 500, "TOTAL": 1750})

    def test_format(self):
        led = Ledger()
        led.add(date(2026, 1, 10), 1250, "food")
        self.assertIn("food", format_report(led, 2026, 1))


if __name__ == "__main__":
    unittest.main()
