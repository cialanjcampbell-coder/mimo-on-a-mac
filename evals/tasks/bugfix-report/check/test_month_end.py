import io
import unittest
from datetime import date

from expenses.dates import month_bounds
from expenses.ledger import Ledger
from expenses.report import monthly_totals


class TestMonthEnd(unittest.TestCase):
    def test_last_day_of_january_counted(self):
        led = Ledger()
        led.add(date(2026, 1, 31), 4000, "groceries")
        self.assertEqual(monthly_totals(led, 2026, 1)["TOTAL"], 4000)

    def test_leap_february(self):
        led = Ledger()
        led.add(date(2028, 2, 29), 100, "x")
        led.add(date(2028, 3, 1), 999, "x")
        self.assertEqual(monthly_totals(led, 2028, 2), {"x": 100, "TOTAL": 100})

    def test_december(self):
        led = Ledger()
        led.add(date(2026, 12, 31), 700, "gifts")
        led.add(date(2027, 1, 1), 1, "gifts")
        self.assertEqual(monthly_totals(led, 2026, 12)["TOTAL"], 700)

    def test_neighbouring_months_excluded(self):
        led = Ledger()
        led.add(date(2026, 4, 1), 10, "a")
        led.add(date(2026, 3, 31), 20, "a")
        led.add(date(2026, 5, 1), 40, "a")
        self.assertEqual(monthly_totals(led, 2026, 4)["TOTAL"], 10)

    def test_single_day_range(self):
        led = Ledger()
        led.add(date(2026, 6, 15), 300, "a")
        self.assertEqual(len(led.entries_between(date(2026, 6, 15), date(2026, 6, 15))), 1)

    def test_month_bounds_unchanged(self):
        self.assertEqual(month_bounds(2026, 2), (date(2026, 2, 1), date(2026, 2, 28)))

    def test_empty_month(self):
        self.assertEqual(monthly_totals(Ledger(), 2026, 7), {"TOTAL": 0})

    def test_visible_suite_passes(self):
        suite = unittest.TestLoader().discover("tests", top_level_dir=".")
        result = unittest.TextTestRunner(stream=io.StringIO()).run(suite)
        self.assertTrue(result.wasSuccessful())
