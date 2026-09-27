import unittest

from importers.csv_import import import_csv
from importers.tsv_import import import_tsv


class TestImporters(unittest.TestCase):
    def test_csv(self):
        self.assertEqual(import_csv("name,email,age\nAda,ada@x.org,36\n"),
                         [{"name": "Ada", "email": "ada@x.org", "age": 36}])

    def test_tsv(self):
        self.assertEqual(import_tsv("Ada\tada@x.org\t36\n"), [{"name": "Ada", "email": "ada@x.org", "age": 36}])

    def test_bad_age(self):
        with self.assertRaises(ValueError):
            import_csv("Ada,ada@x.org,old\n")


if __name__ == "__main__":
    unittest.main()
