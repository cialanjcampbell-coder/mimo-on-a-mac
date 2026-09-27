import ast
import io
import unittest
from pathlib import Path

from importers.csv_import import import_csv
from importers.tsv_import import import_tsv

GOOD = [{"name": "Ada", "email": "ada@x.org", "age": 36}, {"name": "Bob", "email": "b@y.io", "age": 0}]


class TestBehaviour(unittest.TestCase):
    def test_valid_both(self):
        self.assertEqual(import_csv("name,email,age\nAda, ADA@X.ORG ,36\n\nBob,b@y.io,0\n"), GOOD)
        self.assertEqual(import_tsv("name\temail\tage\nAda\t ADA@X.ORG \t36\n\nBob\tb@y.io\t0\n"), GOOD)

    def assertErr(self, fn, text, msg):
        with self.assertRaises(ValueError) as cm:
            fn(text)
        self.assertEqual(str(cm.exception), msg)

    def test_errors_identical_in_both(self):
        for fn, sep in ((import_csv, ","), (import_tsv, "\t")):
            j = sep.join
            with self.subTest(fn=fn.__name__):
                self.assertErr(fn, j(["Ada", "a@b.c"]), "line 1: expected 3 fields, got 2")
                self.assertErr(fn, j([" ", "a@b.c", "3"]), "line 1: missing name")
                self.assertErr(fn, j(["Ada", "nope", "3"]), "line 1: invalid email 'nope'")
                self.assertErr(fn, j(["Ada", "@b.c", "3"]), "line 1: invalid email '@b.c'")
                self.assertErr(fn, j(["Ada", "a@b.c", "x"]), "line 1: invalid age 'x'")
                self.assertErr(fn, j(["Ada", "a@b.c", "151"]), "line 1: invalid age '151'")
                self.assertErr(fn, j(["Ada", "a@b.c", "-1"]), "line 1: invalid age '-1'")
                text = "\n".join([j(["name", "email", "age"]), j(["Ada", "a@b.c", "1"]), "", j(["Bo", "bad", "2"])])
                self.assertErr(fn, text, "line 4: invalid email 'bad'")


class TestStructure(unittest.TestCase):
    def parse(self, path):
        tree = ast.parse(Path(path).read_text())
        funcs = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        return funcs, tree

    def test_records_defines_parse_record(self):
        funcs, _ = self.parse("importers/records.py")
        self.assertIn("parse_record", funcs)

    def test_importers_use_shared_parser_only(self):
        for mod, public in (("importers/csv_import.py", "import_csv"), ("importers/tsv_import.py", "import_tsv")):
            with self.subTest(mod=mod):
                funcs, tree = self.parse(mod)
                self.assertEqual(funcs, {public}, f"{mod} still defines {sorted(funcs - {public})}")
                names = ({n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
                         | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
                         | {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names})
                self.assertIn("parse_record", names)

    def test_visible_suite_passes(self):
        suite = unittest.TestLoader().discover("tests", top_level_dir=".")
        result = unittest.TextTestRunner(stream=io.StringIO()).run(suite)
        self.assertTrue(result.wasSuccessful())
