import io
import unittest

from tmpl import escape, render


class TestHidden(unittest.TestCase):
    def test_defaults_not_mutated(self):
        d = {"b": "2"}
        render("{{a}}{{b}}", {"a": "1"}, d)
        self.assertEqual(d, {"b": "2"})

    def test_calls_independent(self):
        render("{{q}}", {"q": "leak"})
        self.assertEqual(render("[{{q}}]", {}), "[]")

    def test_context_overrides_defaults(self):
        self.assertEqual(render("{{a}}", {"a": "ctx"}, {"a": "def"}), "ctx")

    def test_escape_order(self):
        self.assertEqual(escape("&lt;"), "&amp;lt;")
        self.assertEqual(escape("a<&>b"), "a&lt;&amp;&gt;b")

    def test_escape_non_string(self):
        self.assertEqual(escape(5), "5")

    def test_visible_suite_passes(self):
        suite = unittest.TestLoader().discover("tests", top_level_dir=".")
        result = unittest.TextTestRunner(stream=io.StringIO()).run(suite)
        self.assertTrue(result.wasSuccessful(), [str(f[0]) for f in result.failures + result.errors])
