import unittest

from tmpl import escape, render


class TestEscape(unittest.TestCase):
    def test_angle_brackets(self):
        self.assertEqual(escape("<b>"), "&lt;b&gt;")

    def test_ampersand(self):
        self.assertEqual(escape("fish & chips"), "fish &amp; chips")


class TestRender(unittest.TestCase):
    def test_a_simple(self):
        self.assertEqual(render("Hello {{name}}!", {"name": "Ada"}), "Hello Ada!")

    def test_b_escaped(self):
        self.assertEqual(render("{{x}}", {"x": "<i>"}), "&lt;i&gt;")

    def test_c_raw(self):
        self.assertEqual(render("{{!x}}", {"x": "<i>"}), "<i>")

    def test_d_missing_is_empty(self):
        self.assertEqual(render("Hello {{name}}!", {}), "Hello !")

    def test_e_defaults(self):
        self.assertEqual(render("{{a}}-{{b}}", {"a": "1"}, {"b": "2"}), "1-2")


if __name__ == "__main__":
    unittest.main()
