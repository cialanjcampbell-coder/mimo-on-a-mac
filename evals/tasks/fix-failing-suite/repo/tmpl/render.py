"""Minimal templates: {{name}} is HTML-escaped, {{!name}} is inserted raw; unknown names render as ''."""
import re

PLACEHOLDER = re.compile(r"\{\{(!?)(\w+)\}\}")


def escape(value):
    """HTML-escape &, < and >."""
    s = str(value)
    return s.replace("<", "&lt;").replace(">", "&gt;").replace("&", "&amp;")


def render(template, context, defaults={}):
    """Fill placeholders from context, falling back to defaults."""
    values = defaults
    values.update(context)

    def sub(m):
        raw, key = m.group(1), m.group(2)
        if key not in values:
            return ""
        return str(values[key]) if raw else escape(values[key])

    return PLACEHOLDER.sub(sub, template)
