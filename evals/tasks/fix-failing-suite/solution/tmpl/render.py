"""Minimal templates: {{name}} is HTML-escaped, {{!name}} is inserted raw; unknown names render as ''."""
import re

PLACEHOLDER = re.compile(r"\{\{(!?)(\w+)\}\}")


def escape(value):
    """HTML-escape &, < and > (& first, so entities aren't double-escaped)."""
    s = str(value)
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render(template, context, defaults=None):
    """Fill placeholders from context, falling back to defaults."""
    values = dict(defaults or {})
    values.update(context)

    def sub(m):
        raw, key = m.group(1), m.group(2)
        if key not in values:
            return ""
        return str(values[key]) if raw else escape(values[key])

    return PLACEHOLDER.sub(sub, template)
