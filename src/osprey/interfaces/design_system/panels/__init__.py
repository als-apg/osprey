"""Panel descriptors for the design system.

A *panel* is a self-contained HTML mini-app the web terminal can mount
alongside the chat surface. Each panel ships a small JSON **manifest**
(``manifest.py``) declaring its stable ``id``, human ``label``, and HTML
``entry`` point — the single shared contract consumed by the panel
validator, the reference panel, and (later) runtime panel discovery.
"""
