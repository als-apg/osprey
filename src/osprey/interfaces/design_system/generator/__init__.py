"""Design-token generator: parse, validate, and emit CSS/JS token artifacts.

Pure-stdlib pipeline that loads the hand-authored DTCG JSON token sources
(``model.py``), validates them (``validate.py``), and emits the checked-in
``tokens.css``/``tokens.js``/``theme-boot.js`` artifacts (``emit_css.py``,
``emit_js.py``), orchestrated by ``build.py``.
"""
