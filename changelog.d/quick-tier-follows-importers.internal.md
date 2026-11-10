The quick check tier now follows imports between test modules: a change to a
test module or a test package's `__init__.py` also runs every test that
imports it, directly or through a helper or conftest.
