Python submitted to the executor is no longer refused for merely *containing*
the letters `eval(` or `exec(`. `retrieval(...)`, a pandas `df.eval("a + b")`
and a PyTorch `model.eval()` all ran afoul of the substring check; the gate now
matches an actual call of the builtin. Real `eval`, `exec`, `__import__` and
`compile` calls are still refused.
