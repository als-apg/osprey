``setup_inspect`` masks a secret written into a nested block. A key that names
a secret --- key, token, secret or password --- now masks every literal beneath
it rather than only a string directly under it, so a credential typed into a
nested part of ``config.yml`` or ``.mcp.json`` no longer reaches the
transcript, and an unquoted number under ``password`` is masked too. A
``${VAR}`` placeholder still shows which variable the value comes from, and a
key that merely contains one of those words inside a longer one (``keyword``,
``keystore``) is no longer treated as a secret.
