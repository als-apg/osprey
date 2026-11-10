`tests/docs/test_cli_page_commands.py` reads every page under `docs/source`, not only the CLI
reference: each `osprey` command a page writes must name a registered verb, options the command
declares, and values its fixed `click.Choice` options accept.
