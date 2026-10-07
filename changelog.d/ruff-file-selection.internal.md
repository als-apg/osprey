`ruff check` and `ruff format` check the whole repository in CI, in the local
check scripts and in the contributor docs, and extensionless Python scripts are
named in `extend-include`, so ruff selects the same files as the pre-commit hook.
