The unit lane's macOS cells run as one job each (`check.py run pytest --shard
macos`, an allowlist of the tests whose outcome can differ on macOS) instead of
three shard jobs, so a main push takes 2 of the account's 5 macOS slots and a
pull request 1.
