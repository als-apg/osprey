The unit lane's macOS cells run as one job each, over the whole sweep minus
the test files that only repeat ubuntu (`check.py run pytest --shard macos`),
instead of three shard jobs each. A main push now takes 2 of the account's 5
macOS slots and a pull request 1, where a main push alone wanted 6.
