The secret scan (gitleaks), the workflow lint (actionlint), the documentation
link check and the invariant tests are checks of `scripts/check.py`, run by its
tiers and by CI's lint and docs jobs; a broken documentation link now fails the
docs job instead of being reported and ignored.
