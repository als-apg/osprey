A failed `osprey init` no longer leaves the `.env` it seeded from your shell's
provider keys behind. The refusal already said "Nothing was materialized", but
the seeded secrets file survived it — and left the repo root non-empty, so the
next attempt refused too. An `.env` that was already there is still untouched.
