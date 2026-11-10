`osprey init --revalidate` no longer adds a provider entry or key OSPREY ships
that a deviation marker in `providers.yml` claims absent
(`# DEVIATION: providers.openrouter: …`), so `osprey validate` stops warning
that the marker marks no difference.
