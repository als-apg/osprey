`osprey init --revalidate` treats `providers.yml` the way it treats
`profile.yml`: it keeps every value the file states, an edited entry OSPREY
ships included, and adds only the entries and keys OSPREY has added since that
the file lacks. `osprey init --force` still resets the shipped entries.
