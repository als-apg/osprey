`osprey init --revalidate` treats `providers.yml` the way it treats
`profile.yml`: it keeps every value the file states, an edited entry OSPREY
ships included, and adds only the entries and keys OSPREY ships that the file
lacks. `osprey validate` compares each shipped entry key by key as preset
drift — a changed value is a note, a key or model one side lacks is refused
unless a deviation comment claims it — in place of the old whole-file note.
`osprey init --force` still resets the shipped entries.
