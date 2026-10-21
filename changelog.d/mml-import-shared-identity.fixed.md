`facility import mml` no longer writes a wired channel under two device ids, which
`facility validate` refused. Where one device listed under several families would
resolve to a different id in each, the import stops with `mapping-invalid`,
naming the channel, the families, both ids and the fix. `devices:` takes
`{coordinates: <stem>}`, one device per `[sector, device]` row. `same_as` matches
by `[sector, device]` and may name a family with more devices, or one identified
by address or coordinates. The draft proposes one identity for families that list
the same devices.
