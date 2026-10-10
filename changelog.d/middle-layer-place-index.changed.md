In the middle-layer channel-finder index, a device whose place has no sibling
places — one at the machine itself, in an only-child place, or with no place —
now has the `DeviceList` row `[0, k]`, where it had `[n, 1]` by its position in
the family; no two rows of a family are equal. Each family's `setup` gains
`PlaceList`, every device's place beside its row. `list_channels` takes `place`
(a place id or path prefix) where it took `sectors`, the `device_map` table has
`place` and `place_index` where it had `sector`, and the explorer names each
group and filter chip by its place.
