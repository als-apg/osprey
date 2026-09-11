**Breaking change:** the event-dispatch profile key `dispatch.pv_strip_prefix` is
now `dispatch.channel_strip_prefix` (container env `CHANNEL_STRIP_PREFIX`).
`dispatch:` is a closed block, so a profile that still sets the old name fails
the build with a message naming the key that replaced it. The whole `dispatch:`
block is now documented in the profile reference.
