A Bluesky arm refused for a missing or wrong launch token now answers
`launch_token_invalid`; `launch_token_required` means the deployment has no
launch token configured at all. One shared check, `osprey.utils.launch_token`,
gives the same coded answer on every path.
