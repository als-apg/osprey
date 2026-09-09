The installation page says how to install a pre-release. `uv tool install
osprey-framework` resolves to the newest stable release and skips betas, and a
version pin alone fails on the matching `osprey-connectors` pre-release;
`--prerelease allow` is the flag both need.
