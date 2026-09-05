The new `osprey scaffold personas --from PRESET` command writes one
`personas/<name>.yml` file per persona in another preset's catalog into the
current deployment repo, then repoints the repo's own catalog at those
files. Use it to bring a preset's personas into a repo that was built from a
different preset.
