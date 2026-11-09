The three compose templates that elect one builder among services sharing an
image (bluesky, virtual_accelerator, qmd) go through `own.elect()` in
`services/_own_image.j2` instead of restating its predicate, and target
eligibility and the switch gate read `gateways: []` through the connectors'
one classifier. Renders and verdicts are unchanged.
