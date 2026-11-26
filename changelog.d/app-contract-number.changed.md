**Breaking change:** every `app.yml` declares `contract: 1`, the application
contract it is written for; `osprey build`, `osprey app check` and
`osprey validate` refuse a missing or different number, and `osprey build`
now runs the app import rules too.
