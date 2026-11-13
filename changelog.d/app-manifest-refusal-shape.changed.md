A schema error in an application's `app.yml` is reported as a violation with
its rule (`apps/<app>/app.yml: <field>: <message> [<rule>]`) and counted, by
`osprey app check`, `osprey build` and `osprey validate` alike, instead of as
a profile that does not resolve.
