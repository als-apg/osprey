A `web_terminals` OIDC scope list carrying a blank entry is now refused, with
the offending index named. The blank entry used to be dropped and the remaining
scopes shipped as if they were what you had written.
