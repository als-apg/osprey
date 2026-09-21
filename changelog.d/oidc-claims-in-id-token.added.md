Single sign-on can ask the identity provider to put the claims the login
reads into the ID token. New key
`modules.web_terminals.auth.oidc.claims_in_id_token: true` makes the login
route send the OIDC `claims` request parameter naming the identity claim
(essential), `email_verified` and any role-binding claim (voluntary). It is
the fix for a provider that follows OIDC Core §5.4 strictly and serves
scope-requested claims from UserInfo, which the sidecar never calls: every
login failed with "the ID token carries no usable claim" although the scope
was requested. Off by default; with it off the authorization request is
unchanged.
