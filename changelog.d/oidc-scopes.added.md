Single sign-on deployments can now name the scopes requested at the provider's
authorization endpoint, with `modules.web_terminals.auth.oidc.scopes`. The
previous fixed list (`openid profile email`) is still the default, so a
provider that publishes its identity claim under some other scope no longer
needs a code change. A list that drops `openid` is refused: without it there is
no ID token to check.
