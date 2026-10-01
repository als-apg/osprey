Web terminals and the login service now get each proxy setting that has a value as both
`HTTPS_PROXY` and `https_proxy` (and likewise for `HTTP_PROXY` and `NO_PROXY`), so tools that
read only the lowercase name, such as curl, use the site proxy too. A setting with no value is
no longer written as an empty variable.
