A provider that needs an endpoint and has no source for one now says so,
whichever way it is called. Previously only calls that went through OSPREY's
completion entry point were refused; a direct call to the provider adapter ran
on with no endpoint and failed inside the model client with an unrelated
authentication error. Affects `als-apg`, `amsc-i2` and `cborg`, whose endpoint
comes from `api.providers.<name>.base_url` or the provider's environment
variable.
