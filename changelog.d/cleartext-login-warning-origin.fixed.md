The lint warning about logins over plain HTTP now reads `external_origin`, the address browsers
use, instead of `deploy.fqdn` alone. It appears for a loopback `deploy.fqdn` behind a real
`http://` origin, and it no longer appears behind an `https://` terminator.
