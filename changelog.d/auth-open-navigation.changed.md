**Upgrade note:** ``modules.web_terminals.auth.method: none`` now means an
**open** deployment. nginx stamps each user's operator secret onto every
request it proxies, so the landing page is pure navigation — a card opens its
terminal, with no login page and no per-user login URL. The posture that value
used to select is now spelled ``token``, and is still what an absent ``auth``
stanza means, so only a config that writes ``method: none`` explicitly changes
behaviour on upgrade. An open deployment is refused unless every persona denies
``Bash``, ``WebFetch``, ``WebSearch`` and Playwright, and the refusal names
``token`` as the way back.
