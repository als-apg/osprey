The first ARIEL picture-lane failure in a server process no longer stalls
every request for seconds. Classifying a provider failure used to import
LiteLLM, httpx, requests and psycopg on the spot, on the event loop, when
nothing had loaded them yet; it now only looks at libraries already imported,
which cannot miss a match because an exception from a library implies the
library is loaded.
