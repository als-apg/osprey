A provider's request cap is now `requests_per_minute` on its `providers.yml` entry,
and the in-context channel finder and the benchmark ReAct loop pace their model
calls to it; a provider without the key is not paced. The shipped `cborg` entry
sets 18. A repository whose `providers.yml` was copied by an earlier `osprey init`
runs `cborg` unpaced until `osprey profile expand --providers` refreshes it.
