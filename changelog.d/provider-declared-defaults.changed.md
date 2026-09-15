An adapter's declared default endpoint and its keyless posture now follow from
what the provider itself declares, instead of each adapter opting in by hand. A
provider added to a deployment cannot carry a default endpoint that is silently
ignored, and a provider that declares it needs no API key sends the placeholder
the endpoint expects without its adapter having to repeat it.
