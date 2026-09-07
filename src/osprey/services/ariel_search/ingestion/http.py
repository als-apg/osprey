"""HTTP transport helpers shared by ingestion adapters.

One place decides what TLS verification an ingestion request gets, so an
adapter cannot ship its own quieter answer to that question.
"""

import ssl

__all__ = ["build_ssl_context"]


def build_ssl_context(verify_ssl: bool, ca_bundle: str | None = None) -> ssl.SSLContext | bool:
    """Return the ``ssl=`` argument for an ingestion request.

    Three outcomes, in the order a deployment meets them:

    * ``verify_ssl`` true and no ``ca_bundle`` — ``True``, which leaves aiohttp
      on the trust store the image ships. A site that installs its CA into that
      store therefore needs no ARIEL key at all.
    * ``verify_ssl`` true with a ``ca_bundle`` — a default context pinned to
      that file, for a site CA that lives beside the config rather than in the
      image.
    * ``verify_ssl`` false — verification off. This is an explicit opt-out an
      operator has to write; it is never what an unconfigured deployment gets.

    Args:
        verify_ssl: Whether certificates are verified at all.
        ca_bundle: Path to a PEM bundle to verify against, or ``None`` to use
            the trust store already present.

    Returns:
        ``True``, a verifying context built from *ca_bundle*, or a context with
        hostname checking and verification disabled.
    """
    if verify_ssl:
        if ca_bundle:
            return ssl.create_default_context(cafile=ca_bundle)
        return True

    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context
