"""How a bearer credential is prepared for a constant-time comparison.

One rule, one module, because it is applied at every gate that weighs a
presented bearer against a configured one — the dispatcher's dashboard routes
and MCP transport, the webhook source, the dispatch worker's API. A gate that
spelled it for itself would be the gate that answers 500 where the others
answer 401.

A configured secret has to be ASCII to be usable. The two sides of a gate reach
it by different decodes — a header arrives as the latin-1 text the ASGI server
read, an environment secret as utf-8 with ``surrogateescape`` — and those agree
on ASCII and nowhere else, so a non-ASCII secret is a secret no caller can
present. Such a deployment is refused rather than admitted, which is the safe
end of that asymmetry, and it is why every gate here is a refusal and never a
crash.
"""

from __future__ import annotations

__all__ = ["credential_bytes"]


def credential_bytes(value: str) -> bytes:
    """Encode one side of a bearer comparison for ``compare_digest``.

    ``compare_digest`` refuses a ``str`` argument that is not ASCII-only, and
    both a configured secret and a presented bearer are arbitrary text — a
    header carries whatever bytes a caller chose — so every comparison is made
    on bytes.

    ``surrogatepass`` rather than ``surrogateescape`` because this encode must
    be total. Both handlers carry the lone LOW surrogates ``os.environ`` hands
    back for a secret whose bytes are not valid UTF-8; only ``surrogatepass``
    also carries a lone HIGH surrogate, which no current caller produces but
    which any future one decoding from JSON could. A raise on this path is a
    500 on *every* request that presents a bearer — an unreachable service
    rather than a refused caller — so the encoding is chosen so that there is
    no input at all it can refuse.

    The bytes are for comparison only. They are the same bytes for the same
    text, which is all a comparison needs; they are not a faithful round-trip
    of a secret whose bytes were never valid UTF-8, and nothing reads them back.

    Args:
        value: A presented or configured credential, as text.

    Returns:
        The same credential as bytes. Defined for every ``str``, so neither
        operand of the comparison can raise.
    """
    return value.encode("utf-8", errors="surrogatepass")
