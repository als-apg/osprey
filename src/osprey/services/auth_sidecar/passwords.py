"""Password hashing and credential-generation tags for the auth sidecar.

Passwords are hashed with :func:`hashlib.scrypt` and stored as self-describing
strings::

    scrypt.<n>.<r>.<p>.<salt>.<hash>

Salt and hash are unpadded URL-safe base64. Carrying the cost parameters in the
stored string means they can be raised later without invalidating credentials
minted under the old cost: :func:`verify_password` always reads the parameters
back out of the stored string rather than assuming the current pins.

WHY ``.`` AND NOT THE CONVENTIONAL ``$``
----------------------------------------
The PHC-style ``$`` separator cannot survive the trip to the container. A stored
hash travels to the sidecar as a value in ``.env.auth``, which the rendered
compose overlay hands over as an ``env_file`` — and Docker Compose performs
``${VAR}`` interpolation on those values. Every ``$`` followed by a valid
identifier character is read as a variable reference and replaced with the empty
string, so ``scrypt$16384$8$1$<salt>$<key>`` arrives as ``scrypt$16384$8$1``:
the numeric fields survive (a digit cannot begin a variable name) and the salt
and key — random base64, usually starting with a letter — are silently eaten.
The credential then verifies against nothing and every login is refused.

``.`` is outside the base64url alphabet (``A-Za-z0-9-_``), so it cannot collide
with a salt or key, and no shell or compose layer ascribes meaning to it. The
separator is :data:`FIELD_SEP` rather than a literal so the writer and
:func:`_parse_stored` cannot drift apart.

The module also mints *credential-generation tags* — truncated one-way digests
of a stored-hash string. Session cookies are signed but not encrypted, so the
stored hash must never enter one; a tag lets the sidecar detect that a user's
password has been rotated (the tag no longer matches the digest of the current
stored hash) without keeping server-side session state.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets

SCHEME = "scrypt"
"""Identifier written as the first field of every stored hash."""

FIELD_SEP = "."
"""Separator between stored-hash fields.

Deliberately NOT ``$``: these strings travel to the sidecar through a compose
``env_file``, where ``$`` starts a variable reference and would take the salt
and key with it (see the module docstring). Must stay outside the base64url
alphabet so it cannot occur inside a salt or key.
"""

SCRYPT_N = 2**14
"""CPU/memory cost factor for newly minted hashes."""

SCRYPT_R = 8
"""Block size for newly minted hashes."""

SCRYPT_P = 1
"""Parallelisation factor for newly minted hashes."""

SCRYPT_MAXMEM = 64 * 1024 * 1024
"""Memory ceiling handed to OpenSSL, in bytes.

``hashlib.scrypt``'s default of ``maxmem=0`` leaves OpenSSL's own 32 MB cap in
place, which ``n=2**14, r=8`` already exceeds; without an explicit ceiling the
pinned cost factors raise instead of hashing.
"""

SALT_BYTES = 16
"""Length of the random salt drawn per hash."""

KEY_BYTES = 32
"""Length of the derived key."""

GENERATION_TAG_CHARS = 16
"""Number of hex characters kept from the generation-tag digest."""

_FIELD_COUNT = 6


def _b64encode(raw: bytes) -> str:
    """Encode bytes as unpadded URL-safe base64."""
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64decode(text: str) -> bytes:
    """Decode unpadded URL-safe base64.

    Raises:
        ValueError: If ``text`` is not valid base64.
    """
    padding = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + padding)


def hash_password(
    password: str,
    *,
    n: int = SCRYPT_N,
    r: int = SCRYPT_R,
    p: int = SCRYPT_P,
) -> str:
    """Hash a password into a self-describing stored-hash string.

    Args:
        password: The plaintext password. Must be non-empty.
        n: CPU/memory cost factor. Defaults to the module pin.
        r: Block size. Defaults to the module pin.
        p: Parallelisation factor. Defaults to the module pin.

    Returns:
        A string of the form ``scrypt.n.r.p.salt.hash``.

    Raises:
        ValueError: If ``password`` is empty.
    """
    if not password:
        raise ValueError("password must not be empty")

    salt = secrets.token_bytes(SALT_BYTES)
    derived = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=n,
        r=r,
        p=p,
        maxmem=SCRYPT_MAXMEM,
        dklen=KEY_BYTES,
    )
    return FIELD_SEP.join((SCHEME, str(n), str(r), str(p), _b64encode(salt), _b64encode(derived)))


def _parse_stored(stored: str) -> tuple[int, int, int, bytes, bytes]:
    """Split a stored-hash string into its cost parameters, salt and key.

    Args:
        stored: A string previously produced by :func:`hash_password`.

    Returns:
        A tuple of ``(n, r, p, salt, key)``.

    Raises:
        ValueError: If ``stored`` is not a well-formed scrypt hash string.
    """
    fields = stored.split(FIELD_SEP)
    if len(fields) != _FIELD_COUNT:
        raise ValueError(f"stored hash must have {_FIELD_COUNT} fields")

    scheme, n_text, r_text, p_text, salt_text, key_text = fields
    if scheme != SCHEME:
        raise ValueError(f"unsupported hash scheme: {scheme!r}")

    try:
        n, r, p = int(n_text), int(r_text), int(p_text)
    except ValueError as exc:
        raise ValueError("scrypt parameters must be integers") from exc
    if n < 2 or r < 1 or p < 1:
        raise ValueError("scrypt parameters out of range")

    salt, key = _b64decode(salt_text), _b64decode(key_text)
    if not salt or not key:
        raise ValueError("stored hash carries an empty salt or key")
    return n, r, p, salt, key


def verify_password(password: str, stored: str) -> bool:
    """Check a plaintext password against a stored-hash string.

    The cost parameters come from ``stored``, so hashes minted under an older
    cost keep verifying. Comparison is constant-time. A malformed or unusable
    ``stored`` value verifies as ``False`` rather than raising — the sidecar
    treats an unreadable credential as a failed login, not a crash.

    Args:
        password: The plaintext password to check.
        stored: A string previously produced by :func:`hash_password`.

    Returns:
        ``True`` if the password matches, ``False`` otherwise.
    """
    if not password:
        return False
    try:
        n, r, p, salt, key = _parse_stored(stored)
        candidate = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt,
            n=n,
            r=r,
            p=p,
            maxmem=SCRYPT_MAXMEM,
            dklen=len(key),
        )
    except (ValueError, MemoryError):
        return False
    return hmac.compare_digest(candidate, key)


def generation_tag(stored: str) -> str:
    """Derive the credential-generation tag for a stored-hash string.

    The tag is a truncated SHA-256 digest: one-way, so a cookie carrying it
    discloses nothing about the hash, and specific to one credential, so a
    rotated password yields a different tag.

    Args:
        stored: A stored-hash string as produced by :func:`hash_password`.

    Returns:
        A lowercase hex string of :data:`GENERATION_TAG_CHARS` characters.

    Raises:
        ValueError: If ``stored`` is empty.
    """
    if not stored:
        raise ValueError("stored hash must not be empty")
    digest = hashlib.sha256(stored.encode("utf-8")).hexdigest()
    return digest[:GENERATION_TAG_CHARS]


def verify_generation_tag(tag: str, stored: str) -> bool:
    """Check a session's generation tag against the current stored hash.

    Args:
        tag: The tag carried by the session, or an empty value.
        stored: The user's current stored-hash string.

    Returns:
        ``True`` if ``tag`` was derived from ``stored``, ``False`` otherwise —
        including when either argument is empty, so a password-mode session
        without a tag never authorises.
    """
    if not tag or not stored:
        return False
    return hmac.compare_digest(tag, generation_tag(stored))
