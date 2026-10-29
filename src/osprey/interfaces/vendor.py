# ruff: noqa: T201 -- the osprey vendor fetch reports its progress on stdout
"""Vendor asset management — CDN-by-default with opt-in offline bundling.

All vendor assets (JS libraries, CSS, fonts) are declared in
vendor_manifest.json, which pairs each asset with its CDN URL, a SHA256
checksum, and one or more local target directories. A target is either a
directory relative to this ``interfaces`` package, or an object
``{"dir": ..., "base": "package"}`` naming a directory relative to the whole
``osprey`` package — how a service's own panel (``services/<name>/panel/``)
receives a file it loads from ``vendor/`` beside its page.

By default, interfaces serve CDN URLs directly — no setup required.
Firewalled deployments (e.g. the LBL control room) opt into local bundles
by setting ``OSPREY_OFFLINE=1`` or ``offline: true`` in ``config.yml``,
then run ``osprey vendor fetch`` to populate the target dirs.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from osprey.apps.manifest import AppManifest, PackageVendorEntry, UrlVendorEntry

MANIFEST_PATH = Path(__file__).parent / "vendor_manifest.json"

#: The directory ``"interfaces"``-based targets resolve against by default.
INTERFACES_DIR = Path(__file__).parent

#: The directory ``"package"``-based targets resolve against by default: the
#: installed ``osprey`` package, wherever it was installed.
PACKAGE_DIR = INTERFACES_DIR.parent

#: The two bases a manifest target can name.
_BASES = ("interfaces", "package")

# `osprey vendor fetch` runs during image builds (OSPREY_OFFLINE=1), where a
# momentary DNS or route failure reaching the public CDN would otherwise abort
# the whole build. Retry the errors that a later attempt can plausibly fix.
_FETCH_ATTEMPTS = 3
_BACKOFF_SECONDS = 1.0

# 408 Request Timeout and 429 Too Many Requests are transient despite being 4xx.
_RETRYABLE_HTTP_STATUS = frozenset({408, 429})


def _is_cert_error(exc: BaseException) -> bool:
    """urllib wraps SSLCertVerificationError in URLError(reason=...)."""
    return isinstance(exc, ssl.SSLCertVerificationError) or (
        isinstance(exc, urllib.error.URLError)
        and isinstance(exc.reason, ssl.SSLCertVerificationError)
    )


def _is_retryable(exc: BaseException) -> bool:
    """True for failures a retry can plausibly fix.

    A bad cert, a 404, or a checksum mismatch will fail identically on every
    attempt — retrying those only slows the build down and buries the real
    error behind a delay.
    """
    if _is_cert_error(exc):
        return False
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in _RETRYABLE_HTTP_STATUS or 500 <= exc.code < 600
    # URLError covers DNS failure and ENETUNREACH; both subclass OSError, as do
    # socket.timeout (TimeoutError) and connection resets mid-body.
    return isinstance(exc, (urllib.error.URLError, OSError, http.client.HTTPException))


def _ssl_context(insecure: bool) -> ssl.SSLContext:
    """Build an SSL context, optionally skipping cert verification.

    Skipping verification is safe here because every download is checked
    against a SHA256 in the manifest — TLS adds confidentiality, not
    integrity, and the assets are public CDN files. Useful for corporate
    proxies (e.g. Squid) that intercept TLS with a self-signed CA.

    The verified route for those proxies is ``OSPREY_CA_BUNDLE``: a path to
    the CA bundle the proxy re-signs with, loaded as the context's ``cafile``.
    It exists as OSPREY's own knob so one spelling works wherever OSPREY
    builds an SSL context; unset, the default context still honors Python's
    ``SSL_CERT_FILE``, so either variable does the job here.
    """
    if insecure:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx
    cafile = os.environ.get("OSPREY_CA_BUNDLE")
    if cafile:
        try:
            return ssl.create_default_context(cafile=cafile)
        except (OSError, ssl.SSLError) as exc:
            raise RuntimeError(
                f"OSPREY_CA_BUNDLE points at an unusable CA bundle: {cafile}\n"
                f"  ({exc})\n"
                "  It must be a readable PEM file, e.g. the system bundle at\n"
                "  /etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem (RHEL-family)\n"
                "  or /etc/ssl/certs/ca-certificates.crt (Debian-family)."
            ) from exc
    return ssl.create_default_context()


def _load_manifest() -> dict[str, Any]:
    with open(MANIFEST_PATH) as f:
        return cast(dict[str, Any], json.load(f))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch_one(
    url: str,
    dest: Path,
    expected_sha256: str,
    insecure: bool = False,
    attempts: int = _FETCH_ATTEMPTS,
) -> None:
    """Download a single file and verify its checksum.

    Transient network failures are retried with exponential backoff; a cert
    error, an HTTP 404, or a checksum mismatch fails immediately.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, attempts + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "osprey-vendor/1.0"})
            with urllib.request.urlopen(req, timeout=60, context=_ssl_context(insecure)) as resp:
                data = resp.read()
            dest.write_bytes(data)
            break
        except Exception as exc:
            if dest.exists():
                dest.unlink()
            if _is_cert_error(exc):
                raise RuntimeError(
                    f"TLS cert verification failed fetching {url}: {exc}\n"
                    "  This usually means a corporate proxy (e.g. Squid) is "
                    "intercepting TLS.\n"
                    "  Fix:\n"
                    "    - Point at your CA bundle:\n"
                    "        OSPREY_CA_BUNDLE=/path/to/ca-bundle.pem osprey vendor fetch\n"
                    "      (SSL_CERT_FILE works too; OSPREY_CA_BUNDLE wins if both are set)\n"
                    "    - Or skip verification (safe — SHA256 is still checked):\n"
                    "        osprey vendor fetch --insecure   "
                    "(or OSPREY_VENDOR_INSECURE=1)"
                ) from exc
            if _is_retryable(exc):
                if attempt < attempts:
                    time.sleep(_BACKOFF_SECONDS * 2 ** (attempt - 1))
                    continue
                raise RuntimeError(
                    f"Failed to fetch {url} after {attempts} attempts: {exc}"
                ) from exc
            raise RuntimeError(f"Failed to fetch {url}: {exc}") from exc

    actual = _sha256(dest)
    if actual != expected_sha256:
        dest.unlink()
        raise RuntimeError(
            f"SHA256 mismatch for {dest.name}: expected {expected_sha256[:12]}…, got {actual[:12]}…"
        )


def _target_spec(target: str | dict[str, str]) -> tuple[str, str]:
    """A manifest target as ``(base, dir)``; a bare string is interfaces-based."""
    if isinstance(target, str):
        return "interfaces", target
    base = target.get("base", "interfaces")
    if base not in _BASES:
        raise ValueError(f"vendor manifest target {target!r} names an unknown base {base!r}")
    return base, target["dir"]


def _normalized(directory: str) -> str:
    return directory.strip().strip("/")


def _entries(manifest: dict[str, Any]) -> list[tuple[str, str, str, str, str]]:
    """Every file the manifest delivers, as ``(base, dir, filename, url, sha256)``."""
    entries: list[tuple[str, str, str, str, str]] = []
    for asset in manifest.get("assets", []):
        for target in asset["targets"]:
            base, directory = _target_spec(target)
            entries.append((base, directory, asset["filename"], asset["url"], asset["sha256"]))
    for bundle in manifest.get("font_bundles", []):
        base, directory = _target_spec(bundle["target"])
        for filename, sha256 in bundle["files"].items():
            entries.append((base, directory, filename, bundle["base_url"] + filename, sha256))
    return entries


def _select(
    entries: list[tuple[str, str, str, str, str]], targets: Sequence[str] | None
) -> list[tuple[str, str, str, str, str]]:
    """The entries whose target directory is one of ``targets``; all when ``None``.

    Directories compare without leading/trailing slashes. A requested directory
    no entry delivers to raises, so a misspelt filter fails the build instead of
    fetching nothing.
    """
    if targets is None:
        return entries
    wanted = {_normalized(t) for t in targets}
    known = {_normalized(directory) for _, directory, *_ in entries}
    unknown = sorted(wanted - known)
    if unknown:
        raise ValueError(
            f"unknown vendor target(s): {', '.join(unknown)}. "
            f"Known targets: {', '.join(sorted(known))}"
        )
    return [entry for entry in entries if _normalized(entry[1]) in wanted]


def fetch_all(
    base_dir: Path | None = None,
    quiet: bool = False,
    insecure: bool | None = None,
    targets: Sequence[str] | None = None,
    package_dir: Path | None = None,
) -> list[str]:
    """Download the vendor assets declared in the manifest.

    Idempotent: skips files that already exist with a matching checksum.
    Returns list of files that were actually downloaded (not skipped).

    ``insecure`` skips TLS cert verification (SHA256 still enforced). If
    None, falls back to the ``OSPREY_VENDOR_INSECURE`` env var.

    ``targets`` restricts the fetch to those target directories (as written in
    the manifest, slashes optional); ``None`` fetches every target. An unknown
    directory raises ``ValueError`` before anything is downloaded.

    ``base_dir`` and ``package_dir`` are where interfaces-based and
    package-based targets resolve; they default to :data:`INTERFACES_DIR` and
    :data:`PACKAGE_DIR`.
    """
    roots = {"interfaces": base_dir or INTERFACES_DIR, "package": package_dir or PACKAGE_DIR}
    if insecure is None:
        insecure = bool(os.environ.get("OSPREY_VENDOR_INSECURE"))

    selected = _select(_entries(_load_manifest()), targets)
    downloaded: list[str] = []
    for base, directory, filename, url, sha256 in selected:
        dest = roots[base] / directory / filename
        if dest.exists() and _sha256(dest) == sha256:
            continue
        if not quiet:
            print(f"  {filename} → {directory}")
        _fetch_one(url, dest, sha256, insecure=insecure)
        downloaded.append(str(dest))
    return downloaded


def verify_all(
    base_dir: Path | None = None, package_dir: Path | None = None
) -> tuple[list[str], list[str]]:
    """Verify all vendor assets exist and have correct checksums.

    Returns (ok_files, problems) where problems is a list of human-readable
    error strings. ``base_dir`` and ``package_dir`` resolve targets as in
    :func:`fetch_all`.
    """
    roots = {"interfaces": base_dir or INTERFACES_DIR, "package": package_dir or PACKAGE_DIR}
    ok: list[str] = []
    problems: list[str] = []
    for base, directory, filename, _url, sha256 in _entries(_load_manifest()):
        rel = str(Path(directory) / filename)
        dest = roots[base] / rel
        if not dest.exists():
            problems.append(f"MISSING: {rel}")
        elif _sha256(dest) != sha256:
            problems.append(f"CHECKSUM MISMATCH: {rel}")
        else:
            ok.append(str(dest))
    return ok, problems


class AppVendorError(RuntimeError):
    """An application's vendor file could not be put in place and verified.

    Attributes:
        app: The application whose ``vendor:`` entry failed.
        source: What the entry fetches: ``<package>@<version>/<file>`` for a
            package entry, the URL for a url entry.
    """

    def __init__(self, app: str, source: str, message: str) -> None:
        self.app = app
        self.source = source
        super().__init__(f"apps/{app}: vendor {source}: {message}")


def _entry_source(entry: UrlVendorEntry | PackageVendorEntry) -> str:
    from osprey.apps.manifest import PackageVendorEntry

    if isinstance(entry, PackageVendorEntry):
        return f"{entry.package}@{entry.version}/{entry.file}"
    return entry.source_url


def _app_vendor_dest(app_dir: Path, app: str, entry: UrlVendorEntry | PackageVendorEntry) -> Path:
    """Where an entry's file lands: ``apps/<app>/<into>``, refused outside the app dir.

    The path is resolved, symlinks included, before it is compared, so neither
    ``..`` nor a linked directory can carry a download out of the application.
    """
    into = entry.into
    outside = AppVendorError(app, _entry_source(entry), f"into {into!r} lands outside apps/{app}/")
    if Path(into).is_absolute():
        raise outside
    if into.endswith("/"):
        raise AppVendorError(
            app, _entry_source(entry), f"into {into!r} must name a file, not a directory"
        )
    root = app_dir.resolve()
    dest = (app_dir / into).resolve()
    if dest == root or not dest.is_relative_to(root):
        raise outside
    return dest


#: A version as a vendored file name spells it: dotted numbers, optionally
#: followed by a pre-release or build suffix (``1.6.30``, ``3.0.0-rc.1``).
_VERSION_TOKEN = re.compile(r"(?<![\d.])\d+(?:\.\d+)+(?:[-+][0-9A-Za-z.-]*[0-9A-Za-z])?")


def _names_version(name: str, version: str) -> bool:
    """Whether ``name`` spells ``version`` as a whole token.

    ``1.6.3`` is in ``uplot-1.6.3.min.js`` but not in ``uplot-11.6.3.min.js``,
    ``uplot-1.6.30.min.js`` or ``uplot-1.6.3.1.min.js``.
    """
    pattern = rf"(?<!\d)(?<!\d\.){re.escape(version)}(?!\d)(?!\.\d)"
    return re.search(pattern, name) is not None


def _check_versioned_name(app: str, entry: UrlVendorEntry | PackageVendorEntry) -> None:
    """Refuse an entry whose ``into`` file name does not carry the version it fetches.

    A panel's vendor files are served as immutable, so a new version must reach
    the browser under a new name, the convention of OSPREY's own ``/static/``
    vendor trees (``plotly-3.3.1.min.js``). A package entry's file name carries
    its ``version``; a url entry's carries a version its URL names.
    """
    from osprey.apps.manifest import PackageVendorEntry

    name = Path(entry.into).name
    if isinstance(entry, PackageVendorEntry):
        if _names_version(name, entry.version):
            return
        wanted = f"version {entry.version}"
    else:
        url_path = urllib.parse.urlsplit(entry.url).path
        versions = {m.group() for m in _VERSION_TOKEN.finditer(url_path)}
        if any(_names_version(name, v) for v in versions):
            return
        wanted = (
            "a version its URL names (" + ", ".join(sorted(versions)) + ")"
            if versions
            else "a version, and the URL names none"
        )
    raise AppVendorError(
        app,
        _entry_source(entry),
        f"into file name {name!r} must carry the version it fetches ({wanted}): "
        "a vendor file is cached as immutable, so a new version needs a new name",
    )


def fetch_app_vendor(
    deployment_root: Path,
    manifests: Sequence[AppManifest],
    *,
    offline: bool | None = None,
    insecure: bool | None = None,
    quiet: bool = False,
) -> list[str]:
    """Put every application's ``vendor:`` files in place, each verified by its SHA256.

    Each entry lands at ``apps/<name>/<into>`` under ``deployment_root``,
    fetched from the source it names (its ``url``, or a package file through
    the one CDN template). Every destination is checked before anything is
    downloaded; one outside the application's directory, or whose file name
    does not carry the version it fetches, is refused.

    A file already present with the entry's hash is used as it is, with no
    request. Otherwise an offline deployment, a failed download or a hash
    mismatch raises, so a panel is never built with a missing or unverified
    file.

    Args:
        deployment_root: The deployment directory holding ``apps/``.
        manifests: The applications to fetch for, as loaded by
            :func:`osprey.apps.loader.load`.
        offline: Whether the deployment is offline. ``None`` reads
            :func:`is_offline`; a caller holding the config passes
            :func:`offline_from_config`'s answer.
        insecure: Skip TLS cert verification (the hash is still checked).
            ``None`` reads ``OSPREY_VENDOR_INSECURE``.
        quiet: Suppress the per-file progress lines.

    Returns:
        The files actually downloaded (not those already verified).

    Raises:
        AppVendorError: An entry's target is outside its application or its
            file name lacks its version, or its file is missing or unverified and could not be fetched; the error
            names the application and the package (or URL).
    """
    if offline is None:
        offline = is_offline()
    if insecure is None:
        insecure = bool(os.environ.get("OSPREY_VENDOR_INSECURE"))

    plan = []
    for manifest in manifests:
        for entry in manifest.vendor:
            app_dir = deployment_root / "apps" / manifest.name
            dest = _app_vendor_dest(app_dir, manifest.name, entry)
            _check_versioned_name(manifest.name, entry)
            plan.append((manifest.name, entry, dest))
    downloaded: list[str] = []
    for app, entry, dest in plan:
        if dest.is_file() and _sha256(dest) == entry.sha256:
            continue
        source = _entry_source(entry)
        if offline:
            raise AppVendorError(
                app,
                source,
                f"no verified copy at apps/{app}/{entry.into} and the deployment is offline; "
                "fetch it once with network access, or place the file there",
            )
        if not quiet:
            print(f"  {source} → apps/{app}/{entry.into}")
        try:
            _fetch_one(entry.source_url, dest, entry.sha256, insecure=insecure)
        except (RuntimeError, OSError) as exc:
            raise AppVendorError(app, source, str(exc)) from exc
        downloaded.append(str(dest))
    return downloaded


_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}


def _offline_env() -> bool | None:
    """What ``OSPREY_OFFLINE`` says, or ``None`` when it says nothing.

    Split out so the environment layer is read the same way whether the config
    is loaded from disk (:func:`is_offline`) or already in the caller's hands
    (:func:`offline_from_config`).
    """
    env = os.environ.get("OSPREY_OFFLINE", "").strip().lower()
    if env in _TRUTHY:
        return True
    if env in _FALSY:
        return False
    return None


def offline_from_config(config: dict | None) -> bool:
    """:func:`is_offline`'s answer for a config the caller already holds.

    The deploy path resolves this before anything is running: an image is
    built from a config dict, and ``osprey vendor fetch`` runs INSIDE that
    build, where the deployment's ``config.yml`` is not on disk to be loaded.
    Reading the same two layers in the same order is what keeps the mode the
    image was built for and the mode it later serves in from disagreeing.

    :param config: The configuration to read ``offline`` from. ``None`` and an
        empty mapping both mean "the config says nothing".
    :return: Whether locally bundled vendor assets should be served.
    """
    override = _offline_env()
    if override is not None:
        return override
    return bool((config or {}).get("offline", False))


def is_offline() -> bool:
    """Return True if interfaces should serve locally bundled vendor assets.

    Resolution order:
      1. ``OSPREY_OFFLINE`` environment variable (truthy: 1/true/yes/on;
         falsy: 0/false/no/off)
      2. Top-level ``offline`` key in ``config.yml``

    Env var wins — a developer can flip modes per-shell without editing
    project config.
    """
    override = _offline_env()
    if override is not None:
        return override
    try:
        from osprey_connectors.workspace import load_osprey_config

        cfg = load_osprey_config() or {}
    except Exception:
        return False
    return offline_from_config(cfg)


def asset_cdn_url(name: str) -> str:
    """Return the CDN URL for the asset with the given manifest ``name``.

    Raises ``KeyError`` with a clear message if ``name`` is unknown, which
    catches typos at Jinja render time rather than producing broken ``src``.
    """
    for asset in _load_manifest().get("assets", []):
        if asset.get("name") == name:
            url: str = asset["url"]
            return url
    raise KeyError(
        f"Unknown vendor asset name: {name!r}. Check the 'name' fields in {MANIFEST_PATH}."
    )


def vendor_url(name: str, local_path: str) -> str:
    """Return ``local_path`` in offline mode, else the CDN URL for ``name``.

    ``local_path`` is taken as an explicit arg (rather than derived from the
    manifest) because assets like Plotly and highlight.js ship to multiple
    target dirs — the caller knows which one applies to this page.
    """
    return local_path if is_offline() else asset_cdn_url(name)


# A <script> tag whose src names a Plotly bundle: local or CDN, any version,
# minified or not.
_PLOTLY_SCRIPT_SRC = re.compile(
    rb"""<script\b[^>]*?\bsrc\s*=\s*["']?[^"'\s>]*plotly[^"'\s>]*\.js""",
    re.IGNORECASE,
)
# The banner every distributed plotly.js build opens with, kept when a page
# inlines the bundle (``to_html(include_plotlyjs=True)``).
_PLOTLY_INLINE_MARKER = b"plotly.js v"


def html_has_plotly_bundle(html: bytes) -> bool:
    """True when the page loads Plotly itself, by script tag or inlined bundle.

    :param html: Raw HTML bytes.
    :return: Whether a ``<script src=...plotly...js>`` tag or the inlined
        bundle's ``plotly.js v`` banner is present.
    """
    return _PLOTLY_INLINE_MARKER in html or _PLOTLY_SCRIPT_SRC.search(html) is not None


def html_needs_plotly(html: bytes) -> bool:
    """True when the page calls ``Plotly.`` but never loads the library.

    Such a page renders blank on its own, e.g. a figure exported with
    ``include_plotlyjs=False``.

    :param html: Raw HTML bytes.
    :return: Whether a Plotly bundle must be supplied for the page to render.
    """
    return b"Plotly." in html and not html_has_plotly_bundle(html)
