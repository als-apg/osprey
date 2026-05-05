"""Vendor asset management — CDN-by-default with opt-in offline bundling.

All vendor assets (JS libraries, CSS, fonts) are declared in
vendor_manifest.json, which pairs each asset with its CDN URL, a SHA256
checksum, and one or more local target directories.

By default, interfaces serve CDN URLs directly — no setup required.
Firewalled deployments (e.g. the LBL control room) opt into local bundles
by setting ``OSPREY_OFFLINE=1`` or ``offline: true`` in ``config.yml``,
then run ``osprey vendor fetch`` to populate the target dirs.
"""

from __future__ import annotations

import hashlib
import json
import os
import ssl
import urllib.error
import urllib.request
from pathlib import Path

MANIFEST_PATH = Path(__file__).parent / "vendor_manifest.json"


def _ssl_context(insecure: bool) -> ssl.SSLContext:
    """Build an SSL context, optionally skipping cert verification.

    Skipping verification is safe here because every download is checked
    against a SHA256 in the manifest — TLS adds confidentiality, not
    integrity, and the assets are public CDN files. Useful for corporate
    proxies (e.g. Squid) that intercept TLS with a self-signed CA.
    """
    if insecure:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx
    return ssl.create_default_context()


def _load_manifest() -> dict:
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch_one(url: str, dest: Path, expected_sha256: str, insecure: bool = False) -> None:
    """Download a single file and verify its checksum."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "osprey-vendor/1.0"})
        with urllib.request.urlopen(req, timeout=60, context=_ssl_context(insecure)) as resp:
            data = resp.read()
        dest.write_bytes(data)
    except Exception as exc:
        if dest.exists():
            dest.unlink()
        # urllib wraps SSLCertVerificationError in URLError(reason=...)
        cert_err = isinstance(exc, ssl.SSLCertVerificationError) or (
            isinstance(exc, urllib.error.URLError)
            and isinstance(exc.reason, ssl.SSLCertVerificationError)
        )
        if cert_err:
            raise RuntimeError(
                f"TLS cert verification failed fetching {url}: {exc}\n"
                "  This usually means a corporate proxy (e.g. Squid) is "
                "intercepting TLS.\n"
                "  Fix:\n"
                "    - Point at your CA bundle:\n"
                "        SSL_CERT_FILE=/path/to/ca-bundle.pem osprey vendor fetch\n"
                "    - Or skip verification (safe — SHA256 is still checked):\n"
                "        osprey vendor fetch --insecure   "
                "(or OSPREY_VENDOR_INSECURE=1)"
            ) from exc
        raise RuntimeError(f"Failed to fetch {url}: {exc}") from exc
    actual = _sha256(dest)
    if actual != expected_sha256:
        dest.unlink()
        raise RuntimeError(
            f"SHA256 mismatch for {dest.name}: expected {expected_sha256[:12]}…, got {actual[:12]}…"
        )


def fetch_all(
    base_dir: Path | None = None, quiet: bool = False, insecure: bool | None = None
) -> list[str]:
    """Download all vendor assets declared in the manifest.

    Idempotent: skips files that already exist with a matching checksum.
    Returns list of files that were actually downloaded (not skipped).

    ``insecure`` skips TLS cert verification (SHA256 still enforced). If
    None, falls back to the ``OSPREY_VENDOR_INSECURE`` env var.
    """
    if base_dir is None:
        base_dir = Path(__file__).parent
    if insecure is None:
        insecure = bool(os.environ.get("OSPREY_VENDOR_INSECURE"))

    manifest = _load_manifest()
    downloaded: list[str] = []

    for asset in manifest.get("assets", []):
        for target in asset["targets"]:
            dest = base_dir / target / asset["filename"]
            if dest.exists() and _sha256(dest) == asset["sha256"]:
                continue
            if not quiet:
                print(f"  {asset['filename']} → {target}")
            _fetch_one(asset["url"], dest, asset["sha256"], insecure=insecure)
            downloaded.append(str(dest))

    for bundle in manifest.get("font_bundles", []):
        base_url = bundle["base_url"]
        target_dir = base_dir / bundle["target"]
        for filename, sha256 in bundle["files"].items():
            dest = target_dir / filename
            if dest.exists() and _sha256(dest) == sha256:
                continue
            if not quiet:
                print(f"  {filename} → {bundle['target']}")
            _fetch_one(base_url + filename, dest, sha256, insecure=insecure)
            downloaded.append(str(dest))

    return downloaded


def verify_all(base_dir: Path | None = None) -> tuple[list[str], list[str]]:
    """Verify all vendor assets exist and have correct checksums.

    Returns (ok_files, problems) where problems is a list of human-readable
    error strings.
    """
    if base_dir is None:
        base_dir = Path(__file__).parent

    manifest = _load_manifest()
    ok: list[str] = []
    problems: list[str] = []

    for asset in manifest.get("assets", []):
        for target in asset["targets"]:
            rel = str(Path(target) / asset["filename"])
            dest = base_dir / rel
            if not dest.exists():
                problems.append(f"MISSING: {rel}")
            elif _sha256(dest) != asset["sha256"]:
                problems.append(f"CHECKSUM MISMATCH: {rel}")
            else:
                ok.append(str(dest))

    for bundle in manifest.get("font_bundles", []):
        target_dir = base_dir / bundle["target"]
        for filename, sha256 in bundle["files"].items():
            rel = str(Path(bundle["target"]) / filename)
            dest = target_dir / filename
            if not dest.exists():
                problems.append(f"MISSING: {rel}")
            elif _sha256(dest) != sha256:
                problems.append(f"CHECKSUM MISMATCH: {rel}")
            else:
                ok.append(str(dest))

    return ok, problems


_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}


def is_offline() -> bool:
    """Return True if interfaces should serve locally bundled vendor assets.

    Resolution order:
      1. ``OSPREY_OFFLINE`` environment variable (truthy: 1/true/yes/on;
         falsy: 0/false/no/off)
      2. Top-level ``offline`` key in ``config.yml``

    Env var wins — a developer can flip modes per-shell without editing
    project config.
    """
    env = os.environ.get("OSPREY_OFFLINE", "").strip().lower()
    if env in _TRUTHY:
        return True
    if env in _FALSY:
        return False
    try:
        from osprey.utils.workspace import load_osprey_config

        cfg = load_osprey_config() or {}
    except Exception:
        return False
    return bool(cfg.get("offline", False))


def asset_cdn_url(name: str) -> str:
    """Return the CDN URL for the asset with the given manifest ``name``.

    Raises ``KeyError`` with a clear message if ``name`` is unknown, which
    catches typos at Jinja render time rather than producing broken ``src``.
    """
    for asset in _load_manifest().get("assets", []):
        if asset.get("name") == name:
            return asset["url"]
    raise KeyError(
        f"Unknown vendor asset name: {name!r}. Check the 'name' fields in "
        "src/osprey/interfaces/vendor_manifest.json."
    )


def vendor_url(name: str, local_path: str) -> str:
    """Return ``local_path`` in offline mode, else the CDN URL for ``name``.

    ``local_path`` is taken as an explicit arg (rather than derived from the
    manifest) because assets like Plotly and highlight.js ship to multiple
    target dirs — the caller knows which one applies to this page.
    """
    return local_path if is_offline() else asset_cdn_url(name)
